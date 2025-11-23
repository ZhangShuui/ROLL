# drop-in replacement with step control & blending
import json
import re
from functools import lru_cache
from typing import Optional, Union, Dict, List, Any, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from openai import OpenAI
from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.factory import create_strategy
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import (
    default_reward_model_provider,
    default_tokenizer_provider,
)
from roll.utils.context_managers import state_offload_manger
from roll.utils.prompt import *

QUESTION_FILE = "/project/hdtaccuracy/Personality-Alignment/choice_ver/v10/merged_choice_questions.json"


class PerAlignChoiceSpecificSentenceRewardWorker(Worker):
    """
    GSSPO Reward Worker with STEP CONTROL:
      - Early steps: guided (binary criteria + sentence evidence -> token rewards), scaled by correctness
      - Late steps: default reward (no LLM judge). Strategies:
          * zero: no token reward
          * uniform_answer_scaled: uniform token reward scaled by (correct/incorrect)
          * uniform_bank_score_scaled: uniform token reward scaled by normalized bank score of chosen option
      - Optional linear BLEND across a window: token = alpha * guided + (1 - alpha) * default
      - (Optional) emit scores in default phase for group-relative advantages (off by default)
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

        # ===== Judge config =====
        self.judge_prompt = getattr(self.worker_config, "judge_prompt", None)
        if self.judge_prompt and self.judge_prompt in prompt_maps:
            self.judge_prompt = prompt_maps[self.judge_prompt]
        self.judge_model_type = getattr(self.worker_config, "judge_model_type", "api")
        self.judge_model_name = getattr(self.worker_config, "judge_model_name", None)
        self.judge_api_url = getattr(self.worker_config, "judge_api_url", None)
        self.judge_api_key = getattr(self.worker_config, "judge_api_key", None)

        # ===== New options =====
        # 是否允许负分（-1 指明显违规/反面证据），否则只 - 0/1
        self.allow_negative_scores = bool(getattr(self.worker_config, "allow_negative_scores", True))
        # judge_score_target: "both"|"correct"|"incorrect" - 决定只对哪些样本调用 judge
        self.judge_score_target = getattr(self.worker_config, "judge_score_target", "both")
        # batch-level normalization: None|'minmax'|'zscore'
        self.batch_score_normalization = getattr(self.worker_config, "batch_score_normalization", None)

        # ===== Token reward (guided) =====
        self.sentence_reward_base = float(getattr(self.worker_config, "sentence_reward_value", 0.1))
        self.overlap_aggregation = getattr(self.worker_config, "overlap_aggregation", "sum")  # 'max'|'sum'
        self.smoothing_radius = int(getattr(self.worker_config, "smoothing_radius", 0))  # 0=no smoothing
        self.normalize_mode = getattr(
            self.worker_config, "normalize_mode", "none"
        )  # 'none'|'by_length'|'by_num_sentences'
        self.cap_per_token = float(getattr(self.worker_config, "cap_per_token", 1.0))
        self.max_matches_per_sentence = int(getattr(self.worker_config, "max_matches_per_sentence", 0))  # 0=all
        self.scale_by_num_criteria = bool(getattr(self.worker_config, "scale_by_num_criteria", True))

        # ===== Correctness scaling (shared by guided & default strategies that need it) =====
        self.correct_reward_scale = float(getattr(self.worker_config, "correct_reward_scale", 1.0))
        self.incorrect_reward_scale = float(getattr(self.worker_config, "incorrect_reward_scale", 0.0))

        # ===== STEP CONTROL =====
        # guided 在 [0, guidance_until_step) 有效；若 blend_span>0，则在 [guidance_until_step, guidance_until_step+blend_span) 线性从1衰减到0
        self.guidance_until_step = int(getattr(self.worker_config, "guidance_until_step", 5000))
        self.guidance_blend_span = int(getattr(self.worker_config, "guidance_blend_span", 0))  # 0=no blend
        # 默认阶段 token 奖励策略：
        #  - "zero"                      : 全零
        #  - "uniform_answer_scaled"     : 全段统一值 = sentence_reward_base * (correct/incorrect scale)
        #  - "uniform_bank_score_scaled" : 全段统一值 = sentence_reward_base * norm_bank_score
        self.default_token_reward_strategy = getattr(self.worker_config, "default_token_reward_strategy", "zero")
        # 是否在默认阶段回传 scores（例如给 GRPO 的组内相对优势使用）
        self.emit_scores_in_default_phase = bool(getattr(self.worker_config, "emit_scores_in_default_phase", False))
        # 如果启用 bank score，是否按该题目所有选项的 min-max 归一化到 [0,1]
        self.bank_score_minmax_norm = bool(getattr(self.worker_config, "bank_score_minmax_norm", True))

        # ===== Compatibility: by default we don't emit response-level outputs =====
        self.return_response_level = bool(getattr(self.worker_config, "return_response_level", True))
        self.return_scores = bool(getattr(self.worker_config, "return_scores", False))

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        super().initialize(pipeline_config=pipeline_config)
        if self.judge_model_type == "api":
            self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
            self.logger.info(f"{self.worker_name} initialized with API model")
            self.logger.info(f"  model_name: {self.judge_model_name}")
            self.logger.info(f"  tokenizer path: {self.worker_config.model_args.model_name_or_path}")
        elif self.judge_model_type == "inference":
            if self.strategy is None:
                self.strategy = create_strategy(self)
                self.strategy.initialize(model_provider=default_reward_model_provider)
                self.tokenizer = self.strategy.tokenizer
            self.strategy.offload_states()
        else:
            raise ValueError(f"Unsupported judge model type: {self.judge_model_type}")

    # ===== question bank =====
    @staticmethod
    @lru_cache(maxsize=1)
    def _load_question_bank():
        bank = {}
        try:
            with open(QUESTION_FILE, "r", encoding="utf-8") as f:
                try:
                    # 尝试读取整个JSON文件
                    json_content = json.load(f)
                    if isinstance(json_content, list):
                        questions = json_content
                    elif isinstance(json_content, dict) and "questions" in json_content:
                        questions = json_content["questions"]
                    else:
                        questions = [json_content]

                    for obj in questions:
                        if not isinstance(obj, dict):
                            continue
                        qid = obj.get("qid")
                        if not qid:
                            continue

                        entry = {"gold": None, "scores": {}, "choice_texts": {}}

                        # 获取正确答案
                        gold = obj.get("correct_answer")  # 你的数据中是 "correct_answer"
                        entry["gold"] = gold

                        # 处理choices数组
                        choices = obj.get("choices", [])
                        if isinstance(choices, list):
                            for choice in choices:
                                if not isinstance(choice, dict):
                                    continue

                                label = choice.get("label")
                                if not label:
                                    continue

                                label_upper = label.upper()

                                # 获取分数
                                try:
                                    score = float(choice.get("score", 0.0))
                                except (TypeError, ValueError):
                                    score = 0.0
                                entry["scores"][label_upper] = score

                                # 获取选项文本
                                text = choice.get("text", "")
                                if text:
                                    entry["choice_texts"][label_upper] = text

                        bank[qid] = entry

                except json.JSONDecodeError:
                    # 如果不是标准JSON，尝试按行读取
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            qid = obj.get("qid")
                            if not qid:
                                continue

                            entry = {"gold": None, "scores": {}, "choice_texts": {}}

                            # 获取正确答案
                            gold = obj.get("correct_answer")
                            entry["gold"] = gold

                            # 处理choices数组
                            choices = obj.get("choices", [])
                            if isinstance(choices, list):
                                for choice in choices:
                                    if not isinstance(choice, dict):
                                        continue

                                    label = choice.get("label")
                                    if not label:
                                        continue

                                    label_upper = label.upper()

                                    # 获取分数
                                    try:
                                        score = float(choice.get("score", 0.0))
                                    except (TypeError, ValueError):
                                        score = 0.0
                                    entry["scores"][label_upper] = score

                                    # 获取选项文本
                                    text = choice.get("text", "")
                                    if text:
                                        entry["choice_texts"][label_upper] = text

                            bank[qid] = entry

                        except json.JSONDecodeError:
                            continue

        except FileNotFoundError:
            pass
        return bank

    # ===== parse model's chosen option =====
    @staticmethod
    def _normalize_choice(text: str):
        if not text:
            return None
        t_up = (text or "").strip().upper()
        # 不忽略 THINK，仅去除标签外观，保留文本
        t_up = re.sub(r"<[^>]+>", " ", t_up)
        t_up = re.sub(r"\b(ASSISTANT|SYSTEM|USER)\b", " ", t_up)
        m = re.search(r"/CHOICE\{([ABCDEFG])\}", t_up)
        if m:
            return m.group(1)
        patterns = [
            r"([ABCDEFG])\s*$",
            r'(?:^|\s)(?:ANSWER|OPTION|CHOICE)\s*[:\-]?\s*["\'`(]*\b([ABCDEFG])\b',
            r'^\s*["\'`(]*([ABCDEFG])(?:[\).\s]|$)',
            r"\b([ABCDEFG])\b",
        ]
        for p in patterns:
            m = re.search(p, t_up)
            if m:
                return m.group(1)
        letters = re.findall(r"[ABCDEFG]", t_up)
        if len(letters) == 1:
            return letters[0]
        return None

    @staticmethod
    def _norm_letter(x: Optional[str]) -> Optional[str]:
        if not x:
            return None
        x = str(x).strip().upper()
        return x[0] if x and x[0] in "ABCDEFG" else None

    # ===== judge prompt (with CORRECT_OPTION for context) =====
    def _format_judge_prompt(
        self, prompt: str, response: str, correct_label: Optional[str], correct_text: Optional[str]
    ) -> List[Dict]:
        if not self.judge_prompt:
            co_str = "UNKNOWN"
            if correct_label:
                co_str = f"{correct_label}" + (f": {correct_text}" if correct_text else "")
            if not correct_text:
                self.logger.warning(f"Missing correct_text for correct_label={correct_label}")
            # NOTE: allow negative scores. Judge should output -1/0/1 for each criterion:
            # -1 => explicit violation (penalize), 0 => none, 1 => positive evidence
            formatted_prompt = f"""
    You are an evaluator for a next-turn prediction task.

    GOAL
    Given a PERSONA + multi-turn DIALOGUE (in PROMPT), a list of OPTIONS (for context), the CORRECT_OPTION (for context), and a MODEL_OUTPUT (may include <THINK>...</THINK> plus the final choice), evaluate the MODEL_OUTPUT against specific CRITERIA.

    INSTRUCTIONS
    1) For each criterion below, output 0 or 1 only (binary evaluation)
    2) For each criterion with score=1, provide exact sentence evidence from MODEL_OUTPUT:
    - Extract exact sentences that demonstrate the criterion
    - Extracted sentences should be some analysis or reasoning in the MODEL_OUTPUT that supports your score, don't extract sentences from the PROMPT or OPTIONS or some generic sentence that doesn't support your score
    - Use "ALL" if the entire response satisfies the criterion
    - Provide multiple sentences if applicable
    3) Focus on quality assessment, NOT correctness of the final choice
    4) Consider ALL text in MODEL_OUTPUT including reasoning in <THINK> tags
    5) Output valid JSON only, no additional text

    CRITERIA (can be -1/0/1)
    [1] Persona Consistency: Does the response maintain the character's personality, values, and behavioral patterns?
    [2] Conversational Coherence: Is the response contextually appropriate and follows naturally from the dialogue?
    [3] Tone & Style Match: Does the response match the expected communication style and emotional tone?
    [4] Contextual Grounding: Does the response show understanding of the specific situation and context?
    [5] Affective Alignment: Does the response appropriately reflect emotions and stance consistent with the persona?

    OUTPUT FORMAT (JSON only):
    {{
    "criteria": [
        {{"id": 1, "score": -1, 0 or 1, "sentences": [ "Exact sentence from MODEL_OUTPUT", "Another sentence", ... ]}},
        {{"id": 2, "score": -1. 0 or 1, "sentences": [ ... ]}},
        {{"id": 3, "score": -1, 0 or 1, "sentences": [ ... ]}},
        {{"id": 4, "score": -1, 0 or 1, "sentences": [ ... ]}},
        {{"id": 5, "score": -1, 0 or 1, "sentences": [ ... ]}}
    ],
    "explanation": "Optional brief explanation"
    }}

    CONTEXT
    CORRECT_OPTION (for reference): {co_str}

    PROMPT (PERSONA + DIALOGUE + OPTIONS):
    {prompt}

    MODEL_OUTPUT (to evaluate):
    {response}
    """.strip()
        else:
            formatted_prompt = self.judge_prompt.format(
                PROMPT=prompt,
                RESPONSE=response,
                CORRECT_LABEL=correct_label or "UNKNOWN",
                CORRECT_TEXT=correct_text or "",
            )
        return [{"role": "user", "content": formatted_prompt}]

    def _run_local_inference(self, messages: List[Dict]) -> str:
        if not self.strategy:
            raise ValueError("Strategy not initialized for local inference")
        from roll.datasets.chat_template import get_chat_template

        template_name = "qwen3_nothink"  # fixed for judge
        chat_template_func = get_chat_template(template_name, self.tokenizer)
        text = chat_template_func(messages)
        tokenized = self.tokenizer(text, return_tensors="pt")
        input_ids = tokenized["input_ids"].to("cuda")
        attention_mask = tokenized["attention_mask"].to("cuda")
        generation_config = self.strategy.worker_config.generating_args.to_dict()
        generation_config["eos_token_id"] = [self.tokenizer.eos_token_id]
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id

        # 创建 DataProto 时添加必要的 meta_info
        batch_size = input_ids.size(0)
        infer_batch_size = self.worker_config.infer_batch_size or batch_size
        micro_batch_size = min(batch_size, infer_batch_size)  # 对于单个推理，使用1作为micro_batch_size

        data = DataProto.from_dict(
            tensors={"input_ids": input_ids, "attention_mask": attention_mask},
            meta_info={"micro_batch_size": micro_batch_size},
        ).to("cuda")

        with torch.no_grad():
            output = self.strategy.generate(batch=data, generation_config=generation_config)
            if isinstance(output, torch.Tensor):
                generate_ids = output[:, len(input_ids[0]) :]
            else:
                generate_ids = output.batch["input_ids"][:, len(input_ids[0]) :]
        return self.tokenizer.decode(generate_ids[0], skip_special_tokens=True).strip()

    def _extract_criteria_result(self, llm_response: str) -> Dict[str, Any]:
        """提取LLM响应中的criteria结果，适配新的JSON schema"""
        try:
            start_idx = llm_response.find("{")
            end_idx = llm_response.rfind("}") + 1
            if start_idx == -1 or end_idx == 0:
                return self._fallback_line_parsing(llm_response)

            json_str = llm_response[start_idx:end_idx]
            data = json.loads(json_str)

            # 验证基本结构
            if not isinstance(data, dict):
                return self._fallback_line_parsing(llm_response)

            data.setdefault("criteria", [])
            data.setdefault("explanation", "")
            data["_compat_fallback"] = False

            # 处理criteria数组
            processed_criteria = []
            sentence_items = []  # 用于兼容现有的sentence_items格式

            for i, criterion in enumerate(data.get("criteria", [])):
                if not isinstance(criterion, dict):
                    continue

                # 标准化criterion
                processed_criterion = {
                    "id": criterion.get("id", i + 1),
                    "score": self._normalize_score(criterion.get("score", 0)),
                    "sentences": criterion.get("sentences", []),
                }
                processed_criteria.append(processed_criterion)

                # 如果该criterion得分为1，提取其sentences用于token reward计算
                if processed_criterion["score"] == 1:
                    sentences = processed_criterion["sentences"]
                    if isinstance(sentences, list):
                        for sentence in sentences:
                            if isinstance(sentence, str) and sentence.strip():
                                sentence_items.append(
                                    {
                                        "text": sentence.strip(),
                                        "criteria": [processed_criterion["id"]],  # 记录来源criterion
                                    }
                                )

            data["criteria"] = processed_criteria
            data["sentences"] = sentence_items  # 保持与现有代码的兼容性

            return data

        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON decode failed: {e}, attempting line-by-line parsing")
            self.logger.info(f"LLM response was: {llm_response}")
            return self._fallback_line_parsing(llm_response)
        except Exception as e:
            self.logger.error(f"Failed to extract criteria result: {e}, attempting line-by-line parsing")
            self.logger.info(f"LLM response was: {llm_response}")
            return self._fallback_line_parsing(llm_response)

    def _fallback_line_parsing(self, llm_response: str) -> Dict[str, Any]:
        """回退方案：按行解析提取criteria信息"""
        try:
            lines = llm_response.split("\n")
            criteria = []
            sentence_items = []
            explanation = ""

            current_criterion = None
            in_sentences_array = False
            current_sentences = []

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # 移除行末逗号
                line_clean = line.rstrip(",").strip()

                # 尝试提取criterion信息
                # 匹配 {"id": 1, "score": 1, "sentences": [...]}
                if line_clean.startswith('{"id"') or line_clean.startswith('{"id"'):
                    try:
                        # 尝试直接解析整行
                        criterion_data = json.loads(line_clean)
                        if isinstance(criterion_data, dict):
                            criterion = {
                                "id": criterion_data.get("id", len(criteria) + 1),
                                "score": self._normalize_score(criterion_data.get("score", 0)),
                                "sentences": criterion_data.get("sentences", []),
                            }
                            criteria.append(criterion)

                            # 添加到sentence_items
                            if criterion["score"] == 1:
                                sentences = criterion["sentences"]
                                if isinstance(sentences, list):
                                    for sentence in sentences:
                                        if isinstance(sentence, str) and sentence.strip():
                                            sentence_items.append(
                                                {"text": sentence.strip(), "criteria": [criterion["id"]]}
                                            )
                            continue
                    except json.JSONDecodeError:
                        pass

                # 匹配单独的字段
                # 提取 id
                id_match = re.search(r'"id":\s*(\d+)', line)
                score_match = re.search(r'"score":\s*(-?\d+\.?\d*)', line)

                if id_match and score_match:
                    criterion_id = int(id_match.group(1))
                    try:
                        s_val = float(score_match.group(1))
                    except Exception:
                        s_val = 0.0
                    criterion_score = self._normalize_score(s_val)

                    # 查找sentences
                    sentences_match = re.search(r'"sentences":\s*\[(.*?)\]', line)
                    sentences = []
                    if sentences_match:
                        sentences_content = sentences_match.group(1)
                        # 简单的句子提取（处理引号内的内容）
                        sentence_matches = re.findall(r'"([^"]*)"', sentences_content)
                        sentences = [s.strip() for s in sentence_matches if s.strip()]

                    criterion = {"id": criterion_id, "score": criterion_score, "sentences": sentences}
                    criteria.append(criterion)

                    # 添加到sentence_items
                    if criterion_score == 1 and sentences:
                        for sentence in sentences:
                            if sentence.strip():
                                sentence_items.append({"text": sentence.strip(), "criteria": [criterion_id]})
                    continue

                # 尝试单独提取分数信息
                score_only_match = re.search(r'"score":\s*(-?\d+\.?\d*)', line)
                if score_only_match and not criteria:
                    # 如果还没有criteria，创建一个基础的
                    try:
                        score = self._normalize_score(float(score_only_match.group(1)))
                    except Exception:
                        score = 0
                    criterion = {"id": 1, "score": score, "sentences": ["ALL"] if score == 1 else []}
                    criteria.append(criterion)

                    if score == 1:
                        sentence_items.append({"text": "ALL", "criteria": [1]})

                # 提取explanation
                if '"explanation"' in line:
                    exp_match = re.search(r'"explanation":\s*"([^"]*)"', line)
                    if exp_match:
                        explanation = exp_match.group(1)

            # 如果没有提取到任何criteria，创建默认的
            if not criteria:
                self.logger.warning("No criteria found in line parsing, creating default fallback")
                # 尝试查找任何数字作为分数
                numbers = re.findall(r"-?\d+\.?\d*", llm_response)
                if numbers:
                    for i, num in enumerate(numbers[:5]):  # 最多5个criteria
                        try:
                            score = self._normalize_score(float(num))
                        except Exception:
                            score = 0
                        criterion = {"id": i + 1, "score": score, "sentences": ["ALL"] if score == 1 else []}
                        criteria.append(criterion)

                        if score == 1:
                            sentence_items.append({"text": "ALL", "criteria": [i + 1]})

            # 如果仍然没有criteria，创建一个全0的默认结果
            if not criteria:
                for i in range(5):  # 5个criteria
                    criteria.append({"id": i + 1, "score": 0, "sentences": []})

            result = {
                "criteria": criteria,
                "sentences": sentence_items,
                "explanation": explanation or "Extracted via line parsing fallback",
                "_compat_fallback": True,
                "llm_response": llm_response,
            }

            self.logger.info(f"Line parsing extracted {len(criteria)} criteria, {len(sentence_items)} sentences")
            return result

        except Exception as e:
            self.logger.error(f"Line parsing also failed: {e}")
            return self._create_fallback_result(f"Both JSON and line parsing failed: {str(e)}")

    def _normalize_score(self, score) -> int:
        """标准化分数为0或1"""
        try:
            score_val = float(score)
            if score_val <= -0.5:
                if self.allow_negative_scores:
                    return -1
                else:
                    return 0
            elif score_val >= 0.5:
                return 1
            return 0
        except (TypeError, ValueError, AttributeError):
            return 0

    def _create_fallback_result(self, reason: str) -> Dict[str, Any]:
        """创建fallback结果"""
        return {
            "criteria": [{"id": i + 1, "score": 0, "sentences": []} for i in range(5)],
            "sentences": [],
            "explanation": f"Fallback result: {reason}",
            "_compat_fallback": True,
            "llm_response": "",
        }

    def _get_llm_judgment(
        self, prompt: str, response: str, correct_label: Optional[str], correct_text: Optional[str]
    ) -> Dict:
        messages = self._format_judge_prompt(prompt, response, correct_label, correct_text)
        self.logger.info(f"batch size={len(messages)} messages to judge with LLM")
        if self.judge_model_type == "api":
            llm_response = self._call_api_model(messages)
        elif self.judge_model_type == "inference":
            llm_response = self._run_local_inference(messages)
        else:
            raise ValueError(f"Unsupported judge model type: {self.judge_model_type}")
        out = self._extract_criteria_result(llm_response)
        out["llm_response"] = llm_response
        return out

    # ===== utils =====
    @staticmethod
    def _find_all_subsequences(subseq: List[int], seq: List[int], max_matches: int = 0) -> List[Tuple[int, int]]:
        matches = []
        if not subseq or not seq or len(subseq) > len(seq):
            return matches
        first = subseq[0]
        i = 0
        Ls = len(subseq)
        while i <= len(seq) - Ls:
            if seq[i] == first and seq[i : i + Ls] == subseq:
                matches.append((i, i + Ls))
                if max_matches and len(matches) >= max_matches:
                    break
                i += Ls
            else:
                i += 1
        return matches

    def _smooth_rewards(self, rewards_1d: torch.Tensor, radius: int) -> torch.Tensor:
        if radius <= 0:
            return rewards_1d
        k = 2 * radius + 1
        kernel = torch.ones(1, 1, k, device=rewards_1d.device) / float(k)
        x = rewards_1d.view(1, 1, -1)
        y = F.pad(x, (radius, radius), mode="replicate")
        return F.conv1d(y, kernel).view(-1)

    # ===== guided path: sentence evidence -> token rewards =====
    def _compute_guided_token_rewards(
        self,
        response_tokens_1d: torch.Tensor,
        sentence_items: List[Dict[str, Any]],
        response_text: str,
        answer_scale: float,
        metrics: Dict,
    ) -> torch.Tensor:
        device = response_tokens_1d.device
        token_rewards = torch.zeros_like(response_tokens_1d, dtype=torch.float32, device=device)
        resp_ids = response_tokens_1d.tolist()

        # 统计变量
        n_all = 0
        hit_spans_tokenizer = 0  # tokenizer匹配成功的句子数
        hit_spans_fuzzy = 0  # 模糊匹配成功的句子数
        fallback_uniform_count = 0  # 使用uniform fallback的次数
        total_matched_tokens = 0
        has_any_match = False

        # 为模糊匹配准备：预先计算每个token的文本和字符位置
        token_texts = []
        token_char_spans = []
        char_pos = 0

        for tid in resp_ids:
            token_text = self.tokenizer.decode([tid], skip_special_tokens=False)
            token_texts.append(token_text)

            # 尝试在response_text中找到这个token
            idx = response_text.find(token_text, char_pos)
            if idx >= 0:
                token_char_spans.append((idx, idx + len(token_text)))
                char_pos = idx + len(token_text)
            else:
                # 找不到时使用上一个位置
                token_char_spans.append((char_pos, char_pos))

        self.logger.debug(f"[TOKEN MATCH] Prepared {len(token_char_spans)} token char spans for fuzzy matching")

        for s_idx, s in enumerate(sentence_items):
            text = (s.get("text") or "").strip()
            crit_list = s.get("criteria", [])
            if not text:
                continue

            n_all += 1
            scale = max(1, len(crit_list)) if self.scale_by_num_criteria else 1
            value = float(self.sentence_reward_base) * float(scale) * float(answer_scale)

            self.logger.debug(
                f"[TOKEN MATCH] Processing sentence {s_idx+1}/{len(sentence_items)}: "
                f"'{text[:50]}...' (len={len(text)}, criteria={crit_list}, value={value:.4f})"
            )

            # ALL => 全覆盖
            if text.upper() == "ALL":
                if self.overlap_aggregation == "sum":
                    token_rewards += value
                else:
                    # ✅ 修复: 直接赋值,不用 maximum
                    token_rewards.fill_(value)
                has_any_match = True
                self.logger.debug(f"[TOKEN MATCH] Sentence {s_idx+1}: ALL - applying to all tokens")
                continue

            # ===== 策略1: 优先使用tokenizer匹配 =====
            spans: List[Tuple[int, int]] = []
            sent_ids = self.tokenizer.encode(text, add_special_tokens=False)

            if sent_ids:
                self.logger.debug(
                    f"[TOKEN MATCH] Sentence {s_idx+1}: Tokenized to {len(sent_ids)} tokens, "
                    f"searching in {len(resp_ids)} response tokens"
                )
                spans = self._find_all_subsequences(sent_ids, resp_ids, max_matches=self.max_matches_per_sentence)

            if spans:
                # ===== Tokenizer匹配成功 =====
                has_any_match = True
                hit_spans_tokenizer += 1
                matched_token_count = 0

                for span_idx, (st, ed) in enumerate(spans):
                    if self.overlap_aggregation == "sum":
                        token_rewards[st:ed] += value
                    else:
                        # ✅ 修复: 使用 clamp_min 或直接赋值
                        current_values = token_rewards[st:ed]
                        token_rewards[st:ed] = torch.maximum(current_values, torch.tensor(value, device=device))
                    matched_token_count += ed - st

                total_matched_tokens += matched_token_count
                self.logger.debug(
                    f"[TOKEN MATCH] Sentence {s_idx+1}: TOKENIZER SUCCESS - "
                    f"found {len(spans)} spans, matched {matched_token_count} tokens, "
                    f"reward_value={value:.4f}"  # ✅ 添加 value 日志
                )

            else:
                # ===== 策略2: 模糊文本匹配 =====
                self.logger.debug(
                    f"[TOKEN MATCH] Sentence {s_idx+1}: Tokenizer match failed, trying fuzzy text matching"
                )

                fuzzy_matched_tokens = set()
                char_start = 0
                fuzzy_match_count = 0

                while True:
                    idx = response_text.find(text, char_start)
                    if idx < 0:
                        break

                    match_start_char = idx
                    match_end_char = idx + len(text)
                    fuzzy_match_count += 1

                    # 找到覆盖这个字符范围的所有tokens
                    for token_idx, (tok_start, tok_end) in enumerate(token_char_spans):
                        # 判断token是否与匹配区域重叠
                        if tok_end > match_start_char and tok_start < match_end_char:
                            fuzzy_matched_tokens.add(token_idx)

                    if self.max_matches_per_sentence and fuzzy_match_count >= self.max_matches_per_sentence:
                        break

                    char_start = match_end_char

                if fuzzy_matched_tokens:
                    # ===== 模糊匹配成功 =====
                    has_any_match = True
                    hit_spans_fuzzy += 1
                    matched_token_count = len(fuzzy_matched_tokens)
                    total_matched_tokens += matched_token_count

                    # ✅ 修复: 给匹配的tokens赋reward
                    for token_idx in fuzzy_matched_tokens:
                        if self.overlap_aggregation == "sum":
                            token_rewards[token_idx] += value
                        else:
                            token_rewards[token_idx] = max(token_rewards[token_idx].item(), value)

                    self.logger.debug(
                        f"[TOKEN MATCH] Sentence {s_idx+1}: FUZZY SUCCESS - "
                        f"found {fuzzy_match_count} text matches, matched {matched_token_count} tokens, "
                        f"reward_value={value:.4f}"  # ✅ 添加 value 日志
                    )

                else:
                    # ===== 策略3: 两种匹配都失败,使用uniform fallback =====
                    fallback_uniform_count += 1

                    self.logger.warning(
                        f"[TOKEN MATCH] Sentence {s_idx+1}: BOTH FAILED - "
                        f"applying uniform reward {value:.4f} to all tokens. "  # ✅ 显示具体值
                        f"Sentence: '{text[:100]}...'"
                    )

                    # ✅ 修复: 确保真正赋值
                    if self.overlap_aggregation == "sum":
                        token_rewards += value
                    else:
                        # 对每个token取max
                        token_rewards = torch.maximum(token_rewards, torch.tensor(value, device=device))

                    has_any_match = True

        # ===== 最终Fallback: 如果所有句子都没有匹配到任何东西 =====
        if not has_any_match and sentence_items:
            total_criteria = sum(len(s.get("criteria", [])) for s in sentence_items)
            if total_criteria == 0:
                total_criteria = len(sentence_items)

            scale = max(1, total_criteria) if self.scale_by_num_criteria else 1
            fallback_value = float(self.sentence_reward_base) * float(scale) * float(answer_scale)

            if self.overlap_aggregation == "sum":
                token_rewards += fallback_value
            else:
                token_rewards.fill_(fallback_value)  # ✅ 直接填充

            self.logger.warning(
                f"[TOKEN MATCH] GLOBAL FALLBACK: No matches found for any sentence, "
                f"applying uniform reward {fallback_value:.4f} to all tokens"
            )

        # ===== 后处理 =====
        if self.smoothing_radius > 0:
            token_rewards = self._smooth_rewards(token_rewards, self.smoothing_radius)
            self.logger.debug(f"[TOKEN MATCH] Applied smoothing with radius {self.smoothing_radius}")

        if self.normalize_mode == "by_length":
            length = (token_rewards > 0).float().sum().clamp_min(1.0)
            token_rewards = token_rewards / length
            self.logger.debug(f"[TOKEN MATCH] Normalized by length: {length.item()}")
        elif self.normalize_mode == "by_num_sentences":
            num_sents = max(1, len(sentence_items))
            token_rewards = token_rewards / float(num_sents)
            self.logger.debug(f"[TOKEN MATCH] Normalized by num_sentences: {num_sents}")

        if self.cap_per_token is not None:
            token_rewards = torch.clamp(token_rewards, max=self.cap_per_token)
            self.logger.debug(f"[TOKEN MATCH] Applied cap: {self.cap_per_token}")

        # ===== 统计指标 =====
        nonzero_tokens = (token_rewards > 0).sum().item()
        total_tokens = token_rewards.numel()

        metrics["reward/token_coverage"] = nonzero_tokens / max(1, total_tokens)
        metrics["reward/token_mean"] = token_rewards.mean().item()
        metrics["reward/token_max"] = float(token_rewards.max().item() if token_rewards.numel() else 0.0)
        metrics["reward/total_sentences"] = float(n_all)
        metrics["reward/tokenizer_matched_sentences"] = float(hit_spans_tokenizer)
        metrics["reward/fuzzy_matched_sentences"] = float(hit_spans_fuzzy)
        metrics["reward/uniform_fallback_sentences"] = float(fallback_uniform_count)
        metrics["reward/total_matched_tokens"] = float(total_matched_tokens)

        # ===== 增强日志 =====
        self.logger.info(
            f"[TOKEN MATCH] Summary: {n_all} sentences processed - "
            f"Tokenizer: {hit_spans_tokenizer}, Fuzzy: {hit_spans_fuzzy}, "
            f"Uniform Fallback: {fallback_uniform_count}, "
            f"Token coverage: {nonzero_tokens}/{total_tokens} ({metrics['reward/token_coverage']*100:.1f}%), "
            f"Mean reward: {metrics['reward/token_mean']:.6f}, Max: {metrics['reward/token_max']:.6f}"
        )

        # ✅ 新增: 如果有匹配但reward仍为0,特别警告
        if (hit_spans_tokenizer > 0 or hit_spans_fuzzy > 0) and nonzero_tokens == 0:
            self.logger.error(
                f"[TOKEN MATCH] CRITICAL BUG: Had {hit_spans_tokenizer + hit_spans_fuzzy} matches "
                f"but all rewards are ZERO! Check value calculation. "
                f"sentence_reward_base={self.sentence_reward_base}, answer_scale={answer_scale}"
            )

        if hit_spans_tokenizer == 0 and n_all > 0:
            self.logger.warning(
                f"[TOKEN MATCH] WARNING: No tokenizer matches! "
                f"All {n_all} sentences used fuzzy ({hit_spans_fuzzy}) or uniform fallback ({fallback_uniform_count}). "
                f"This may indicate tokenizer mismatch between judge and training model."
            )

        return token_rewards.to(dtype=torch.float16)

    # ===== default path: compute token rewards without LLM judge =====
    def _get_bank_norm_score(self, entry: Dict, label: Optional[str]) -> float:
        if not entry or not label:
            return 0.0
        scores = entry.get("scores", {}) or {}
        if label not in scores and label.upper() in scores:
            s = scores[label.upper()]
        else:
            s = scores.get(label, 0.0)
        try:
            s = float(s)
        except (TypeError, ValueError):
            s = 0.0
        if not self.bank_score_minmax_norm:
            return max(0.0, s)
        # min-max 归一化到 [0,1]（按该题的所有选项）
        vals = [float(v) for v in scores.values()] if scores else []
        if not vals:
            return 0.0
        vmin, vmax = min(vals), max(vals)
        if vmax <= vmin:
            return 0.0
        return (s - vmin) / (vmax - vmin)

    def _compute_default_token_rewards(
        self, response_tokens_1d: torch.Tensor, answer_scale: float, bank_norm_score: float
    ) -> torch.Tensor:
        """
        Default strategies:
          - zero                      : 全零
          - uniform_answer_scaled     : 全段= sentence_reward_base * answer_scale
          - uniform_bank_score_scaled : 全段= sentence_reward_base * bank_norm_score
        """
        T = response_tokens_1d.size(0)
        device = response_tokens_1d.device
        if self.default_token_reward_strategy == "zero":
            return torch.zeros(T, dtype=torch.float16, device=device)
        elif self.default_token_reward_strategy == "uniform_answer_scaled":
            v = float(self.sentence_reward_base) * float(answer_scale)
            return torch.full((T,), v, dtype=torch.float16, device=device)
        elif self.default_token_reward_strategy == "uniform_bank_score_scaled":
            v = float(self.sentence_reward_base) * float(bank_norm_score)
            return torch.full((T,), v, dtype=torch.float16, device=device)
        else:
            # fallback to zero
            return torch.zeros(T, dtype=torch.float16, device=device)

    # ===== main =====
    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data: DataProto):
        is_offload_states = data.meta_info.get("is_offload_states", True)
        metrics = {}
        if self.judge_model_type == "inference" and self.strategy:
            with state_offload_manger(
                strategy=self.strategy,
                metrics=metrics,
                metric_infix=f"{self.cluster_name}/compute_rewards",
                is_offload_states=is_offload_states,
            ):
                return self._compute_rewards_impl(data, metrics)
        else:
            return self._compute_rewards_impl(data, metrics)

    def _compute_weighted_llm_score(self, criteria_result: Dict[str, Any]) -> float:
        """
        根据LLM评分结果计算加权平均值

        Args:
            criteria_result: LLM判断的结果，包含criteria数组

        Returns:
            float: 加权平均分数 [0, 0.5]
        """
        criteria = criteria_result.get("criteria", [])
        if not criteria:
            return 0.0

        # 定义每个criterion的权重 (可以通过config配置)
        criterion_weights = getattr(
            self.worker_config,
            "criterion_weights",
            {
                1: 0.25,  # Persona Consistency
                2: 0.20,  # Conversational Coherence
                3: 0.15,  # Tone & Style Match
                4: 0.20,  # Contextual Grounding
                5: 0.20,  # Affective Alignment
            },
        )

        total_weighted_score = 0.0
        total_weight = 0.0

        for criterion in criteria:
            if not isinstance(criterion, dict):
                continue

            criterion_id = criterion.get("id")
            score = criterion.get("score", 0)

            # 获取该criterion的权重
            weight = criterion_weights.get(criterion_id, 0.2)  # 默认权重0.2

            # 标准化分数到-1 - 1
            normalized_score = float(self._normalize_score(score))

            total_weighted_score += normalized_score * weight
            total_weight += weight

        # 计算加权平均值
        if total_weight > 0:
            weighted_avg = total_weighted_score / total_weight
        else:
            weighted_avg = 0.0

        # 支持负分：weighted_avg 可能在 [-1,1], 映射到 [-0.5,0.5]
        raw = float(weighted_avg)
        return max(-1, min(1, raw)) if self.allow_negative_scores else max(0, min(1, raw))

    def _compute_rewards_impl(self, data: DataProto, metrics: Dict):
        global_step = int(data.meta_info.get("global_step", 0))

        # 解码
        resp_token_batch = data.batch["responses"]
        prompts_text_list = self.tokenizer.batch_decode(data.batch["prompts"], skip_special_tokens=True)
        decoded_responses = self.tokenizer.batch_decode(resp_token_batch, skip_special_tokens=True)

        if len(prompts_text_list) != len(decoded_responses):
            raise ValueError("prompts 与 responses 数量不一致")

        non_tensor = data.non_tensor_batch
        if "qid" in non_tensor:
            qids = non_tensor["qid"]
        elif "question_id" in non_tensor:
            qids = non_tensor["question_id"]
        else:
            raise ValueError("This worker requires 'qid' or 'question_id' in non_tensor_batch")
        if len(qids) != len(decoded_responses) or len(qids) != len(prompts_text_list):
            raise ValueError("qid / prompts / responses 数量不一致")

        bank = self._load_question_bank()

        # ===== 计算 alpha =====
        if global_step < self.guidance_until_step:
            alpha = 1.0
        elif self.guidance_blend_span > 0 and global_step < self.guidance_until_step + self.guidance_blend_span:
            remain = (self.guidance_until_step + self.guidance_blend_span) - global_step
            alpha = max(0.0, float(remain) / float(self.guidance_blend_span))
        else:
            alpha = 0.0

        # ✅ 关键: 提前判断是否需要LLM judge (包含 blend 阶段!)
        need_llm_judge = alpha > 0.0

        device = resp_token_batch.device
        B, T = resp_token_batch.size(0), resp_token_batch.size(1)
        token_level_rewards = torch.zeros_like(resp_token_batch, dtype=torch.float16, device=device)
        response_level_rewards = torch.zeros(B, dtype=torch.float16, device=device)
        out_scores = torch.zeros(B, dtype=torch.float16, device=device) if self.emit_scores_in_default_phase else None

        # ===== 预处理 =====
        sample_info = []
        samples_with_answers = []
        no_answer_count = 0

        for i, (qid, prompt_txt, response_txt) in enumerate(zip(qids, prompts_text_list, decoded_responses)):
            entry = bank.get(qid, {}) or {}
            gold = self._norm_letter(entry.get("gold"))
            choice_texts: Dict[str, str] = entry.get("choice_texts", {}) or {}
            correct_text = choice_texts.get(gold) if gold else None
            pred = self._normalize_choice(response_txt)

            has_answer = pred is not None
            if not has_answer:
                no_answer_count += 1

            is_correct = int(pred == gold) if (pred and gold) else 0
            bank_norm_score = self._get_bank_norm_score(entry, pred)

            info = {
                "index": i,
                "qid": qid,
                "prompt_txt": prompt_txt,
                "response_txt": response_txt,
                "gold": gold,
                "correct_text": correct_text,
                "pred": pred,
                "has_answer": has_answer,
                "is_correct": is_correct,
                "bank_norm_score": bank_norm_score,
                "entry": entry,
            }

            sample_info.append(info)

            # ✅ 修复: 应用 judge_score_target 过滤
            if need_llm_judge and has_answer:
                # 根据 judge_score_target 决定是否需要判断
                should_judge = False
                if self.judge_score_target == "both":
                    should_judge = True
                elif self.judge_score_target == "correct":
                    should_judge = bool(is_correct)
                elif self.judge_score_target == "incorrect":
                    should_judge = not bool(is_correct)
                else:
                    # 默认行为: both
                    should_judge = True

                if should_judge:
                    samples_with_answers.append(info)

        # 统计
        metrics["answer_check/total_samples"] = float(len(sample_info))
        metrics["answer_check/no_answer_count"] = float(no_answer_count)
        metrics["answer_check/has_answer_count"] = float(len(samples_with_answers)) if need_llm_judge else 0.0
        metrics["answer_check/no_answer_ratio"] = float(no_answer_count) / max(1, len(sample_info))

        # ===== Guided阶段: LLM判断 (✅ 只调用一次!) =====
        # apply judge_target filter when collecting samples_with_answers earlier
        if need_llm_judge and samples_with_answers:
            self.logger.info(
                f"[REWARD DEBUG] GUIDED PHASE (alpha={alpha:.3f}): "
                f"Starting LLM judge for {len(samples_with_answers)} samples"
            )

            # ✅ 只调用一次!
            batch_criteria_results = self._get_batch_llm_judgment(samples_with_answers)

            valid_results = sum(1 for r in batch_criteria_results if r and not r.get("_compat_fallback", False))
            fallback_results = sum(1 for r in batch_criteria_results if r and r.get("_compat_fallback", False))
            self.logger.info(f"[REWARD DEBUG] LLM results - Valid: {valid_results}, Fallback: {fallback_results}")

            # 创建索引映射
            answer_sample_to_result = {}
            for info, criteria_result in zip(samples_with_answers, batch_criteria_results):
                answer_sample_to_result[info["index"]] = criteria_result

            # ===== compute weighted raw scores for batch and optionally normalize across batch =====
            raw_scores = []
            for r in batch_criteria_results:
                if r is None:
                    raw_scores.append(0.0)
                else:
                    raw_scores.append(self._compute_weighted_llm_score(r))

            # batch normalization
            norm_scores = raw_scores
            if self.batch_score_normalization in ("minmax", "min-max", "min_max"):
                import numpy as _np

                arr = _np.array(raw_scores, dtype=_np.float32)
                mn, mx = float(arr.min()), float(arr.max())
                if mx > mn:
                    # map to [0, 2]
                    norm_scores = (2.0 * (arr - mn) / (mx - mn)).tolist()
                else:
                    norm_scores = [1.0 for _ in raw_scores]  # 中点
            elif self.batch_score_normalization in ("zscore", "z-score"):
                import numpy as _np

                arr = _np.array(raw_scores, dtype=_np.float32)
                mu, sd = float(arr.mean()), float(arr.std())
                if sd > 1e-6:
                    # map to approximately [0, 2] via tanh, centered at 1.0
                    norm_scores = (1.0 + _np.tanh((arr - mu) / sd)).tolist()
                else:
                    norm_scores = [1.0 for _ in raw_scores]  # 中点

            # 将最终分数回写到每个 result，保持原始值也存下
            for idx, r in enumerate(batch_criteria_results):
                if r is None:
                    continue
                r["llm_weighted_raw"] = raw_scores[idx]
                r["llm_weighted_score"] = float(norm_scores[idx])

            # 计算guided tokens
            for info in sample_info:
                i = info["index"]

                if info["has_answer"] and i in answer_sample_to_result:
                    criteria_result = answer_sample_to_result[i]
                    sentence_items = criteria_result.get("sentences", [])
                    answer_scale = self.correct_reward_scale if info["is_correct"] else self.incorrect_reward_scale

                    guided_tokens = self._compute_guided_token_rewards(
                        resp_token_batch[i], sentence_items, info["response_txt"], answer_scale, metrics
                    )

                    # 使用已经归一化后的分数（若未归一化，_get_batch_llm_judgment 之后会有 raw -> score）
                    info["llm_weighted_score"] = float(
                        criteria_result.get("llm_weighted_score", criteria_result.get("llm_weighted_raw", 0.0))
                    )
                else:
                    # ✅ 没有答案: guided给零 (会在混合时被default替代)
                    guided_tokens = torch.zeros(T, dtype=torch.float16, device=device)
                    info["llm_weighted_score"] = 0.0

                info["guided_tokens"] = guided_tokens
        else:
            # ✅ Pure default阶段: 跳过LLM judge
            self.logger.info(
                f"[REWARD DEBUG] DEFAULT PHASE (alpha={alpha:.3f}): "
                f"Skipping LLM judge, strategy={self.default_token_reward_strategy}"
            )

            for info in sample_info:
                info["guided_tokens"] = torch.zeros(T, dtype=torch.float16, device=device)
                info["llm_weighted_score"] = 0.0

        # ===== Default path & 最终混合 =====
        llm_scores = []

        for info in sample_info:
            i = info["index"]

            # ✅ 总是计算default tokens (作为backup)
            if info["has_answer"]:
                answer_scale = self.correct_reward_scale if info["is_correct"] else self.incorrect_reward_scale
                default_tokens = self._compute_default_token_rewards(
                    response_tokens_1d=resp_token_batch[i],
                    answer_scale=answer_scale,
                    bank_norm_score=info["bank_norm_score"],
                )
            else:
                default_tokens = self._compute_default_error_token_rewards(
                    resp_token_batch[i], self.incorrect_reward_scale
                )

            # ✅ 修复: 平滑混合 (避免跳跃)
            guided_tokens = info["guided_tokens"]

            if alpha >= 1.0:
                # 纯 guided
                final_tokens = guided_tokens
                blend_type = "100\% guided"
            elif alpha <= 0.0:
                # 纯 default
                final_tokens = default_tokens
                blend_type = "100\% default"
            else:
                # ✅ 关键修复: 确保平滑过渡
                final_tokens = (
                    alpha * guided_tokens.to(torch.float32) + (1.0 - alpha) * default_tokens.to(torch.float32)
                ).to(torch.float16)
                blend_type = f"{alpha*100:.1f}% guided + {(1-alpha)*100:.1f}% default"

            token_level_rewards[i] = final_tokens

            # ✅ Response-level: 也使用平滑插值
            llm_score = info.get("llm_weighted_score", 0.0)

            # Guided response reward (基于LLM score + correctness)
            if info["is_correct"]:
                guided_response_reward = 1.0 + llm_score  # [1.0, 1.5]
            else:
                guided_response_reward = 0.0 + llm_score  # [0.0, 0.5]

            # Default response reward (仅基于correctness或bank_score)
            if info["has_answer"]:
                if self.default_token_reward_strategy == "uniform_bank_score_scaled":
                    # 使用bank_score归一化到[0,1]
                    default_response_reward = float(info["bank_norm_score"])
                else:
                    # 使用correctness: correct=1.0, incorrect=0.0
                    default_response_reward = 1.0 if info["is_correct"] else 0.0
            else:
                # 没有答案: 给0
                default_response_reward = 0.0

            # ✅ 平滑插值
            if alpha >= 1.0:
                # 纯guided
                response_reward = guided_response_reward
            elif alpha <= 0.0:
                # 纯default
                response_reward = default_response_reward
            else:
                # 混合: alpha * guided + (1-alpha) * default
                response_reward = alpha * guided_response_reward + (1.0 - alpha) * default_response_reward

            response_level_rewards[i] = response_reward
            llm_scores.append(llm_score)

            # Scores (for GRPO)
            if self.emit_scores_in_default_phase:
                if info["has_answer"]:
                    score_val = (
                        info["bank_norm_score"]
                        if self.default_token_reward_strategy == "uniform_bank_score_scaled"
                        else float(info["is_correct"])
                    )
                else:
                    score_val = 0.0
                out_scores[i] = float(score_val)

            # ✅ 增强日志: 显示guided/default/final的值
            if i < 3:
                self.logger.info(
                    f"[REWARD BLEND] Sample {i}: {blend_type}, "
                    f"Has answer: {info['has_answer']}, Correct: {info['is_correct']}, "
                    f"Token mean: {final_tokens.mean().item():.6f}, "
                    f"Response - Guided: {guided_response_reward:.6f}, "
                    f"Default: {default_response_reward:.6f}, "
                    f"Final: {response_reward:.6f}"
                )

        # ===== Metrics =====
        if need_llm_judge and llm_scores:
            metrics["llm_score/mean"] = float(np.mean(llm_scores))
            metrics["llm_score/std"] = float(np.std(llm_scores))
            metrics["llm_score/min"] = float(np.min(llm_scores))
            metrics["llm_score/max"] = float(np.max(llm_scores))
            metrics["llm_score/non_zero_count"] = float(sum(1 for s in llm_scores if s > 0))
            metrics["llm_score/non_zero_ratio"] = float(sum(1 for s in llm_scores if s > 0)) / max(1, len(llm_scores))

        tensors = {"token_level_rewards": token_level_rewards}

        if self.return_response_level:
            tensors["response_level_rewards"] = response_level_rewards

        if self.emit_scores_in_default_phase and out_scores is not None:
            tensors["scores"] = out_scores

        if self.return_scores and "scores" in data.batch and not self.emit_scores_in_default_phase:
            tensors["scores"] = data.batch["scores"]

        # ===== 增强日志 =====
        self.logger.info(f"[REWARD DEBUG] ========== SUMMARY ==========")
        self.logger.info(f"[REWARD DEBUG] Global step: {global_step}, Alpha: {alpha:.3f}")
        self.logger.info(f"[REWARD DEBUG] Phase: {'GUIDED/BLEND' if alpha > 0 else 'PURE DEFAULT'}")
        self.logger.info(f"[REWARD DEBUG] Default strategy: {self.default_token_reward_strategy}")
        self.logger.info(
            f"[REWARD DEBUG] Total samples: {len(sample_info)}, With answers: {len(sample_info) - no_answer_count}"
        )

        if need_llm_judge:
            self.logger.info(f"[REWARD DEBUG] LLM judge: {len(samples_with_answers)} samples")
        else:
            self.logger.info(f"[REWARD DEBUG] LLM judge: SKIPPED")

        # Token统计
        total_nonzero = (token_level_rewards > 0).sum().item()
        total_tokens = token_level_rewards.numel()
        metrics["reward_debug/total_token_coverage"] = total_nonzero / max(1, total_tokens)

        self.logger.info(
            f"[REWARD DEBUG] Token coverage: {total_nonzero}/{total_tokens} "
            f"({total_nonzero/max(1,total_tokens)*100:.1f}%)"
        )

        # Response-level统计
        nonzero_responses = (response_level_rewards > 0).sum().item()
        self.logger.info(f"[REWARD DEBUG] Response rewards - Nonzero: {nonzero_responses}/{B}")
        self.logger.info(f"[REWARD DEBUG] Response mean: {response_level_rewards.mean().item():.6f}")

        # 正确vs错误对比
        correct_indices = [i for i, info in enumerate(sample_info) if info["is_correct"]]
        incorrect_indices = [i for i, info in enumerate(sample_info) if not info["is_correct"]]

        if correct_indices:
            correct_rewards = response_level_rewards[correct_indices]
            self.logger.info(
                f"[REWARD DEBUG] Correct ({len(correct_indices)}): "
                f"mean={correct_rewards.mean().item():.6f}, max={correct_rewards.max().item():.6f}"
            )

        if incorrect_indices:
            incorrect_rewards = response_level_rewards[incorrect_indices]
            self.logger.info(
                f"[REWARD DEBUG] Incorrect ({len(incorrect_indices)}): "
                f"mean={incorrect_rewards.mean().item():.6f}, max={incorrect_rewards.max().item():.6f}"
            )

        self.logger.info(f"[REWARD DEBUG] ==========================================")

        output = DataProto.from_dict(tensors=tensors)
        output.meta_info = {"metrics": metrics}

        return output

    def _compute_default_error_token_rewards(
        self, response_tokens_1d: torch.Tensor, error_scale: float
    ) -> torch.Tensor:
        """
        为没有答案的回复计算默认错误token rewards
        """
        T = response_tokens_1d.size(0)
        device = response_tokens_1d.device

        # 使用错误scale给出统一的负面或零reward
        v = float(self.sentence_reward_base) * float(error_scale)
        return torch.full((T,), v, dtype=torch.float16, device=device)

    def _call_api_model(self, messages: List[Dict], retry_times=3) -> str:
        """改进的API调用方法，支持更好的错误处理"""
        if not self.judge_api_url or not self.judge_api_key:
            raise ValueError("API URL and API key must be provided for API model type")

        output = ""
        last_exception = None

        for attempt in range(retry_times):
            try:
                # 添加随机延迟避免并发时的rate limiting
                if attempt > 0:
                    import time
                    import random

                    delay = random.uniform(0.1, 0.5) * (2**attempt)
                    time.sleep(delay)

                client = OpenAI(api_key=self.judge_api_key, base_url=self.judge_api_url)

                # 使用responses.create (如o3等新模型)
                response = client.responses.create(
                    model=self.judge_model_name,
                    input=messages,
                    reasoning={"effort": "low"},
                    max_output_tokens=2048,
                    temperature=0.3,
                )
                output = response.output_text.strip()

                if output:
                    break

            except Exception as e:
                last_exception = e
                self.logger.warning(f"API call attempt {attempt + 1} failed: {e}")
                continue

        if not output and last_exception:
            self.logger.error(f"All API call attempts failed. Last error: {last_exception}")
            # 不抛出异常，返回空字符串让上层处理

        return output or ""

    def _run_batch_api_inference(self, batch_messages: List[List[Dict]]) -> List[Dict]:
        """批量API推理（并行），与local inference逻辑对齐"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time

        def call_api_sync(index_messages_pair) -> Tuple[int, Dict]:
            """同步API调用封装，返回索引和结果"""
            index, messages = index_messages_pair
            try:
                llm_response = self._call_api_model(messages)
                result = self._extract_criteria_result(llm_response)
                result["llm_response"] = llm_response

                # 记录日志（与local inference保持一致）
                # if self.rank_info.rank == 0:
                #     self.logger.info(f"API LLM result for batch item {index}: {llm_response} => {result}")

                return index, result
            except Exception as e:
                self.logger.error(f"API call failed for batch item {index}: {e}")
                # 创建fallback结果
                fallback_result = self._create_fallback_result(f"API call failed: {str(e)}")
                return index, fallback_result

        # 限制并发数，避免API rate limiting
        max_workers = min(len(batch_messages), getattr(self.worker_config, "api_max_workers", 8))
        results = [None] * len(batch_messages)

        self.logger.info(
            f"Starting batch API inference with {len(batch_messages)} requests, max_workers: {max_workers}"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务，传递索引确保结果顺序
            index_messages_pairs = [(i, messages) for i, messages in enumerate(batch_messages)]
            future_to_index = {executor.submit(call_api_sync, pair): pair[0] for pair in index_messages_pairs}

            # 收集结果
            completed_count = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result_index, result = future.result()
                    results[result_index] = result
                    completed_count += 1

                    if completed_count % 10 == 0:  # 每10个请求记录一次进度
                        self.logger.info(f"Completed {completed_count}/{len(batch_messages)} API requests")

                except Exception as e:
                    self.logger.error(f"Future execution failed for batch item {index}: {e}")
                    # 创建fallback结果
                    results[index] = self._create_fallback_result(f"Future execution failed: {str(e)}")

        self.logger.info(f"Batch API inference completed: {len(batch_messages)} requests processed")

        # 确保所有位置都有结果
        for i, result in enumerate(results):
            if result is None:
                self.logger.warning(f"Missing result for batch item {i}, creating fallback")
                results[i] = self._create_fallback_result("Missing result")

        return results

    def _run_batch_local_inference(self, batch_messages: List[List[Dict]]) -> List[Dict]:
        """批量本地推理，优化日志和错误处理"""
        if not self.strategy:
            raise ValueError("Strategy not initialized for local inference")

        from roll.datasets.chat_template import get_chat_template

        template_name = "qwen3_nothink"
        chat_template_func = get_chat_template(template_name, self.tokenizer)

        # 准备批量文本
        batch_texts = []
        for messages in batch_messages:
            text = chat_template_func(messages)
            batch_texts.append(text)

        # 批量tokenize
        max_length = getattr(self.worker_config, "judge_max_length", 3584)
        tokenized = self.tokenizer(
            batch_texts, max_length=max_length, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = tokenized["input_ids"].to("cuda")
        attention_mask = tokenized["attention_mask"].to("cuda")

        generation_config = self.strategy.worker_config.generating_args.to_dict()
        generation_config["eos_token_id"] = [self.tokenizer.eos_token_id]
        generation_config["pad_token_id"] = self.tokenizer.pad_token_id

        # 确定批量大小
        batch_size = input_ids.size(0)
        infer_batch_size = getattr(self.worker_config, "infer_batch_size", batch_size)
        micro_batch_size = min(batch_size, infer_batch_size)

        self.logger.info(
            f"Local batch inference - batch_size: {batch_size}, "
            f"infer_batch_size: {infer_batch_size}, micro_batch_size: {micro_batch_size}"
        )

        # 创建DataProto
        data = DataProto.from_dict(
            tensors={"input_ids": input_ids, "attention_mask": attention_mask},
            meta_info={"micro_batch_size": micro_batch_size},
        ).to("cuda")

        # 批量推理
        try:
            with torch.no_grad():
                output = self.strategy.generate(batch=data, generation_config=generation_config)
                prompt_lengths = (attention_mask.sum(dim=1)).tolist()
                generated_texts = []

                if isinstance(output, torch.Tensor):
                    # output shape: [B, total_length]
                    for i in range(batch_size):
                        prompt_len = prompt_lengths[i]
                        generate_ids = output[i, prompt_len:]
                        # 移除padding tokens
                        if generation_config["pad_token_id"] in generate_ids:
                            pad_idx = (generate_ids == generation_config["pad_token_id"]).nonzero()
                            if len(pad_idx) > 0:
                                generate_ids = generate_ids[: pad_idx[0].item()]
                        text = self.tokenizer.decode(generate_ids, skip_special_tokens=True).strip()
                        generated_texts.append(text)
                else:
                    # 处理其他格式的输出
                    batch_output = output.batch["input_ids"]
                    for i in range(batch_size):
                        prompt_len = prompt_lengths[i]
                        generate_ids = batch_output[i, prompt_len:]
                        if generation_config["pad_token_id"] in generate_ids:
                            pad_idx = (generate_ids == generation_config["pad_token_id"]).nonzero()
                            if len(pad_idx) > 0:
                                generate_ids = generate_ids[: pad_idx[0].item()]
                        text = self.tokenizer.decode(generate_ids, skip_special_tokens=True).strip()
                        generated_texts.append(text)

        except Exception as e:
            self.logger.error(f"Local batch inference failed: {e}")
            # 创建fallback结果
            generated_texts = [""] * batch_size

        # 处理每个生成的回复
        results = []
        for i, llm_response in enumerate(generated_texts):
            try:
                result = self._extract_criteria_result(llm_response)
                result["llm_response"] = llm_response

                # 记录日志
                if self.rank_info.rank == 0:
                    self.logger.info(f"Local LLM result for batch item {i}: {llm_response} => {result}")

                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process result for batch item {i}: {e}")
                fallback_result = self._create_fallback_result(f"Result processing failed: {str(e)}")
                results.append(fallback_result)

        self.logger.info(f"Local batch inference completed: {len(results)} results generated")
        return results

    def _get_batch_llm_judgment(self, sample_info: List[Dict]) -> List[Dict]:
        """批量处理LLM判断，统一API和local推理的接口"""
        if not sample_info:
            return []

        # 准备批量消息
        batch_messages = []
        for info in sample_info:
            messages = self._format_judge_prompt(
                info["prompt_txt"], info["response_txt"], info["gold"], info["correct_text"]
            )
            batch_messages.append(messages)

        self.logger.info(
            f"Starting batch LLM judgment for {len(batch_messages)} samples using {self.judge_model_type}"
        )

        if self.judge_model_type == "api":
            # API模式：并行调用
            return self._run_batch_api_inference(batch_messages)
        elif self.judge_model_type == "inference":
            # 本地推理模式：真正的批量处理
            return self._run_batch_local_inference(batch_messages)
        else:
            raise ValueError(
                f"Unsupported judge model type: {self.judge_model_type}"
            )  # def _call_api_model(self, messages: List[Dict], retry_times=3) -> str:

    #     """改进的API调用方法，支持更好的错误处理"""
    #     from openai import OpenAI
    #     import time
    #     import random

    #     if not self.judge_api_url or not self.judge_api_key:
    #         raise ValueError("API URL and API key must be provided for API model type")

    #     output = ""
    #     last_exception = None

    #     for attempt in range(retry_times):
    #         try:
    #             # 添加随机延迟避免并发时的rate limiting
    #             if attempt > 0:
    #                 delay = random.uniform(0.1, 0.5) * (2**attempt)
    #                 time.sleep(delay)

    #             client = OpenAI(api_key=self.judge_api_key, base_url=self.judge_api_url)
    #             completion = client.chat.completions.create(
    #                 model=self.judge_model_name, messages=messages, timeout=30  # 设置超时
    #             )
    #             output = completion.choices[0].message.content
    #             if output:
    #                 break

    #         except Exception as e:
    #             last_exception = e
    #             self.logger.warning(f"API call attempt {attempt + 1} failed: {e}")
    #             continue

    #     if not output and last_exception:
    #         self.logger.error(f"All API call attempts failed. Last error: {last_exception}")
    #         # 不抛出异常，返回空字符串让上层处理

    #     return output or ""

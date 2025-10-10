import json
import re
from functools import lru_cache
from typing import Optional, Union

import torch

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

QUESTION_FILE = (
    "/project/hdtaccuracy/Personality-Alignment/choice_ver/v10/train_data_enhanced/merged_choice_questions.json"
)
# 如果题目里有 choices.score，则按 score/5.0 归一化，否则用 gold 二分类 1/0
SCORE_NORMALIZE_DENOM = 5.0


class PerAlignChoiceRewardWorker(Worker):
    """
    读取题库行式 JSON：
      {"qid": "...", "answer": "A", "choices":[{"label":"A","score":3.0,...}, ...]}
    根据模型输出解析出预测选项字母 (A/B/C/D)。
    优先使用 choices 的 score；若无则回退到 gold 匹配 (正确=1 否则 0)。
    返回:
      token_level_rewards: 全 0 (可后续扩展)
      response_level_rewards: 与 scores 相同
      scores: 标量奖励 (float16)
    期望 data.non_tensor_batch 中包含 "qid" 或 "question_id"。
    """

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        # 如需在分布式场景构建策略：
        if self.strategy is None:
            self.strategy = create_strategy(
                model_provider=default_reward_model_provider,
                tokenizer_provider=lambda *a, **k: self.tokenizer,
                worker_config=self.worker_config,
            )

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_question_bank():
        bank = {}
        try:
            with open(QUESTION_FILE, "r", encoding="utf-8") as f:
                # 优先按结构化 JSON 读取（list 或包含 "questions" 的 dict）
                try:
                    json_content = json.load(f)
                    if isinstance(json_content, list):
                        questions = json_content
                    elif isinstance(json_content, dict) and "questions" in json_content:
                        questions = json_content["questions"]
                    else:
                        questions = [json_content]  # 单条对象

                    for obj in questions:
                        if not isinstance(obj, dict):
                            continue
                        qid = obj.get("qid")
                        if not qid:
                            continue

                        entry = {"gold": None, "scores": {}}

                        # gold (兼容 answer/output 或 choices 中 is_correct)
                        gold = obj.get("answer") or obj.get("output")
                        if not gold and isinstance(obj.get("choices"), list):
                            for ch in obj["choices"]:
                                if isinstance(ch, dict) and ch.get("is_correct"):
                                    entry["gold"] = ch.get("label")
                                    break
                        else:
                            entry["gold"] = gold

                        # 收集各选项分数
                        choices = obj.get("choices")
                        if isinstance(choices, list):
                            for ch in choices:
                                if not isinstance(ch, dict):
                                    continue
                                label = ch.get("label")
                                if label and isinstance(label, str):
                                    try:
                                        sc_raw = ch.get("score", 0.0)
                                        sc = float(sc_raw)
                                    except (TypeError, ValueError):
                                        sc = 0.0
                                    entry["scores"][label.upper()] = sc

                        bank[qid] = entry

                except json.JSONDecodeError:
                    # 回退为 JSONL(行式 JSON)
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        qid = obj.get("qid")
                        if not qid:
                            continue

                        entry = {"gold": None, "scores": {}}

                        gold = obj.get("answer") or obj.get("output")
                        if not gold and isinstance(obj.get("choices"), list):
                            for ch in obj["choices"]:
                                if isinstance(ch, dict) and ch.get("is_correct"):
                                    entry["gold"] = ch.get("label")
                                    break
                        else:
                            entry["gold"] = gold

                        choices = obj.get("choices")
                        if isinstance(choices, list):
                            for ch in choices:
                                if not isinstance(ch, dict):
                                    continue
                                label = ch.get("label")
                                if label and isinstance(label, str):
                                    try:
                                        sc = float(ch.get("score", 0.0))
                                    except (TypeError, ValueError):
                                        sc = 0.0
                                    entry["scores"][label.upper()] = sc

                        bank[qid] = entry
        except FileNotFoundError:
            pass
        return bank

    @staticmethod
    def _normalize_choice(text: str):
        if not text:
            return None
        t_up = (text or "").strip().upper()
        t_up = re.sub(r"<THINK>.*?</THINK>", " ", t_up, flags=re.DOTALL)
        t_up = re.sub(r"<[^>]+>", " ", t_up)
        t_up = re.sub(r"\b(ASSISTANT|SYSTEM|USER)\b", " ", t_up)

        # Extract /choice{A} pattern
        m = re.search(r"/CHOICE\{([ABCDEFG])\}", t_up)
        if m:
            return m.group(1)

        m = re.search(r"([ABCDEFG])\s*$", t_up)
        if m:
            return m.group(1)
        m = re.search(r'(?:^|\s)(?:ANSWER|OPTION|CHOICE)\s*[:\-]?\s*["\'`(]*\b([ABCDEFG])\b', t_up)
        if m:
            return m.group(1)
        m = re.match(r'^\s*["\'`(]*([ABCDEFG])(?:[\).\s]|$)', t_up)
        if m:
            return m.group(1)
        m = re.search(r"\b([ABCDEFG])\b", t_up)
        if m:
            return m.group(1)
        letters = re.findall(r"[ABCDEFG]", t_up)
        if len(letters) == 1:
            return letters[0]
        return None

    def _score_one(self, qid: str, response_text: str):
        bank = self._load_question_bank()
        info = bank.get(qid)
        pred = self._normalize_choice(response_text)

        if info is None or pred is None:
            return 0.0

        scores_map = info.get("scores") or {}
        if scores_map:
            if pred in scores_map:
                return float(scores_map[pred]) / SCORE_NORMALIZE_DENOM
            return 0.0
        # 回退 gold
        gold = self._normalize_choice(info.get("gold"))
        return 1.0 if (gold is not None and pred == gold) else 0.0

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data):
        # responses: (batch, seq_len)
        resp_token_batch = data.batch["responses"]
        decoded_responses = self.tokenizer.batch_decode(resp_token_batch, skip_special_tokens=True)

        # 取 qid
        non_tensor = data.non_tensor_batch
        if "qid" in non_tensor:
            qids = non_tensor["qid"]
        elif "question_id" in non_tensor:
            qids = non_tensor["question_id"]
        else:
            raise ValueError("PerAlignChoiceRewardWorker requires 'qid' or 'question_id' in non_tensor_batch")

        if len(qids) != len(decoded_responses):
            raise ValueError(f"qid 数量 {len(qids)} 与 responses 数量 {len(decoded_responses)} 不一致")

        rewards = []
        for qid, resp in zip(qids, decoded_responses):
            r = self._score_one(qid, resp)
            rewards.append(r)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float16)
        # token 层暂置 0；response_level_rewards 与 scores 一致
        token_level_rewards = torch.zeros_like(resp_token_batch, dtype=torch.float16)
        response_level_rewards = rewards_tensor.clone()
        scores = rewards_tensor  # 保持接口一致

        # 记录 debug
        try:
            for qid, resp, r in zip(qids, decoded_responses, rewards):
                self.logger.debug(
                    json.dumps(
                        {
                            "qid": qid,
                            "pred_response": resp,
                            "reward": r,
                        },
                        ensure_ascii=False,
                    )
                )
        except Exception:
            pass

        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": response_level_rewards,
                "scores": scores,
            }
        )
        return output

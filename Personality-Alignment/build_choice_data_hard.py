"""
change_dataset_5_version.py
生成符合格式要求的混淆选项，并将原始数据集转换成 ABCD 四选格式。
7.31version: 错误选项似乎都是一样的，需要进行特殊调整。
修改版本：支持多种干扰项生成模式，支持指定model/batchsize，支持GPT API模型
优化版本：基于build_choice_data.py进行架构优化
"""

import os
import json
import random
import torch
import argparse
import openai
import requests
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from tqdm import tqdm
import time
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ========================= 1. Prompt 模板 ========================= #

DISTRACTOR_PROMPTS = {
    "style_violation": [
        {
            "role": "system",
            "content": "You are a distractor generator for multiple-choice questions. Given a TARGET sentence a person would say, generate exactly one sentence that is realistic and close to the TARGET in language, sentence type, and length. Keep the same sentence type as TARGET (e.g., question → question; statement → statement). Make it sound natural, but ensure it is definitely incorrect with respect to the TARGET's intent (e.g., subtly contradict a key detail, flip a polarity, pick a wrong entity, omit a crucial constraint). Do not be too similar to the TARGET (avoid verbatim copying or long phrase reuse). Do NOT explain or add quotes; output only the sentence.",
        },
        {"role": "user", "content": "TARGET: {correct_output}"},
    ],
    "topic_violation": [
        {
            "role": "system",
            "content": "You are a distractor generator for multiple-choice questions. Given a TARGET sentence, generate exactly one sentence that keeps the same language and sentence type, stays near the topic, but shifts focus to a closely related yet incorrect entity/attribute/option. Make it realistic and similar in length, clearly wrong for the intended answer while not being too similar to the TARGET. Do NOT explain or add quotes; output only the sentence.",
        },
        {"role": "user", "content": "TARGET: {correct_output}"},
    ],
    "richness_violation": [
        {
            "role": "system",
            "content": "You are a distractor generator for multiple-choice questions. Given a TARGET sentence, generate exactly one sentence with the same language and sentence type. If TARGET is detailed, produce a concise variant that omits a crucial condition so it becomes wrong; if TARGET is brief, produce a more elaborate variant that adds an incorrect detail. Keep it realistic and similar in length range, not too similar verbatim, and clearly incorrect. Do NOT explain or add quotes; output only the sentence.",
        },
        {"role": "user", "content": "TARGET: {correct_output}"},
    ],
    "free_violation": [
        {
            "role": "system",
            "content": "You are a distractor generator for multiple-choice questions. Given a TARGET sentence, generate exactly one sentence that remains close in language, sentence type, and style, but conveys a different, clearly incorrect intention/fact compared with the TARGET. It should be realistic and plausible in context, not too similar verbatim, and still definitely wrong. Do NOT explain or add quotes; output only the sentence.",
        },
        {"role": "user", "content": "TARGET: {correct_output}"},
    ],
    "profile_violation_w": [
        {
            "role": "system",
            "content": "You are a distractor generator for multiple-choice questions. Given a TARGET sentence and a PROFILE, generate exactly one sentence that is realistic, keeps the same language and sentence type as TARGET, but clearly contradicts the PROFILE (opposite trait/preference/stance). Use first person ('I', 'me', 'my') as if you are that person. Keep it near the TARGET in style and length, avoid verbatim copying, and ensure it is definitely incompatible with the PROFILE. Do NOT explain or add quotes; output only the sentence.",
        },
        {"role": "user", "content": "TARGET: {correct_output}\nPROFILE: {profile}"},
    ],
    "conversation_violation_w": [
        {
            "role": "system",
            "content": "You are a distractor generator for multiple-choice questions. Given a TARGET sentence and CONVERSATION HISTORY, generate exactly one sentence that is realistic and keeps the same language and sentence type as TARGET, but subtly disregards the conversation requirement (e.g., answers a related but different question, ignores a key constraint, wrong perspective/recipient). Keep it near-topic (not random), similar in length, not too similar verbatim, and definitely inappropriate for the conversation. Use first person when natural. Do NOT explain or add quotes; output only the sentence.",
        },
        {"role": "user", "content": "TARGET: {correct_output}\nCONVERSATION: {conversation}"},
    ],
    "both_violation_w": [
        {
            "role": "system",
            "content": "You are a distractor generator for multiple-choice questions. Given a TARGET sentence, PROFILE, and CONVERSATION HISTORY, generate exactly one sentence that is realistic, keeps the same language and sentence type as TARGET, but clearly violates BOTH the PROFILE and the conversation context. Keep it close in style and length to the TARGET, avoid verbatim copying, and ensure it is definitely wrong for both constraints. Use first person when natural. Do NOT explain or add quotes; output only the sentence.",
        },
        {"role": "user", "content": "TARGET: {correct_output}\nPROFILE: {profile}\nCONVERSATION: {conversation}"},
    ],
    "profile_violation_w/o": [
        {
            "role": "system",
            "content": "You are a distractor generator for multiple-choice questions. Given a PROFILE, generate exactly one realistic sentence that a person would say which clearly contradicts the PROFILE (opposite trait/preference/stance). Keep the output natural, moderate in length, and plausible in everyday context. Use first person ('I', 'me', 'my') as if you are that person. Avoid extreme/off-topic content and avoid meta text. Do NOT explain or add quotes; output only the sentence.",
        },
        {"role": "user", "content": "PROFILE: {profile}"},
    ],
    "conversation_violation_w/o": [
        {
            "role": "system",
            "content": "You are a distractor generator for multiple-choice questions. Given a CONVERSATION HISTORY, generate exactly one realistic sentence that appears plausible but subtly disregards the conversation flow or a key constraint (e.g., answers a related but different question, wrong recipient, ignores an instruction). Keep it natural, moderate in length, near-topic (not random), and avoid meta text. Do NOT explain or add quotes; output only the sentence.",
        },
        {"role": "user", "content": "CONVERSATION: {conversation}"},
    ],
    "both_violation_w/o": [
        {
            "role": "system",
            "content": "You are a distractor generator for multiple-choice questions. Given a PROFILE and CONVERSATION HISTORY, generate exactly one realistic sentence that clearly violates BOTH the PROFILE and the conversation context. Keep it natural, moderate in length, near-topic (not random), and avoid meta text or verbatim copying. Use first person when natural. Do NOT explain or add quotes; output only the sentence.",
        },
        {"role": "user", "content": "PROFILE: {profile}\nCONVERSATION: {conversation}"},
    ],
}

# ========================= 2. 配置参数 ========================= #


class Config:
    def __init__(self):
        # 数据路径
        self.data_path = "/project/hdtaccuracy/Personality-Alignment/split_data_v6_filtered/filtered_dataset.jsonl"
        self.save_path = "/project/hdtaccuracy/Personality-Alignment/choice_ver/raw_choice_data_v7_hard.jsonl"

        # 模型配置
        self.model_type = "local"  # "local", "gpt", 或 "gemini"
        self.model_path = "/project/hdtaccuracy/models/base/Qwen3-8B"
        self.gpt_model = "gpt-3.5-turbo"
        self.gpt_api_key = None
        self.gpt_base_url = None

        # Gemini配置
        self.gemini_model = "gemini-1.5-flash"
        self.gemini_api_key = None

        # 生成配置
        self.batch_size = 64
        self.max_retries = 3
        self.retry_delay = 1
        self.quantize = False
        self.device_map = "auto"

        # 数据配置
        self.data_limit = None


# ========================= 3. 读取原始数据 ========================= #
def load_data(config: Config) -> List[dict]:
    """加载数据"""
    data: list[dict] = []
    with open(config.data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    if config.data_limit:
        data = data[: config.data_limit]

    return data


# ========================= 4. 模型相关函数 ========================= #


class ModelWrapper:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.gemini_model = None

        if config.model_type == "local":
            self._load_local_model()
        elif config.model_type == "gpt":
            self._setup_gpt_client()
        elif config.model_type == "gemini":
            self._setup_gemini_client()

    def _load_local_model(self):
        """加载本地模型"""
        print(f"Loading local model: {self.config.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"

        quant_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if self.config.quantize
            else None
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            device_map=self.config.device_map,
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
            trust_remote_code=True,
        )

    def _setup_gpt_client(self):
        """设置GPT客户端"""
        if not self.config.gpt_api_key:
            raise ValueError("GPT API key is required when using GPT models")

        openai.api_key = self.config.gpt_api_key
        if self.config.gpt_base_url:
            openai.base_url = self.config.gpt_base_url

        print(f"Using GPT model: {self.config.gpt_model}")

    def _setup_gemini_client(self):
        """设置Gemini客户端"""
        if not self.config.gemini_api_key:
            raise ValueError("Gemini API key is required when using Gemini models")

        self.gemini_base_url = "https://api.apiplus.org/v1beta/models"
        print(f"Using Gemini model: {self.config.gemini_model}")

    def generate_batch(self, prompts: List[str], max_new_tokens: int = 100) -> List[str]:
        """批量生成文本"""
        if self.config.model_type == "local":
            return self._generate_local_batch(prompts, max_new_tokens)
        elif self.config.model_type == "gpt":
            return self._generate_gpt_batch(prompts, max_new_tokens)
        elif self.config.model_type == "gemini":
            return self._generate_gemini_batch(prompts, max_new_tokens)

    def _generate_local_batch(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        """使用本地模型批量生成"""
        input_tokens = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=get_max_length("default"),
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **input_tokens,
                max_new_tokens=max_new_tokens,
                # do_sample=True,
                # temperature=0.7,
                # top_p=0.9,
            )

        results = []
        for j in range(len(prompts)):
            input_length = len(input_tokens["input_ids"][j])
            generated_tokens = outputs[j][input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            results.append(clean_generated_text(generated_text))

        return results

    def _generate_gpt_batch(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        """使用GPT API批量生成（多进程模式）"""

        def generate_single_prompt(prompt: str) -> str:
            """生成单个prompt的响应"""
            client = openai.OpenAI(api_key=self.config.gpt_api_key, base_url=self.config.gpt_base_url)

            for attempt in range(self.config.max_retries):
                try:
                    response = client.responses.create(
                        model=self.config.gpt_model,
                        input=prompt,
                        max_output_tokens=max_new_tokens,
                        temperature=0.7,
                        reasoning={"effort": "low"},
                    )

                    generated_text = response.output_text
                    return clean_generated_text(generated_text)

                except Exception as e:
                    print(f"GPT API error (attempt {attempt + 1}): {e}")
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay)
                    else:
                        return f"GPT_API_ERROR_{threading.current_thread().ident}"

        # 使用ThreadPoolExecutor进行并发处理
        max_workers = min(len(prompts), 16)
        results = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(generate_single_prompt, prompt): i for i, prompt in enumerate(prompts)}

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    print(f"Future execution error for prompt {index}: {e}")
                    results[index] = f"FUTURE_ERROR_{index}"

        return results

    def _generate_gemini_batch(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        """使用Gemini API批量生成（多线程模式）"""

        def generate_single_prompt(prompt: str) -> str:
            """生成单个prompt的响应"""
            for attempt in range(self.config.max_retries):
                try:
                    url = f"{self.gemini_base_url}/{self.config.gemini_model}:generateContent"
                    params = {"key": self.config.gemini_api_key}
                    headers = {"Content-Type": "application/json", "x-goog-api-key": self.config.gemini_api_key}

                    data = {
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.7,
                            "maxOutputTokens": max_new_tokens,
                            "thinkingConfig": {"includeThoughts": False, "thinkingBudget": 0},
                        },
                    }

                    response = requests.post(url=url, params=params, headers=headers, json=data, timeout=60)

                    if response.status_code == 200:
                        response_data = response.json()

                        if "candidates" in response_data and len(response_data["candidates"]) > 0:
                            candidate = response_data["candidates"][0]
                            if "content" in candidate and "parts" in candidate["content"]:
                                parts = candidate["content"]["parts"]
                                if len(parts) > 0 and "text" in parts[0]:
                                    generated_text = parts[0]["text"]
                                    return clean_generated_text(generated_text)

                        print(
                            f"Gemini API returned unexpected response format (attempt {attempt + 1}): {response_data}"
                        )
                        if attempt < self.config.max_retries - 1:
                            time.sleep(self.config.retry_delay)
                        else:
                            return f"GEMINI_UNEXPECTED_FORMAT_{threading.current_thread().ident}"

                    else:
                        print(
                            f"Gemini API HTTP error (attempt {attempt + 1}): {response.status_code} - {response.text}"
                        )
                        if attempt < self.config.max_retries - 1:
                            time.sleep(self.config.retry_delay)
                        else:
                            return f"GEMINI_HTTP_ERROR_{response.status_code}_{threading.current_thread().ident}"

                except requests.exceptions.RequestException as e:
                    print(f"Gemini API request error (attempt {attempt + 1}): {e}")
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay)
                    else:
                        return f"GEMINI_REQUEST_ERROR_{threading.current_thread().ident}"
                except Exception as e:
                    print(f"Gemini API unexpected error (attempt {attempt + 1}): {e}")
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay)
                    else:
                        return f"GEMINI_UNKNOWN_ERROR_{threading.current_thread().ident}"

        # 使用ThreadPoolExecutor进行并发处理
        max_workers = min(len(prompts), 8)
        results = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {executor.submit(generate_single_prompt, prompt): i for i, prompt in enumerate(prompts)}

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    print(f"Future execution error for prompt {index}: {e}")
                    results[index] = f"FUTURE_ERROR_{index}"

        return results


def extract_profile_and_history(prompt: str) -> tuple:
    """
    Extract profile and conversation history from the prompt
    """
    profile = ""
    conversation_history = ""

    # Extract profile
    if "[Profile Begin]" in prompt and "[Profile End]" in prompt:
        start = prompt.find("[Profile Begin]") + len("[Profile Begin]")
        end = prompt.find("[Profile End]")
        profile = prompt[start:end].strip()

    # Extract conversation history
    if "[Conversation History Begin]" in prompt and "[Conversation History End]" in prompt:
        start = prompt.find("[Conversation History Begin]") + len("[Conversation History Begin]")
        end = prompt.find("[Conversation History End]")
        conversation_history = prompt[start:end].strip()

    return profile, conversation_history


def get_max_length(mode: str) -> int:
    """根据模式返回最大长度"""
    if mode in ["style_violation", "topic_violation", "richness_violation", "free_violation"]:
        return 512
    elif mode in ["profile_violation_w", "profile_violation_w/o"]:
        return 1024
    elif mode in ["conversation_violation_w", "conversation_violation_w/o", "both_violation_w", "both_violation_w/o"]:
        return 8192
    else:
        return 1024


def generate_distractors_by_mode(data_items, mode, model_wrapper: ModelWrapper, batch_size=8):
    """根据指定模式生成干扰项"""
    all_distractors = []

    # 构造所有输入
    all_inputs = []
    for item in data_items:
        correct_output = item["output"]
        profile, conversation = extract_profile_and_history(item["prompt"])

        # 根据模式构造prompt
        if mode in ["style_violation", "topic_violation", "richness_violation", "free_violation"]:
            messages = [
                DISTRACTOR_PROMPTS[mode][0],
                {
                    "role": "user",
                    "content": DISTRACTOR_PROMPTS[mode][1]["content"].format(correct_output=correct_output),
                },
            ]
        elif mode == "profile_violation_w":
            messages = [
                DISTRACTOR_PROMPTS[mode][0],
                {
                    "role": "user",
                    "content": DISTRACTOR_PROMPTS[mode][1]["content"].format(
                        correct_output=correct_output, profile=profile
                    ),
                },
            ]
        elif mode == "conversation_violation_w":
            messages = [
                DISTRACTOR_PROMPTS[mode][0],
                {
                    "role": "user",
                    "content": DISTRACTOR_PROMPTS[mode][1]["content"].format(
                        correct_output=correct_output, conversation=conversation
                    ),
                },
            ]
        elif mode == "both_violation_w":
            messages = [
                DISTRACTOR_PROMPTS[mode][0],
                {
                    "role": "user",
                    "content": DISTRACTOR_PROMPTS[mode][1]["content"].format(
                        correct_output=correct_output, profile=profile, conversation=conversation
                    ),
                },
            ]
        elif mode == "profile_violation_w/o":
            messages = [
                DISTRACTOR_PROMPTS[mode][0],
                {
                    "role": "user",
                    "content": DISTRACTOR_PROMPTS[mode][1]["content"].format(profile=profile),
                },
            ]
        elif mode == "conversation_violation_w/o":
            messages = [
                DISTRACTOR_PROMPTS[mode][0],
                {
                    "role": "user",
                    "content": DISTRACTOR_PROMPTS[mode][1]["content"].format(conversation=conversation),
                },
            ]
        elif mode == "both_violation_w/o":
            messages = [
                DISTRACTOR_PROMPTS[mode][0],
                {
                    "role": "user",
                    "content": DISTRACTOR_PROMPTS[mode][1]["content"].format(
                        profile=profile, conversation=conversation
                    ),
                },
            ]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 根据模型类型格式化prompt
        if model_wrapper.config.model_type == "local" and model_wrapper.tokenizer:
            prompt_text = model_wrapper.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        elif model_wrapper.config.model_type == "gemini":
            prompt_text = f"{messages[0]['content']}\n\n{messages[1]['content']}"
        else:
            prompt_text = f"{messages[0]['content']}\n\n{messages[1]['content']}"

        all_inputs.append(prompt_text)

    # 分批生成
    generated_distractors = []
    for i in tqdm(range(0, len(all_inputs), batch_size), desc=f"Generating {mode} distractors"):
        batch_inputs = all_inputs[i : i + batch_size]

        # 使用模型包装器生成
        batch_outputs = model_wrapper.generate_batch(batch_inputs, max_new_tokens=100)

        # 处理每个输出
        for j, generated_text in enumerate(batch_outputs):
            # 确保生成的干扰项不是正确答案
            correct_output = data_items[i + j]["output"]
            if generated_text != correct_output and generated_text.strip():
                generated_distractors.append(generated_text)
            else:
                print(f"Warning: Generated distractor failed for mode {mode}: {generated_text}")
                generated_distractors.append(f"Failed_{mode}_distractor_{len(generated_distractors)}")

    return generated_distractors


def clean_generated_text(text: str) -> str:
    """清理生成的文本"""
    text = text.strip()
    if "</think>" in text:
        text = text[text.index("</think>") + len("</think>") :].strip()
    if text.startswith("[Assistant]"):
        text = text[text.index("[Assistant]") + len("[Assistant]") :].strip()
    if text.endswith("</s>"):
        text = text[: -len("</s>")].strip()
    if "TARGET:" in text:
        text = text.split("TARGET:")[-1].strip()
    if "OUTPUT:" in text:
        text = text.split("OUTPUT:")[-1].strip()
    return text.strip()


def generate_all_distractors_batch(
    data_items, model_wrapper: ModelWrapper, batch_size=8, start_index=None, end_index=None, modes_to_generate=None
):
    """为每个数据项生成所有10种模式的干扰项"""
    all_modes = [
        "style_violation",
        "topic_violation",
        "richness_violation",
        "free_violation",
        "profile_violation_w",
        "conversation_violation_w",
        "both_violation_w",
        "profile_violation_w/o",
        "conversation_violation_w/o",
        "both_violation_w/o",
    ]

    # 确定要生成的模式
    if modes_to_generate is not None:
        modes = [mode for mode in modes_to_generate if mode in all_modes]
        if not modes:
            raise ValueError(f"No valid modes specified. Valid modes: {all_modes}")
    else:
        modes = all_modes

    # 确定数据范围
    if start_index is not None or end_index is not None:
        start_idx = start_index if start_index is not None else 0
        end_idx = end_index if end_index is not None else len(data_items)
        start_idx = max(0, start_idx)
        end_idx = min(len(data_items), end_idx)

        if start_idx >= end_idx:
            raise ValueError(f"Invalid index range: start_index({start_idx}) >= end_index({end_idx})")

        data_subset = data_items[start_idx:end_idx]
        print(f"处理数据范围: [{start_idx}:{end_idx}] (共 {len(data_subset)} 条)")
    else:
        data_subset = data_items
        start_idx = 0
        print(f"处理全部数据: {len(data_subset)} 条")

    # 为每种模式生成干扰项
    all_mode_distractors = {}
    for mode in modes:
        print(f"\n正在生成 {mode} 模式的干扰项...")
        mode_distractors = generate_distractors_by_mode(data_subset, mode, model_wrapper, batch_size)
        all_mode_distractors[mode] = mode_distractors

    return all_mode_distractors, start_idx


def process_original_prompt(prompt: str) -> str:
    """处理原始提示，提取需要的部分，并重构"""
    profile, conversation_history = extract_profile_and_history(prompt)
    new_prompt = f"[Profile Begin]{profile}[Profile End]\n"
    new_prompt += f"[Conversation History Begin]{conversation_history}[Conversation History End]\n"

    return new_prompt


def process_data_batch(
    data, model_wrapper: ModelWrapper, batch_size=8, start_index=None, end_index=None, modes_to_generate=None
):
    """批量处理数据:生成干扰项并保存所有模式的结果"""
    # 确定数据范围
    if start_index is not None or end_index is not None:
        start_idx = start_index if start_index is not None else 0
        end_idx = end_index if end_index is not None else len(data)
        start_idx = max(0, start_idx)
        end_idx = min(len(data), end_idx)

        if start_idx >= end_idx:
            raise ValueError(f"Invalid index range: start_index({start_idx}) >= end_index({end_idx})")

        data_subset = data[start_idx:end_idx]
    else:
        data_subset = data
        start_idx = 0

    correct_outputs = [item["output"] for item in data_subset]
    qids = [item["qid"] for item in data_subset]

    print("开始批量生成多模式干扰项…")
    all_mode_distractors, actual_start_idx = generate_all_distractors_batch(
        data_subset,
        model_wrapper,
        batch_size,
        start_index=None,
        end_index=None,
        modes_to_generate=modes_to_generate,
    )

    print("构建新数据集…")
    new_data: list[dict] = []

    for i, (qid, original_prompt, correct_output) in enumerate(
        tqdm(
            zip(qids, [item["prompt"] for item in data_subset], correct_outputs),
            total=len(data_subset),
            desc="构建新数据集",
        )
    ):
        prompt_new = f"{process_original_prompt(original_prompt)}\n" "Your choice: /no_think"

        # 构建数据项，包含所有模式的干扰项
        data_item = {
            "qid": qid,
            "prompt": prompt_new,
            "output": correct_output,
            "original_index": start_idx + i,
        }

        # 添加所有模式的干扰项
        for mode, mode_distractors in all_mode_distractors.items():
            if i < len(mode_distractors):
                data_item[f"{mode}_distractor"] = mode_distractors[i]
            else:
                data_item[f"{mode}_distractor"] = f"Missing_{mode}_distractor"

        # 检查每种模式的干扰项内容相互不一致
        distractors = []
        for mode in all_mode_distractors.keys():
            if i < len(all_mode_distractors[mode]):
                distractors.append(all_mode_distractors[mode][i])

        # 检查是否有重复的干扰项
        unique_distractors = set(distractors)
        if len(unique_distractors) != len(distractors):
            print(f"Warning: QID {qid} (index {start_idx + i}) has duplicate distractors")
            for j, d in enumerate(distractors):
                if distractors.count(d) > 1:
                    print(f"  Duplicate: '{d}' appears {distractors.count(d)} times")

        # 检查干扰项是否与正确答案相同
        for mode, distractor in zip(all_mode_distractors.keys(), distractors):
            if distractor == correct_output:
                print(f"Warning: QID {qid} (index {start_idx + i}), {mode} distractor is same as correct output")

        new_data.append(data_item)

    return new_data


def test_single_mode(data, mode, model_wrapper: ModelWrapper, batch_size=8):
    """测试单一模式的干扰项生成效果"""
    print(f"\n{'='*50}")
    print(f"测试模式: {mode}")
    print(f"{'='*50}")

    # 取前10条数据进行测试
    test_data = data[:10]
    distractors = generate_distractors_by_mode(test_data, mode, model_wrapper, batch_size)

    for i, (item, distractor) in enumerate(zip(test_data, distractors)):
        print(f"\n--- 样本 {i+1} ---")
        print(f"正确答案: {item['output']}")
        print(f"{mode} 干扰项: {distractor}")

        if i >= 4:  # 只显示前5个样本
            break

    return distractors


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生成Hard级别选择题数据")

    # 数据配置
    parser.add_argument(
        "--data_path",
        type=str,
        default="/project/hdtaccuracy/Personality-Alignment/dialogue_dataset_all_v9_summarized_cleaned.jsonl",
        help="输入数据路径",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/project/hdtaccuracy/Personality-Alignment/choice_ver/v10/raw_choice_data_v10_hard.jsonl",
        help="输出数据路径",
    )
    parser.add_argument("--data_limit", type=int, default=None, help="限制处理的数据条数 (用于测试)")

    # 索引范围配置
    parser.add_argument("--start_index", type=int, default=None, help="开始处理的索引 (包含)")
    parser.add_argument("--end_index", type=int, default=None, help="结束处理的索引 (不包含)")

    # 模式选择配置
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "style_violation",
            "topic_violation",
            "richness_violation",
            "free_violation",
            "profile_violation_w",
            "conversation_violation_w",
            "both_violation_w",
            "profile_violation_w/o",
            "conversation_violation_w/o",
            "both_violation_w/o",
        ],
        default=None,
        help="指定要生成的模式列表 (如果不指定则生成所有模式)",
    )

    # 模型配置
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["local", "gpt", "gemini"],
        default="local",
        help="模型类型: local, gpt 或 gemini",
    )
    parser.add_argument(
        "--model_path", type=str, default="/project/hdtaccuracy/models/base/Qwen3-8B", help="本地模型路径"
    )
    parser.add_argument("--gpt_model", type=str, default="gpt-3.5-turbo", help="GPT模型名称")
    parser.add_argument("--gpt_api_key", type=str, default=None, help="GPT API密钥")
    parser.add_argument("--gpt_base_url", type=str, default=None, help="GPT API基础URL")

    # Gemini配置
    parser.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-1.5-flash",
        help="Gemini模型名称 (gemini-1.5-flash 或 gemini-1.5-pro)",
    )
    parser.add_argument("--gemini_api_key", type=str, default=None, help="Gemini API密钥")

    # 生成配置
    parser.add_argument("--batch_size", type=int, default=64, help="批处理大小")
    parser.add_argument("--quantize", action="store_true", help="是否使用4bit量化")
    parser.add_argument("--device_map", type=str, default="auto", help="设备映射")
    parser.add_argument("--max_retries", type=int, default=3, help="API调用最大重试次数")
    parser.add_argument("--retry_delay", type=float, default=1.0, help="重试延迟时间(秒)")

    # 测试模式
    parser.add_argument("--test_mode", action="store_true", help="启用测试模式，只测试各种干扰项生成效果")

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建配置
    config = Config()
    config.data_path = args.data_path
    config.save_path = args.save_path
    config.data_limit = args.data_limit
    config.model_type = args.model_type
    config.model_path = args.model_path
    config.gpt_model = args.gpt_model
    config.gpt_api_key = args.gpt_api_key
    config.gpt_base_url = args.gpt_base_url
    config.gemini_model = args.gemini_model
    config.gemini_api_key = args.gemini_api_key
    config.batch_size = args.batch_size
    config.quantize = args.quantize
    config.device_map = args.device_map
    config.max_retries = args.max_retries
    config.retry_delay = args.retry_delay

    print("=" * 60)
    print("Hard级别选择题数据生成工具")
    print("=" * 60)
    print(f"模型类型: {config.model_type}")
    if config.model_type == "local":
        print(f"模型路径: {config.model_path}")
    elif config.model_type == "gpt":
        print(f"GPT模型: {config.gpt_model}")
    elif config.model_type == "gemini":
        print(f"Gemini模型: {config.gemini_model}")
    print(f"批处理大小: {config.batch_size}")
    print(f"数据路径: {config.data_path}")
    print(f"保存路径: {config.save_path}")
    if config.data_limit:
        print(f"数据限制: {config.data_limit}")

    # 显示索引范围和模式信息
    if args.start_index is not None or args.end_index is not None:
        start_idx = args.start_index if args.start_index is not None else 0
        end_idx = args.end_index if args.end_index is not None else "END"
        print(f"处理索引范围: [{start_idx}:{end_idx}]")

    if args.modes:
        print(f"指定生成模式: {', '.join(args.modes)}")
    else:
        print("生成所有模式")
    print("=" * 60)

    # 加载数据
    print("加载数据...")
    data = load_data(config)
    print(f"加载了 {len(data)} 条数据")

    # 验证索引范围
    if args.start_index is not None or args.end_index is not None:
        start_idx = args.start_index if args.start_index is not None else 0
        end_idx = args.end_index if args.end_index is not None else len(data)

        if start_idx < 0 or start_idx >= len(data):
            raise ValueError(f"start_index ({start_idx}) 超出数据范围 [0, {len(data)})")
        if end_idx < 0 or end_idx > len(data):
            raise ValueError(f"end_index ({end_idx}) 超出数据范围 [0, {len(data)}]")
        if start_idx >= end_idx:
            raise ValueError(f"start_index ({start_idx}) 必须小于 end_index ({end_idx})")

        actual_count = end_idx - start_idx
        print(f"实际处理范围: [{start_idx}:{end_idx}] (共 {actual_count} 条)")

    # 初始化模型
    print("初始化模型...")
    model_wrapper = ModelWrapper(config)

    if args.test_mode:
        # 测试模式
        print("\n进入测试模式...")
        test_modes = (
            args.modes
            if args.modes
            else [
                "style_violation",
                "topic_violation",
                "richness_violation",
                "profile_violation_w",
                "conversation_violation_w",
                "both_violation_w",
            ]
        )

        # 在测试模式中也支持索引范围
        test_data = data
        if args.start_index is not None or args.end_index is not None:
            start_idx = args.start_index if args.start_index is not None else 0
            end_idx = args.end_index if args.end_index is not None else len(data)
            test_data = data[start_idx : min(end_idx, start_idx + 10)]
        else:
            test_data = data[:10]

        for mode in test_modes:
            test_single_mode(test_data, mode, model_wrapper, config.batch_size)
    else:
        # 正式处理数据
        actual_data_count = len(data)
        if args.start_index is not None or args.end_index is not None:
            start_idx = args.start_index if args.start_index is not None else 0
            end_idx = args.end_index if args.end_index is not None else len(data)
            actual_data_count = end_idx - start_idx

        print(f"\n开始处理 {actual_data_count} 条数据...")
        new_data = process_data_batch(
            data,
            model_wrapper,
            batch_size=config.batch_size,
            start_index=args.start_index,
            end_index=args.end_index,
            modes_to_generate=args.modes,
        )

        # 保存结果 - 如果是部分处理，在文件名中加入范围信息
        save_path = config.save_path
        if not save_path.endswith(".jsonl"):
            save_path += "result.jsonl"
        directory_path = os.path.dirname(save_path)
        if directory_path and not os.path.exists(directory_path):
            os.makedirs(directory_path)

        if args.start_index is not None or args.end_index is not None:
            start_idx = args.start_index if args.start_index is not None else 0
            end_idx = args.end_index if args.end_index is not None else len(data)

            # 在文件名中插入范围信息
            base_name, ext = save_path.rsplit(".", 1) if "." in save_path else (save_path, "jsonl")
            save_path = f"{base_name}_range_{start_idx}_{end_idx}.{ext}"

        if args.modes:
            # 在文件名中加入模式信息
            base_name, ext = save_path.rsplit(".", 1) if "." in save_path else (save_path, "jsonl")
            modes_str = "_".join(args.modes)
            save_path = f"{base_name}_modes_{modes_str}.{ext}"

        print("保存新数据集...")
        with open(save_path, "w", encoding="utf-8") as f:
            for item in new_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"处理完成！共生成 {len(new_data)} 条新数据")
        print(f"结果已保存到: {save_path}")

        # 统计信息
        print(f"\n生成模式说明：")
        print(f"1. style_violation: 违反句式结构")
        print(f"2. topic_violation: 违反话题内容")
        print(f"3. richness_violation: 违反内容丰富度")
        print(f"4. free_violation: 自由违反（表达不同意思/意图）")
        print(f"5. profile_violation_w: 违反个性档案 (with target)")
        print(f"6. conversation_violation_w: 违反对话上下文 (with target)")
        print(f"7. both_violation_w: 同时违反档案和对话 (with target)")
        print(f"8. profile_violation_w/o: 违反个性档案 (without target)")
        print(f"9. conversation_violation_w/o: 违反对话上下文 (without target)")
        print(f"10. both_violation_w/o: 同时违反档案和对话 (without target)")


if __name__ == "__main__":
    main()

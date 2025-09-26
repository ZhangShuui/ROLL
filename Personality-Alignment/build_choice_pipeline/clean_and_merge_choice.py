#!/usr/bin/env python3

import json
import os
import glob
import argparse
import random
import time
import threading
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import re
from dataclasses import dataclass


@dataclass
class FixTask:
    """修复任务的数据结构"""

    question_idx: int
    choice_idx: int
    original_text: str
    correct_answer: str
    profile: str
    conversation_history: str
    existing_distractors: List[str]
    question_id: str  # 用于日志


class ThreadSafeCounter:
    """线程安全的计数器"""

    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def increment(self):
        with self.lock:
            self.value += 1
            return self.value

    def get(self):
        with self.lock:
            return self.value


class PromptLoader:
    """处理原始prompt JSONL文件的加载和查询"""

    def __init__(self, prompt_file_path: str = None):
        self.prompt_data = {}
        if prompt_file_path:
            self.load_prompt_file(prompt_file_path)

    def load_prompt_file(self, file_path: str):
        """加载原始prompt JSONL文件"""
        print(f"Loading prompt file: {file_path}")

        if not os.path.exists(file_path):
            print(f"Warning: Prompt file not found: {file_path}")
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        qid = data.get("qid")
                        if qid:
                            self.prompt_data[qid] = data
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}: {e}")
                        continue

            print(f"Loaded {len(self.prompt_data)} prompts from {file_path}")

        except Exception as e:
            print(f"Error loading prompt file {file_path}: {e}")

    def get_prompt_by_qid(self, qid: str) -> str:
        """根据qid获取对应的prompt"""
        if qid in self.prompt_data:
            # 尝试不同的可能字段名
            for field in ["prompt", "input", "text", "question"]:
                if field in self.prompt_data[qid]:
                    return self.prompt_data[qid][field]

            # 如果没有找到标准字段，返回整个数据的字符串表示
            return str(self.prompt_data[qid])

        return ""

    def extract_profile_conv_from_qid(self, qid: str) -> Tuple[str, str]:
        """根据qid提取profile和conversation history"""
        prompt = self.get_prompt_by_qid(qid)
        if not prompt:
            return "", ""

        return self.extract_profile_conv_from_prompt(prompt)

    def extract_profile_conv_from_prompt(self, prompt: str) -> Tuple[str, str]:
        """从prompt文本中提取profile和conversation history - 使用与build_choice_training_data.py相同的逻辑"""
        profile = ""
        conversation_history = ""

        # 方法1: 查找标准标记
        if "[Profile Begin]" in prompt and "[Profile End]" in prompt:
            start = prompt.find("[Profile Begin]") + len("[Profile Begin]")
            end = prompt.find("[Profile End]")
            if end > start:
                profile = prompt[start:end].strip()

        if "[Conversation History Begin]" in prompt and "[Conversation History End]" in prompt:
            start = prompt.find("[Conversation History Begin]") + len("[Conversation History Begin]")
            end = prompt.find("[Conversation History End]")
            if end > start:
                conversation_history = prompt[start:end].strip()

        # 方法2: 如果没找到标准标记，尝试其他常见模式
        if not profile and not conversation_history:
            # 尝试查找其他可能的分隔符
            patterns = [
                (r"Profile:?\s*(.*?)\s*(?:Conversation|History|$)", "profile"),
                (r"Person:?\s*(.*?)\s*(?:Conversation|History|$)", "profile"),
                (r"Character:?\s*(.*?)\s*(?:Conversation|History|$)", "profile"),
                (r"(?:Conversation|History):?\s*(.*?)(?:$|\n\n)", "conversation"),
            ]

            for pattern, field_type in patterns:
                matches = re.findall(pattern, prompt, re.DOTALL | re.IGNORECASE)
                if matches:
                    content = matches[0].strip()
                    if content and len(content) > 10:  # 确保内容有意义
                        if field_type == "profile":
                            profile = content
                        else:
                            conversation_history = content

        # 方法3: 如果仍然没找到，尝试从JSON结构中提取
        if not profile and not conversation_history:
            try:
                # 尝试解析为JSON
                if prompt.startswith("{") or prompt.startswith("["):
                    data = json.loads(prompt)
                    if isinstance(data, dict):
                        # 查找可能的profile字段
                        for key in ["profile", "person", "character", "user_profile"]:
                            if key in data and data[key]:
                                profile = str(data[key]).strip()
                                break

                        # 查找可能的conversation字段
                        for key in ["conversation", "history", "chat_history", "messages"]:
                            if key in data and data[key]:
                                if isinstance(data[key], list):
                                    conversation_history = "\n".join(str(msg) for msg in data[key])
                                else:
                                    conversation_history = str(data[key]).strip()
                                break
            except (json.JSONDecodeError, Exception):
                pass

        return profile, conversation_history


class ResultMergerParallel:
    """并行版本的合并和修复工具类"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.apiplus.org/v1",
        max_workers: int = 8,
        prompt_loader: PromptLoader = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.max_workers = max_workers
        self.prompt_loader = prompt_loader or PromptLoader()
        self.problematic_patterns = [
            "I cannot assist with that request",
            "I can't assist with that",
            "I cannot help with that",
            "I'm unable to assist",
            "I cannot provide",
            "I can't provide",
        ]
        self.success_counter = ThreadSafeCounter()
        self.failure_counter = ThreadSafeCounter()

    def create_client(self) -> OpenAI:
        """为每个线程创建独立的OpenAI客户端"""
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def find_json_files(self, directory: str, pattern: str = "*.json") -> List[str]:
        """查找匹配模式的JSON文件"""
        search_pattern = os.path.join(directory, pattern)
        files = glob.glob(search_pattern)
        # 按文件名排序确保正确顺序
        files.sort()
        return files

    def load_and_merge_files(self, file_paths: List[str]) -> List[Dict]:
        """并行加载并合并多个JSON文件"""
        all_questions = []

        print(f"并行加载 {len(file_paths)} 个文件...")

        def load_single_file(file_path: str) -> List[Dict]:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    else:
                        print(f"警告: {file_path} 不是列表格式，跳过")
                        return []
            except Exception as e:
                print(f"加载文件 {file_path} 时出错: {e}")
                return []

        # 并行加载文件
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(file_paths))) as executor:
            future_to_file = {executor.submit(load_single_file, fp): fp for fp in file_paths}

            for future in tqdm(as_completed(future_to_file), total=len(file_paths), desc="Loading files"):
                file_path = future_to_file[future]
                try:
                    questions = future.result()
                    all_questions.extend(questions)
                except Exception as e:
                    print(f"处理文件 {file_path} 结果时出错: {e}")

        # 按original_index排序保持顺序
        print("排序问题...")
        all_questions.sort(key=lambda x: x.get("original_index", 0))

        print(f"总共合并了 {len(all_questions)} 个问题")
        return all_questions

    def check_problematic_choices(self, questions: List[Dict]) -> Tuple[List[Dict], List[FixTask]]:
        """并行检查有问题的选项"""
        fix_tasks = []

        print("并行检查有问题的选项...")

        def check_single_question(q_idx_and_question: Tuple[int, Dict]) -> List[FixTask]:
            q_idx, question = q_idx_and_question
            tasks = []

            if "choices" not in question:
                return tasks

            # 使用qid从prompt文件中提取上下文信息
            qid = question.get("qid", "")
            if qid and self.prompt_loader:
                profile, conversation_history = self.prompt_loader.extract_profile_conv_from_qid(qid)
            else:
                # 回退到从original_prompt字段提取
                prompt = question.get("original_prompt", "")
                profile, conversation_history = self.prompt_loader.extract_profile_conv_from_prompt(prompt)

            # 找到正确答案
            correct_answer = ""
            for choice in question["choices"]:
                if choice.get("is_correct", False):
                    correct_answer = choice["text"]
                    break

            # 收集现有的干扰项
            existing_distractors = []
            for choice in question["choices"]:
                if not choice.get("is_correct", False):
                    existing_distractors.append(choice["text"])

            # 检查每个选项
            for c_idx, choice in enumerate(question["choices"]):
                if "text" not in choice or choice.get("is_correct", False):
                    continue

                choice_text = choice["text"].strip()

                # 检查是否包含问题模式
                for pattern in self.problematic_patterns:
                    if pattern.lower() in choice_text.lower():
                        # 创建修复任务
                        task = FixTask(
                            question_idx=q_idx,
                            choice_idx=c_idx,
                            original_text=choice_text,
                            correct_answer=correct_answer,
                            profile=profile,
                            conversation_history=conversation_history,
                            existing_distractors=[d for d in existing_distractors if d != choice_text],
                            question_id=question.get("qid", f"q_{q_idx}"),
                        )
                        tasks.append(task)
                        break

            return tasks

        # 并行检查所有问题
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            question_items = list(enumerate(questions))
            future_to_idx = {executor.submit(check_single_question, item): item[0] for item in question_items}

            for future in tqdm(as_completed(future_to_idx), total=len(question_items), desc="Checking questions"):
                try:
                    tasks = future.result()
                    fix_tasks.extend(tasks)
                except Exception as e:
                    q_idx = future_to_idx[future]
                    print(f"检查问题 {q_idx} 时出错: {e}")

        print(f"发现 {len(fix_tasks)} 个有问题的选项")
        return questions, fix_tasks

    def generate_replacement_distractor_single(
        self, task: FixTask, model_name: str = "gpt-5-mini-2025-08-07", max_retries: int = 3, thread_id: int = 0
    ) -> Optional[Tuple[FixTask, str]]:
        """单个任务的干扰项生成 - 线程安全版本"""

        client = self.create_client()

        # 创建上下文描述
        context_parts = []
        if task.profile:
            context_parts.append(f"Person's Profile:\n{task.profile}")
        if task.conversation_history:
            context_parts.append(f"Conversation History:\n{task.conversation_history}")

        context_description = "\n\n".join(context_parts) if context_parts else "No specific context provided."

        # 格式化现有干扰项
        existing_list = []
        for i, text in enumerate(task.existing_distractors):
            existing_list.append(f"{i+1}. {text}")

        enhancement_prompt = f"""You are tasked with creating one highly challenging distractor for a personality-based multiple-choice question. 

The original distractor was problematic and contained refusal text: "{task.original_text}"

Context:
{context_description}

Correct Answer: {task.correct_answer}

Current Other Distractors:
{chr(10).join(existing_list) if existing_list else "None"}

Create ONE new distractor that is:

1. **Highly Similar**: Close enough to the correct answer that it requires deep understanding to distinguish
2. **Personality-Specific**: Represents how someone with different but plausible personality traits might respond  
3. **Contextually Perfect**: Fits the situation and conversation flow naturally
4. **Psychologically Sound**: Based on real personality differences
5. **Length and Style Matched**: Similar in length and style to the correct answer
6. **NO REFUSAL**: Must be a proper response, not a refusal or "I cannot assist" type message

The distractor should represent a plausible alternative that someone with different personality traits, values, or communication styles might actually give in this exact situation.

Please provide exactly ONE enhanced distractor, with no additional formatting or explanation."""

        for attempt in range(max_retries):
            try:
                print(f"线程 {thread_id}: 生成替换干扰项 {task.question_id} (尝试 {attempt + 1}/{max_retries})")

                response = client.responses.create(
                    model=model_name,
                    input=[
                        {
                            "role": "system",
                            "content": "You are a master of psychological assessment and personality theory. You excel at creating sophisticated distractors that test nuanced understanding of individual differences.",
                        },
                        {"role": "user", "content": enhancement_prompt},
                    ],
                    max_output_tokens=2500,
                    temperature=0.7,
                    reasoning={"effort": "medium"},
                    timeout=50,
                )

                new_distractor = response.output_text.strip()

                # 清理响应
                if new_distractor.startswith(("1.", "-", "*", "•")):
                    new_distractor = new_distractor[2:].strip()

                # 验证新的干扰项不包含问题模式
                has_problem = False
                for pattern in self.problematic_patterns:
                    if pattern.lower() in new_distractor.lower():
                        has_problem = True
                        break

                if not has_problem and new_distractor != task.correct_answer.strip():
                    print(f"线程 {thread_id}: ✅ 成功生成 {task.question_id}")
                    return (task, new_distractor)
                else:
                    print(f"线程 {thread_id}: 生成的干扰项仍有问题，重试...")

            except Exception as e:
                print(f"线程 {thread_id}: 生成失败 (尝试 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))

        return None

    def fix_problematic_questions_parallel(self, questions: List[Dict], fix_tasks: List[FixTask]) -> List[Dict]:
        """并行修复有问题的问题"""

        if not fix_tasks:
            print("没有发现需要修复的问题")
            return questions

        print(f"开始并行修复 {len(fix_tasks)} 个有问题的选项 (使用 {self.max_workers} 个线程)...")

        fixed_questions = [q.copy() for q in questions]
        results = []

        # 并行处理所有修复任务
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {}
            for i, task in enumerate(fix_tasks):
                future = executor.submit(
                    self.generate_replacement_distractor_single, task=task, thread_id=i % self.max_workers
                )
                future_to_task[future] = task

            # 收集结果
            for future in tqdm(as_completed(future_to_task), total=len(fix_tasks), desc="Fixing questions"):
                original_task = future_to_task[future]
                try:
                    result = future.result()
                    if result:
                        task, new_distractor = result
                        results.append((task, new_distractor))
                        self.success_counter.increment()
                    else:
                        print(f"❌ 无法修复 {original_task.question_id}")
                        self.failure_counter.increment()

                except Exception as e:
                    print(f"修复任务 {original_task.question_id} 时出错: {e}")
                    self.failure_counter.increment()

        # 应用修复结果
        print("应用修复结果...")
        for task, new_distractor in tqdm(results, desc="Applying fixes"):
            q_idx = task.question_idx
            c_idx = task.choice_idx

            # 更新选项
            fixed_questions[q_idx]["choices"][c_idx]["text"] = new_distractor
            fixed_questions[q_idx]["choices"][c_idx]["type"] = "enhanced_distractor_fixed"
            fixed_questions[q_idx]["choices"][c_idx]["bank"] = "model_generated_gpt-5-mini"

            # 添加修复记录
            if "metadata" not in fixed_questions[q_idx]:
                fixed_questions[q_idx]["metadata"] = {}
            if "fixed_choices" not in fixed_questions[q_idx]["metadata"]:
                fixed_questions[q_idx]["metadata"]["fixed_choices"] = []

            fixed_questions[q_idx]["metadata"]["fixed_choices"].append(
                {
                    "choice_index": c_idx,
                    "original_text": task.original_text,
                    "new_text": new_distractor,
                    "fix_reason": "contained_refusal_text",
                }
            )

        success_count = self.success_counter.get()
        failure_count = self.failure_counter.get()

        print(f"\n并行修复完成!")
        print(f"成功修复: {success_count}/{len(fix_tasks)} 个选项")
        print(f"修复失败: {failure_count}/{len(fix_tasks)} 个选项")

        return fixed_questions

    def save_results(self, questions: List[Dict], output_path: str, fix_tasks: List[FixTask] = None):
        """保存结果"""
        print(f"保存合并结果到: {output_path}")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)

        # 保存问题报告
        if fix_tasks:
            report_path = output_path.replace(".json", "_fix_report.json")
            report = {
                "total_questions": len(questions),
                "problematic_items_found": len(fix_tasks),
                "successfully_fixed": self.success_counter.get(),
                "failed_to_fix": self.failure_counter.get(),
                "parallel_processing": {"max_workers": self.max_workers, "total_tasks": len(fix_tasks)},
                "prompt_source": {
                    "has_prompt_loader": self.prompt_loader is not None,
                    "loaded_prompts": len(self.prompt_loader.prompt_data) if self.prompt_loader else 0,
                },
                "problematic_details": [
                    {
                        "question_index": task.question_idx,
                        "choice_index": task.choice_idx,
                        "question_id": task.question_id,
                        "problematic_text": task.original_text,
                    }
                    for task in fix_tasks
                ],
            }

            print(f"保存修复报告到: {report_path}")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description="并行合并多选题结果并修复有问题的选项")
    parser.add_argument("--input-dir", required=True, help="包含结果文件的目录")
    parser.add_argument(
        "--pattern", default="choice_questions_part_*.json", help="文件匹配模式 (默认: choice_questions_part_*.json)"
    )
    parser.add_argument("--output", required=True, help="合并后的输出文件路径")
    parser.add_argument(
        "--api-key", default="sk-K6tq07IP2UM744DR1YkZSqZ3MGpab7bJ6IImmBoUWxoT2Jpa", help="OpenAI API key"
    )
    parser.add_argument("--base-url", default="https://yunwu.zeabur.app/v1", help="OpenAI API base URL")
    parser.add_argument("--max-workers", type=int, default=8, help="最大并行线程数 (默认: 8)")
    parser.add_argument("--fix-problems", action="store_true", help="修复有问题的选项")
    parser.add_argument("--dry-run", action="store_true", help="仅检查问题不进行修复")
    # 新增：原始prompt文件参数
    parser.add_argument("--prompt-file", help="原始prompt JSONL文件路径，用于根据qid提取profile和conversation")

    args = parser.parse_args()

    # 初始化prompt加载器
    prompt_loader = None
    if args.prompt_file:
        if os.path.exists(args.prompt_file):
            prompt_loader = PromptLoader(args.prompt_file)
        else:
            print(f"警告: 指定的prompt文件不存在: {args.prompt_file}")
            print("将使用原有的从original_prompt字段提取的方法")
            prompt_loader = PromptLoader()
    else:
        print("未指定prompt文件，将使用原有的从original_prompt字段提取的方法")
        prompt_loader = PromptLoader()

    # 初始化并行合并器
    merger = ResultMergerParallel(
        api_key=args.api_key, base_url=args.base_url, max_workers=args.max_workers, prompt_loader=prompt_loader
    )

    print(f"使用 {args.max_workers} 个并行线程")
    if prompt_loader and prompt_loader.prompt_data:
        print(f"已加载 {len(prompt_loader.prompt_data)} 个原始prompts")

    # 查找文件
    files = merger.find_json_files(args.input_dir, args.pattern)
    if not files:
        print(f"在目录 {args.input_dir} 中没有找到匹配 {args.pattern} 的文件")
        return

    print(f"找到 {len(files)} 个文件:")
    for f in files:
        print(f"  - {os.path.basename(f)}")

    # 并行合并文件
    questions = merger.load_and_merge_files(files)

    # 并行检查问题
    questions, fix_tasks = merger.check_problematic_choices(questions)

    if fix_tasks:
        print(f"\n发现的问题选项:")
        for task in fix_tasks[:5]:  # 只显示前5个
            print(f"  问题 {task.question_idx}, 选项 {task.choice_idx}: {task.original_text[:80]}...")
        if len(fix_tasks) > 5:
            print(f"  ... 还有 {len(fix_tasks) - 5} 个问题")

    # 并行修复问题 (如果启用)
    if args.fix_problems and not args.dry_run and fix_tasks:
        questions = merger.fix_problematic_questions_parallel(questions, fix_tasks)
    elif args.dry_run:
        print("\n干运行模式: 不进行实际修复")

    # 保存结果
    merger.save_results(questions, args.output, fix_tasks)

    print(f"\n✅ 并行处理完成!")
    print(f"合并了 {len(questions)} 个问题")
    if fix_tasks:
        print(f"发现 {len(fix_tasks)} 个有问题的选项")
        if args.fix_problems and not args.dry_run:
            print(f"成功修复: {merger.success_counter.get()}")
            print(f"修复失败: {merger.failure_counter.get()}")


if __name__ == "__main__":
    main()

import json
import time
from typing import Dict, List, Any, Optional
import argparse
from pathlib import Path
from openai import OpenAI
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class GenerationTask:
    """数据生成任务的数据结构"""

    data_item: Dict
    temperature: float
    variation_id: int
    task_id: str


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


class CoTRLDataGeneratorAPI:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.apiplus.org/v1",
        model_name: str = "gpt-4o",
        max_workers: int = 8,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.max_workers = max_workers

        print(f"Using API model: {model_name}")
        print(f"API base URL: {base_url}")
        print(f"Max workers: {max_workers}")

        # 计数器
        self.success_counter = ThreadSafeCounter()
        self.failure_counter = ThreadSafeCounter()

    def create_client(self) -> OpenAI:
        """为每个线程创建独立的OpenAI客户端"""
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate_cot_prompt(self, original_data: Dict) -> Optional[List[Dict]]:
        """Generate Chain-of-Thought prompt based on original data"""
        conversations = original_data.get("conversations", [])

        # Find the user's question (last user message)
        user_question = None
        for conv in conversations:
            if conv["role"] == "user":
                user_question = conv["content"]

        if not user_question:
            return None

        # Create CoT prompt
        cot_prompt = f"""You are tasked with solving a multiple-choice question step by step. Please provide detailed reasoning before giving your final answer.

Original Question:
{user_question}

Please follow this format:
1. **Analysis**: First, analyze the question and context carefully
2. **Reasoning**: Think through each option step by step
3. **Evaluation**: Compare the options against the given criteria
4. **Conclusion**: State your final choice with justification

After your detailed reasoning, provide your final answer in the format /choice{{letter}}.

Now, please solve this step by step:"""

        return [
            {
                "role": "system",
                "content": "You are an AI assistant that provides detailed step-by-step reasoning for multiple-choice questions.",
            },
            {"role": "user", "content": cot_prompt},
        ]

    def call_api_single(self, task: GenerationTask, max_retries: int = 3, thread_id: int = 0) -> Optional[Dict]:
        """单个任务的API调用 - 线程安全版本"""

        client = self.create_client()
        messages = self.generate_cot_prompt(task.data_item)

        if not messages:
            return None

        for attempt in range(max_retries):
            try:
                print(
                    f"线程 {thread_id}: 生成 {task.task_id} temp={task.temperature} (尝试 {attempt + 1}/{max_retries})"
                )

                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=4096,
                    temperature=task.temperature,
                    timeout=60,
                )

                content = response.choices[0].message.content.strip()

                if content:
                    variation = {
                        "qid": task.data_item.get("qid", "") + f"_cot_var_{task.variation_id}",
                        "original_qid": task.data_item.get("qid", ""),
                        "variation_id": task.variation_id,
                        "temperature": task.temperature,
                        "conversations": [
                            messages[0],  # system message
                            messages[1],  # user question
                            {"role": "assistant", "content": content},
                        ],
                        "original_answer": self._extract_answer(task.data_item),
                        "metadata": {
                            "generation_type": "cot_rl",
                            "model": self.model_name,
                            "timestamp": time.time(),
                            "api_generated": True,
                        },
                    }

                    print(f"线程 {thread_id}: ✅ 成功生成 {task.task_id}")
                    return variation
                else:
                    print(f"线程 {thread_id}: API返回空内容，重试...")

            except Exception as e:
                print(f"线程 {thread_id}: API调用失败 (尝试 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))

        return None

    def _extract_answer(self, original_data: Dict) -> str:
        """Extract the original answer from the data"""
        conversations = original_data.get("conversations", [])
        for conv in conversations:
            if conv["role"] == "assistant":
                content = conv["content"]
                if "/choice{" in content:
                    return content.strip()
        return ""

    def generate_batch_variations_parallel(self, data_batch: List[Dict], num_variations: int = 3) -> List[Dict]:
        """并行生成一批数据的变体"""
        all_tasks = []

        # 创建所有任务
        for data_item in data_batch:
            for temp_idx in range(num_variations):
                temperature = 0.3 + (temp_idx * 0.3)  # 0.3, 0.6, 0.9
                task = GenerationTask(
                    data_item=data_item,
                    temperature=temperature,
                    variation_id=temp_idx + 1,
                    task_id=data_item.get("qid", f"unknown_{temp_idx}"),
                )
                all_tasks.append(task)

        if not all_tasks:
            return []

        print(f"开始并行生成 {len(all_tasks)} 个变体任务...")

        results = []

        # 并行处理所有任务
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {}
            for i, task in enumerate(all_tasks):
                future = executor.submit(self.call_api_single, task=task, thread_id=i % self.max_workers)
                future_to_task[future] = task

            # 收集结果
            for future in tqdm(as_completed(future_to_task), total=len(all_tasks), desc="Generating variations"):
                original_task = future_to_task[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        self.success_counter.increment()
                    else:
                        print(f"❌ 生成失败: {original_task.task_id}")
                        self.failure_counter.increment()

                except Exception as e:
                    print(f"处理任务 {original_task.task_id} 时出错: {e}")
                    self.failure_counter.increment()

        return results

    def process_dataset(
        self, input_file: str, output_file: str, num_variations: int = 3, max_samples: int = None, batch_size: int = 10
    ):
        """Process the entire dataset and generate CoT RL data using parallel API calls"""
        print(f"Loading dataset from {input_file}")

        with open(input_file, "r", encoding="utf-8") as f:
            original_data = json.load(f)

        # Handle both single dict and list of dicts
        if isinstance(original_data, dict):
            data_list = [original_data]
        elif isinstance(original_data, list):
            data_list = original_data
        else:
            print("Error: Input file should contain a dict or list of dicts")
            return

        if max_samples:
            data_list = data_list[:max_samples]

        all_variations = []

        # Process data in batches
        for batch_start in range(0, len(data_list), batch_size):
            batch_end = min(batch_start + batch_size, len(data_list))
            data_batch = data_list[batch_start:batch_end]

            print(f"\n处理批次 {batch_start//batch_size + 1}/{(len(data_list) + batch_size - 1)//batch_size}")
            print(f"批次项目: {[item.get('qid', 'unknown') for item in data_batch]}")

            # Generate variations for this batch using parallel processing
            batch_variations = self.generate_batch_variations_parallel(data_batch, num_variations)
            all_variations.extend(batch_variations)

            print(f"本批次生成了 {len(batch_variations)} 个变体")

        # Save results
        print(f"\n保存 {len(all_variations)} 个变体到 {output_file}")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_variations, f, indent=2, ensure_ascii=False)

        success_count = self.success_counter.get()
        failure_count = self.failure_counter.get()
        total_expected = len(data_list) * num_variations

        print(f"\n✅ 数据生成完成!")
        print(f"成功生成: {success_count}/{total_expected} 个样本")
        print(f"生成失败: {failure_count}/{total_expected} 个样本")
        print(f"成功率: {success_count/total_expected*100:.1f}%")

        # Generate summary
        self._generate_summary(all_variations, output_file.replace(".json", "_summary.txt"))

    def _generate_summary(self, variations: List[Dict], summary_file: str):
        """Generate a summary of the generated data"""
        if not variations:
            print("没有生成的变体，跳过摘要生成")
            return

        summary = f"""CoT RL Data Generation Summary (API Version)
=============================================
Total variations generated: {len(variations)}
Unique original questions: {len(set(v.get('original_qid', '') for v in variations))}
Average variations per question: {len(variations) / len(set(v.get('original_qid', '') for v in variations)) if variations else 0:.1f}
Model: {self.model_name}
API URL: {self.base_url}
Max workers: {self.max_workers}

Success rate: {self.success_counter.get()}/{self.success_counter.get() + self.failure_counter.get()} ({self.success_counter.get()/(self.success_counter.get() + self.failure_counter.get())*100:.1f}%)

Temperature distribution:
"""

        temp_counts = {}
        for v in variations:
            temp = v.get("temperature", 0)
            temp_counts[temp] = temp_counts.get(temp, 0) + 1

        for temp, count in sorted(temp_counts.items()):
            summary += f"  {temp}: {count} samples\n"

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write(summary)

        print(f"摘要保存到 {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate CoT RL data using OpenAI API with parallel processing")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--base-url", default="https://api.apiplus.org/v1", help="OpenAI API base URL")
    parser.add_argument("--model-name", default="gpt-4o", help="Model name to use")
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum number of parallel workers")
    parser.add_argument("--batch-size", "-b", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--variations", "-v", type=int, default=3, help="Number of variations per question")
    parser.add_argument("--max-samples", "-m", type=int, default=None, help="Maximum number of samples to process")

    args = parser.parse_args()

    # Check if input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} does not exist")
        return

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize generator with API
    generator = CoTRLDataGeneratorAPI(
        api_key=args.api_key, base_url=args.base_url, model_name=args.model_name, max_workers=args.max_workers
    )

    # Process dataset
    generator.process_dataset(
        input_file=args.input,
        output_file=args.output,
        num_variations=args.variations,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

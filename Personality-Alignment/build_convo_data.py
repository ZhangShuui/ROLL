import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.multiprocessing import Pool
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import copy
import traceback


def load_profiles(profile_path: str) -> dict:
    """profile.json → {user_id: profile_text}"""
    with open(profile_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_qwen3_8b(model_path="Qwen/Qwen3-8B", device_map="auto", quantize=False):
    """
    加载 Qwen3-8B 模型和分词器

    参数:
    model_path: 模型路径 (Hugging Face ID 或本地路径)
    device_map: 设备映射 ("auto", "cuda", "cpu")
    quantize: 是否使用 4-bit 量化 (减少显存需求)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"  # 确保分词器左侧填充
    quantization_config = None
    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        # device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model = model.eval()  # 设置为评估模式
    model.to("cuda:0")

    return model, tokenizer


def load_qwen3_8b_multi_gpu(model_path="Qwen/Qwen3-8B", devices=["cuda:0", "cuda:1"], quantize=False):
    """
    在多个GPU上加载 Qwen3-8B 模型和分词器

    参数:
    model_path: 模型路径 (Hugging Face ID 或本地路径)
    devices: GPU设备列表
    quantize: 是否使用 4-bit 量化 (减少显存需求)

    返回:
    models: 模型列表
    tokenizer: 分词器
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"  # 确保分词器左侧填充
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 设置pad_token

    quantization_config = None
    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )

    models = []
    for device in devices:
        print(f"🔄 正在加载模型到 {device}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        model = model.eval()  # 设置为评估模式
        model.to(device)
        models.append(model)
        print(f"✅ 模型已加载到 {device}")

    return models, tokenizer


def summarize_model_response(model, tokenizer, response_text: str, max_length: int = 500) -> str:
    """
    使用 Qwen3-8B 模型对过长的 model response 进行精简

    参数:
    model: 加载的 Qwen3-8B 模型
    tokenizer: 对应的分词器
    response_text: 需要精简的 model response 文本
    max_length: 精简后的最大长度

    返回:
    精简后的文本
    """
    messages = [
        {
            "role": "user",
            "content": f"""Please condense the following assistant response to no more than {max_length} characters while keeping the key information and maintaining the same tone and intent:

Original response:
{response_text}

Condensed response:""",
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    # Tokenize
    input_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8000).to(model.device)
    input_length = input_tokens["input_ids"].shape[1]

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **input_tokens,
            max_new_tokens=600,  # 稍微多一点空间以生成完整回答
            do_sample=True,
            temperature=0.3,  # 较低的温度保证精简质量
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode - 只解码生成的新部分
    generated_ids = outputs[0][input_length:]
    condensed = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # 如果生成的精简版本仍然太长，则进行安全截断
    if len(condensed) > max_length:
        condensed = condensed[:max_length].rsplit(" ", 1)[0] + "..."

    return condensed


def summarize_model_responses_batch(
    model, tokenizer, response_texts: list, max_length: int = 500, batch_size: int = 8
) -> list:
    """
    批量使用 Qwen3-8B 模型对过长的 model responses 进行精简

    参数:
    model: 加载的 Qwen3-8B 模型
    tokenizer: 对应的分词器
    response_texts: 需要精简的 model response 文本列表
    max_length: 精简后的最大长度
    batch_size: 批量大小

    返回:
    精简后的文本列表
    """
    all_condensed = []
    total_batches = (len(response_texts) + batch_size - 1) // batch_size

    progress_bar = tqdm(range(0, len(response_texts), batch_size), desc="批量精简model回复", total=total_batches)

    for i in progress_bar:
        batch_texts = response_texts[i : i + batch_size]
        batch_prompts = []
        for text in batch_texts:
            messages = [
                {
                    "role": "user",
                    "content": f"""Please condense the following assistant response to no more than {max_length} characters while keeping the key information and maintaining the same tone and intent:

Original response:
{text}

Condensed response:""",
                }
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            batch_prompts.append(prompt)

        # 批量tokenize
        input_tokens = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=8000
        ).to(model.device)

        # 批量生成
        with torch.no_grad():
            outputs = model.generate(
                **input_tokens,
                max_new_tokens=600,
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 批量解码
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # 处理这一批的结果
        batch_condensed = []
        for j, text in enumerate(generated_texts):
            condensed = text.replace(batch_prompts[j], "").strip()
            # 如果生成的精简版本仍然太长，则进行安全截断
            if len(condensed) > max_length:
                condensed = condensed[:max_length].rsplit(" ", 1)[0] + "..."
            batch_condensed.append(condensed)

        all_condensed.extend(batch_condensed)

        # 更新进度条信息
        progress_bar.set_postfix(
            {"已完成": len(all_condensed), "总数": len(response_texts), "批次大小": len(batch_texts)}
        )

    return all_condensed


def summarize_history_batch(
    model, tokenizer, history_texts: list, max_summary_length: int = 1000, batch_size: int = 8
) -> list:
    """
    批量使用 Qwen3-8B 模型对历史对话进行总结

    参数:
    model: 加载的 Qwen3-8B 模型
    tokenizer: 对应的分词器
    history_texts: 需要总结的历史对话文本列表
    max_summary_length: 总结的最大长度
    batch_size: 批量大小

    返回:
    总结后的文本列表
    """
    all_summaries = []
    total_batches = (len(history_texts) + batch_size - 1) // batch_size

    progress_bar = tqdm(range(0, len(history_texts), batch_size), desc="批量总结历史对话", total=total_batches)

    for i in progress_bar:
        batch_texts = history_texts[i : i + batch_size]
        batch_prompts = []
        for text in batch_texts:
            messages = [
                {
                    "role": "user",
                    "content": f"""Please summarize the following conversation history in no more than {max_summary_length} characters. Keep the key information especially about the target simulate person's personality, preferences, and important context:

{text}

Summary:""",
                }
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            batch_prompts.append(prompt)

        # 批量tokenize
        input_tokens = tokenizer(
            batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=10000
        ).to(model.device)

        # 批量生成
        with torch.no_grad():
            outputs = model.generate(
                **input_tokens,
                max_new_tokens=1000,
                do_sample=True,
                temperature=0.3,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 批量解码
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # 处理这一批的结果
        batch_summaries = []
        for j, text in enumerate(generated_texts):
            summary = text.replace(batch_prompts[j], "").strip()
            # 确保总结不超过指定长度
            if len(summary) > max_summary_length:
                summary = summary[:max_summary_length].rsplit(" ", 1)[0] + "..."
            batch_summaries.append(summary)

        all_summaries.extend(batch_summaries)

        # 更新进度条信息
        progress_bar.set_postfix(
            {"已完成": len(all_summaries), "总数": len(history_texts), "批次大小": len(batch_texts)}
        )

    return all_summaries


def summarize_history(model, tokenizer, history_text: str, max_summary_length: int = 1000) -> str:
    """
    使用 Qwen3-8B 模型对历史对话进行总结

    参数:
    model: 加载的 Qwen3-8B 模型
    tokenizer: 对应的分词器
    history_text: 需要总结的历史对话文本
    max_summary_length: 总结的最大长度

    返回:
    总结后的文本
    """
    messages = [
        {
            "role": "user",
            "content": f"""Please summarize the following conversation history in no more than {max_summary_length} characters. Keep the key information especially about the target simulate person's personality, preferences, and important context:

{history_text}

Summary:""",
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    # Tokenize
    input_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8000).to(model.device)
    input_length = input_tokens["input_ids"].shape[1]

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **input_tokens,
            max_new_tokens=1000,  # 控制总结长度
            do_sample=True,
            temperature=0.3,  # 较低的温度保证总结质量
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode - 只解码生成的新部分
    generated_ids = outputs[0][input_length:]
    summary = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # 确保总结不超过指定长度
    if len(summary) > max_summary_length:
        summary = summary[:max_summary_length].rsplit(" ", 1)[0] + "..."

    return summary


def iterate_messages(record: dict):
    """
    根据常见字段名把一条对话里的 message 列表取出来。
    你可以按需要再补别名。
    """
    return record["conversations"]


def clean_user_text(model, tokenizer, user_text: str) -> str:
    """
    使用 Qwen3-8B 模型对用户文本进行清理（修正错误）

    参数:
    model: 加载的 Qwen3-8B 模型
    tokenizer: 对应的分词器
    user_text: 需要清理的用户文本

    返回:
    清理后的文本
    """
    messages = [
        {
            "role": "user",
            "content": f"""Please fix spelling errors, grammar mistakes, and formatting issues in the following text while preserving the original meaning:

Original text: {user_text}

Requirements:
1. Fix obvious spelling errors
2. Correct grammar issues
3. Standardize punctuation usage
4. Maintain the original semantics and tone
5. Return only the corrected text without any explanations

Corrected text:""",
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    # Tokenize
    input_tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=500).to(model.device)
    input_length = input_tokens["input_ids"].shape[1]

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **input_tokens,
            max_new_tokens=400,  # 控制清理后文本长度
            do_sample=True,
            temperature=0.2,  # 较低的温度保证清理质量
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode - 只解码生成的新部分
    generated_ids = outputs[0][input_length:]
    cleaned = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # 如果清理后的文本为空或过短，返回原文本
    if not cleaned or len(cleaned.strip()) < 3:
        return user_text

    return cleaned


def clean_user_texts_batch(model, tokenizer, user_texts: list, batch_size: int = 8) -> list:
    """
    批量使用 Qwen3-8B 模型对用户文本进行清理

    参数:
    model: 加载的 Qwen3-8B 模型
    tokenizer: 对应的分词器
    user_texts: 需要清理的用户文本列表
    batch_size: 批量大小

    返回:
    清理后的文本列表
    """
    all_cleaned = []
    total_batches = (len(user_texts) + batch_size - 1) // batch_size

    progress_bar = tqdm(range(0, len(user_texts), batch_size), desc="批量清理用户文本", total=total_batches)

    for i in progress_bar:
        batch_texts = user_texts[i : i + batch_size]
        batch_prompts = []
        for text in batch_texts:
            messages = [
                {
                    "role": "user",
                    "content": f"""Please fix spelling errors, grammar mistakes, and formatting issues in the following text while preserving the original meaning:

Original text: {text}

Requirements:
1. Fix obvious spelling errors
2. Correct grammar issues
3. Standardize punctuation usage
4. Maintain the original semantics and tone
5. Return only the corrected text without any explanations

Corrected text:""",
                }
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            batch_prompts.append(prompt)

        # 批量tokenize
        input_tokens = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=500).to(
            model.device
        )

        # 批量生成
        with torch.no_grad():
            outputs = model.generate(
                **input_tokens,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.2,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 批量解码
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # 处理这一批的结果
        batch_cleaned = []
        for j, (text, prompt, original) in enumerate(zip(generated_texts, batch_prompts, batch_texts)):
            cleaned = text.replace(prompt, "").strip()
            # 如果清理后的文本为空或过短，使用原文本
            if not cleaned or len(cleaned.strip()) < 3:
                cleaned = original
            batch_cleaned.append(cleaned)

        all_cleaned.extend(batch_cleaned)

        # 更新进度条信息
        progress_bar.set_postfix({"已完成": len(all_cleaned), "总数": len(user_texts), "批次大小": len(batch_texts)})

    return all_cleaned


def clean_user_texts_batch_multi_gpu(models, tokenizer, user_texts: list, batch_size: int = 8) -> list:
    """
    使用多GPU批量清理用户文本

    参数:
    models: 模型列表
    tokenizer: 分词器
    user_texts: 需要清理的用户文本列表
    batch_size: 批量大小

    返回:
    清理后的文本列表
    """
    if not user_texts:
        return []

    # 准备批次数据
    batches = []
    for i in range(0, len(user_texts), batch_size):
        batch_texts = user_texts[i : i + batch_size]
        batch_prompts = []
        for text in batch_texts:
            messages = [
                {
                    "role": "user",
                    "content": f"""Please fix spelling errors, grammar mistakes, and formatting issues in the following text while preserving the original meaning:

Original text: {text}

Requirements:
1. Fix obvious spelling errors
2. Correct grammar issues
3. Standardize punctuation usage
4. Maintain the original semantics and tone
5. Return only the corrected text without any explanations

Corrected text:""",
                }
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            batch_prompts.append(prompt)
        batches.append((i // batch_size, batch_prompts, batch_texts))  # 添加原文本用于备选

    # 创建队列
    input_queue = Queue()
    output_queue = Queue()

    # 将批次数据放入输入队列
    for batch in batches:
        input_queue.put(batch)

    # 添加结束信号
    for _ in models:
        input_queue.put(None)

    # 启动工作线程
    threads = []
    for i, model in enumerate(models):
        device = model.device
        thread = threading.Thread(
            target=clean_inference_worker, args=(model, tokenizer, device, input_queue, output_queue)
        )
        thread.start()
        threads.append(thread)

    # 收集结果
    results = {}
    total_batches = len(batches)

    progress_bar = tqdm(range(total_batches), desc="批量清理用户文本 (多GPU)")
    completed_batches = 0

    while completed_batches < total_batches:
        try:
            batch_idx, batch_results = output_queue.get(timeout=30)
            if batch_results is not None:
                results[batch_idx] = batch_results
            completed_batches += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                {"GPU数量": len(models), "已完成批次": completed_batches, "总批次": total_batches}
            )
        except Exception as e:
            print(f"⚠️ 获取结果超时: {e}")
            break

    progress_bar.close()

    # 等待所有线程结束
    for thread in threads:
        thread.join()

    # 按顺序组装最终结果
    all_cleaned = []
    for batch_idx in sorted(results.keys()):
        all_cleaned.extend(results[batch_idx])

    return all_cleaned


def clean_inference_worker(model, tokenizer, device, input_queue, output_queue):
    """
    用户文本清理的GPU推理工作线程

    参数:
    model: 模型实例
    tokenizer: 分词器
    device: 设备名称
    input_queue: 输入队列
    output_queue: 输出队列
    """
    while True:
        try:
            batch_data = input_queue.get(timeout=10)
            if batch_data is None:  # 结束信号
                break

            batch_idx, batch_prompts, original_texts = batch_data

            # 批量tokenize
            input_tokens = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=6000
            ).to(device)

            # 获取输入长度用于后续切片
            # input_lengths = input_tokens["attention_mask"].sum(dim=1).detach().cpu().tolist()

            # 批量生成
            with torch.no_grad():
                outputs = model.generate(
                    **input_tokens,
                    max_new_tokens=400,
                    do_sample=True,
                    temperature=0.2,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # 处理结果 - 根据输入长度正确提取生成的部分
            batch_results = []
            for generated_ids, input_id, original in zip(outputs, input_tokens.input_ids, original_texts):
                # 只取生成的新token部分
                input_length = len(input_id)
                generated_ids = generated_ids[int(input_length) :]
                cleaned = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                # print(f"原文本: {original}\n清理后: {cleaned}\n---")
                # 如果清理后的文本为空或过短，使用原文本
                if not cleaned or len(cleaned.strip()) < 3:
                    cleaned = original
                batch_results.append(cleaned)

            output_queue.put((batch_idx, batch_results))

        except Exception as e:
            print(f"⚠️ GPU {device} 清理推理错误: {e}")
            traceback.print_exc()
            output_queue.put((batch_idx, None))  # 错误标记


def format_history_prepare(messages: list, max_length: int = 4000) -> tuple:
    """
    准备格式化对话历史，返回是否需要总结和相关信息

    新逻辑：
    1. 收集需要精简的model responses
    2. 收集需要清理的user texts
    3. 当整个对话长度超过4000时，只保留最近1/3轮对话，并对其他的历史对话做总结

    参数:
    messages: 消息列表
    max_length: 触发总结的最大长度阈值 (默认4000)

    返回:
    (formatted_history, needs_summary, early_text, recent_history_formatted, long_model_responses, long_response_indices, user_texts, user_text_indices)
    """
    history = []
    long_model_responses = []  # 存储需要精简的长回复
    long_response_indices = []  # 存储需要精简的回复在history中的索引
    user_texts = []  # 存储需要清理的用户文本
    user_text_indices = []  # 存储需要清理的用户文本在history中的索引

    for msg in messages:
        if not msg.get("if_chosen", True):
            continue
        if msg["content"].strip() == "EMPTY STRING":
            continue

        content = msg["content"]

        # 检查model回复是否需要精简
        if msg.get("role") == "model" and len(content) > 500:
            long_model_responses.append(content)
            long_response_indices.append(len(history))  # 记录在history中的位置

        # 收集用户文本进行清理
        if msg.get("role") == "user":
            user_texts.append(content)
            user_text_indices.append(len(history))  # 记录在history中的位置
            history.append(f"<Target person>: {content}")
        elif msg.get("role") == "model":
            history.append(f"<LLM assistant>: {content}")
        else:
            raise ValueError(f"Unknown condition: {msg.get('role')}")

    # 组合历史记录
    full_history = "\n".join(history)

    # 检查是否需要总结（阈值改为3000）
    needs_summary = len(full_history) > max_length
    early_text = None
    recent_history_formatted = None

    if needs_summary:
        # 计算需要保留的最近消息数量（保留最后1/3轮对话）
        total_msgs = len(history)
        keep_recent = max(3, total_msgs // 3)  # 至少保留2条消息

        # 分离早期历史和最近历史
        early_history = history[:-keep_recent]
        recent_history = history[-keep_recent:]

        if early_history:  # 只有当有早期历史时才进行总结
            early_text = ""
            for line in early_history:
                if len(early_text) > 3000:
                    early_text += "...\n"
                    break
                if len(line) > 700:
                    early_text += line[:1000].rsplit(" ", 1)[0] + "...\n"
                else:
                    early_text += line + "\n"
            recent_history_formatted = recent_history

    return (
        full_history,
        needs_summary,
        early_text,
        recent_history_formatted,
        long_model_responses,
        long_response_indices,
        user_texts,
        user_text_indices,
    )


def format_history(messages: list, model=None, tokenizer=None, max_length: int = 4000) -> str:
    """
    格式化对话历史为字符串，当长度超过阈值时使用模型进行总结。
    当model回复过长时使用模型进行精简。

    参数:
    messages: 消息列表
    model: Qwen3-8B 模型（可选，用于总结和精简）
    tokenizer: 对应的分词器（可选，用于总结和精简）
    max_length: 触发总结的最大长度阈值 (默认4000)
    """
    history = []

    # 先收集需要精简的model回复
    long_responses = []
    long_response_msg_indices = []

    for msg_idx, msg in enumerate(messages):
        if msg["content"].strip() == "EMPTY STRING":
            continue
        if msg.get("role") == "model" and len(msg["content"]) > 500:
            long_responses.append(msg["content"])
            long_response_msg_indices.append(msg_idx)

    # 如果有需要精简的回复且模型可用，进行批量精简
    condensed_responses = {}
    if long_responses and model is not None and tokenizer is not None:
        condensed_list = summarize_model_responses_batch(model, tokenizer, long_responses, max_length=500)
        condensed_responses = dict(zip(long_response_msg_indices, condensed_list))

    # 构建历史记录，使用精简后的回复
    for msg_idx, msg in enumerate(messages):
        if msg["content"].strip() == "EMPTY STRING":
            continue

        content = msg["content"]
        # 如果这条消息有精简版本，使用精简版本
        if msg_idx in condensed_responses:
            content = condensed_responses[msg_idx]

        if msg.get("role") == "user":
            history.append(f"<Target person>: {content}")
        elif msg.get("role") == "model":
            history.append(f"<LLM assistant>: {content}")

    # 组合历史记录
    full_history = "\n".join(history)
    summarized = False

    # 检查是否需要总结（阈值改为4000）
    if len(full_history) > max_length and model is not None and tokenizer is not None:
        # 计算需要保留的最近消息数量（保留最后1/3轮对话）
        total_msgs = len(history)
        keep_recent = max(2, total_msgs // 3)  # 至少保留2条消息

        # 分离早期历史和最近历史
        early_history = history[:-keep_recent]
        recent_history = history[-keep_recent:]

        if early_history:  # 只有当有早期历史时才进行总结
            early_text = "\n".join(early_history)
            summary = summarize_history(model, tokenizer, early_text)
            summarized = True
            # 组合总结和最近历史
            result_parts = (
                ["[Earlier conversation summary]", summary, "[End of summary]", "", "[Recent conversation]"]
                + recent_history
                + ["[End of recent conversation]"]
            )

            return "\n".join(result_parts), summarized

    return full_history, summarized


def format_history_with_summary(summary: str, recent_history_formatted: list) -> str:
    """
    使用提供的总结来格式化历史记录
    """
    result_parts = (
        ["[Earlier conversation summary]", summary, "[End of summary]", "", "[Recent conversation]"]
        + recent_history_formatted
        + ["[End of recent conversation]"]
    )
    return "\n".join(result_parts)


def format_conversation_history(history_str: str, record: dict) -> str:
    """
    格式化对话历史为完整的对话字符串。
    假设 history_str 是格式化后的对话历史字符串，record 包含其他信息。
    """
    opening_prompt = record.get("opening_prompt", "")
    conversation_type = record.get("conversation_type", "")
    description = None
    if conversation_type == "unguided":
        description = " This is an unguided conversation without any specific topic. The person could ask, request or talk to the model about anything."
    elif conversation_type == "values guided":
        description = " This is a value guided conversation. The person is required to ask, request or talk to the model about something important to it or that represents its values. This could be related to work, religion, family and relationship, politics or culture."
    elif conversation_type == "controversy guided":
        description = " This is a controversy guided conversation. The person is required to ask, request or talk to the model about something controversial or where people would disagree in its community, culture or country"
    else:
        raise ValueError(f"Unknown conversation type: {conversation_type}")
    return (
        f"Below is the conversation history between the person and an LLM assistant.\n{description}\n"
        "[Conversation History Begin]\n"
        f"{history_str}\n"
        "[Conversation History End]\n"
    )


def inference_worker(model, tokenizer, device, input_queue, output_queue, task_type="condense"):
    """
    单个GPU的推理工作线程

    参数:
    model: 模型实例
    tokenizer: 分词器
    device: 设备名称
    input_queue: 输入队列
    output_queue: 输出队列
    task_type: 任务类型 ("condense" 或 "summarize")
    """
    while True:
        try:
            batch_data = input_queue.get(timeout=1)
            if batch_data is None:  # 结束信号
                break

            batch_idx, batch_prompts = batch_data

            # 批量tokenize
            max_length = 8000 if task_type == "condense" else 10000
            input_tokens = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(device)

            # 获取输入长度用于后续切片

            # 批量生成
            max_new_tokens = 800 if task_type == "condense" else 500
            with torch.no_grad():
                outputs = model.generate(
                    **input_tokens,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.3,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # 处理结果 - 根据输入长度正确提取生成的部分
            batch_results = []
            max_result_length = 500 if task_type == "condense" else 1000

            for j, (generated_ids, input_id) in enumerate(zip(outputs, input_tokens.input_ids)):
                # 只取生成的新token部分
                input_length = len(input_id)
                generated_ids = generated_ids[int(input_length) :]
                result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                # print(f"GPU {device} 处理批次 {batch_idx}, 条目 {j}: {result}...")  # 打印前100字符
                # 如果生成的结果仍然太长，则进行安全截断
                if len(result) > max_result_length:
                    result = result[:max_result_length].rsplit(" ", 1)[0] + "..."
                batch_results.append(result)

            output_queue.put((batch_idx, batch_results))

        except Exception as e:
            print(f"⚠️ GPU {device} 推理错误: {e}")
            output_queue.put((batch_idx, None))  # 错误标记


def summarize_model_responses_batch_multi_gpu(
    models, tokenizer, response_texts: list, max_length: int = 300, batch_size: int = 8
) -> list:
    """
    使用多GPU批量精简model responses

    参数:
    models: 模型列表
    tokenizer: 分词器
    response_texts: 需要精简的 model response 文本列表
    max_length: 精简后的最大长度
    batch_size: 批量大小

    返回:
    精简后的文本列表
    """
    if not response_texts:
        return []

    # 准备批次数据
    batches = []
    for i in range(0, len(response_texts), batch_size):
        batch_texts = response_texts[i : i + batch_size]
        batch_prompts = []
        for text in batch_texts:
            messages = [
                {
                    "role": "user",
                    "content": f"""Please condense the following assistant response to no more than {max_length} characters while keeping the key information and maintaining the same tone and intent:

Original response:
{text}

Condensed response:""",
                }
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            batch_prompts.append(prompt)
        batches.append((i // batch_size, batch_prompts))

    # 创建队列
    input_queue = Queue()
    output_queue = Queue()

    # 将批次数据放入输入队列
    for batch in batches:
        input_queue.put(batch)

    # 添加结束信号
    for _ in models:
        input_queue.put(None)

    # 启动工作线程
    threads = []
    for i, model in enumerate(models):
        device = model.device
        thread = threading.Thread(
            target=inference_worker, args=(model, tokenizer, device, input_queue, output_queue, "condense")
        )
        thread.start()
        threads.append(thread)

    # 收集结果
    results = {}
    total_batches = len(batches)

    progress_bar = tqdm(range(total_batches), desc="批量精简model回复 (多GPU)")
    completed_batches = 0

    while completed_batches < total_batches:
        try:
            batch_idx, batch_results = output_queue.get(timeout=60)
            if batch_results is not None:
                results[batch_idx] = batch_results
            completed_batches += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                {"GPU数量": len(models), "已完成批次": completed_batches, "总批次": total_batches}
            )
        except Exception as e:
            print(f"⚠️ 获取结果超时: {e}")
            break

    progress_bar.close()

    # 等待所有线程结束
    for thread in threads:
        thread.join()

    # 按顺序组装最终结果
    all_condensed = []
    for batch_idx in sorted(results.keys()):
        all_condensed.extend(results[batch_idx])

    return all_condensed


def summarize_history_batch_multi_gpu(
    models, tokenizer, history_texts: list, max_summary_length: int = 1000, batch_size: int = 8
) -> list:
    """
    使用多GPU批量总结历史对话

    参数:
    models: 模型列表
    tokenizer: 分词器
    history_texts: 需要总结的历史对话文本列表
    max_summary_length: 总结的最大长度
    batch_size: 批量大小

    返回:
    总结后的文本列表
    """
    if not history_texts:
        return []

    # 准备批次数据
    batches = []
    for i in range(0, len(history_texts), batch_size):
        batch_texts = history_texts[i : i + batch_size]
        batch_prompts = []
        for text in batch_texts:
            messages = [
                {
                    "role": "user",
                    "content": f"""Please summarize the following conversation history in no more than {max_summary_length} characters. Keep the key information especially about the target simulate person's personality, preferences, and important context:

{text}

Summary:""",
                }
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            batch_prompts.append(prompt)
        batches.append((i // batch_size, batch_prompts))

    # 创建队列
    input_queue = Queue()
    output_queue = Queue()

    # 将批次数据放入输入队列
    for batch in batches:
        input_queue.put(batch)

    # 添加结束信号
    for _ in models:
        input_queue.put(None)

    # 启动工作线程
    threads = []
    for i, model in enumerate(models):
        device = model.device
        thread = threading.Thread(
            target=inference_worker, args=(model, tokenizer, device, input_queue, output_queue, "summarize")
        )
        thread.start()
        threads.append(thread)

    # 收集结果
    results = {}
    total_batches = len(batches)

    progress_bar = tqdm(range(total_batches), desc="批量总结历史对话 (多GPU)")
    completed_batches = 0

    while completed_batches < total_batches:
        try:
            batch_idx, batch_results = output_queue.get(timeout=60)
            if batch_results is not None:
                results[batch_idx] = batch_results
            completed_batches += 1
            progress_bar.update(1)
            progress_bar.set_postfix(
                {"GPU数量": len(models), "已完成批次": completed_batches, "总批次": total_batches}
            )
        except Exception as e:
            print(f"⚠️ 获取结果超时: {e}")
            break

    progress_bar.close()

    # 等待所有线程结束
    for thread in threads:
        thread.join()

    # 按顺序组装最终结果
    all_summaries = []
    for batch_idx in sorted(results.keys()):
        all_summaries.extend(results[batch_idx])

    return all_summaries


def build_dataset(
    base_path,
    roleplay_path: str,
    profile_path: str,
    output_path: str,
    use_summarization: bool = True,
    use_cleaning: bool = True,  # 新增参数控制是否清理用户文本
    batch_size: int = 32,
    gpu_devices: list = ["cuda:0", "cuda:1"],  # 新增参数
):
    """
    构建数据集 - 支持多GPU并行推理

    参数:
    roleplay_path: 角色扮演数据路径
    profile_path: 配置文件路径
    output_path: 输出路径
    use_summarization: 是否使用总结功能
    use_cleaning: 是否使用用户文本清理功能
    batch_size: 批量处理大小
    gpu_devices: GPU设备列表
    """
    max_data_num = 1000000
    profile_path = base_path + profile_path
    roleplay_path = base_path + roleplay_path
    profiles = load_profiles(profile_path)

    # 第一步：统计总行数（用于进度显示）
    print("🔄 第一步：统计数据总量...")
    total_lines = 0
    with open(roleplay_path, "r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1
    print(f"📊 数据文件总行数：{total_lines}")

    # 第二步：收集所有需要处理的数据
    print("🔄 第二步：收集所有需要处理的消息...")
    all_processing_items = []

    with open(roleplay_path, "r", encoding="utf-8") as f:
        progress_bar = tqdm(enumerate(f), total=total_lines, desc="收集消息")
        for line_idx, line in progress_bar:
            if not line.strip():
                continue
            if max_data_num > 0 and len(all_processing_items) >= max_data_num:
                print(f"⚠️ 已达到最大数据量限制：{max_data_num}，停止收集")
                break
            try:
                record = json.loads(line)
                messages = iterate_messages(record)

                # 取 user_id；按需补充或修改字段名
                user_id = record.get("user_id") or record.get("uid") or record.get("profile_id") or record.get("user")
                profile_text = profiles.get(str(user_id), "")  # 若找不到可留空

                # 遍历 message，找到第一条 role=user 作为输出
                for msg_idx, msg in enumerate(messages):
                    if msg_idx == 0:
                        continue
                    if msg.get("role") == "user":
                        history_msgs = messages[:msg_idx]

                        # 准备格式化历史 - 更新调用以处理新的返回值
                        result = format_history_prepare(history_msgs)
                        full_history, needs_summary, early_text, recent_history_formatted = result[:4]
                        long_model_responses = result[4] if len(result) > 4 else []
                        long_response_indices = result[5] if len(result) > 5 else []
                        user_texts = result[6] if len(result) > 6 else []
                        user_text_indices = result[7] if len(result) > 7 else []

                        item = {
                            "qid": f"{user_id}_{line_idx}_{msg_idx}",
                            "user_id": user_id,
                            "profile_text": profile_text,
                            "record": record,
                            "output": msg["content"],
                            "original_output": msg["content"],  # 保存原始输出用于清理
                            "history_msgs": history_msgs,  # 保存原始消息用于后续处理
                            "full_history": full_history,
                            "needs_summary": needs_summary,
                            "early_text": early_text,
                            "recent_history_formatted": recent_history_formatted,
                            "long_model_responses": long_model_responses,
                            "long_response_indices": long_response_indices,
                            "user_texts": user_texts,
                            "user_text_indices": user_text_indices,
                        }
                        all_processing_items.append(item)

                # 更新进度条信息
                progress_bar.set_postfix(
                    {
                        "已收集": len(all_processing_items),
                        "当前用户": str(user_id)[:10] + "..." if len(str(user_id)) > 10 else str(user_id),
                    }
                )
            except json.JSONDecodeError as e:
                print(f"⚠️ 第{line_idx+1}行JSON解析错误: {e}")
                continue
            except Exception as e:
                print(f"⚠️ 第{line_idx+1}行处理错误: {e}")
                continue

    print(f"✅ 收集完成，共收集到 {len(all_processing_items)} 条消息需要处理")

    # 第三步：批量处理
    condensed_responses = {}
    cleaned_user_texts = {}
    cleaned_outputs = {}  # 新增：存储清理后的输出文本
    summaries = {}

    if use_summarization or use_cleaning:
        print(f"🔄 第三步：加载模型到多GPU ({', '.join(gpu_devices)})...")
        models, tokenizer = load_qwen3_8b_multi_gpu(
            model_path="/project/hdtaccuracy/models/base/Qwen3-4B", devices=gpu_devices, quantize=False
        )
        print("✅ 多GPU模型加载完成")

        # 第三步A：收集并精简过长的model回复
        all_long_responses = []
        response_item_mapping = []

        for item in all_processing_items:
            if item["long_model_responses"]:
                for response in item["long_model_responses"]:
                    all_long_responses.append(response)
                    response_item_mapping.append(item["qid"])

        if all_long_responses and use_summarization:
            print(f"🔄 第三步A：精简过长的model回复（共 {len(all_long_responses)} 条）...")
            condensed_list = summarize_model_responses_batch_multi_gpu(
                models, tokenizer, all_long_responses, batch_size=batch_size
            )

            # 将精简结果按item分组
            for response_idx, (condensed, item_qid) in enumerate(zip(condensed_list, response_item_mapping)):
                if item_qid not in condensed_responses:
                    condensed_responses[item_qid] = []
                condensed_responses[item_qid].append(condensed)

            print("✅ model回复精简完成")

        # 第三步B：收集并清理用户文本（包括历史中的用户文本和最终输出）
        all_user_texts = []
        user_text_item_mapping = []
        all_output_texts = []  # 新增：收集所有需要清理的输出文本
        output_item_mapping = []  # 新增：输出文本的映射

        for item in all_processing_items:
            if use_cleaning:
                # 收集历史中的用户文本
                if item["user_texts"]:
                    for user_text in item["user_texts"]:
                        all_user_texts.append(user_text)
                        user_text_item_mapping.append(item["qid"])

                # 收集当前输出文本进行清理
                all_output_texts.append(item["original_output"])
                output_item_mapping.append(item["qid"])

        if all_user_texts and use_cleaning:
            print(f"🔄 第三步B：清理历史用户文本（共 {len(all_user_texts)} 条）...")
            cleaned_list = clean_user_texts_batch_multi_gpu(models, tokenizer, all_user_texts, batch_size=batch_size)

            # 将清理结果按item分组
            for text_idx, (cleaned, item_qid) in enumerate(zip(cleaned_list, user_text_item_mapping)):
                if item_qid not in cleaned_user_texts:
                    cleaned_user_texts[item_qid] = []
                cleaned_user_texts[item_qid].append(cleaned)

            print("✅ 历史用户文本清理完成")

        if all_output_texts and use_cleaning:
            print(f"🔄 第三步B2：清理输出文本（共 {len(all_output_texts)} 条）...")
            cleaned_output_list = clean_user_texts_batch_multi_gpu(
                models, tokenizer, all_output_texts, batch_size=batch_size
            )

            # 将清理结果映射到对应的item
            for cleaned_output, item_qid in zip(cleaned_output_list, output_item_mapping):
                cleaned_outputs[item_qid] = cleaned_output

            print("✅ 输出文本清理完成")

        # 第三步C：重新构建early_text并进行对话历史总结
        if use_summarization:
            print("🔄 第三步C：重新构建历史文本并进行总结...")

            # 更新所有items的early_text，使用精简后的回复和清理后的用户文本
            items_need_summary = []
            updated_early_texts = []

            for item in all_processing_items:
                if item["needs_summary"]:
                    # 重新构建early_text，使用精简和清理后的内容
                    history_msgs = copy.deepcopy(item["history_msgs"])

                    # 应用精简的回复
                    if item["qid"] in condensed_responses:
                        condensed_list = condensed_responses[item["qid"]]
                        condensed_idx = 0
                        for msg in history_msgs:
                            if msg.get("role") == "model" and len(msg["content"]) > 500:
                                if condensed_idx < len(condensed_list):
                                    msg["content"] = condensed_list[condensed_idx]
                                    condensed_idx += 1

                    # 应用清理的用户文本
                    if item["qid"] in cleaned_user_texts:
                        cleaned_list = cleaned_user_texts[item["qid"]]
                        cleaned_idx = 0
                        for msg in history_msgs:
                            if msg.get("role") == "user":
                                if cleaned_idx < len(cleaned_list):
                                    msg["content"] = cleaned_list[cleaned_idx]
                                    cleaned_idx += 1

                    # 重新构建history并确定需要总结的部分
                    history_list = []
                    for msg in history_msgs:
                        if not msg.get("if_chosen", True):
                            continue
                        if msg["content"].strip() == "EMPTY STRING":
                            continue

                        if msg.get("role") == "user":
                            history_list.append(f"<Target person>: {msg['content']}")
                        elif msg.get("role") == "model":
                            history_list.append(f"<LLM assistant>: {msg['content']}")

                    # 分离早期历史（需要总结的部分）
                    total_msgs = len(history_list)
                    keep_recent = max(3, total_msgs // 3)
                    early_history = history_list[:-keep_recent]

                    if early_history:  # 只有当有早期历史时才进行总结
                        updated_early_text = ""
                        for line in early_history:
                            if len(updated_early_text) > 7000:
                                break
                            if len(line) > 700:
                                updated_early_text += line[:1000].rsplit(" ", 1)[0] + "...\n"
                            else:
                                updated_early_text += line + "\n"

                        items_need_summary.append(item)
                        updated_early_texts.append(updated_early_text)

            if items_need_summary:
                print(f"🔄 批量总结历史对话（共 {len(items_need_summary)} 条需要总结）...")

                # 批量总结 - 使用多GPU
                batch_summaries = summarize_history_batch_multi_gpu(
                    models, tokenizer, updated_early_texts, batch_size=batch_size
                )

                # 将总结结果映射回对应的项目
                for item, summary in zip(items_need_summary, batch_summaries):
                    summaries[item["qid"]] = summary

                print("✅ 批量总结完成")
            else:
                print("ℹ️ 没有需要总结的历史对话")

        # 清理模型以释放显存
        for model in models:
            del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("🗑️ 多GPU模型已清理，显存已释放")
    else:
        print("ℹ️ 跳过总结和清理功能")

    # 第四步：构建最终数据集
    print("🔄 第四步：构建最终数据集...")
    new_records = []

    progress_bar = tqdm(all_processing_items, desc="构建数据集")
    for item in progress_bar:
        # 使用format_history函数处理，但传入预处理的精简回复和清理的用户文本
        history_msgs = copy.deepcopy(item["history_msgs"])  # 深拷贝以避免修改原数据

        # 如果有精简的回复，替换原消息内容
        if item["qid"] in condensed_responses:
            condensed_list = condensed_responses[item["qid"]]
            condensed_idx = 0
            for msg_idx, msg in enumerate(history_msgs):
                if msg.get("role") == "model" and len(msg["content"]) > 500:
                    if condensed_idx < len(condensed_list):
                        msg["content"] = condensed_list[condensed_idx]
                        condensed_idx += 1

        # 如果有清理的用户文本，替换原消息内容
        if item["qid"] in cleaned_user_texts:
            cleaned_list = cleaned_user_texts[item["qid"]]
            cleaned_idx = 0
            for msg_idx, msg in enumerate(history_msgs):
                if msg.get("role") == "user":
                    if cleaned_idx < len(cleaned_list):
                        msg["content"] = cleaned_list[cleaned_idx]
                        cleaned_idx += 1

        # 重新格式化历史（此时已经使用了精简的回复和清理的用户文本）
        if item["needs_summary"] and use_summarization and item["qid"] in summaries:
            # 手动构建使用总结的历史
            history_list = []
            for msg in history_msgs:
                if msg["content"].strip() == "EMPTY STRING" or not msg.get("if_chosen", True):
                    continue
                if msg.get("role") == "user":
                    history_list.append(f"<Target person>: {msg['content']}")
                elif msg.get("role") == "model":
                    history_list.append(f"<LLM assistant>: {msg['content']}")

            total_msgs = len(history_list)
            keep_recent = max(2, total_msgs // 3)
            recent_history = history_list[-keep_recent:]

            history_str = format_history_with_summary(summaries[item["qid"]], recent_history)
            summarized = True
        else:
            # 使用完整历史（已经包含精简的回复和清理的用户文本）
            history_list = []
            for msg in history_msgs:
                if msg["content"].strip() == "EMPTY STRING" or not msg.get("if_chosen", True):
                    continue
                if msg.get("role") == "user":
                    history_list.append(f"<Target person>: {msg['content']}")
                elif msg.get("role") == "model":
                    history_list.append(f"<LLM assistant>: {msg['content']}")

            history_str = "\n".join(history_list)
            summarized = False

        conversation_history = format_conversation_history(history_str, item["record"])

        # 使用清理后的输出（如果有的话）
        final_output = cleaned_outputs.get(item["qid"], item["original_output"])

        new_records.append(
            {
                "qid": item["qid"],
                "prompt": (
                    "Now, you are required to simulate the person with profile below:\n"
                    "[Profile Begin]\n"
                    f"{item['profile_text']}\n\n"
                    "[Profile End]\n"
                    f"{conversation_history}\n"
                    "Now you should generate a response as if you are the person.\n"
                    "Your output should align with the profile of the person and the conversation history.\n"
                    "Now, your output:"
                ),
                "output": final_output,  # 使用清理后的输出
                "summarized": summarized,
                "condensed": item["qid"] in condensed_responses,  # 标记是否使用了精简
                "cleaned": item["qid"] in cleaned_outputs,  # 标记输出是否被清理了
            }
        )

        # 更新进度条信息
        progress_bar.set_postfix(
            {
                "已处理": len(new_records),
                "已总结": sum(1 for rec in new_records if rec.get("summarized", False)),
                "已精简": sum(1 for rec in new_records if rec.get("condensed", False)),
                "已清理": sum(1 for rec in new_records if rec.get("cleaned", False)),
            }
        )

    # 写出新数据集
    print("🔄 第五步：保存数据集...")
    output_path = base_path + output_path
    if not Path(output_path).parent.exists():
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"📂 输出路径：{output_path}")

    with open(output_path, "w", encoding="utf-8") as out_f:
        progress_bar = tqdm(new_records, desc="保存数据")
        for rec in progress_bar:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 统计信息
    summarized_count = sum(1 for rec in new_records if rec.get("summarized", False))
    condensed_count = sum(1 for rec in new_records if rec.get("condensed", False))
    cleaned_count = sum(1 for rec in new_records if rec.get("cleaned", False))
    print(f"✅ 生成完成：{output_path}")
    print(f"📊 总记录数：{len(new_records)}")
    print(f"📊 使用总结的记录数：{summarized_count}")
    print(f"📊 使用精简的记录数：{condensed_count}")
    print(f"📊 使用清理的记录数：{cleaned_count}")
    print(f"📊 总结比例：{summarized_count/len(new_records)*100:.1f}%")
    print(f"📊 精简比例：{condensed_count/len(new_records)*100:.1f}%")
    print(f"📊 清理比例：{cleaned_count/len(new_records)*100:.1f}%")


if __name__ == "__main__":
    # 可以通过 use_summarization 和 use_cleaning 参数控制功能
    base_path = "/project/hdtaccuracy/Personality-Alignment/"
    build_dataset(
        base_path,
        "roleplay_dataset_en_new.jsonl",
        "profile.json",
        "dialogue_dataset_all_v9_summarized_cleaned.jsonl",
        use_summarization=True,
        use_cleaning=True,  # 启用用户文本清理
        gpu_devices=["cuda:0", "cuda:1"],  # 指定使用的GPU
    )
    print("✅ 生成完成：dialogue_dataset_all_v9_summarized_cleaned.jsonl")

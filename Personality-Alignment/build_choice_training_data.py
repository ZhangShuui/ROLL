#!/usr/bin/env python3
import argparse
import json
import os
import sys
import random
import re
from typing import Dict, List, Tuple, Optional

try:
    from transformers import AutoTokenizer

    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("Warning: transformers not available. Token statistics will be disabled.", file=sys.stderr)

# Different prompt templates
PROMPT_TEMPLATES = {
    "no_think": [
        {
            "role": "system",
            "content": "You are an answer agent for multiple-choice questions. Your task is to output the correct choice in the format /choice{letter}. Do not explain, do not copy the option, do not output anything about conversation. Just output /choice{letter} where letter is A, B, C, D, E, F, or G.",
        },
        {
            "role": "user",
            "content": "Choose the correct answer for the following question: {question}\n{reference_data}\nYour output should be in the format /choice{{letter}} where letter is the correct choice (A, B, C, D, E, F, or G). Do not explain, do not copy the option, do not output anything about conversation.\nNow, your output is:",
        },
    ],
    "with_think": [
        {
            "role": "system",
            "content": "You are an answer agent for multiple-choice questions. Your task is to analyze the question carefully and then output the correct choice in the format /choice{letter}. First think through your reasoning, then provide your final answer as /choice{letter}.",
        },
        {
            "role": "user",
            "content": "Choose the correct answer for the following question: {question}\n{reference_data}\nPlease think step by step:\n1. Analyze the person's profile and conversation context\n2. Consider what response would be most appropriate for this person\n3. Evaluate each option against the person's characteristics\n4. Choose the best option\n\nAfter your analysis, provide your final answer in the format /choice{{letter}} where letter is A, B, C, D, E, F, or G.\n\nNow, please provide your reasoning and answer:",
        },
    ],
}

# For backwards compatibility
GENERATION_MESSAGE = PROMPT_TEMPLATES["no_think"]


def load_tokenizer(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Load tokenizer for token counting"""
    if not TOKENIZER_AVAILABLE:
        print("Tokenizer not available - transformers library not installed", file=sys.stderr)
        return None
    try:
        print(f"Attempting to load tokenizer from: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"Successfully loaded tokenizer: {type(tokenizer).__name__}")
        return tokenizer
    except Exception as e:
        print(f"Warning: Failed to load tokenizer {model_name}: {e}", file=sys.stderr)
        return None


def count_tokens(tokenizer, messages: List[dict]) -> int:
    """Count tokens in messages using tokenizer's chat template"""
    if not tokenizer:
        return 0
    try:
        formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer.encode(formatted_text)
        return len(tokens)
    except Exception as e:
        print(f"Warning: Failed to count tokens: {e}", file=sys.stderr)
        try:
            raw_text = "\n".join([msg.get("content", "") for msg in messages])
            tokens = tokenizer.encode(raw_text)
            return len(tokens)
        except Exception:
            return 0


def count_output_tokens(tokenizer, output_text: str) -> int:
    """Count tokens in output text"""
    if not tokenizer:
        return 0
    try:
        tokens = tokenizer.encode(output_text)
        return len(tokens)
    except Exception as e:
        print(f"Warning: Failed to count output tokens: {e}", file=sys.stderr)
        return 0


def batch_count_tokens(tokenizer, messages_list: List[List[dict]], batch_size: int = 32) -> List[int]:
    """Batch count tokens for multiple messages using tokenizer's chat template"""
    if not tokenizer:
        return [0] * len(messages_list)

    token_counts = []

    for i in range(0, len(messages_list), batch_size):
        batch = messages_list[i : i + batch_size]
        batch_texts = []

        for messages in batch:
            try:
                formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                batch_texts.append(formatted_text)
            except Exception as e:
                print(f"Warning: Failed to format messages: {e}", file=sys.stderr)
                raw_text = "\n".join([msg.get("content", "") for msg in messages])
                batch_texts.append(raw_text)

        try:
            batch_encodings = tokenizer(
                batch_texts, padding=False, truncation=False, return_tensors=None, add_special_tokens=False
            )
            for encoding in batch_encodings["input_ids"]:
                token_counts.append(len(encoding))
        except Exception as e:
            print(f"Warning: Failed to batch encode: {e}", file=sys.stderr)
            for text in batch_texts:
                try:
                    tokens = tokenizer.encode(text)
                    token_counts.append(len(tokens))
                except Exception:
                    token_counts.append(0)

    return token_counts


def batch_count_output_tokens(tokenizer, output_texts: List[str], batch_size: int = 64) -> List[int]:
    """Batch count tokens for multiple output texts"""
    if not tokenizer:
        return [0] * len(output_texts)

    token_counts = []

    for i in range(0, len(output_texts), batch_size):
        batch = output_texts[i : i + batch_size]
        try:
            batch_encodings = tokenizer(
                batch, padding=False, truncation=False, return_tensors=None, add_special_tokens=False
            )
            for encoding in batch_encodings["input_ids"]:
                token_counts.append(len(encoding))
        except Exception as e:
            print(f"Warning: Failed to batch encode outputs: {e}", file=sys.stderr)
            for text in batch:
                try:
                    tokens = tokenizer.encode(text)
                    token_counts.append(len(tokens))
                except Exception:
                    token_counts.append(0)

    return token_counts


def analyze_token_stats(tokenizer, items: List[dict], batch_size: int = 32) -> dict:
    """Analyze token statistics for the dataset using batch processing"""
    if not tokenizer:
        print("No tokenizer available for token analysis", file=sys.stderr)
        return {}

    if not items:
        print("No items to analyze", file=sys.stderr)
        return {}

    print(f"Analyzing token statistics for {len(items)} items using batch processing...")
    messages_list = [item.get("messages", []) for item in items]
    output_texts = [item.get("output", "") for item in items]

    print(f"Batch processing {len(messages_list)} prompt messages...")
    prompt_lengths = batch_count_tokens(tokenizer, messages_list, batch_size=batch_size)

    print(f"Batch processing {len(output_texts)} output texts...")
    output_lengths = batch_count_output_tokens(tokenizer, output_texts, batch_size=batch_size * 2)

    total_lengths = [p + o for p, o in zip(prompt_lengths, output_lengths)]
    print(f"Token analysis completed for {len(items)} items.")

    def get_stats(lengths):
        if not lengths:
            return {}
        lengths_sorted = sorted(lengths)
        n = len(lengths_sorted)
        return {
            "count": n,
            "min": min(lengths_sorted),
            "max": max(lengths_sorted),
            "mean": sum(lengths_sorted) / n,
            "median": lengths_sorted[n // 2] if n > 0 else 0,
            "p90": lengths_sorted[int(n * 0.9)] if n > 0 else 0,
            "p95": lengths_sorted[int(n * 0.95)] if n > 0 else 0,
            "p99": lengths_sorted[int(n * 0.99)] if n >= 100 else (lengths_sorted[-1] if lengths_sorted else 0),
        }

    return {
        "prompt_tokens": get_stats(prompt_lengths),
        "output_tokens": get_stats(output_lengths),
        "total_tokens": get_stats(total_lengths),
    }


def print_token_stats(stats: dict, prompt_type: str):
    """Print token statistics in a readable format"""
    if not stats:
        print("Token statistics not available (tokenizer not loaded)")
        return

    print(f"\n=== Token Statistics (Prompt Type: {prompt_type}) ===")
    for section, section_stats in stats.items():
        if not section_stats:
            continue
        section_name = section.replace("_", " ").title()
        print(f"\n{section_name}:")
        print(f"  Count: {section_stats['count']}")
        print(f"  Min: {section_stats['min']}")
        print(f"  Max: {section_stats['max']}")
        print(f"  Mean: {section_stats['mean']:.1f}")
        print(f"  Median: {section_stats['median']}")
        print(f"  P90: {section_stats['p90']}")
        print(f"  P95: {section_stats['p95']}")
        print(f"  P99: {section_stats['p99']}")


def load_questions(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("questions file must be a JSON array")
        return data


def load_prompts(path: str) -> Dict[str, str]:
    """Load prompts from JSONL: each line must have {'qid': ..., 'prompt': ...}"""
    prompts: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[warn] prompts line {lineno} JSON parse error: {e}", file=sys.stderr)
                continue
            qid = str(obj.get("qid", "")).strip()
            prompt_text = obj.get("prompt")
            if not qid or not isinstance(prompt_text, str):
                print(f"[warn] prompts line {lineno} missing qid or prompt", file=sys.stderr)
                continue
            prompts[qid] = prompt_text
    return prompts


def extract_profile_conv_from_prompt(prompt: str) -> Tuple[str, str]:
    profile = ""
    conversation_history = ""
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
    return profile, conversation_history


def create_multi_choice_question_text(question: dict, profile: str, conversation_history: str) -> str:
    """Create question text supporting 2-7 choices (A-G)"""
    parts: List[str] = []
    parts.append(
        "Given the following information about a person and a multiple-choice question, select the most appropriate response from the provided options."
    )
    if profile:
        parts.append(f"[Profile Begin]\n{profile}\n[Profile End]\n")
    if conversation_history:
        parts.append(f"[Conversation History Begin]\n{conversation_history}\n[Conversation History End]\n")
    parts.append("Which response is most appropriate for this person in this context?\n")

    choices = question.get("choices", [])
    sorted_choices = sorted(choices, key=lambda x: x.get("label", ""))

    for choice in sorted_choices:
        label = (choice.get("label") or "").strip()
        text = (choice.get("text") or "").strip()
        if label and text:
            parts.append(f"{label}. {text}")

    return "\n".join(parts).rstrip() + "\n"


def get_correct_answer_letter(question: dict) -> Optional[str]:
    """Get correct answer letter, supporting A-G"""
    letter = question.get("correct_answer")
    if isinstance(letter, str) and letter.strip():
        return letter.strip().upper()

    for ch in question.get("choices", []):
        if ch.get("is_correct") is True:
            lbl = ch.get("label")
            if isinstance(lbl, str) and lbl.strip():
                return lbl.strip().upper()
    return None


def parse_choice_letter_from_output(output: str) -> Optional[str]:
    """Parse choice letter from output like '/choice{A}' or 'A'"""
    if not isinstance(output, str):
        return None
    m = re.search(r"choice\s*\{([A-G])\}", output, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    out = output.strip().upper()
    if out in list("ABCDEFG"):
        return out
    return None


def normalize_existing_choices(question_dict: dict) -> Tuple[Optional[List[dict]], Optional[str]]:
    """
    Normalize existing choices:
    - Keep all provided options with non-empty text, up to 7 options (A-G).
    - Determine the correct option from is_correct/correct_answer/output.
    - Re-label sequentially A.. and ensure exactly one is_correct=True.
    Returns (choices, correct_letter) or (None, None) if cannot determine.
    """
    raw_choices = question_dict.get("choices", [])
    if not isinstance(raw_choices, list) or not raw_choices:
        return None, None

    norm = []
    for ch in raw_choices:
        text = (ch.get("text") or "").strip()
        if text == "":
            continue
        label = (ch.get("label") or "").strip().upper()
        is_correct = bool(ch.get("is_correct", False))
        norm.append({"text": text, "label": label, "is_correct": is_correct})

    if not norm:
        return None, None

    idx_correct = None
    # 1) from is_correct flag
    for i, c in enumerate(norm):
        if c.get("is_correct"):
            idx_correct = i
            break

    # 2) from correct_answer (letter or text)
    if idx_correct is None:
        ca = question_dict.get("correct_answer")
        if isinstance(ca, str) and ca.strip():
            ca_str = ca.strip()
            # try text match first
            for i, c in enumerate(norm):
                if c["text"].strip() == ca_str:
                    idx_correct = i
                    break
            # fallback letter on existing labels
            if idx_correct is None:
                ca_letter = ca_str.upper()
                if ca_letter in list("ABCDEFG"):
                    for i, c in enumerate(norm):
                        if (c.get("label") or "").upper() == ca_letter:
                            idx_correct = i
                            break

    # 3) from output (letter or text)
    if idx_correct is None:
        out = question_dict.get("output") or ""
        ca_letter = parse_choice_letter_from_output(out)
        if ca_letter:
            for i, c in enumerate(norm):
                if (c.get("label") or "").upper() == ca_letter:
                    idx_correct = i
                    break
        if idx_correct is None and isinstance(out, str) and out.strip():
            for i, c in enumerate(norm):
                if c["text"].strip() == out.strip():
                    idx_correct = i
                    break

    if idx_correct is None:
        return None, None

    # Limit to at most 7 choices (A-G), keep correct and the earliest others
    MAX_CHOICES = 7
    if len(norm) > MAX_CHOICES:
        kept = [norm[idx_correct]]
        for i, c in enumerate(norm):
            if i == idx_correct:
                continue
            if len(kept) >= MAX_CHOICES:
                break
            kept.append(c)
        norm = kept
        idx_correct = 0

    # Assign labels A.. and set is_correct
    for i, c in enumerate(norm):
        c["label"] = chr(65 + i)  # A, B, ...
        c["is_correct"] = i == idx_correct

    correct_letter = chr(65 + idx_correct)
    return norm, correct_letter


def validate_question_format(question: dict) -> bool:
    """Validate that question has proper format and choices (2-7 choices)"""
    choices = question.get("choices", [])
    if not choices or len(choices) < 2 or len(choices) > 7:
        return False

    labels = set()
    has_correct = False
    valid_labels = set("ABCDEFG")

    for choice in choices:
        label = (choice.get("label") or "").strip().upper()
        text = (choice.get("text") or "").strip()

        if not label or not text:
            continue

        if label not in valid_labels:
            return False

        if label in labels:
            return False
        labels.add(label)

        if choice.get("is_correct"):
            has_correct = True

    return 2 <= len(labels) <= 7 and has_correct


def build_messages(question_text: str, reference_data: str = "", prompt_type: str = "no_think") -> List[dict]:
    """Build messages using specified prompt template"""
    if prompt_type not in PROMPT_TEMPLATES:
        raise ValueError(f"Unsupported prompt type: {prompt_type}. Available types: {list(PROMPT_TEMPLATES.keys())}")

    template = PROMPT_TEMPLATES[prompt_type]
    return [
        template[0],
        {
            "role": "user",
            "content": template[1]["content"].format(question=question_text, reference_data=reference_data or ""),
        },
    ]


def get_expected_output_format(prompt_type: str, correct_letter: str) -> str:
    """Get expected output format based on prompt type"""
    if prompt_type in ("no_think", "with_think"):
        return "/choice{" + correct_letter + "}"
    return f"/choice{correct_letter}"


def export_dataset(items: List[dict], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if out_path.lower().endswith(".json"):
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    else:
        with open(out_path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")


def parse_qid(qid: str) -> Tuple[str, int, int]:
    """
    解析qid，返回(user_id, line_idx, msg_idx)
    若解析失败，则将整个qid作为user_id，索引置0。
    """
    try:
        parts = str(qid).split("_")
        if len(parts) >= 3:
            user_id = "_".join(parts[:-2])
            line_idx = int(parts[-2])
            msg_idx = int(parts[-1])
        else:
            user_id, line_idx, msg_idx = str(qid), 0, 0
        return user_id, line_idx, msg_idx
    except Exception:
        return str(qid), 0, 0


def random_split_items(items: List[dict], test_ratio: float = 0.2, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """随机划分 items"""
    rng = random.Random(seed)
    arr = items.copy()
    rng.shuffle(arr)
    n_test = int(len(arr) * test_ratio)
    test_items = arr[:n_test]
    train_items = arr[n_test:]
    return train_items, test_items


def user_based_split_items(items: List[dict], test_ratio: float = 0.2) -> Tuple[List[dict], List[dict]]:
    """
    按用户划分：每个用户末尾 test_ratio 的样本作测试集（按 line_idx, msg_idx 排序）
    """
    buckets: Dict[str, List[Tuple[dict, int, int]]] = {}
    for it in items:
        qid = it.get("qid", "")
        user_id, line_idx, msg_idx = parse_qid(qid)
        buckets.setdefault(user_id, []).append((it, line_idx, msg_idx))

    train_items: List[dict] = []
    test_items: List[dict] = []
    for _, arr in buckets.items():
        arr.sort(key=lambda x: (x[1], x[2]))
        n = len(arr)
        n_test = max(1, int(n * test_ratio)) if n > 0 else 0
        user_test = arr[-n_test:] if n_test > 0 else []
        user_train = arr[:-n_test] if n_test > 0 else arr
        train_items.extend([x[0] for x in user_train])
        test_items.extend([x[0] for x in user_test])
    return train_items, test_items


def user_partial_split_items(
    items: List[dict],
    test_ratio: float = 0.2,
    user_subset_ratio: float = 0.3,
    seed: int = 42,
) -> Tuple[List[dict], List[dict]]:
    """
    新语义：
    - 随机抽取 user_subset_ratio 比例的用户，这些用户的所有样本全部进入测试集，标记 test_tag="fully"
      （若计算得到数量 >= 总用户且总用户>1，则减少 1，保证不是所有用户都被选为 fully）
    - 其余用户：各自末尾 test_ratio 比例(至少1条若该用户样本数>0) 进入测试集，标记 test_tag="partially"，其余进入训练集
    返回: (train_items, test_items)
    注意：测试集中每条样本都带有字段 "test_tag": "fully" 或 "partially"
    """
    # 分桶
    buckets: Dict[str, List[Tuple[dict, int, int]]] = {}
    for it in items:
        qid = it.get("qid", "")
        user_id, line_idx, msg_idx = parse_qid(qid)
        buckets.setdefault(user_id, []).append((it, line_idx, msg_idx))

    users = list(buckets.keys())
    if not users:
        return items, []

    rng = random.Random(seed)
    rng.shuffle(users)

    k = max(1, int(len(users) * user_subset_ratio))
    if k >= len(users) and len(users) > 1:
        k = len(users) - 1  # 避免全部用户都 fully
    fully_users = set(users[:k])

    train_items: List[dict] = []
    test_items: List[dict] = []

    for uid, arr in buckets.items():
        arr.sort(key=lambda x: (x[1], x[2]))  # 按 (line_idx, msg_idx)
        if uid in fully_users:
            for rec, _, _ in arr:
                rec["test_tag"] = "fully"
                test_items.append(rec)
        else:
            n = len(arr)
            if n == 0:
                continue
            n_test = max(1, int(n * test_ratio))
            tail = arr[-n_test:]
            head = arr[:-n_test]
            for rec, _, _ in head:
                train_items.append(rec)
            for rec, _, _ in tail:
                rec["test_tag"] = "partially"
                test_items.append(rec)

    return train_items, test_items


def kfold_split(items: List[dict], k: int, mode: str, seed: int = 42):
    """
    返回列表 folds，每个元素 (train_items, test_items)
    mode: 'random' or 'user'
    """
    if k <= 1:
        raise ValueError("k must be > 1 for k-fold")
    if mode == "random":
        rng = random.Random(seed)
        shuffled = items.copy()
        rng.shuffle(shuffled)
        fold_size = max(1, len(shuffled) // k)
        folds_raw = [shuffled[i * fold_size : (i + 1) * fold_size] for i in range(k - 1)]
        folds_raw.append(shuffled[(k - 1) * fold_size :])
    elif mode == "user":
        # 按用户分桶，再把用户打乱后均匀分配到 k 个折
        buckets: Dict[str, List[dict]] = {}
        for it in items:
            uid, _, _ = parse_qid(it.get("qid", ""))
            buckets.setdefault(uid, []).append(it)
        users = list(buckets.keys())
        rng = random.Random(seed)
        rng.shuffle(users)
        folds_raw = [[] for _ in range(k)]
        for idx, uid in enumerate(users):
            folds_raw[idx % k].extend(buckets[uid])
    else:
        raise ValueError("unsupported kfold mode")
    folds = []
    for i in range(k):
        test_items = folds_raw[i]
        train_items = []
        for j in range(k):
            if j != i:
                train_items.extend(folds_raw[j])
        folds.append((train_items, test_items))
    return folds


def split_validation(
    test_items: List[dict],
    val_ratio: float = 0.5,
    seed: int = 42,
    balance_test_tag: bool = False,
) -> Tuple[List[dict], List[dict]]:
    """
    从 test_items 中抽取 val_ratio 比例作为验证集。
    balance_test_tag=True 时，对 test_tag 分层（用于 user_partial）。
    返回 (val_items, remaining_test_items)
    """
    if not test_items or val_ratio <= 0:
        return [], test_items
    rng = random.Random(seed)
    if not balance_test_tag:
        idxs = list(range(len(test_items)))
        rng.shuffle(idxs)
        cut = max(1, int(len(idxs) * val_ratio))
        val_set_idx = set(idxs[:cut])
        val_items = [test_items[i] for i in range(len(test_items)) if i in val_set_idx]
        remaining = [test_items[i] for i in range(len(test_items)) if i not in val_set_idx]
        return val_items, remaining
    # 分层：按 test_tag
    buckets: Dict[str, List[dict]] = {}
    for it in test_items:
        tag = it.get("test_tag", "_none")
        buckets.setdefault(tag, []).append(it)
    val_items: List[dict] = []
    remaining: List[dict] = []
    for tag, arr in buckets.items():
        arr_copy = arr[:]  # 不打乱原顺序
        rng.shuffle(arr_copy)
        take = max(1, int(len(arr_copy) * val_ratio)) if len(arr_copy) > 0 else 0
        val_items.extend(arr_copy[:take])
        remaining.extend(arr_copy[take:])
    return val_items, remaining


def derive_split_paths(out_path: str) -> Tuple[str, str]:
    """
    基于 --out 生成 train/test 输出路径：
    - 若以 .json/.jsonl 结尾：追加 _train/_test
    - 否则视为目录：写入 train.jsonl / test.jsonl
    """
    base, ext = os.path.splitext(out_path)
    if ext.lower() in {".json", ".jsonl"}:
        return f"{base}_train{ext}", f"{base}_test{ext}"
    os.makedirs(out_path, exist_ok=True)
    return os.path.join(out_path, "train.jsonl"), os.path.join(out_path, "test.jsonl")


def derive_split_paths_with_val(out_path: str) -> Tuple[str, str, str]:
    """
    生成 train / val / test 输出路径
    """
    base, ext = os.path.splitext(out_path)
    if ext.lower() in {".json", ".jsonl"}:
        return f"{base}_train{ext}", f"{base}_val{ext}", f"{base}_test{ext}"
    os.makedirs(out_path, exist_ok=True)
    return (
        os.path.join(out_path, "train.jsonl"),
        os.path.join(out_path, "val.jsonl"),
        os.path.join(out_path, "test.jsonl"),
    )


def analyze_dataset_stats(questions: List[dict]) -> dict:
    """Analyze question statistics from input questions"""
    choice_counts = {}
    valid_questions = 0
    for q in questions:
        choices = q.get("choices", [])
        if not isinstance(choices, list) or not choices:
            continue
        num_choices = len([c for c in choices if (c.get("text") or "").strip()])
        if num_choices <= 0:
            continue
        choice_counts[num_choices] = choice_counts.get(num_choices, 0) + 1
        valid_questions += 1
    return {
        "total_questions": len(questions),
        "valid_questions": valid_questions,
        "choice_count_distribution": choice_counts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build {qid, messages, output} dataset for multi-choice QA using all existing options per question (2-7 choices)."
    )
    parser.add_argument("--questions", required=True, help="Path to questions JSON array")
    parser.add_argument("--prompts", required=True, help="Path to prompts JSONL")
    parser.add_argument("--out", required=True, help="Output path (.json or .jsonl, or a directory)")
    parser.add_argument("--max_items", type=int, default=0, help="Optional cap on number of items")
    parser.add_argument("--skip_missing_prompt", action="store_true", help="Skip items without prompt")
    parser.add_argument("--skip_invalid_format", action="store_true", help="Skip questions with invalid format")
    parser.add_argument(
        "--prompt_type",
        choices=list(PROMPT_TEMPLATES.keys()),
        default="no_think",
        help="Type of prompt template to use",
    )
    parser.add_argument(
        "--split_mode",
        choices=["none", "random", "user", "user_partial", "kfold_random", "kfold_user"],
        default="none",
        help="Split dataset mode",
    )
    parser.add_argument("--k_folds", type=int, default=5, help="For kfold_* modes: number of folds")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    parser.add_argument(
        "--user_subset_ratio",
        type=float,
        default=0.2,
        help="For user_partial: ratio of users to fully hold out",
    )
    parser.add_argument("--make_val", action="store_true", help="Split half (val_ratio) of test set as validation set")
    parser.add_argument("--val_ratio", type=float, default=0.5, help="Portion of test set to become validation set")
    parser.add_argument("--show_stats", action="store_true", help="Show dataset statistics")
    parser.add_argument("--show_token_stats", action="store_true", help="Show token statistics using tokenizer")
    parser.add_argument(
        "--tokenizer_model", default="Qwen/Qwen2.5-7B-Instruct", help="Tokenizer model to use for token counting"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for token counting")
    parser.add_argument(
        "--start_index", type=int, default=0, help="Start index for processing questions (inclusive, 0-based)"
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=-1,
        help="End index for processing questions (exclusive, -1 means process until end)",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    questions = load_questions(args.questions)
    prompts_map = load_prompts(args.prompts)

    # Apply index range filtering
    total_questions = len(questions)
    start_idx = max(0, args.start_index)
    end_idx = total_questions if args.end_index == -1 else min(total_questions, args.end_index)
    if start_idx >= end_idx:
        print(f"ERROR: Invalid index range. start_index ({start_idx}) must be less than end_index ({end_idx})")
        print(f"Total questions available: {total_questions}")
        sys.exit(1)

    original_total = len(questions)
    questions = questions[start_idx:end_idx]
    print(f"Processing questions {start_idx} to {end_idx-1} (total: {len(questions)} out of {original_total})")

    # Load tokenizer if token stats requested
    tokenizer = None
    if args.show_token_stats:
        if not TOKENIZER_AVAILABLE:
            print("ERROR: Cannot show token statistics - transformers library not available!")
            print("Please install transformers: pip install transformers")
            sys.exit(1)
        print(f"Loading tokenizer: {args.tokenizer_model}")
        tokenizer = load_tokenizer(args.tokenizer_model)
        if not tokenizer:
            print("ERROR: Failed to load tokenizer! Token statistics will be disabled.")
        else:
            print("Tokenizer loaded successfully!")

    print(f"Using prompt type: {args.prompt_type}")
    print(f"Available prompt types: {list(PROMPT_TEMPLATES.keys())}")

    # Show statistics if requested (on filtered range)
    if args.show_stats:
        stats = analyze_dataset_stats(questions)
        print("\n=== Dataset Statistics (Filtered Range) ===")
        print(f"Index range: {start_idx} to {end_idx-1}")
        print(f"Total questions in range: {stats['total_questions']}")
        print(f"Valid questions in range: {stats['valid_questions']}")
        print("Choice count distribution:")
        for num_choices, count in sorted(stats["choice_count_distribution"].items()):
            percentage = count / stats["valid_questions"] * 100 if stats["valid_questions"] > 0 else 0
            print(f"  {num_choices} choices: {count} questions ({percentage:.1f}%)")
        print()

    items: List[dict] = []
    skipped_no_prompt = 0
    skipped_invalid_format = 0
    skipped_no_correct = 0
    skipped_no_choices = 0

    for i, q in enumerate(questions):
        actual_idx = start_idx + i
        qid = str(q.get("qid", "")).strip()
        if not qid:
            continue

        prompt_text = prompts_map.get(qid)
        if not prompt_text:
            if args.skip_missing_prompt:
                skipped_no_prompt += 1
                continue
            else:
                prompt_text = ""

        profile, conv = extract_profile_conv_from_prompt(prompt_text)

        # Normalize choices from existing data only
        choices, correct_letter = normalize_existing_choices(q)
        if choices is None or correct_letter is None:
            # No usable choices or cannot determine correct option
            if not q.get("choices"):
                skipped_no_choices += 1
            else:
                skipped_no_correct += 1
            continue

        normalized_question = q.copy()
        normalized_question.update(
            {
                "choices": choices,
                "correct_answer": correct_letter,
                "num_choices": len(choices),
            }
        )

        # Validate if requested
        if args.skip_invalid_format and not validate_question_format(normalized_question):
            skipped_invalid_format += 1
            continue

        question_text = create_multi_choice_question_text(normalized_question, profile, conv)
        messages = build_messages(question_text, reference_data="", prompt_type=args.prompt_type)

        output = get_expected_output_format(args.prompt_type, correct_letter)

        item_data = {
            "qid": qid,
            "messages": messages,
            "output": output,
            "num_choices": len(choices),
            "prompt_type": args.prompt_type,
            "choices": choices,  # Include choices for reference
            "correct_answer": correct_letter,
            "original_index": actual_idx,
        }
        items.append(item_data)

        if args.max_items and len(items) >= args.max_items:
            print(f"Reached max_items limit ({args.max_items}), stopping at index {actual_idx}")
            break

    # Update output filename to include range if not full dataset
    if start_idx != 0 or end_idx != original_total:
        base_out = args.out
        name, ext = os.path.splitext(base_out)
        if ext:
            range_suffix = f"_range_{start_idx}_{end_idx-1}"
            args.out = f"{name}{range_suffix}{ext}"
        else:
            args.out = f"{base_out}_range_{start_idx}_{end_idx-1}"
        print(f"Output path updated to include range: {args.out}")

    # Show token statistics if requested
    if args.show_token_stats and tokenizer and items:
        print("\n" + "=" * 50)
        print(f"ANALYZING TOKEN STATISTICS FOR RANGE {start_idx}-{end_idx-1}")
        print("=" * 50)
        token_stats = analyze_token_stats(tokenizer, items, batch_size=args.batch_size)
        if token_stats:
            print_token_stats(token_stats, args.prompt_type)
        else:
            print("Failed to generate token statistics")

    # Export with or without split
    if args.split_mode == "none":
        if args.make_val:
            print("WARNING: --make_val ignored when split_mode=none (no test set).")
        export_dataset(items, args.out)
        print(f"Built {len(items)} items from range {start_idx}-{end_idx-1} -> {args.out}")
    else:
        if args.split_mode == "random":
            train_items, test_items = random_split_items(items, test_ratio=args.test_ratio, seed=args.seed)
        elif args.split_mode == "user":
            train_items, test_items = user_based_split_items(items, test_ratio=args.test_ratio)
        elif args.split_mode in ("kfold_random", "kfold_user"):
            mode = "random" if args.split_mode == "kfold_random" else "user"
            folds = kfold_split(items, k=args.k_folds, mode=mode, seed=args.seed)
            base_out = args.out
            os.makedirs(base_out, exist_ok=True)
            for i, (tr, te) in enumerate(folds):
                val_items: List[dict] = []
                if args.make_val and te:
                    val_items, te = split_validation(
                        te, val_ratio=args.val_ratio, seed=args.seed + i, balance_test_tag=False
                    )
                fold_dir = os.path.join(base_out, f"fold_{i}")
                os.makedirs(fold_dir, exist_ok=True)
                export_dataset(tr, os.path.join(fold_dir, "train.jsonl"))
                if args.make_val:
                    export_dataset(val_items, os.path.join(fold_dir, "val.jsonl"))
                export_dataset(te, os.path.join(fold_dir, "test.jsonl"))
                print(
                    f"Fold {i}: train={len(tr)}, "
                    + (f"val={len(val_items)}, " if args.make_val else "")
                    + f"test={len(te)}"
                )
            print(f"K-fold ({args.k_folds}, mode={mode}) saved under {base_out}")
            return
        else:  # user_partial
            train_items, test_items = user_partial_split_items(
                items,
                test_ratio=args.test_ratio,
                user_subset_ratio=args.user_subset_ratio,
                seed=args.seed,
            )

        if args.show_token_stats and tokenizer:
            if train_items:
                print("\n" + "=" * 50)
                print(f"TRAINING SET TOKEN STATISTICS (Range {start_idx}-{end_idx-1})")
                print("=" * 50)
                train_token_stats = analyze_token_stats(tokenizer, train_items, batch_size=args.batch_size)
                if train_token_stats:
                    print_token_stats(train_token_stats, args.prompt_type)
            if test_items:
                print("\n" + "=" * 50)
                print(f"TEST SET TOKEN STATISTICS (Range {start_idx}-{end_idx-1})")
                print("=" * 50)
                test_token_stats = analyze_token_stats(tokenizer, test_items, batch_size=args.batch_size)
                if test_token_stats:
                    print_token_stats(test_token_stats, args.prompt_type)

        if args.make_val:
            balance = args.split_mode == "user_partial"
            val_items, test_items = split_validation(
                test_items, val_ratio=args.val_ratio, seed=args.seed, balance_test_tag=balance
            )
            train_out, val_out, test_out = derive_split_paths_with_val(args.out)
            export_dataset(train_items, train_out)
            export_dataset(val_items, val_out)
            export_dataset(test_items, test_out)
            print(
                f"Built {len(items)} items from range {start_idx}-{end_idx-1}; train={len(train_items)}, val={len(val_items)}, test={len(test_items)}"
            )
            print(f"Saved train -> {train_out}")
            print(f"Saved val   -> {val_out}")
            print(f"Saved test  -> {test_out}")
        else:
            train_out, test_out = derive_split_paths(args.out)
            export_dataset(train_items, train_out)
            export_dataset(test_items, test_out)
            print(
                f"Built {len(items)} items from range {start_idx}-{end_idx-1}; train={len(train_items)}, test={len(test_items)}"
            )
            print(f"Saved train -> {train_out}")
            print(f"Saved test  -> {test_out}")

    if skipped_no_prompt:
        print(f"Skipped {skipped_no_prompt} items due to missing prompt", file=sys.stderr)
    if skipped_invalid_format:
        print(f"Skipped {skipped_invalid_format} items due to invalid question format", file=sys.stderr)
    if skipped_no_correct:
        print(f"Skipped {skipped_no_correct} items due to undetermined correct answer", file=sys.stderr)
    if skipped_no_choices:
        print(f"Skipped {skipped_no_choices} items due to missing or empty choices", file=sys.stderr)

    print(f"\nGenerated dataset with prompt type: {args.prompt_type}")
    print(f"Processed questions from index {start_idx} to {end_idx-1}")


if __name__ == "__main__":
    main()

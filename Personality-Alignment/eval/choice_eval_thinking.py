#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import argparse
from collections import Counter, defaultdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch.distributed as dist
import torch.distributed as dist


def init_dist():
    """Initialize torch.distributed from torchrun env vars."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        is_main = rank == 0
        return rank, world_size, local_rank, is_main
    # fallback: 单进程调试
    return 0, 1, 0, True


# ========== 基础工具 ==========
def setup_ddp():
    """Init torch.distributed if launched with torchrun; return (rank, world_size, local_rank, is_main)."""
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # 单进程降级模式（便于本地调试）
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank, (rank == 0)


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def normalize_choice(response: str):
    if not response:
        return None
    t = response.strip()
    patterns = [
        r"/choice\s*[{(]?\s*([A-G])\s*[})]?",  # /choice{A}
        r"\bchoice\s*[:\-]?\s*([A-G])\b",  # choice: A
        r"\banswer\s*[:\-]?\s*([A-G])\b",  # answer: A
        r"\boption\s*[:\-]?\s*([A-G])\b",  # option: A
        r'^\s*["\'`(]*([A-G])(?:[\).\s]|$)',  # 文首单字母
        r"\b([A-G])\b",  # 任意独立字母
    ]
    for p in patterns:
        m = re.search(p, t, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def load_test_dataset(dataset_path: str):
    if dataset_path.endswith(".jsonl"):
        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompts_from_messages(tokenizer, messages_batch):
    prompts = []
    for messages in messages_batch:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # 某些分支不支持则删去：
            enable_thinking=False,
        )
        prompts.append(prompt)
    return prompts


@torch.inference_mode()
def generate_response_batch(
    model,
    tokenizer,
    messages_list,
    *,
    max_new_tokens=128,
    batch_size=32,
    temperature=0.7,
    top_p=0.9,
    max_input_len=2500,
    device="cuda",
):
    all_responses = []
    for i in range(0, len(messages_list), batch_size):
        batch_messages = messages_list[i : i + batch_size]
        prompts = build_prompts_from_messages(tokenizer, batch_messages)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_len,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        except torch.cuda.OutOfMemoryError:
            # 简单 OOM 回退：减半 batch 继续
            torch.cuda.empty_cache()
            small_bs = max(1, len(batch_messages) // 2)
            for j in range(0, len(batch_messages), small_bs):
                sub_msgs = batch_messages[j : j + small_bs]
                sub_prompts = build_prompts_from_messages(tokenizer, sub_msgs)
                sub_inputs = tokenizer(
                    sub_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_input_len
                )
                sub_inputs = {k: v.to(device) for k, v in sub_inputs.items()}
                sub_out = model.generate(
                    **sub_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )
                attn = sub_inputs["attention_mask"]
                for r in range(sub_out.size(0)):
                    in_len = int(attn[r].sum().item())
                    gen_tokens = sub_out[r, in_len:]
                    text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                    all_responses.append(text)
            continue

        # 正常路径：用 attention_mask 剥离输入前缀
        attn = inputs["attention_mask"]
        for j in range(outputs.size(0)):
            in_len = int(attn[j].sum().item())
            gen_tokens = outputs[j, in_len:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            all_responses.append(text)

    return all_responses


def list_model_dirs(model_dirs, model_globs, models_file, max_models=None):
    paths = []

    for d in model_dirs or []:
        if os.path.isdir(d):
            paths.append(os.path.abspath(d))

    for g in model_globs or []:
        for d in glob.glob(g):
            if os.path.isdir(d):
                paths.append(os.path.abspath(d))

    if models_file:
        with open(models_file, "r", encoding="utf-8") as f:
            for line in f:
                d = line.strip()
                if d and os.path.isdir(d):
                    paths.append(os.path.abspath(d))

    def looks_like_hf(d):
        has_cfg = os.path.isfile(os.path.join(d, "config.json"))
        has_tok = os.path.isfile(os.path.join(d, "tokenizer.json"))
        has_wt = len(glob.glob(os.path.join(d, "model*.safetensors"))) > 0
        return has_cfg and has_tok and has_wt

    paths = [p for p in sorted(set(paths)) if looks_like_hf(p)]
    if max_models:
        paths = paths[:max_models]
    return paths


def parse_args():
    ap = argparse.ArgumentParser("Distributed (multi-GPU) evaluation for HF models on multi-choice tasks (A–G).")
    ap.add_argument("--dataset", required=True, help="JSON/JSONL with fields: messages, output")
    ap.add_argument("--save_dir", default="choice_eval_dist_outputs", help="Where to save outputs")

    group = ap.add_mutually_exclusive_group(required=False)
    group.add_argument("--models_file", help="Text file: one model dir per line")
    ap.add_argument("--model_dir", action="append", help="Repeatable exact HF model dir")
    ap.add_argument("--model_glob", action="append", help="Repeatable glob, e.g. /path/.../checkpoint-*")
    ap.add_argument(
        "--default_globs",
        action="store_true",
        help="Use built-in globs for your qwen3 0.6B/1.7B/4B under converted_hf",
    )

    ap.add_argument("--max_models", type=int, default=None)

    ap.add_argument("--batch_size", type=int, default=32, help="Per-GPU batch size")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.01)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--sample", type=int, default=None, help="Evaluate first N samples only")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max_input_len", type=int, default=2500)

    return ap.parse_args()


def evaluate_one_model_dist(
    model_dir,
    dataset,
    *,
    batch_size,
    max_new_tokens,
    temperature,
    top_p,
    sample,
    dtype,
    max_input_len,
    save_dir,
    rank,
    world_size,
    local_rank,
    is_main,
):

    # 每个 rank 只跑自己负责的样本：i % world_size == rank
    all_indices = list(range(len(dataset))) if sample is None else list(range(min(sample, len(dataset))))
    shard_indices = [i for i in all_indices if (i % world_size) == rank]
    if is_main:
        print(
            f"[{os.path.basename(model_dir)}] world_size={world_size}, each rank samples≈{len(all_indices)//world_size}, rank{rank} -> {len(shard_indices)}"
        )

    # 加载 tokenizer + model 到本 rank 的 GPU
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]

    # 关键改动：用 device_map 把权重直接加载到本 rank 的 GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map={"": local_rank},
        attn_implementation="flash_attention_2",  # 如果环境不支持，把这行删掉
    )
    model.eval()

    # 本 rank 的数据
    shard_msgs = [dataset[i]["messages"] for i in shard_indices]
    shard_gold_raw = [dataset[i].get("output", "") for i in shard_indices]
    shard_gold = [normalize_choice(x) for x in shard_gold_raw]

    # 推理
    shard_preds_raw = []
    for k in tqdm(range(0, len(shard_msgs), batch_size), desc=f"rank{rank} gen"):
        batch = shard_msgs[k : k + batch_size]
        batch_resps = generate_response_batch(
            model=model,
            tokenizer=tokenizer,
            messages_list=batch,
            max_new_tokens=max_new_tokens,
            batch_size=len(batch),
            temperature=temperature,
            top_p=top_p,
            max_input_len=max_input_len,
            device=device,
        )
        shard_preds_raw.extend(batch_resps)

    shard_preds = [normalize_choice(x) for x in shard_preds_raw]

    # 统计与明细（本 rank）
    shard_detailed = []
    shard_gold_counter = Counter()
    shard_pred_counter = Counter()
    shard_confusion = defaultdict(lambda: Counter())
    shard_correct = 0
    shard_invalid = 0

    for local_idx, (g, pr) in enumerate(zip(shard_gold, shard_preds)):
        global_idx = shard_indices[local_idx]
        shard_gold_counter[g] += 1
        if pr is None:
            shard_invalid += 1
        else:
            shard_pred_counter[pr] += 1
            shard_confusion[g][pr] += 1
            if (g is not None) and (pr == g):
                shard_correct += 1
        shard_detailed.append(
            {
                "index": global_idx,
                "gold_raw": shard_gold_raw[local_idx],
                "gold": g,
                "prediction_raw": shard_preds_raw[local_idx],
                "prediction": pr,
                "is_correct": (g is not None and pr == g),
            }
        )

    # 写本 rank 的明细到磁盘，rank0 稍后汇总
    model_tag = os.path.basename(model_dir.rstrip("/")) or "model"
    model_save_dir = os.path.join(save_dir, model_tag)
    os.makedirs(model_save_dir, exist_ok=True)
    part_path = os.path.join(model_save_dir, f"part_rank{rank}.json")
    with open(part_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "rank": rank,
                "indices": shard_indices,
                "detailed": shard_detailed,
                "gold_counter": dict(shard_gold_counter),
                "pred_counter": dict(shard_pred_counter),
                "confusion": {k: dict(v) for k, v in shard_confusion.items()},
                "correct": shard_correct,
                "invalid": shard_invalid,
                "total": len(shard_indices),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 同步
    barrier()

    if not is_main:
        return  # 只有 rank0 做合并

    # rank0 汇总
    parts = []
    for r in range(world_size):
        p = os.path.join(model_save_dir, f"part_rank{r}.json")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                parts.append(json.load(f))

    detailed_all = []
    gold_counter = Counter()
    pred_counter = Counter()
    confusion = defaultdict(lambda: Counter())
    correct = 0
    invalid = 0
    seen = set()

    for part in parts:
        correct += part["correct"]
        invalid += part["invalid"]
        for k, v in part["gold_counter"].items():
            gold_counter[k] += v
        for k, v in part["pred_counter"].items():
            pred_counter[k] += v
        for g, d in part["confusion"].items():
            for pr, cnt in d.items():
                confusion[g][pr] += cnt
        # 保持全局索引唯一
        for row in part["detailed"]:
            if row["index"] in seen:
                continue
            seen.add(row["index"])
            detailed_all.append(row)

    total = len(load_test_dataset.__self__) if hasattr(load_test_dataset, "__self__") else None  # 无法直接获全局N
    # 直接用 parts 的 total 之和
    total = sum(p["total"] for p in parts)

    accuracy = (correct / total) if total > 0 else 0.0
    per_class = {}
    for c in list("ABCDEFG"):
        g = gold_counter.get(c, 0)
        per_class[c] = {
            "gold": g,
            "correct": confusion[c][c],
            "accuracy": (confusion[c][c] / g) if g > 0 else None,
        }

    summary = {
        "model_dir": model_dir,
        "total": total,
        "valid_predictions": total - invalid,
        "invalid_predictions": invalid,
        "correct": correct,
        "accuracy": accuracy,
        "gold_distribution": dict(gold_counter),
        "pred_distribution": dict(pred_counter),
        "per_class": per_class,
        "confusion": {g: dict(confusion[g]) for g in confusion},
        "world_size": world_size,
    }

    # 排序明细（按原始 index）
    detailed_all.sort(key=lambda x: x["index"])

    # 保存最终结果
    final_path = os.path.join(model_save_dir, f"choice_eval_{model_tag}.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": detailed_all}, f, ensure_ascii=False, indent=2)

    # 额外写一个汇总表
    agg_path = os.path.join(save_dir, "choice_eval_results_models.jsonl")
    with open(agg_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"model_dir": model_dir, "accuracy": accuracy, "correct": correct, "total": total, "invalid": invalid},
                ensure_ascii=False,
            )
            + "\n"
        )

    print(f"[rank0] {model_tag} accuracy={accuracy:.4f} ({correct}/{total}, invalid={invalid})")
    print(f"[rank0] saved: {final_path}")


def main():
    rank, world_size, local_rank, is_main = init_dist()

    args = parse_args()

    if args.default_globs and not args.model_glob and not args.model_dir and not args.models_file:
        args.model_glob = [
            "/project/hdtaccuracy/trains/converted_hf/qwen3_0_6b_choice_v3/checkpoint-*",
            "/project/hdtaccuracy/trains/converted_hf/qwen3_1_7b_choice_v3/checkpoint-*",
            "/project/hdtaccuracy/trains/converted_hf/qwen3_4b_choice_v3/checkpoint-*",
        ]

    model_paths = list_model_dirs(args.model_dir, args.model_glob, args.models_file, args.max_models)
    if is_main:
        if not model_paths:
            raise SystemExit(
                "No valid HF model dirs. Use --model_dir / --model_glob / --models_file / --default_globs."
            )
        print(f"Found {len(model_paths)} model dirs:")
        for p in model_paths:
            print("  -", p)
        os.makedirs(args.save_dir, exist_ok=True)
        # 清空旧的汇总文件（可选）
        agg_path = os.path.join(args.save_dir, "choice_eval_results_models.jsonl")
        if os.path.exists(agg_path):
            os.remove(agg_path)
    barrier()

    # 数据集全体在每个 rank 本地加载（只是一份 Python list，开销不大）
    dataset = load_test_dataset(args.dataset)

    # 依次评测各模型（每个模型的计算在多卡并行）
    for mdir in model_paths:
        evaluate_one_model_dist(
            model_dir=mdir,
            dataset=dataset,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            sample=args.sample,
            dtype=args.dtype,
            max_input_len=args.max_input_len,
            save_dir=args.save_dir,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            is_main=is_main,
        )
        barrier()  # 保证所有 rank 结束后再进入下一个模型

    if is_main:
        print("All done.")

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    # 若使用 torchrun，会自动初始化进程组；这里不做强制 init，兼容单进程
    main()

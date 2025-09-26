import os
import json
import argparse
from collections import defaultdict


def load_tags(dataset_path, tag_field="test_tag"):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tags = [item.get(tag_field) for item in data]
    return tags


def safe_div(a, b):
    return a / b if b else 0.0


def calc_per_tag(detailed_results, tags):
    per_tag = defaultdict(lambda: {"total": 0, "correct": 0})
    n_dataset = len(tags)
    for i, rec in enumerate(detailed_results):
        idx = rec.get("index", i)
        if idx >= n_dataset:
            continue
        tag = tags[idx]
        if tag is None:
            continue
        per_tag[tag]["total"] += 1
        if rec.get("is_correct"):
            per_tag[tag]["correct"] += 1
    out = {}
    for tg, st in per_tag.items():
        out[tg] = {
            "total": st["total"],
            "correct": st["correct"],
            "accuracy": safe_div(st["correct"], st["total"]),
        }
    return out


def process_file(path, tags):
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        return {"error": f"load_failed: {e}"}
    detailed = obj.get("results") or []
    per_tag = calc_per_tag(detailed, tags)
    # 也附带整体（如 summary 中没有 per_tag）
    summary = obj.get("summary", {})
    overall_accuracy = summary.get("accuracy")
    if overall_accuracy is None:
        # 重新算一次
        total = len(detailed)
        correct = sum(1 for r in detailed if r.get("is_correct"))
        overall_accuracy = safe_div(correct, total)
    return {
        "lora_path": obj.get("lora_path"),
        "overall_accuracy": overall_accuracy,
        "per_tag": per_tag,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="包含 test_tag 的原始数据集 (例如 v7_hard_test.json)")
    ap.add_argument("--results_dir", required=True, help="choice_eval_*.json 所在目录")
    ap.add_argument("--out", default="per_tag_accuracy_summary.json", help="输出汇总 JSON 文件名")
    ap.add_argument("--tag_field", default="test_tag", help="数据集中 tag 字段名 (默认 test_tag)")
    ap.add_argument("--pattern_prefix", default="choice_eval_", help="结果文件前缀过滤")
    ap.add_argument("--pattern_suffix", default=".json", help="结果文件后缀过滤")
    ap.add_argument("--skip_files", nargs="*", default=["choice_eval_results.json"], help="需要跳过的文件名")
    args = ap.parse_args()

    tags = load_tags(args.dataset, args.tag_field)

    summaries = {}
    for fname in sorted(os.listdir(args.results_dir)):
        if not fname.startswith(args.pattern_prefix) or not fname.endswith(args.pattern_suffix):
            continue
        if fname in args.skip_files:
            continue
        path = os.path.join(args.results_dir, fname)
        res = process_file(path, tags)
        summaries[fname] = res

    out_path = os.path.join(args.results_dir, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    # 控制台打印简表
    print(f"Saved per-tag accuracies to: {out_path}")
    for fname, info in summaries.items():
        print(f"\nFile: {fname}")
        if "error" in info:
            print("  ERROR:", info["error"])
            continue
        print(f"  Overall: {info['overall_accuracy']:.4f}")
        for tg, st in info["per_tag"].items():
            print(f"    {tg:<10} {st['correct']}/{st['total']} = {st['accuracy']:.4f}")


if __name__ == "__main__":
    main()

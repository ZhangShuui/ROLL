#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys


def main_json():
    if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} input.json output.json")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]

    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # 期望为 list[dict]，每个包含: qid, messages([system,user]), output

    out = []
    for rec in data:
        msgs = rec.get("messages", [])
        system = None
        user = None
        for m in msgs:
            role = m.get("role")
            if role == "system" and system is None:
                system = m.get("content", "")
            elif role == "user" and user is None:
                user = m.get("content", "")

        gpt = rec.get("output", "")
        if not user or not gpt:
            continue  # 跳过不完整样本

        item = {
            "qid": rec.get("qid"),
            "conversations": [
                {"role": "system", "content": system.strip()},
                {"role": "user", "content": user.strip()},
                {"role": "assistant", "content": gpt.strip()},
            ],
        }
        # if system and system.strip():
        #     item["system"] = system.strip()

        out.append(item)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(out)} samples -> {out_path}")


def main_jsonl():
    if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} input.jsonl output.json")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]

    out = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)  # 期望为 dict，包含: qid, messages([system,user]), output

            msgs = rec.get("messages", [])
            system = None
            user = None
            for m in msgs:
                role = m.get("role")
                if role == "system" and system is None:
                    system = m.get("content", "")
                elif role == "user" and user is None:
                    user = m.get("content", "")

            gpt = rec.get("output", "")
            if not user or not gpt:
                continue  # 跳过不完整样本

            item = {
                "qid": rec.get("qid"),
                "conversations": [
                    {"role": "system", "content": system.strip()},
                    {"role": "user", "content": user.strip()},
                    {"role": "assistant", "content": gpt.strip()},
                ],
            }
            # if system and system.strip():
            #     item["system"] = system.strip()

            out.append(item)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Converted {len(out)} samples -> {out_path}")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python3 {sys.argv[0]} input.jsonl output.json")
        sys.exit(1)

    in_path = sys.argv[1]
    if in_path.endswith(".jsonl"):
        main_jsonl()
    elif in_path.endswith(".json"):
        main_json()
    else:
        print("Input file must be .json or .jsonl")
        sys.exit(1)


if __name__ == "__main__":
    main()

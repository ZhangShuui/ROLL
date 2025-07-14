import json
from pathlib import Path


def load_profiles(profile_path: str) -> dict:
    """profile.json → {user_id: profile_text}"""
    with open(profile_path, "r", encoding="utf-8") as f:
        return json.load(f)


def iterate_messages(record: dict):
    """
    根据常见字段名把一条对话里的 message 列表取出来。
    你可以按需要再补别名。
    """
    return record["conversations"]


def build_dataset(roleplay_path: str, profile_path: str, output_path: str):
    profiles = load_profiles(profile_path)
    new_records = []

    with open(roleplay_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = iterate_messages(record)

            # 取 user_id；按需补充或修改字段名
            user_id = (
                record.get("user_id")
                or record.get("uid")
                or record.get("profile_id")
                or record.get("user")
            )

            profile_text = profiles.get(str(user_id), "")  # 若找不到可留空

            # 遍历 message，找到第一条 role=user 作为输出
            for msg_idx, msg in enumerate(messages):
                if msg_idx == 0:
                    continue
                if msg.get("role") == "user":
                    history_msgs = messages[:msg_idx]
                    history_str = "\n".join(
                        f"{m['role']}: {m['content']}" for m in history_msgs
                    )
                    if msg.get("content").endswith("?"):
                        continue  # 跳过问题
                    new_records.append(
                        {
                            "qid": record.get("qid")
                            or record.get("id")
                            or f"r{line_idx}_{msg_idx}",
                            "prompt": (
                                "Now, you are required to simulate the person with profile below:\n"
                                f"{profile_text}\n\n"
                                "Your conversation history are:\n"
                                f"{history_str}\n"
                                "Your output should align with the profile of the person and the conversation history.\n"
                                "Now, your output:"
                            ),
                            "output": msg["content"],
                        }
                    )

    # 写出新数据集
    with open(output_path, "w", encoding="utf-8") as out_f:
        for rec in new_records:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    build_dataset("roleplay_dataset_en.jsonl", "profile.json", "dialogue_dataset.jsonl")
    print("✅ 生成完成：dialogue_dataset.jsonl")

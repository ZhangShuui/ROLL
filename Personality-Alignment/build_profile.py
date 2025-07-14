import json
from pathlib import Path

# 1) Codebook 中“人格滑条”含义映射
PERSONALITY_KEY_MEANINGS = {
    "values": "reflects my values or cultural perspectives",
    "creativity": "produces responses that are creative and inspiring",
    "fluency": "produces responses that are well-written and coherent",
    "factuality": "produces factual and informative responses",
    "diversity": "summarises multiple viewpoints or different worldviews",
    "safety": "produces responses that are safe and do not risk harm",
    "personalisation": "learns from our conversations and feels personalised to me",
    "helpfulness": "produces responses that are helpful and relevant to my requests",
}


def build_profile_text(record: dict) -> str:
    """把 survey 里的 profile dict 拼成完整可读文本"""
    parts = []

    # a) 受访者自述
    if record.get("self_description"):
        parts.append(record["self_description"].strip())

    # b) 对 LLM 的系统指令
    if record.get("system_string"):
        parts.append(f"Preferred system behaviour: {record['system_string'].strip()}")

    # c) Stated preferences（人格相关滑条）
    stated = record.get("stated_prefs", {})
    if stated:
        pref_lines = []
        for key, meaning in PERSONALITY_KEY_MEANINGS.items():
            if key in stated:
                score = stated[key]
                pref_lines.append(
                    f"{key.capitalize()} ({meaning}): {score}/100 importance"
                )
        if pref_lines:
            parts.append("Stated preferences:\n" + "\n".join(pref_lines))

    return "\n\n".join(parts)


def main():
    output = {}
    survey_path = Path("data/survey.jsonl")
    with survey_path.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            user_id = record["user_id"]  # e.g. "user0"
            profile_text = build_profile_text(record)
            output[user_id] = profile_text

    with open("profile.json", "w", encoding="utf-8") as fout:
        json.dump(output, fout, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

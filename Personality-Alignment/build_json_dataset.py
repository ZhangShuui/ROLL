import pandas as pd
from src.utils.data_loader import load_data
import json


def main():
    # Load all data splits
    data = load_data(survey=True, conversations=True, metadata=True)
    survey_df = data["survey"]
    conversations_df = data["conversations"]
    metadata_df = data["metadata"]

    # Filter metadata for English language
    english_convo_ids = set(
        metadata_df[metadata_df["en_flag"] == True]["conversation_id"]
    )
    # Filter conversations for English only
    english_conversations = conversations_df[
        conversations_df["conversation_id"].isin(english_convo_ids)
    ]

    # Build user_id to profile mapping
    user_profiles = {}
    for _, row in survey_df.iterrows():
        user_profiles[row["user_id"]] = row.to_dict()

    # Build output dataset
    output = []
    for _, convo in english_conversations.iterrows():
        user_id = convo["user_id"]
        profile = user_profiles.get(user_id, {})
        conversations = convo["conversation_history"]
        conversation_type = convo.get("conversation_type", "")
        output.append(
            {
                "user_id": user_id,
                "conversations": conversations,
                "conversation_type": conversation_type,
            }
        )

    # Write to file
    with open("roleplay_dataset_en.json", "w") as f:
        for item in output:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

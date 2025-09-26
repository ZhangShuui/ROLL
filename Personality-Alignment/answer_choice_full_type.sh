python answer_choice_four_type.py \
    --question_file /project/hdtaccuracy/Personality-Alignment/choice_ver/four_choice_questions.json \
    --prompt_file /project/hdtaccuracy/Personality-Alignment/choice_ver/v9/dialogue_dataset_all_v8_summarized_cleaned.jsonl \
    --model_path /project/hdtaccuracy/models/base/Qwen3-8B \
    --batch_size 32 \
    --output_file /project/hdtaccuracy/Personality-Alignment/choice_ver/four_results/results.json
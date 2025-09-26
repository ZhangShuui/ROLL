# python build_choice_training_data.py \
#     --questions /project/hdtaccuracy/Personality-Alignment/choice_ver/four_choices_question_v7/all_questions.json \
#     --prompts /project/hdtaccuracy/Personality-Alignment/choice_ver/raw_choice_data_v6.jsonl \
#     --out /project/hdtaccuracy/Personality-Alignment/choice_ver/four_choices_question_v7/v7.json \
#     --split_mode user \
#     --test_ratio 0.2 \
#     --skip_missing_prompt
# python build_choice_training_data.py \
#   --questions /project/hdtaccuracy/Personality-Alignment/choice_ver/v9/bank/multi_choice_questions_ram.json \
#   --prompts /project/hdtaccuracy/Personality-Alignment/choice_ver/v9/dialogue_dataset_all_v8_summarized_cleaned.jsonl \
#   --out /project/hdtaccuracy/Personality-Alignment/choice_ver/v9/train_data_ram/training_data_nothink.jsonl \
#   --prompt_type no_think \
#   --skip_invalid_format \
#   --split_mode user_partial \
#   --user_subset_ratio 0.3 \
#   --test_ratio 0.3 \
#   --make_val \
#   --val_ratio 0.1 \
#   --show_stats \
#   --skip_missing_prompt
  # --show_token_stats \
  # --tokenizer_model "/project/hdtaccuracy/models/base/Qwen3-4B" \
# python build_choice_training_data.py \
#   --questions /project/hdtaccuracy/Personality-Alignment/choice_ver/v9/choice_enhanced/merged_choice_questions.json \
#   --prompts /project/hdtaccuracy/Personality-Alignment/choice_ver/v9/dialogue_dataset_all_v8_summarized_cleaned.jsonl \
#   --out /project/hdtaccuracy/Personality-Alignment/choice_ver/v9/train_data_enhanced/training_data_nothink.jsonl \
#   --prompt_type no_think \
#   --skip_invalid_format \
#   --split_mode user_partial \
#   --user_subset_ratio 0.3 \
#   --test_ratio 0.3 \
#   --make_val \
#   --val_ratio 0.1 \
#   --show_stats \
#   --skip_missing_prompt
python build_choice_training_data.py \
  --questions /project/hdtaccuracy/Personality-Alignment/choice_ver/v10/train_data_enhanced/merged_choice_questions.json \
  --prompts /project/hdtaccuracy/Personality-Alignment/choice_ver/v10/dialogue_dataset_all_v9_summarized_cleaned.jsonl \
  --out /project/hdtaccuracy/Personality-Alignment/choice_ver/v10/train_data_enhanced/training_data_nothink.jsonl \
  --prompt_type no_think \
  --skip_invalid_format \
  --split_mode user_partial \
  --user_subset_ratio 0.3 \
  --test_ratio 0.3 \
  --make_val \
  --val_ratio 0.1 \
  --show_stats \
  --skip_missing_prompt \
  --show_token_stats \
  --tokenizer_model "/project/hdtaccuracy/models/base/Qwen3-4B"
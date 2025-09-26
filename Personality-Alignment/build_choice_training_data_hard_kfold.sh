python build_choice_training_data.py \
    --questions /project/hdtaccuracy/Personality-Alignment/choice_ver/four_choices_question_v7_hard/all_questions.json \
    --prompts /project/hdtaccuracy/Personality-Alignment/choice_ver/raw_choice_data_v7_hard.jsonl \
    --out /project/hdtaccuracy/Personality-Alignment/choice_ver/four_choices_question_v7_hard_kfold/v7_hard.json \
    --split_mode kfold_user \
    --k_folds 5 \
    --skip_missing_prompt
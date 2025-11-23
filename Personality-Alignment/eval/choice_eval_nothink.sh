python choice_eval.py \
    --base_model /project/hdtaccuracy/models/base/Qwen3-4B \
    --lora_dir /project/hdtaccuracy/trains/choice-sft/qwen3-4b-lora-sft-v3-filtered\
    --dataset /project/hdtaccuracy/Personality-Alignment/choice_ver/v11/train_data_nothink/training_data_nothink_test.jsonl \
    --batch_size 32 \
    --save_dir /home/szhangfa/ROLL/Personality-Alignment/eval/qwen3_4b_sft_v3_filtered_nothink
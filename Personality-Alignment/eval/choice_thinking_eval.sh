# python choice_eval_thinking.py \
#   --model_glob "/project/hdtaccuracy/trains/converted_hf/qwen3_0_6b_choice_v3/checkpoint-*" \
#   --dataset /project/hdtaccuracy/Personality-Alignment/choice_ver/v10/train_data_with_thinking/training_data_with_think_test.jsonl \
#   --save_dir ./qwen3_0_6b_thinking_grpo \
#   --batch_size 8 \
#   --max_new_tokens 1536 \
#   --max_input_len 1024 \
#   --dtype bf16

torchrun --nproc_per_node=8 choice_eval_thinking.py \
  --model_glob "/project/hdtaccuracy/trains/converted_hf/qwen3_0_6b_choice_v3/checkpoint-*" \
  --dataset /project/hdtaccuracy/Personality-Alignment/choice_ver/v10/train_data_with_thinking/training_data_with_think_test.jsonl \
  --save_dir ./qwen3_0_6b_thinking_grpo \
  --batch_size 4 \
  --max_new_tokens 1536 \
  --max_input_len 1024 \
  --dtype bf16

torchrun --nproc_per_node=8 choice_eval_thinking.py \
  --model_glob "/project/hdtaccuracy/trains/converted_hf/qwen3_1_7b_choice_v3/checkpoint-*" \
  --dataset /project/hdtaccuracy/Personality-Alignment/choice_ver/v10/train_data_with_thinking/training_data_with_think_test.jsonl \
  --save_dir ./qwen3_1_7b_thinking_grpo \
  --batch_size 4 \
  --max_new_tokens 1536 \
  --max_input_len 1024 \
  --dtype bf16

torchrun --nproc_per_node=8 choice_eval_thinking.py \
  --model_glob "/project/hdtaccuracy/trains/converted_hf/qwen3_4b_choice_v3/checkpoint-*" \
  --dataset /project/hdtaccuracy/Personality-Alignment/choice_ver/v10/train_data_with_thinking/training_data_with_think_test.jsonl \
  --save_dir ./qwen3_4b_thinking_grpo \
  --batch_size 4 \
  --max_new_tokens 1536 \
  --max_input_len 1024 \
  --dtype bf16

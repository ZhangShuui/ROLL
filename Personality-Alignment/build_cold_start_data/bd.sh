#!/bin/bash

# Example usage script with batch processing
echo "Starting CoT RL data generation with batch processing..."

# Run the generation script
# python build_data.py \
#     --input /project/hdtaccuracy/Personality-Alignment/choice_ver/v10/train_data_with_thinking/llama/train.json \
#     --output /project/hdtaccuracy/Personality-Alignment/choice_ver/v10/train_data_with_thinking/cold_start_llama/cot_rl.json \
#     --model-name "/project/hdtaccuracy/models/base/Qwen3-80B-A3B" \
#     --batch-size 4 \
#     --variations 3 \
#     --max-samples 20

python build_data.py \
  --input /project/hdtaccuracy/Personality-Alignment/choice_ver/v10/train_data_with_thinking/llama/train.json \
  --output /project/hdtaccuracy/Personality-Alignment/choice_ver/v10/train_data_with_thinking/cold_start_llama/cot_rl.json \
  --api-key "sk-K6tq07IP2UM744DR1YkZSqZ3MGpab7bJ6IImmBoUWxoT2Jpa" \
  --base-url "https://api.apiplus.org/v1" \
  --model-name "gpt-5-2025-08-07" \
  --max-workers 8 \
  --batch-size 10 \
  --max-samples 1000 \
  --variations 3
echo "Generation completed!"
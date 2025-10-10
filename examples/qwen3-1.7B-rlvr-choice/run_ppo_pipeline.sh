#!/bin/bash
set +x

CONFIG_PATH="qwen3-1.7B-rlvr-choice"
python -m examples.start_rlvr_pipeline --config_path $CONFIG_PATH  --config_name ppo_config_mega

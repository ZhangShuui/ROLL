#!/bin/bash
set +x

CONFIG_PATH="qwen3-4B-rlvr-choice"
python -m examples.start_rlvr_pipeline --config_path $CONFIG_PATH  --config_name gspo_config_mega

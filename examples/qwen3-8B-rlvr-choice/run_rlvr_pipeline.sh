#!/bin/bash
set +x

CONFIG_PATH="qwen3-8B-rlvr-choice"
python -m examples.start_rlvr_pipeline --config_path $CONFIG_PATH  --config_name rlvr_config_ds

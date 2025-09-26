#!/bin/bash
#SBATCH --job-name=gpt_choice_data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=dgx-34
#SBATCH --time=10:00:00
#SBATCH --account=hdtaccuracy
#SBATCH --partition=cpu

cd /home/szhangfa/ROLL/Personality-Alignment
# Set script directory and paths
SCRIPT_DIR="/home/szhangfa/ROLL/Personality-Alignment"
DATA_PATH="/project/hdtaccuracy/Personality-Alignment/split_data_v6_filtered/filtered_dataset.jsonl"
SAVE_PATH="/project/hdtaccuracy/Personality-Alignment/choice_ver/raw_choice_data_v6_gpt.jsonl"

# GPT API configuration
GPT_API_KEY="sk-K6tq07IP2UM744DR1YkZSqZ3MGpab7bJ6IImmBoUWxoT2Jpa"  # Replace with your actual API key
GPT_BASE_URL="https://api.apiplus.org/v1"  # Optional: set if using custom endpoint
GPT_MODEL="gpt-5-2025-08-07"  # or "gpt-4"

# Processing configuration
BATCH_SIZE=16  # Smaller batch size for API calls
DATA_LIMIT=""  # Set to limit data for testing, empty for all data
MAX_RETRIES=5
RETRY_DELAY=2.0

# Check if API key is set
if [ "$GPT_API_KEY" = "your_api_key_here" ]; then
    echo "Error: Please set your GPT API key in the script"
    echo "Edit the GPT_API_KEY variable in this script"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$SAVE_PATH")
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="${OUTPUT_DIR}/gpt_generation_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "GPT Choice Data Generation Script"
echo "=========================================="
echo "Data Path: $DATA_PATH"
echo "Save Path: $SAVE_PATH"
echo "GPT Model: $GPT_MODEL"
echo "Batch Size: $BATCH_SIZE"
echo "Max Retries: $MAX_RETRIES"
echo "Log File: $LOG_FILE"
echo "=========================================="

# Build command
CMD="/home/szhangfa/.conda/envs/local/bin/python $SCRIPT_DIR/build_choice_data.py \
    --model_type gpt \
    --gpt_model $GPT_MODEL \
    --gpt_api_key $GPT_API_KEY \
    --batch_size $BATCH_SIZE \
    --max_retries $MAX_RETRIES \
    --retry_delay $RETRY_DELAY \
    --data_path $DATA_PATH \
    --save_path $SAVE_PATH"

# Add optional parameters
if [ -n "$GPT_BASE_URL" ]; then
    CMD="$CMD --gpt_base_url $GPT_BASE_URL"
fi

if [ -n "$DATA_LIMIT" ]; then
    CMD="$CMD --data_limit $DATA_LIMIT"
fi

# Execute command with logging
echo "Starting GPT data generation..."
echo "Command: $CMD"
echo ""

# Run the command and capture both stdout and stderr
{
    echo "=== GPT Choice Data Generation Started at $(date) ==="
    echo "Command: $CMD"
    echo ""
    
    eval $CMD
    
    echo ""
    echo "=== GPT Choice Data Generation Completed at $(date) ==="
} 2>&1 | tee "$LOG_FILE"

# Check if the process was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "SUCCESS: GPT data generation completed!"
    echo "Output file: $SAVE_PATH"
    echo "Log file: $LOG_FILE"
    echo "=========================================="
    
    # Display file size and line count if output exists
    if [ -f "$SAVE_PATH" ]; then
        LINES=$(wc -l < "$SAVE_PATH")
        SIZE=$(du -h "$SAVE_PATH" | cut -f1)
        echo "Generated $LINES data entries"
        echo "Output file size: $SIZE"
    fi
else
    echo ""
    echo "=========================================="
    echo "ERROR: GPT data generation failed!"
    echo "Check the log file for details: $LOG_FILE"
    echo "=========================================="
    exit 1
fi

# Optional: Test mode
if [ "$1" = "--test" ]; then
    echo ""
    echo "Running in test mode..."
    TEST_CMD="/home/szhangfa/.conda/envs/local/bin/python $SCRIPT_DIR/build_choice_data.py \
        --model_type gpt \
        --gpt_model $GPT_MODEL \
        --gpt_api_key $GPT_API_KEY \
        --batch_size 2 \
        --data_path $DATA_PATH \
        --data_limit 10 \
        --test_mode"
    
    echo "Test command: $TEST_CMD"
    eval $TEST_CMD
fi
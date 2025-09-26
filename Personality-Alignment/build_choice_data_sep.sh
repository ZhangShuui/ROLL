#!/bin/bash
#SBATCH --job-name=pchoice_array
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --account=hdtaccuracy
#SBATCH --array=0-8%8  # 10 个任务，每次最多 10 个并行
#SBATCH --output=build_choice_data_%A_%a.out
#SBATCH --error=build_choice_data_%A_%a.err

# 配置参数
TOTAL_DATA=19161
CHUNK_SIZE=2500

# 计算当前任务的数据范围
START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END_IDX=$(((SLURM_ARRAY_TASK_ID + 1) * CHUNK_SIZE))

# 确保不超过总数据量
if [ $END_IDX -gt $TOTAL_DATA ]; then
    END_IDX=$TOTAL_DATA
fi

echo "任务 ID: $SLURM_ARRAY_TASK_ID"
echo "处理范围: $START_IDX - $END_IDX"
echo "开始时间: $(date)"

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=$(( 20000 + RANDOM % 10000 ))

srun --export=ALL \
    --container-image=/home/szhangfa/containers/roll.img \
    --container-mounts=/home/szhangfa:/home/szhangfa \
    --container-workdir=/home/szhangfa/ROLL/Personality-Alignment \
    --container-writable \
    bash -c "
cd /home/szhangfa/ROLL/Personality-Alignment
python build_choice_data.py \
    --start_index $START_IDX \
    --end_index $END_IDX \
    --save_path /project/hdtaccuracy/Personality-Alignment/choice_ver/v8/raw_choice_data_v8_chunk_$(printf '%03d' $SLURM_ARRAY_TASK_ID).jsonl \
    --data_path /project/hdtaccuracy/Personality-Alignment/v8/dialogue_dataset_all_v8_summarized_cleaned.jsonl
"

echo "结束时间: $(date)"
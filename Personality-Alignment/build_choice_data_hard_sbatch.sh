#!/bin/bash
#SBATCH --job-name=pchoice_hard_array
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --account=hdtaccuracy
#SBATCH --array=0-7%6  # 8个任务，每次最多6个并行
#SBATCH --output=logs/build_choice_hard_%A_%a.out
#SBATCH --error=logs/build_choice_hard_%A_%a.err

# 创建日志目录
mkdir -p logs

# 配置参数
TOTAL_DATA=19161
CHUNK_SIZE=2500  # 每个任务处理的数据量

# 计算当前任务的数据范围
START_IDX=$((SLURM_ARRAY_TASK_ID * CHUNK_SIZE))
END_IDX=$(((SLURM_ARRAY_TASK_ID + 1) * CHUNK_SIZE))

# 确保不超过总数据量
if [ $END_IDX -gt $TOTAL_DATA ]; then
    END_IDX=$TOTAL_DATA
fi

echo "========================================"
echo "Hard级别选择题数据生成 - 任务 $SLURM_ARRAY_TASK_ID"
echo "========================================"
echo "任务 ID: $SLURM_ARRAY_TASK_ID"
echo "处理范围: $START_IDX - $END_IDX"
echo "数据量: $((END_IDX - START_IDX))"
echo "节点: $SLURM_NODELIST"
echo "开始时间: $(date)"
echo "========================================"

# 设置网络环境变量
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=$(( 20000 + RANDOM % 10000 ))

# 运行容器化任务
srun --export=ALL \
    --container-image=/home/szhangfa/containers/roll.img \
    --container-mounts=/home/szhangfa:/home/szhangfa \
    --container-workdir=/home/szhangfa/ROLL/Personality-Alignment \
    --container-writable \
    bash -c "
# 进入工作目录
cd /home/szhangfa/ROLL/Personality-Alignment

# 运行 hard 级别选择题生成
python build_choice_data_hard.py \
    --model_type local \
    --model_path /project/hdtaccuracy/models/base/Qwen3-8B \
    --batch_size 16 \
    --start_index $START_IDX \
    --end_index $END_IDX \
    --data_path /project/hdtaccuracy/Personality-Alignment/v8/dialogue_dataset_all_v8_summarized_cleaned.jsonl \
    --save_path /project/hdtaccuracy/Personality-Alignment/choice_ver/v9/raw_choice_data_v9_hard_chunk_$(printf '%03d' $SLURM_ARRAY_TASK_ID).jsonl

echo ''
echo '任务完成！'
echo '输出文件: /project/hdtaccuracy/Personality-Alignment/choice_ver/v9/raw_choice_data_v9_hard_chunk_$(printf '%03d' $SLURM_ARRAY_TASK_ID).jsonl'
"

echo "========================================"
echo "任务 $SLURM_ARRAY_TASK_ID 完成"
echo "结束时间: $(date)"
echo "========================================"
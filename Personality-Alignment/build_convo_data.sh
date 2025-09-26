#!/bin/bash
#SBATCH --job-name=bconvo
#SBATCH --nodes=1
#SBATCH --gpus=2            # 一行就够
#SBATCH --ntasks=1          # 只起 1 个进程
#SBATCH --cpus-per-task=8  # 视节点 CPU 数而定
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --account=hdtaccuracy
#SBATCH --output=build_convo_data.out

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=$(( 20000 + RANDOM % 10000 ))

srun --export=ALL \
    --container-image=/home/szhangfa/containers/roll.img \
    --no-container-mount-home \
    --container-mounts=/home/szhangfa:/home/szhangfa \
    --container-workdir=/home/szhangfa/ROLL/Personality-Alignment \
    --container-writable \
    bash -c "
cd /home/szhangfa/ROLL/Personality-Alignment
python build_convo_data.py
"


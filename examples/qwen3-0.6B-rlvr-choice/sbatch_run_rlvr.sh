#!/bin/bash
#SBATCH --job-name=hdtgrpo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=dgx-34
#SBATCH --time=30:00:00
#SBATCH --account=hdtaccuracy
#SBATCH --partition=normal
##SBATCH --container-writable
##SBATCH --container-image /home/szhangfa/containers/roll.img
##SBATCH --container-save /home/szhangfa/containers/roll.img

# cd /home/szhangfa/LLaMA-Factory
export WANDB_API_KEY=dce12064d30900b2cc538f73e82997de5aafbb96

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=$(( 20000 + RANDOM % 10000 ))

srun --export=WANDB_API_KEY,MASTER_ADDR,MASTER_PORT \
    --container-image=/project/hdtaccuracy/images/roll.img \
    --container-mounts=/project/hdtaccuracy:/project/hdtaccuracy,/home/szhangfa/ROLL/:/home/szhangfa/ROLL \
    --no-container-mount-home \
    --container-env=PYXI_DISABLE_DEFAULT_MOUNTS=1 \
    --container-workdir=/home/szhangfa/ROLL \
    --container-writable \
    bash -c "
set -euo pipefail

# === 容器内启动 GPU 显存监控 ===
GPU_LOG_INTERVAL=\"\${GPU_LOG_INTERVAL:-5}\"
LOG_DIR=\"/home/szhangfa/ROLL/logs/gpu\"
mkdir -p \"\$LOG_DIR\"

HOST=\"\$(hostname)\"
JOBID=\"\${SLURM_JOB_ID:-manual}\"
TS=\"\$(date +%Y%m%d_%H%M%S)\"
GPU_LOG=\"\$LOG_DIR/gpu_\${HOST}_\${JOBID}_\${TS}.csv\"

# 只记录显存使用/总量（每 \${GPU_LOG_INTERVAL}s）
nvidia-smi --query-gpu=timestamp,index,uuid,name,memory.used,memory.total \
  --format=csv -l \"\$GPU_LOG_INTERVAL\" >> \"\$GPU_LOG\" &
GPU_MON_PID=\$!

trap 'kill \$GPU_MON_PID 2>/dev/null || true' EXIT INT TERM
# =================================

cd /home/szhangfa/ROLL
python3 -m pip install -r requirements_torch260_vllm.txt
bash examples/qwen3-0.6B-rlvr-choice/run_rlvr_pipeline.sh
"
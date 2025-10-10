#!/bin/bash
#SBATCH --job-name=hdtgrpo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=dgx-34,dgx-20
#SBATCH --time=30:00:00
#SBATCH --account=hdtaccuracy
#SBATCH --partition=preempt
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

bash examples/qwen3-0.6B-rlvr-choice/run_gspo_pipeline.sh
"
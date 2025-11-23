#!/bin/bash
#SBATCH --job-name=hdtgrpo
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=dgx-34,dgx-20
#SBATCH --time=30:00:00
#SBATCH --account=hdtaccuracy
#SBATCH --partition=normal
##SBATCH --container-writable
##SBATCH --container-image /home/szhangfa/containers/roll.img
##SBATCH --container-save /home/szhangfa/containers/roll.img

srun --export=WANDB_API_KEY,MASTER_ADDR,MASTER_PORT \
    --container-image=/project/hdtaccuracy/images/roll.img \
    --container-mounts=/project/hdtaccuracy:/project/hdtaccuracy,/home/szhangfa/ROLL/:/home/szhangfa/ROLL \
    --no-container-mount-home \
    --container-env=PYXI_DISABLE_DEFAULT_MOUNTS=1 \
    --container-workdir=/home/szhangfa/ROLL \
    --container-writable \
    bash -c "
set -euo pipefail

cd /home/szhangfa/ROLL
# python3 -m pip install -r requirements_torch260_vllm.txt
bash examples/qwen3-1.7B-rlvr-choice/run_gsspo_seq_correct.sh
"
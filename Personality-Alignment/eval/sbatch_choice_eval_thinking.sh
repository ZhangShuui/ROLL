#!/bin/bash
#SBATCH --job-name=hdteval_gspo
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

srun --export=WANDB_API_KEY,MASTER_ADDR,MASTER_PORT \
    --container-image=/project/hdtaccuracy/images/roll.img \
    --container-mounts=/project/hdtaccuracy:/project/hdtaccuracy,/home/szhangfa/ROLL/:/home/szhangfa/ROLL \
    --no-container-mount-home \
    --container-env=PYXI_DISABLE_DEFAULT_MOUNTS=1 \
    --container-workdir=/home/szhangfa/ROLL \
    --container-writable \
    bash -c "
cd /home/szhangfa/ROLL/Personality-Alignment/eval

torchrun --nproc_per_node=8 choice_eval_thinking.py \
  --model_dir /project/hdtaccuracy/trains/converted_hf/grpo/qwen3_0_6b_choice_v4/checkpoint-399 \
  --dataset /project/hdtaccuracy/Personality-Alignment/choice_ver/v10/train_data_with_thinking/training_data_with_think_test.jsonl \
  --save_dir ./qwen3_0_6b_thinking_grpo \
  --batch_size 32 \
  --max_new_tokens 2048 \
  --max_input_len 1024 \
  --dtype bf16
"
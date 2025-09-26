#!/bin/bash
#SBATCH --job-name=hdteval
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=dgx-34
#SBATCH --time=10:00:00
#SBATCH --account=hdtaccuracy
#SBATCH --partition=normal

# cd /home/szhangfa/LLaMA-Factory
export WANDB_API_KEY="dce12064d30900b2cc538f73e82997de5aafbb96"

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n1)
export MASTER_PORT=$(( 20000 + RANDOM % 10000 ))

srun --export=ALL \
    --container-image=/project/hdtaccuracy/images/llama.img \
    --container-mounts=/home/szhangfa:/home/szhangfa \
    --container-workdir=/home/szhangfa/LLaMA-Factory \
    --container-writable \
    bash -c "
cd /home/szhangfa/ROLL/Personality-Alignment/eval
python choice_eval.py \
    --base_model /project/hdtaccuracy/models/base/Qwen3-4B \
    --lora_dir /project/hdtaccuracy/trains/choice-sft/qwen3-4b-lora-sft-hard-plus\
    --dataset /project/hdtaccuracy/Personality-Alignment/choice_ver/four_choices_question_v7_hard_final/v7_hard_test.json \
    --batch_size 16 \
    --save_dir /home/szhangfa/ROLL/Personality-Alignment/eval/qwen3_4b_outputs_hard_final
"
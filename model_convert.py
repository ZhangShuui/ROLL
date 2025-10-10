import os
import subprocess

# 需要转换的根目录
base_dirs = [
    "/project/hdtaccuracy/trains/grpo/qwen3_0_6b_choice_v3/output/actor_train-1",
    "/project/hdtaccuracy/trains/grpo/qwen3_1_7b_choice_v3/output/actor_train-0",
    "/project/hdtaccuracy/trains/grpo/qwen3_4b_choice_v3/output/actor_train-0",
]

# 输出目录根路径
output_root = "/project/hdtaccuracy/trains/converted_hf"

# 确保输出目录存在
os.makedirs(output_root, exist_ok=True)

for base_dir in base_dirs:
    model_name = os.path.basename(os.path.dirname(base_dir))  # 例如 qwen3_4b_choice_v3
    for ckpt in sorted(os.listdir(base_dir)):
        ckpt_path = os.path.join(base_dir, ckpt)
        if os.path.isdir(ckpt_path) and ckpt.startswith("checkpoint-"):
            output_dir = os.path.join(output_root, model_name, ckpt)
            os.makedirs(output_dir, exist_ok=True)

            cmd = [
                "python",
                "mcore_adapter/tools/convert.py",
                "--checkpoint_path",
                ckpt_path,
                "--output_path",
                output_dir,
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

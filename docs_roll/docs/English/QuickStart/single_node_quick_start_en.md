# Quickstart: Singel Node Deployment Guide

## Environment Preparation
1. Purchase a machine and install GPU drivers simultaneously
2. Connect remotely to the GPU instance and access the machine terminal
3. Install Docker environment and NVIDIA Container Toolkit
```shell
curl -fsSL https://your-domain.com/install_docker_aliyun.sh | sudo bash  
https://github.com/alibaba/ROLL/blob/main/examples/quick_start/install_docker_nvidia_container_toolkit.sh  
```

## Environment Configuration
```shell
# 1. Pull Docker image
sudo docker pull <image_address>

# Image Addresses (choose based on your needs)
# torch2.6.0 + SGlang0.4.6: roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-sglang046
# torch2.6.0 + vLLM0.8.4: roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-vllm084
# torch2.5.1 + SGlang0.4.3: roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch251-sglang043
# torch2.5.1 + vLLM0.7.3: roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch251-vllm073

# 2. Start a Docker container with GPU support, expose the port, and keep the container running
sudo docker images
sudo docker run -dit \
  --gpus all \
  --network=host \
  --ipc=host \
  --shm-size=2gb \
  <image_id> \
  /bin/bash

# 3. Enter the Docker container (execute this command every time you reconnect)
sudo docker ps
sudo docker exec -it <container_id> /bin/bash

# 4. Verify GPU visibility
nvidia-smi

# 5. Download the code

# Install git（this image is ubuntu-based）and clone the repo
apt update && apt install git -y
git clone https://github.com/alibaba/ROLL.git

# If Github is not accessible, download the zip file directly and unzip
wget https://github.com/alibaba/ROLL/archive/refs/heads/main.zip
unzip main.zip

# 5. Install dependencies (select the requirements file corresponding to your chosen image)
cd ROLL-main
pip install -r requirements_torch260_sglang.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## Pipeline Execution
```shell
bash examples/quick_start/run_agentic_pipeline_frozen_lake_single_node_demo.sh  
```

Example Log Screenshots during Pipeline Execution:
![log1](../../../static/img/log_1.png)

![log2](../../../static/img/log_2.png)

![log3](../../../static/img/log_3.png)


## Reference: V100 Single-GPU Memory Configuration Optimization
```yaml
# Reduce the system's expected number of GPUs from 8 to your actual 1 V100
num_gpus_per_node: 1 
# Training processes are now mapped only to GPU 0
actor_train.device_mapping: list(range(0,1))
# Inference processes are now mapped only to GPU 0
actor_infer.device_mapping: list(range(0,1))
# Reference model processes are now mapped only to GPU 0
reference.device_mapping: list(range(0,1))

# Significantly reduce the batch sizes for Rollout and Validation stages to prevent out-of-memory errors on a single GPU
rollout_batch_size: 16
val_batch_size: 16

# V100 has better native support for FP16 than BF16 (unlike A100/H100). Switching to FP16 improves compatibility and stability, while also saving GPU memory.
actor_train.model_args.dtype: fp16
actor_infer.model_args.dtype: fp16
reference.model_args.dtype: fp16

# Switch the large model training framework from DeepSpeed to Megatron-LM. Parameters can be sent in batches, resulting in faster execution.
strategy_name: megatron_train
strategy_config:
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  expert_model_parallel_size: 1
  use_distributed_optimizer: true
  recompute_granularity: full

# In megatron training the global train batch size is equivalent to per_device_train_batch_size * gradient_accumulation_steps * world_size
actor_train.training_args.per_device_train_batch_size: 1
actor_train.training_args.gradient_accumulation_steps: 16  

# Reduce the maximum number of actions per trajectory, making each Rollout trajectory shorter that reduces the length of LLM-generated content.
max_actions_per_traj: 10    

# Reduce the number of parallel training and validation environment groups to accommodate single-GPU resources.
train_env_manager.env_groups: 1
train_env_manager.n_groups: 1
val_env_manager.env_groups: 2
val_env_manager.n_groups: [1, 1]
val_env_manager.tags: [SimpleSokoban, FrozenLake]

# Reduce the total number of training steps for quicker full pipeline runs, useful for rapid debugging.
max_steps: 100
```

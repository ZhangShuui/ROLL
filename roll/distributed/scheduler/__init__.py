"""
跟rpc控制系统有关的
跟集群资源有关的
"""

default_envs = {
    # "RAY_DEBUG": "legacy"
    "TORCHINDUCTOR_COMPILE_THREADS": "2",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "NCCL_CUMEM_ENABLE": "0",   # https://github.com/NVIDIA/nccl/issues/1234
    "NCCL_NVLS_ENABLE": "0",
    # "ACCL_TUNING_LEVEL": "1",
}

#!/usr/bin/env bash
set -euo pipefail

# Optional: ensure running as root (Docker build stage usually runs as root)
if [[ "$(id -u)" -ne 0 ]]; then
  echo "Please run as root (or inside the Docker build stage)." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
export PIP_ROOT_USER_ACTION=ignore
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64

# 1. Upgrade pip toolchain (use mirror)
pip install --upgrade pip setuptools wheel \
  --trusted-host mirrors.aliyun.com \
  --index-url https://mirrors.aliyun.com/pypi/simple/

# 2. Remove preinstalled conflicting libs
pip uninstall -y torch torchvision torchaudio torch-tensorrt || true
pip uninstall -y flash_attn transformer-engine || true
pip uninstall -y cudf dask-cuda cugraph cugraph-service-server cuml raft-dask cugraph-dgl cugraph-pyg dask-cudf || true

# 3. Install specific PyTorch (CUDA 12.4)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

# 4. Reinstall clean OpenCV headless
pip uninstall -y opencv opencv-python opencv-python-headless || true
rm -rf /usr/local/lib/python3.10/dist-packages/cv2/ || true
pip install opencv-python-headless==4.11.0.86 \
  --trusted-host mirrors.aliyun.com \
  --index-url https://mirrors.aliyun.com/pypi/simple/

# 5. Core Python deps
pip install \
  "numpy==1.26.4" "optree>=0.13.0" "spacy==3.7.5" "weasel==0.4.1" \
  transformer-engine[pytorch]==2.2.0 megatron-core==0.11.0 deepspeed==0.16.4 \
  --trusted-host mirrors.aliyun.com \
  --index-url https://mirrors.aliyun.com/pypi/simple/

# 6. FlashAttention (prebuilt wheel)
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 7. vLLM
pip install vllm==0.8.4 \
  --trusted-host mirrors.aliyun.com \
  --index-url https://mirrors.aliyun.com/pypi/simple/

# 8. (Optional) build apex
APEX_URL="git+https://github.com/NVIDIA/apex.git@25.04"
pip uninstall -y apex || true
MAX_JOBS=32 NINJA_FLAGS="-j32" NVCC_APPEND_FLAGS="--threads 32" \
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 32" "${APEX_URL}"

# 9. Switch APT mirror to Aliyun (backup first)
if [[ ! -f /etc/apt/sources.list.bak ]]; then
  cp /etc/apt/sources.list /etc/apt/sources.list.bak
fi
cat >/etc/apt/sources.list <<'EOF'
deb https://mirrors.aliyun.com/ubuntu/ jammy main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ jammy-security main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ jammy-updates main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu/ jammy-backports main restricted universe multiverse
EOF

apt-get update
apt-get install -y zip
apt-get install -y openjdk-21-jdk

echo "Environment setup"
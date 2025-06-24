FROM registry.cn-hangzhou.aliyuncs.com/sais-public/pytorch:2.0.0-py3.9.12-cuda11.8.0-u22.04

ADD . /app

WORKDIR /app

# --- 修复步骤：添加NVIDIA CUDA和cuDNN官方APT源 ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gnupg2 ca-certificates wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update

# --- 安装cuDNN 8.6.x for CUDA 11.x ---
RUN apt-get install -y --no-install-recommends \
    libcudnn8=8.6.0.*-1+cuda11.8 \
    libcudnn8-dev=8.6.0.*-1+cuda11.8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# --- 设置库路径（显式指定路径）---
ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 设置pip国内源
COPY . /app
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install -r /app/requirements.txt
# RUN dos2unix /app/run.sh && chmod +x /app/run.sh

# 定义容器启动时默认执行的命令
CMD ["sh", "/app/run.sh"]


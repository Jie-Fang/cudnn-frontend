FROM nvcr.io/nvidia/pytorch:25.12-py3

# Set working directory
WORKDIR /workspace

# Update libcudnn9-cuda-13
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get -y install cudnn && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Clone cudnn_frontend
ARG GITLAB_ACCESS_TOKEN
ARG BRANCH=develop
RUN git clone -b ${BRANCH} https://vagarwalla:${GITLAB_ACCESS_TOKEN}@gitlab-master.nvidia.com/cudnn/cudnn_frontend.git

# Clone flash-attention
RUN pip uninstall -y flash-attn && \
    git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    sed -i 's/^    import flash_attn_2_cuda as flash_attn_gpu$/    pass/' /workspace/flash-attention/flash_attn/flash_attn_interface.py
ENV PYTHONPATH=/workspace/flash-attention

# Install additional dependencies for benchmarking
RUN pip install seaborn nvidia-cutlass-dsl apache-tvm-ffi quack-kernels

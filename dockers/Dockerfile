ARG DLFW_MONTH

FROM nvcr.io/nvidia/pytorch:${DLFW_MONTH}-py3

ARG CUDNN_VERSION
ARG CUDA_VERSION
ARG SKIP_CUDNN=false

ENV \
    DEBIAN_FRONTEND="noninteractive" \
    TZ="Etc/UTC" \
    CUDNN_VERSION=${CUDNN_VERSION} \
    CUDA_VERSION=${CUDA_VERSION}

SHELL ["/bin/bash", "-c"]

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update -qq --fix-missing

RUN CUDNN_VERSION_M=${CUDNN_VERSION%%.*} && \
    CUDA_VERSION_M=${CUDA_VERSION%%.*} && \
    if [ "${CUDNN_VERSION_M}" == "9" ]; then \
        if [ "${SKIP_CUDNN}" = false ]; then \
            CUDNN_PACKAGE_NAME="${CUDNN_VERSION}-1" && \
            apt update -qq --fix-missing && \
            apt upgrade -y --allow-change-held-packages \
            libcudnn9-cuda-${CUDA_VERSION_M}=${CUDNN_PACKAGE_NAME} \
            libcudnn9-dev-cuda-${CUDA_VERSION_M}=${CUDNN_PACKAGE_NAME} \
            && \
            rm -rf /root/.cache && \
            rm -rf /var/lib/apt/lists/*; \
        else \
            apt-get purge -y libcudnn9-cuda-12; \
        fi \
    else \
        CUDNN_PACKAGE_NAME="${CUDNN_VERSION}-1+cuda${CUDA_VERSION%.*}" && \
        apt update -qq --fix-missing && \
        apt upgrade -y --allow-downgrades --allow-change-held-packages \
        libcudnn8=${CUDNN_PACKAGE_NAME} \
        libcudnn8-dev=${CUDNN_PACKAGE_NAME} \
        && \
        rm -rf /root/.cache && \
        rm -rf /var/lib/apt/lists/*; \
    fi

COPY requirements.txt /tmp/
RUN pip install -U --no-cache-dir -r /tmp/requirements.txt
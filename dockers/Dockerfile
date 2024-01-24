ARG UBUNTU_VERSION="22.04"
ARG CUDA_VERSION="12.2.0"

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG CUDNN_VERSION="8.9.7.29"
ARG PYTHON_VERSION="3.10"
ARG TORCH_VERSION="2.1.2"

ENV \
    DEBIAN_FRONTEND="noninteractive" \
    TZ="Etc/UTC" \
    CUDNN_VERSION=${CUDNN_VERSION}

SHELL ["/bin/bash", "-c"]
RUN apt-get update -qq --fix-missing && \
    apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
        build-essential \
        ca-certificates \
        software-properties-common \
        pkg-config \
        cmake \
        ninja-build \
        git \
        wget \
        curl \
        unzip \
    && \
    # Install python
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-distutils \
        python${PYTHON_VERSION}-dev \
    && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 && \
    curl https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION} && \
    # Cleaning
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/* 

RUN CUDNN_PACKAGE_NAME="${CUDNN_VERSION}-1+cuda${CUDA_VERSION%.*}" && \
    apt update -qq --fix-missing && \
    apt upgrade -y --allow-change-held-packages \
      libcudnn8=${CUDNN_PACKAGE_NAME} \
      libcudnn8-dev=${CUDNN_PACKAGE_NAME} \
    && \
    rm -rf /root/.cache && \
    rm -rf /var/lib/apt/lists/*

RUN CUDA_VERSION_M=${CUDA_VERSION%%.*} && \
    if [ "${CUDA_VERSION_M}" == "11" ]; then \
        pip install "torch==${TORCH_VERSION}" \
            --index-url="https://download.pytorch.org/whl/cu118"; \
    else \
        pip install "torch==${TORCH_VERSION}" \
            --index-url="https://download.pytorch.org/whl/cu121"; \
    fi

COPY requirements.txt /tmp/
RUN pip install -U -r /tmp/requirements.txt
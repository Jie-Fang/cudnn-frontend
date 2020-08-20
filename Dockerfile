ARG FROM_IMAGE_NAME=ubuntu:18.04
ARG FROM_SCRIPTS_IMAGE=gitlab-master.nvidia.com:5005/dl/devops/build-scripts:latest


FROM ${FROM_SCRIPTS_IMAGE} AS build-scripts
FROM ${FROM_IMAGE_NAME}

RUN export DEBIAN_FRONTEND=noninteractive \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        ca-certificates \
        curl \
        patch \
        wget \
        jq \
        gnupg \
        git \
        dkms \
        libtcmalloc-minimal4 && \
        curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
        echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
        rm -rf /var/lib/apt/lists/*

COPY --from=build-scripts build-scripts /nvidia/build-scripts

# There seems to be a problem with release version on installCUDA.sh so not using 11.0.167
ARG CUDA_VERSION="11.0.167" 
ARG CUDA_BRANCH="11.0" 
ARG CUDA_DRIVER_VERSION="450.51.06"
ARG CUFFT_VERSION="10.2.1.245"
ARG CURAND_VERSION="10.2.1.245"
ARG CUDNN_VERSION="8.0.2.39"

ENV CUDA_VERSION=${CUDA_VERSION} \
    CUDA_DRIVER_VERSION=${CUDA_DRIVER_VERSION} \
    CUDA_CACHE_DISABLE=1

RUN pwd

RUN INTERNAL=1 DEVEL=1 /nvidia/build-scripts/installCUDA.sh

ENV CUFFT_VERSION=${CUFFT_VERSION} \
    CURAND_VERSION=${CURAND_VERSION} \
    CUDA_BRANCH=${CUDA_BRANCH} \
    CUDNN_VERSION=${CUDNN_VERSION}

RUN echo $CUDA_BRANCH
RUN wget "http://cuda-repo/release-candidates/repos/cudnn_r8.0_CUDA$CUDA_BRANCH/8.0.2/ubuntu1804/x86_64/libcudnn8_$CUDNN_VERSION-1+cuda11.0_amd64.deb"
RUN wget "http://cuda-repo/release-candidates/repos/cudnn_r8.0_CUDA$CUDA_BRANCH/8.0.2/ubuntu1804/x86_64/libcudnn8-dev_$CUDNN_VERSION-1+cuda11.0_amd64.deb"
RUN dpkg -i libcudnn*.deb
RUN /nvidia/build-scripts/installLIBS.sh

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
 && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
    # can't run ldconfig here because /usr/local/nvidia doesn't get populated until runtime

# set paths for nvdocker1 and CUDA compat
# set metadata for nvidia-container-runtime (nvdocker2)
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=${_CUDA_COMPAT_PATH}/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

COPY . .

RUN make -C samples

RUN pwd
RUN /samples/conv_sample

#!/bin/bash
set -e

GPU_CC=`nvidia-smi --query-gpu=compute_cap --format=csv | grep -v compute_cap | cut -d"." -f1`
CUDNN_BE_VERSION=`python3 -c "import cudnn;print(cudnn.backend_version())"`

export LD_LIBRARY_PATH=/debug_cudnn/lib64

if [ "${GPU_CC}" -ge "9" ]; then
    jupyter execute samples/python/0*
    jupyter execute samples/python/2[0-8]*
    jupyter execute samples/python/50*
    jupyter execute samples/python/51*
else
    jupyter execute samples/python/00*
    jupyter execute samples/python/02*
    jupyter execute samples/python/50*
    jupyter execute samples/python/51*
fi

# Run paged attention for SM>=80 and cuDNN >= 9.5
if [ "${CUDNN_BE_VERSION}" -ge "90500" ] && [ "${GPU_CC}" -ge "8" ]; then
    jupyter execute samples/python/52*
fi

if [ "${CUDNN_BE_VERSION}" -ge "91000" ] && [ "${GPU_CC}" -ge "8" ]; then
    jupyter execute samples/python/29*
fi

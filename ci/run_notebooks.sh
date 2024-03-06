#!/bin/bash
set -e

GPU_CC=`nvidia-smi --query-gpu=compute_cap --format=csv | grep -v compute_cap | cut -d"." -f1`

export LD_LIBRARY_PATH=/debug_cudnn/lib64

if [ "${GPU_CC}" == "9" ]; then
    jupyter execute samples/python/*
else
    jupyter execute samples/python/02*
    jupyter execute samples/python/50*
    jupyter execute samples/python/51*
fi

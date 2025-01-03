#!/bin/bash
set -e

function display_header() {
    nvidia-smi
    echo $CUDNN_VERSION
    echo $CUDA_VERSION
}

function run_python_tests() {
    export PYTHONPATH=build
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/debug_cudnn/lib64
    
    pytest test/python -n 4 --junit-xml=result-junit.xml
}

display_header
run_python_tests

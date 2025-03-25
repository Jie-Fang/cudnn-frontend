#!/bin/bash
set -e

function display_header() {
    nvidia-smi
    echo "Installed cudnn version" $CUDNN_VERSION_
    echo "Installed cuda version" $CUDA_VERSION_
}

function run_python_tests() {
    export PYTHONPATH=build
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/debug_cudnn/lib64
    
    pytest test/python -n 4 --junit-xml=result-junit.xml --tb=short
}

display_header
run_python_tests

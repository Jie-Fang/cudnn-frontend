#!/bin/bash
set -e

function display_header() {
    nvidia-smi
    echo $CUDNN_VERSION
    echo $CUDA_VERSION
}

function run_python_tests() {
    export PYTHONPATH=build
    
    pytest test/python_fe -n 4 -v
}

display_header
run_python_tests

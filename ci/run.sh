#!/bin/bash

function get_python_requirements() {
    apt-get update
    apt install python3-pip -y
    pip install numpy
    
    # Get the CUDA version from the output of nvcc --version
    cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-3)

    # Check if the CUDA version is 11
    if [[ "$cuda_version" == "11" ]]; then
        pip install cupy-cuda11x
    # Check if the CUDA version is 12
    elif [[ "$cuda_version" == "12" ]]; then
        pip install cupy-cuda12x
    fi
}

function display_header() {
    nvidia-smi
}

function run_cpp_tests() {
    cd build
    bin/samples --reporter JUnit::out=result-junit.xml --reporter console::out=-::colour-mode=ansi
}

function run_python_tests() {
    get_python_requirements
    PYTHONPATH=./python_bindings python3 ../samples/python/conv_bias.py
    PYTHONPATH=./python_bindings python3 ../samples/python/matmul_bias_relu.py
}

display_header
run_cpp_tests
run_python_tests
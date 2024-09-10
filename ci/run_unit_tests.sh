#!/bin/bash
set -e

function display_header() {
    nvidia-smi
    echo $CUDNN_VERSION
    echo $CUDA_VERSION
}

function run_unit_tests() {
    cd build
    bin/unit_tests --reporter JUnit::out=result-junit.xml --reporter console::out=-::colour-mode=ansi
}

display_header
run_unit_tests

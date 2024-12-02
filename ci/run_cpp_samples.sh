#!/bin/bash
set -e

function display_header() {
    nvidia-smi
    echo $CUDNN_VERSION
    echo $CUDA_VERSION
}

function run_cpp_samples() {
    build/bin/samples --reporter JUnit::out=result-junit.xml --reporter console::out=-::colour-mode=ansi
    build/bin/legacy_samples --reporter JUnit::out=result-junit.xml --reporter console::out=-::colour-mode=ansi
}

display_header
run_cpp_samples

#!/bin/bash
set -e

function display_header() {
    nvidia-smi
    echo "Installed cudnn version" $CUDNN_VERSION_
    echo "Installed cuda version" $CUDA_VERSION_
}

function run_cpp_samples() {
    build/bin/samples --reporter JUnit::out=result-junit.xml --reporter console::out=-::colour-mode=ansi
    build/bin/legacy_samples --reporter JUnit::out=result-junit.xml --reporter console::out=-::colour-mode=ansi
}

display_header
run_cpp_samples

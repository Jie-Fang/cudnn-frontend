#!/bin/bash
set -e

input_file="./ci/L4_testlist.txt"
total_lines=$(wc -l < "$input_file")
stream_group_size=16

function display_header() {
    nvidia-smi
    echo $CUDNN_VERSION
    echo $CUDA_VERSION
}

function run_python_streams() {
    export CUDNN_FRONTEND_LOG_FILE=stdout
    export CUDNN_FRONTEND_LOG_INFO=1
    export PYTHONPATH=build/python_bindings

    # TODO: replace this with the commented out code if we want to run a single process for all tests (this ensures all tests are run even if errors are encountered)
    for ((i = 1; i <= total_lines; i += stream_group_size)); do
        echo "Running tests $i - $stream_group_size out of $total_lines"
        ./test/pycudnnTest.py -RgrStream --stream_start "$i" --stream_group_size "$stream_group_size" < "$input_file"
    done
    #./test/pycudnnTest.py -RgrStream < "$input_file"
}

display_header
run_python_streams
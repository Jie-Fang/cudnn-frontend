#!/bin/bash
set -e

function display_header() {
    nvidia-smi
    echo $CUDNN_VERSION
    echo $CUDA_VERSION
}

function run_python_tests() {
    export CUDNN_FRONTEND_LOG_FILE=stdout
    export CUDNN_FRONTEND_LOG_INFO=1
    export PYTHONPATH=build/python_bindings
    
    # Graph tests from json
    python test/pycudnnTest.py --testPath test/json_graph_defs/graphTests.json --testName ConvRelu1
    # Python defined graph tests
    python test/pycudnnTest.py --testPath test/python_graph_defs/basic_tests.py
    # Explicit tests (not using test_graph)
    pytest test/explicit_test.py
}

display_header
run_python_tests

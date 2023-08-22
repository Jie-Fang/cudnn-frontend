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
    #Legacy graph test (TODO(@mbreughe): add to a list instead)
    python test/pycudnnTest.py --testPath test/json_graph_defs/fusionGraphTests.json -s --graphRunnerArgs "-R:graphRunner -jsonTestName=:DgradAdd_abstract -x: -dimA:8,64,64,64 -filtA:64,64,3,3 -padA:1,1 -convStrideA:1,1 -dilationA:1,1 -Pin:s -Pcomp:s -Pout:s -rtol:5e-3 -atol:5e-3 -minDevVer:800 -formatAll:1 -gpuRef: -d:0"
    # Failing on hopper:
    #python test/pycudnnTest.py --testPath test/json_graph_defs/fusionGraphTests.json -s --graphRunnerArgs "-R:graphRunner -jsonTestName=:ConvAdd_abstract -x: -dimA:8,64,64,64 -filtA:64,64,3,3 -padA:1,1 -convStrideA:1,1 -dilationA:1,1 -Pin:s -Pcomp:s -Pout:s -rtol:5e-3 -atol:5e-3 -minDevVer:800 -formatAll:1 -gpuRef: -d:0"
    #python test/pycudnnTest.py --testPath test/json_graph_defs/fusionGraphTests.json -s --graphRunnerArgs "-R:graphRunner -jsonTestName=:ConvRelu_abstract -x: -dimA:8,64,64,64 -filtA:64,64,3,3 -padA:1,1 -convStrideA:1,1 -dilationA:1,1 -Pin:s -Pcomp:s -Pout:s -rtol:5e-3 -atol:5e-3 -minDevVer:800 -formatAll:1 -gpuRef: -d:0"

}

display_header
run_python_tests

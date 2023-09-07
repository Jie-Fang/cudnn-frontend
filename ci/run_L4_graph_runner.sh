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
    
    #Legacy graph test (TODO(@mbreughe): add to a list instead)
    python test/pycudnnTest.py -jsonPath test/json_graph_defs/fusionGraphTests.json -RgraphRunner -jsonTestName=DgradAdd_abstract -x -dimA=8,64,64,64 -filtA=64,64,3,3 -padA=1,1 -convStrideA=1,1 -dilationA=1,1 -Pin s -Pcomp s -Pout s -rtol 5e-3 -atol 5e-3 -minDevVer 800 -formatAll 1
    python test/pycudnnTest.py -jsonPath test/json_graph_defs/fusionGraphTests.json -R graphRunner -jsonTestName=ConvAdd_abstract -x -dimA 8,64,64,64 -filtA 64,64,3,3 -padA 1,1 -convStrideA 1,1 -dilationA 1,1 -Pin s -Pcomp s -Pout s -rtol 5e-3 -atol 5e-3 -minDevVer 800 -formatAll 1
    python test/pycudnnTest.py -jsonPath test/json_graph_defs/fusionGraphTests.json -R graphRunner -jsonTestName ConvRelu_abstract -x -dimA 8,64,64,64 -filtA 64,64,3,3 -padA 1,1 -convStrideA 1,1 -dilationA 1,1 -Pin s -Pcomp s -Pout s -rtol 5e-3 -atol 5e-3 -minDevVer 800 -formatAll 1
}

display_header
run_python_tests

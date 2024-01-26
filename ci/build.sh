#!/bin/bash

function build_commands() {
    mkdir build
    cd build
    cmake -DCUDNN_FRONTEND_KEEP_PYBINDS_IN_BINARY_DIR=ON ../
    cmake --build . -j16 --verbose
}

build_commands

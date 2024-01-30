#!/bin/bash

function build_commands() {
    mkdir build
    cd build
    cmake ../
    cmake --build . -j16 --verbose
}

build_commands

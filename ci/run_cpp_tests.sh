#!/bin/bash
set -e

function display_header() {
    nvidia-smi
}

function run_cpp_tests() {
    cd build
    bin/samples --reporter JUnit::out=result-junit.xml --reporter console::out=-::colour-mode=ansi
}

display_header
run_cpp_tests

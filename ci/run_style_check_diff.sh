#!/bin/bash
set -e
clang-format --version
find include/ samples/ -regex '.*\.\(cpp\|h\)' | xargs clang-format --dry-run -Werror
#!/bin/bash
set -e
clang-format --version
black --version
find include/ samples/ python/ -regex '.*\.\(cpp\|h\)$' | xargs clang-format --dry-run -Werror
find test/ python/ -regex '.*\.py$' | xargs black --check --line-length 160
find ./ -maxdepth 1 -regex '.*\.py$' | xargs black --check --line-length 160
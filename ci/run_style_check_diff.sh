#!/bin/bash
set -e
clang-format --version
find include/ samples/ python/ -regex '.*\.\(cpp\|h\)$' | xargs clang-format --dry-run -Werror
find test/ python/ -regex '.*\.py$' | xargs black --check
find ./ -maxdepth 1 -regex '.*\.py$' | xargs black --check
#!/bin/bash
set -e

clang-format --version

# Wrap everything in a subshell so we can propagate the exit status.
(
find include/ samples/ -regex '.*\.\(cpp\|h\)' -exec clang-format --dry-run -Werror {} \;
)

exit_status=$?

exit ${exit_status}

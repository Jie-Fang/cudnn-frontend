#!/bin/bash
set -e

clang-format --version

# Wrap everything in a subshell so we can propagate the exit status.
(
git diff -U0 --no-color HEAD~1 | clang-format-diff -p1 > format-diff.log
)
exit_status=$?

[ ${exit_status} == 0 ] || exit ${exit_status}

format_diff="$(<format-diff.log)"

if [ -n "${format_diff}" ]; then
    cat format-diff.log
    exit 1
fi


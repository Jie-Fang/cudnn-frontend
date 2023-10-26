#!/bin/bash

input_file="./ci/L4_testlist.txt"
total_lines=$(wc -l < "$input_file")
stream_group_size=8

for ((i = 1; i <= total_lines; i += stream_group_size)); do
    ./test/pycudnnTest.py -RgrStream --stream_start "$i" --stream_group_size "$stream_group_size" < "$input_file"
done
#!/usr/bin/python3 -u

# Merge the pre binary files in parallel pairwise

import argparse
import os
import time

parser = argparse.ArgumentParser(
    description='')
parser.add_argument('-args_file_path', '--args_file_path',
                    metavar='args_file_path',
                    dest='args_file_path',
                    required=True,
                    help='Specify the tmp file that stores the pair of file to merge.')
args = parser.parse_args()

start_time = time.time()

# ***********************************************
# extract necessary info from the args
# ***********************************************

# get partition idx from tmp file name
partition_idx = os.path.basename(args.args_file_path).split(".")[0].split("_")[-1]

# parse the partition info
binary_paths = ""
with open(args.args_file_path, "r") as tmp_file:
    binary_paths = tmp_file.read().split(",")

assert len(binary_paths) == 2
binary_path_1 = binary_paths[0]
binary_path_2 = binary_paths[1]


# ***********************************************
# perform the merging
# ***********************************************

if not os.path.exists(binary_path_1) and not os.path.exists(binary_path_2):
    print("None of %s and %s exist: " % (binary_path_1, binary_path_2))

elif os.path.exists(binary_path_1) and not os.path.exists(binary_path_2):
    print("Not exist: ", binary_path_2)

elif not os.path.exists(binary_path_1) and os.path.exists(binary_path_2):

    # rename `binary_path_2` to `binary_path_1`
    print("Not exist: ", binary_path_2)
    print("Renaming %s to %s" % (binary_path_2, binary_path_1))
    os.rename(binary_path_2, binary_path_1)

else:
    print("Merging %s and %s" % (binary_path_1, binary_path_2))

    with open(binary_path_2, 'r') as binary_2:
        with open(binary_path_1, 'a') as binary_1:
            binary_1.write(binary_2.read())
    
    if os.path.exists(binary_path_2):
        os.remove(binary_path_2)
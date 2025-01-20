#!/usr/bin/python3

# Merge the artifacts (new_api_db and pre_binary files) from each node (dispatcher) into one
# Convert the pre binary files (plain test flags pre line) into binary files (compressed test flags)
# Delete the `to_delete` (those no longer tracked) and `to_update` (those that will be resampled 
#   due to code changes, engine config changes, etc.) cases from the existing persistent database

import sys
import argparse
import re
import os
import time

from heur_diff_utility import map_partition, BinaryWriter, build_to_delete_map, delete_row_from_persistent_db_partition
from multiprocessing import Pool, current_process, cpu_count


parser = argparse.ArgumentParser(
    description='merge the shelves and binaries from all query and diff jobs to one')
parser.add_argument('-job_count', '--job_count',
                    metavar='job_count',
                    dest='job_count',
                    required=True,
                    help='Specify the number of query and diff partitions.')
parser.add_argument('-old_api_db_file', '--old_api_db_file',
                    metavar='old_api_db_file',
                    dest='old_api_db_file',
                    required=True,
                    help='Specify the path to previous api db')
parser.add_argument('-persistent_db_path', '--persistent_db_path',
                    metavar='persistent_db_path',
                    dest='persistent_db_path',
                    required=True,
                    help='Specify the path to the persistent db')
args = parser.parse_args()

args.job_count = int(args.job_count)


# ***********************************************
# Merge the artifacts (new_api_db and pre_binary files) from each node (dispatcher) into one
# ***********************************************

# merge new_api_db pairwise
start_time = time.time()
merge_tasks_count = args.job_count
while merge_tasks_count > 1:

    # prepare pairs of merge db paths
    db_path_pairs = []
    for i in range(0, merge_tasks_count // 2):
        to_merge_1 = os.path.join(os.path.dirname(
            args.old_api_db_file), "new_api_db_job" + str(i) + ".sqlite3")
        if merge_tasks_count % 2 == 0:
            to_merge_2 = os.path.join(os.path.dirname(
                args.old_api_db_file), "new_api_db_job" + str(i+(merge_tasks_count//2)) + ".sqlite3")
        else:
            to_merge_2 = os.path.join(os.path.dirname(
                args.old_api_db_file), "new_api_db_job" + str(i+(merge_tasks_count//2)+1) + ".sqlite3")
        db_path_pairs.append(to_merge_1 + "," + to_merge_2)

    common_args_list = []
    try:
        map_partition(merge_tasks_count // 2, "./persistent_perf/merge_shelve_worker.py",
                    "-args_file_path", db_path_pairs, common_args_list, unique_str=("job"+str(args.job_count)+"_"), args_seperator="\n")
    except:
        print("Failed to merge the api dbs")

    print("merge workers: ", merge_tasks_count // 2)

    if merge_tasks_count % 2 == 0:
        merge_tasks_count //= 2
    else:
        merge_tasks_count = (merge_tasks_count // 2) + 1

# rename final merged file
old_partition_0_db_path = os.path.join(os.path.dirname(
    args.old_api_db_file), "new_api_db_job0.sqlite3")
new_api_db_path = os.path.join(os.path.dirname(
    args.old_api_db_file), "new_api_db.sqlite3")
if os.path.exists(new_api_db_path):
    os.remove(new_api_db_path)
if os.path.exists(old_partition_0_db_path):
    os.rename(old_partition_0_db_path, new_api_db_path)
    print("renamed %s to %s" % (old_partition_0_db_path, new_api_db_path))
else:
    raise RuntimeError("new_api_db partition 0 does not exist on %s" % (old_partition_0_db_path))

print("[%s] merge sqlite3 total time: %s seconds" % (os.path.basename(__file__), time.time() - start_time))

# merge binary file
start_time = time.time()
merge_tasks_count = args.job_count
while merge_tasks_count > 1:
    
    # prepare pairs of merge db paths
    binary_path_pairs = []
    for i in range(0, merge_tasks_count // 2):
        to_merge_1 = os.path.join(os.path.dirname(
            args.old_api_db_file), "main_job" + str(i) + ".txt")
        if merge_tasks_count % 2 == 0:
            to_merge_2 = os.path.join(os.path.dirname(
                args.old_api_db_file), "main_job" + str(i+(merge_tasks_count//2)) + ".txt")
        else:
            to_merge_2 = os.path.join(os.path.dirname(
                args.old_api_db_file), "main_job" + str(i+(merge_tasks_count//2)+1) + ".txt")
        binary_path_pairs.append(to_merge_1 + "," + to_merge_2)

    common_args_list = []
    try:
        map_partition(merge_tasks_count // 2, "./persistent_perf/merge_pre_binary_worker.py",
                    "-args_file_path", binary_path_pairs, common_args_list, unique_str=("job"+str(args.job_count)+"_"), args_seperator="\n")
    except:
        print("Failed to merge the main txt file")

    print("merge workers: ", merge_tasks_count // 2)

    if merge_tasks_count % 2 == 0:
        merge_tasks_count //= 2
    else:
        merge_tasks_count = (merge_tasks_count // 2) + 1

# rename final merged file
old_partition_0_binary_path = os.path.join(os.path.dirname(
    args.old_api_db_file), "main_job0.txt")
new_binary_path = os.path.join(os.path.dirname(
    args.old_api_db_file), "main.txt")
if os.path.exists(new_binary_path):
    os.remove(new_binary_path)
if os.path.exists(old_partition_0_binary_path):
    os.rename(old_partition_0_binary_path, new_binary_path)
    print("renamed %s to %s" % (old_partition_0_binary_path, new_binary_path))
else:
    raise RuntimeError("pre binary partition 0 does not exist on %s" % (old_partition_0_binary_path))

print("[%s] merge binary total time: %s seconds" % (os.path.basename(__file__), time.time() - start_time))


# ***********************************************
# Convert the pre binary files (plain test flags pre line) into binary files (compressed test flags)
# ***********************************************
converted_binary_path = os.path.join(os.path.dirname(
    args.old_api_db_file), "main.binary")
if os.path.exists(converted_binary_path):
    os.remove(converted_binary_path)
binary_writer = BinaryWriter(converted_binary_path, append=False)
binary_writer.open_segment_files(file_name_postpend_str=os.path.basename(converted_binary_path).split(".")[0])

convert_start_time = time.time()
start_time = time.time()
binary_case_count = 0
with open(new_binary_path, 'r') as pre_binary_file:
    for line in pre_binary_file.readlines():
        line = line.strip()
        if line == "":
            continue
        layer_descs, engine_descs, knob_descs, layer_name = line.split(";")
        binary_writer.write_one_case(layer_descs, engine_descs, knob_descs, layer_name)
        binary_case_count += 1
        if binary_case_count % 100000 == 0:
            print("[%s] [%s done] convert binary time: %s seconds" % (os.path.basename(__file__), 
                                                                      binary_case_count, 
                                                                      time.time() - start_time))
            start_time = time.time()

# close the segment files
binary_writer.close_segment_files()

# merge the segment files
binary_writer.merge_segments()

print("[%s] convert binary total time: %s seconds" % (os.path.basename(__file__), time.time() - convert_start_time))


# ***********************************************
# Delete rows from persistent db before sweeping
# ***********************************************

# save all `to_delete` files into a dict: aug_layer_name -> (api_db_key, dirty_bit)
try:
    to_delete_map = build_to_delete_map(os.path.dirname(args.old_api_db_file))
except:
    print("Failed to build to_delete map from logs")

# delete rows of "to_update" cases and "to_delete" cases in persistent db
try:
    deleted_to_update_count, deleted_to_delete_count, to_ignore_count, persistent_db_row_count = delete_row_from_persistent_db_partition(args.persistent_db_path, to_delete_map)

    print("deleted \'to_updated\' cases: %s" % (deleted_to_update_count))
    print("deleted \'to_delete\' cases: %s" % (deleted_to_delete_count))
    print("\'to_ignore\' cases: %s" % (to_ignore_count))
    print("persistent_db queried: %s" % (persistent_db_row_count))

except:
    print("Failed to delete rows from persistent db on %s" % (args.persistent_db_path))

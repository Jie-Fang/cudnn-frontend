#!/usr/bin/python3 -u

# Diff the newly generated sass db with the sass db from the last run
# Find the kernel functions that have changed since last run and save to diff sass dbs

import sys
import argparse
import re
import os
import collections
import time
from shutil import copyfile

from heur_diff_utility import PeresistentSQLDB

def resolve_call_chain(callee, callee_to_caller_dict, visited):
    """callee is the kernel that has been changed
    given the `callee_to_caller_dict`, return all kernels that should be updated
    """

    visited.append(callee)
    update_list = []
    if callee in callee_to_caller_dict:
        for caller in callee_to_caller_dict[callee]:

            # if there is a cycle, stop and check next
            if caller in visited:
                continue
            update_list.append(caller)
            update_list.extend(resolve_call_chain(caller, callee_to_caller_dict, visited))
    return update_list

parser = argparse.ArgumentParser(
    description='')
parser.add_argument('-args_file_path', '--args_file_path',
                    metavar='args_file_path',
                    dest='args_file_path',
                    required=True,
                    help='Specify the tmp file that stores the pair of file to merge.')
args = parser.parse_args()

start_time = time.time()

# get partition idx from tmp file name
partition_idx = os.path.basename(args.args_file_path).split(".")[0].split("_")[-1]

# parse the partition info
args_list = ""
with open(args.args_file_path, "r") as tmp_file:
    args_list = tmp_file.read().split(",")

# the old sass db not exist, this is a new arch
# in this case, just copy the new sass db to diff
if len(args_list) == 2:
    arch = args_list[0]
    new_sassdb_path = args_list[1]
    diff_sassdb_path = os.path.join(os.path.dirname(new_sassdb_path), "diff_sass_" + arch + ".sqlite3")
    copyfile(new_sassdb_path, diff_sassdb_path)

elif len(args_list) != 2 and len(args_list) != 3:
    raise RuntimeError("parameter not valid: ", args_list)

arch = args_list[0]
new_sassdb_path = args_list[1]
old_sassdb_path = args_list[2]


# ***********************************************
# perform the diff: between new and old sass db of same arch
# ***********************************************

# handle CAL instruction: callee -> caller dict
# while diffing, if `CAL` instruction exists, add caller's key to caller dict
# after diffing, for each callee, recursively add caller to diff if callee changed
# e.g., a -> b -> c, a,b not changed, but c changed, the dict will be {b:[a], c:[b]}
# the kernels needs to be updated will be [a,b]
callee_to_caller_dict = {}
pattern_CAL = re.compile(r'^.*CAL `\(\$(.*?)\).*$')

diff_sassdb_path = os.path.join(os.path.dirname(
    new_sassdb_path), "diff_" + os.path.basename(new_sassdb_path).split(".")[0] + ".sqlite3")

# load new sass sqlite3 to mem
new_sass_db = PeresistentSQLDB(new_sassdb_path, "sass_table",
                    elem_names=["fun_name", "sass_content"],
                    type_names=["TEXT PRIMARY KEY", "TEXT"], load_to_mem=False, mode='r')
new_db = {}
for row in new_sass_db.get_gen(["fun_name", "sass_content"]):
    new_db[row["fun_name"]] = row["sass_content"]
new_sass_db.close()

# load old sass sqlite3 to mem
old_sass_db = PeresistentSQLDB(old_sassdb_path, "sass_table",
                    elem_names=["fun_name", "sass_content"],
                    type_names=["TEXT PRIMARY KEY", "TEXT"], load_to_mem=False, mode='r')
old_db = {}
for row in old_sass_db.get_gen(["fun_name", "sass_content"]):
    old_db[row["fun_name"]] = row["sass_content"]
old_sass_db.close()


diff_sass_db = PeresistentSQLDB(diff_sassdb_path, "sass_table",
                    elem_names=["fun_name", "sass_content"],
                    type_names=["TEXT PRIMARY KEY", "TEXT"], load_to_mem=False, mode='n')

# create in-mem diff db for faster access
diff_db = {}

print("(%s) Diffing [%s] with [%s]" % (partition_idx, new_sassdb_path, old_sassdb_path))
print("(%s) diff db path: %s" % (partition_idx, diff_sassdb_path))

count = 0
for key in new_db:
    count += 1
    if count % 100 == 0:
        print("(%s) %d/%d" % (partition_idx, count, len(new_db)))

    # check new key and key change
    # if this is a new kernel or kernel signature changed, add directly to diff
    if key not in old_db:
        print("Found new key: %s" % (key))
        diff_db[key] = None


    # if key exists in both old and new, check for content
    else:

        # add to diff if content has different length
        if len(new_db[key]) != len(old_db[key]):
            print("Found key with diff length: %s" % (key))
            diff_db[key] = None
        else:

            # if this val contains `CAL`, it is not compressed
            if isinstance(new_db[key], str):
                for line_idx, cubin_line in enumerate(new_db[key].split("\n")):

                    # if the content does not match, add the diff and break
                    if old_db[key] != new_db[key]:
                        print("Found key with diff content: %s" % (key))
                        diff_db[key] = None
                        break
                    else:
                        if pattern_CAL.match(cubin_line):
                            callee = pattern_CAL.match(cubin_line).groups()[0]
                            if callee not in callee_to_caller_dict:
                                callee_to_caller_dict[callee] = []
                            callee_to_caller_dict[callee].append(key)
            else:
                if new_db[key] != old_db[key]:
                    print("Found key with diff content: %s" % (key))
                    diff_db[key] = None

# handle CAL instruction
for callee in callee_to_caller_dict:

    # if callee has been changed, get all affected kernels and add to diff
    if callee in diff_db:
        for kernel_to_update in resolve_call_chain(callee, callee_to_caller_dict, []):
            print("Found resolved call chain keys: %s" % (kernel_to_update))
            if kernel_to_update not in diff_db:
                diff_db[kernel_to_update] = None

# write in-mem diff db to sqlite3
for key in diff_db:
    diff_sass_db.add({"fun_name": key})

diff_sass_db.close()
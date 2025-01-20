#!/usr/bin/python3 -u

# Worker script for merging sample db partitions into the corresponding persistent db
# The worker will work on the partitons idx given by the dispatcher

import sys
import argparse
import re
import os
import time

# to import helpers from parent packages
sys.path.append(os.path.abspath('..'))

from heur_diff_utility import PeresistentSQLDB, FlagApiIndexer, create_persistent_db


def save_indexer_to_persistent_db(indexer, persistent_db_idx_table):
    """create (reset) a table in persistent db named `idx_table_name`
    write flattern indexer to table
    """

    # reset table
    persistent_db_idx_table.reset_table()

    key_to_index_dict = indexer.get_key_to_index_dict()

    for key in key_to_index_dict:
        persistent_db_idx_table.add({"idx": key_to_index_dict[key], "val": key})


parser = argparse.ArgumentParser(
    description='worker script to parse query results and diff')
parser.add_argument('-args_file_path', '--args_file_path',
                    metavar='args_file_path',
                    dest='args_file_path',
                    required=True,
                    help='Specify the tmp file that stores a list of api db keys to diff.')
parser.add_argument('-sampling_count', '--sampling_count',
                    metavar='sampling_count',
                    dest='sampling_count',
                    required=True,
                    help='Specify the number of sampling partitions.')
parser.add_argument('-input_sample_db_dir', '--input_sample_db_dir',
                    metavar='input_sample_db_dir',
                    dest='input_sample_db_dir',
                    required=True,
                    help='Specify the path to the persistent db')
parser.add_argument('-persistent_db', '--persistent_db',
                    metavar='persistent_db',
                    dest='persistent_db',
                    required=True,
                    help='Specify the path to the persistent db')
parser.add_argument('-new_api_db_path', '--new_api_db_path',
                    metavar='new_api_db_path',
                    dest='new_api_db_path',
                    required=True,
                    help='Specify the path to the new api db')
args = parser.parse_args()

start_time = time.time()

# ***********************************************
# prepare the data and extract basic infomation
# ***********************************************

# get partition idx from tmp file name
partition_idx = os.path.basename(args.args_file_path).split(".")[
    0].split("_")[-1]

# read indexer from new api db
new_idx_table_db = PeresistentSQLDB(args.new_api_db_path, "idx_table",
                    elem_names=["idx", "val"],
                    type_names=["INTEGER PRIMARY KEY", "TEXT"], load_to_mem=False, mode='r')
if new_idx_table_db.is_table_exist():
    indexer = FlagApiIndexer()
    key_to_index = {}
    index_to_key = {}
    for row in new_idx_table_db.get_gen(["idx", "val"]):
        key_to_index[row["val"]] = int(row["idx"])
        index_to_key[int(row["idx"])] = row["val"]
    indexer.load(key_to_index, index_to_key)
new_idx_table_db.close()

# parse the partition info
hash_codes = []
with open(args.args_file_path, "r") as tmp_file:
    hash_codes = [section for section in tmp_file.read().split(",") if section != ""]
print("[%s] (%s) hash codes assigned %s" % (os.path.basename(
            __file__), partition_idx, hash_codes))


pattern_sample_db_partition = re.compile(r'^.*_([0-9a-f][0-9a-f])\.sqlite3$')
for idx, hash_code in enumerate(hash_codes):

    # walk through the `input_sample_db_dir` to get a list of all samples db of this partition hash code
    db_list = []
    for i in range(int(args.sampling_count)):
        db_list.append(os.path.join(args.input_sample_db_dir, "samples_%s_%s.sqlite3" % (i, hash_code)))
    persistent_db_partition_path = ".".join(args.persistent_db.split(".")[0:-1]) + "_" + hash_code + ".sqlite3"
    if not os.path.exists(persistent_db_partition_path):
        create_persistent_db(persistent_db_partition_path)

    # write new indexer to the persistent db
    persistent_db_idx_table = PeresistentSQLDB(persistent_db_partition_path, "idx_table",
                            elem_names=["idx", "val"],
                            type_names=["INTEGER PRIMARY KEY", "TEXT"], 
                            load_to_mem=False, mode='w')
    save_indexer_to_persistent_db(indexer, persistent_db_idx_table)
    persistent_db_idx_table.close()

    persistent_db_partition = PeresistentSQLDB(persistent_db_partition_path, "ConvData", load_to_mem=False, mode='w')
    print("[%s] (%s) (%s/%s) start merging persistent db %s" % (os.path.basename(
            __file__), partition_idx, idx, len(hash_codes), persistent_db_partition_path))

    merge_start_time = time.time()
    start_time = time.time()
    for db in db_list:
        if not os.path.exists(db):
            continue
        persistent_db_partition.insert_from_db_file_name(db)
        print("[%s] (%s) done merging %s: %s secs" % (os.path.basename(
            __file__), partition_idx, db, time.time() - start_time))
        start_time = time.time()
    print("[%s] (%s) done merging persistentn db %s: %s secs" % (os.path.basename(
            __file__), partition_idx, persistent_db_partition_path, time.time() - merge_start_time))

    persistent_db_partition.close()
#!/usr/bin/python3 -u

# Perform diffing on the queries distributed by the dispatcher
# Each work works on limited number (default 20) of queries per launch, because each query will
#   will generate ~1k cudnnTest flags to run
# The diffing has the following stages (for each cudnnTest flag)
#   1. check if this flag has been sampled before
#   2. if sampled, check if same kernels and cuda apis have been launched
#   3. if same kernels and cuda api been launched, check if these functions have been modified (through SASS comparison)

import sys
import argparse
import re
import os
import collections
from sys import modules
import time
from types import prepare_class
import hashlib

# to import helpers from parent packages
sys.path.append(os.path.abspath('..'))

from scripts.helpers.cudnn_interface import LogExtractor
from scripts.helpers.parser_utility import LogFlagReader, LogParser, use_parser_on_output
from scripts.helpers.utility import get_unique, flags_from_descs_str
from heur_diff_utility import PeresistentSQLDB, FlagApiIndexer, ApiDbWriter, print_used_mem
from collections import OrderedDict, namedtuple, defaultdict


def append_to_flags_per_engine(flags_per_engine, engine_config_str):
    """append a new case to flags_per_engine
    """

    # Get Flags() form from for this engine config
    engine_config_flags = flags_from_descs_str(engine_config_str)[0]

    # Get engine index
    engine = int(engine_config_flags["backendEngine"][0])

    # Append the flags for this engine config to all flags with the same engine
    flags_per_engine[engine].append(engine_config_flags)


parser = argparse.ArgumentParser(
    description='worker script to parse query results and diff')
parser.add_argument('-args_file_path', '--args_file_path',
                    metavar='args_file_path',
                    dest='args_file_path',
                    required=True,
                    help='Specify the tmp file that stores a list of api db keys to diff.')
parser.add_argument('-persistent_db_path', '--persistent_db_path',
                    metavar='persistent_db_path',
                    dest='persistent_db_path',
                    required=True,
                    help='Specify the path to the persistent db')
parser.add_argument('-old_api_db_file', '--old_api_db_file',
                    metavar='old_api_db_file',
                    dest='old_api_db_file',
                    required=True,
                    help='Specify the path to previous api db')
parser.add_argument('-diff_sass_db_file', '--diff_sass_db_file',
                    metavar='diff_sass_db_file',
                    dest='diff_sass_db_file',
                    required=True,
                    help='Specify the path to the diff sass db')
parser.add_argument('-job_idx', '--job_idx',
                    metavar='job_idx',
                    dest='job_idx',
                    required=True,
                    help='Specify the index of current job')
args = parser.parse_args()

start_time = time.time()

query_parser = LogParser(LogExtractor.getExtractorList())

log_dir_path = os.path.dirname(args.diff_sass_db_file)

# ***********************************************
# prepare the data and extract basic infomation
# ***********************************************

# get partition idx from tmp file name
partition_idx = os.path.basename(args.args_file_path).split(".")[
    0].split("_")[-1]

# parse the partition info
parsed_cudnnTest_backendQuery_output_list = []
with open(args.args_file_path, "r") as tmp_file:
    parsed_cudnnTest_backendQuery_output_list = [use_parser_on_output(
        section, query_parser) for section in tmp_file.read().split("\n\n\n") if section != ""]

# ***********************************************
# parse the query and perform diff
# ***********************************************

# do nothing if there is no tasks to do
if len(parsed_cudnnTest_backendQuery_output_list) == 0:
    print("No tasks on partition: %s" % (partition_idx))

else:

    # write the idx to delete to a file
    to_delete_idx_file_path = os.path.join(log_dir_path, "to_delete_job" + str(args.job_idx) + "_part" + str(partition_idx) + ".log")
    to_delete_idx_file = open(to_delete_idx_file_path, "a")

    # write pre-binary to txt
    pre_binary_file_path = os.path.join(log_dir_path, "main_job" + str(args.job_idx) + "_part" + str(partition_idx) + ".txt")
    if not os.path.exists(pre_binary_file_path):
        pre_binary_file = open(pre_binary_file_path, 'w')
    else:
        pre_binary_file = open(pre_binary_file_path, 'a')

    # get indexer from persistent_db partition 00 if exists, 
    # this step ensures that new_api_db shares the same indexing/compressing with persistent db
    # all partitions have the same idx_table
    old_indexer = None
    persistent_db_partition_00_path = ".".join(args.persistent_db_path.split(".")[0:-1]) + "_00.sqlite3"
    old_idx_table_db = PeresistentSQLDB(persistent_db_partition_00_path, "idx_table",
                    elem_names=["idx", "val"],
                    type_names=["INTEGER PRIMARY KEY", "TEXT"], load_to_mem=False, mode='r')
    if old_idx_table_db.is_table_exist():
        old_indexer = FlagApiIndexer()
        key_to_index = {}
        index_to_key = {}
        for row in old_idx_table_db.get_gen(["idx", "val"]):
            key_to_index[row["val"]] = int(row["idx"])
            index_to_key[int(row["idx"])] = row["val"]
        old_indexer.load(key_to_index, index_to_key)
    old_idx_table_db.close()

    # open old_api_db and diff_sass_db in read-only mode
    diff_db_sqlite = PeresistentSQLDB(args.diff_sass_db_file, "sass_table",
                    elem_names=["fun_name", "sass_content"],
                    type_names=["TEXT", "TEXT"], load_to_mem=False, mode='r')
    diff_db = {}
    for row in diff_db_sqlite.get_gen(["fun_name"]):
        diff_db[row["fun_name"]] = None

    # init the new_api_db writer, each worker writes to its own tmp partition file
    api_db_writer = ApiDbWriter(os.path.join(log_dir_path, "new_api_db_job" + str(args.job_idx) + "_part" + str(partition_idx) + ".sqlite3"))

    # inherit the indexer from old_api_db if exists
    api_db_writer.open_and_load_indexers(db_mode='c', indexer=old_indexer)

    check_time = 0
    query_time = 0

    for row in parsed_cudnnTest_backendQuery_output_list:

        # Get all info from query

        layer_name = row.get("layer_name", None)
        test_flags_str = row.get("test_flags", None)
        unique_flags_str = row.get("unique_flags", None)
        query = row.get("query", None)

        # Skip any layer name with None (this is stuff like cudnnTest -g)
        if layer_name == None or query == None:
            continue

        # Get Flags() from test_flags_str
        test_flags = flags_from_descs_str(test_flags_str)[0]

        # Get Flags() from unique_flags_str
        unique_flags = flags_from_descs_str(unique_flags_str)[0]

        # Get string form of Flags() in format of "Rconv_n128_..."
        unique_flags_aug_str = unique_flags.get_str(prefix='', delimiter='_', seperator='')

        # Start augmented layer name with just layer_name (as a base)
        aug_layer_name = layer_name

        # Augment layer name with unique flags string (to ensure name is unique)
        # Without augmentation, layer name will be re-used multiple times
        # If unique_flags_aug_str is empty, it means this layer has no duplicates (so no augmentation is necessary)
        if unique_flags_aug_str != '':
            aug_layer_name += "_" + unique_flags_aug_str

        # query all flag_idx and api_list data of the layer_name from persistent db
        # typically, each worker works on limited number of tasks, e.g., 10
        #   so there will be only 10 queries to the persistent db, the overhead and mem cost is ignorable
        #   by comparison, if we do query per engine config, 10 tasks will generate as many as 10,000 queries

        # flag_idx -> api_list
        flag_idx_dict = {}

        # query persistent db partitions for all records from this `aug_layer_name`
        query_start_time = time.time()
        first_two_hash_code = hashlib.md5(aug_layer_name.encode('utf-8')).hexdigest()[0:2]
        persistent_db_partition_path = ".".join(args.persistent_db_path.split(".")[0:-1]) + "_" + first_two_hash_code + ".sqlite3"
        persistent_db_partition = PeresistentSQLDB(persistent_db_partition_path, "ConvData", load_to_mem=False, mode='r')
        
        try:
            for row in persistent_db_partition.get_gen(retrieve_names=["flag_idx", "api_list"], conditionals=[Conditional("layer_name", "=", aug_layer_name)]):
                flag_idx_dict[row["flag_idx"]] = tuple(row["api_list"].split(","))
        except:
            print("Failed to query layer_name %s from persistent db partition" % (aug_layer_name))
            
        persistent_db_partition.close()
        print("(%s) [%s] layer_name: %s, len(flag_idx_dict): %s, partition: %s, query time: %s" % 
              (partition_idx, print_used_mem(), aug_layer_name, len(flag_idx_dict), 
              persistent_db_partition_path, (time.time() - query_start_time)))
        query_time += (time.time() - query_start_time)

        # Get routine from flags
        routine = test_flags["R"][0]

        # In following loop, accumulate all knob configs with identical engine index
        flags_per_engine = collections.defaultdict(list)

        splitted_query = query.split('\\n')

        # parse the query content and api_idx_dict from this query
        api_list_content = splitted_query[:-1]
        api_idx_dict_str = splitted_query[-1]

        # build revert dict provided by the backendQuery: idx -> kernel name
        api_reverse_idx_dict_from_query = {}
        if len(api_idx_dict_str.split(":")) == 2 and api_idx_dict_str.split(":")[0] == "ApiDict":
            api_idx_dict_str_content_list = api_idx_dict_str.split(":")[
                1].split(",")
            for key_idx in range(0, len(api_idx_dict_str_content_list), 2):
                if key_idx+1 < len(api_idx_dict_str_content_list):
                    api_reverse_idx_dict_from_query[api_idx_dict_str_content_list[key_idx+1]
                                                    ] = api_idx_dict_str_content_list[key_idx]

        # Loop through each engine config (seperated by new line)
        # there are four scenarios for one cudnnTest case (engine config)
        #   1. it is brand new (not in persistent db aka old_api_db): "to_add", 
        #      since it is never seen in previous runs, the flag will be newly indexed
        #   2. it exists in both this run and old run (persistent db), it will be indexed using existing indexer
        #       a. either api_list changes ot sass changes: "to_update"
        #       b. nothing changed: "to_ignore"
        #   3. it only exists in old run (persistent db): "to_delete", 
        #      this scenario will not appear here since it only exists in old_api_db
        # e.g. when you run a layer, previously it launches two cudnnTests with flag "-a -b -c" and "-a -b", 
        #      in this run, it launches flag "-a -b" and "-a -c", and "-a -b" invokes a new kernel, then 
        #      "-a -c" is "to_add", "-a -b" is "to_update", "-a -b -c" is "to_delete"
        for config_str in api_list_content:

            # state variable decsribing the action to take for each case
            #   0 for "to_add" state
            #   1 for "to_update" state
            #   2 for "to_ignore" state (default)
            #   3 for "to_delete" state
            action_state = 2

            # get the encoded api db key
            try:
                api_db_key, new_apis = api_db_writer.get_api_db_key_val_from_backendquery(
                    test_flags, config_str, api_reverse_idx_dict_from_query)
            except:
                print("Failed to calculate api_list and flag_idx from backendquery %s" % (test_flags))
            api_db_val_len = len(new_apis)

            check_start_time = time.time()
            if not api_db_key in flag_idx_dict:
                action_state = 0
                check_time += (time.time() - check_start_time)

            # if the flag exist, diff the apis invoked by the flag
            else:
                old_apis = flag_idx_dict[api_db_key]
                check_time += (time.time() - check_start_time)

                # check if they are in same length
                if len(new_apis) != len(old_apis):
                    action_state = 1

                # compare each api calls individually
                else:
                    found_diff = False
                    
                    for api_idx in range(len(new_apis)):
                        if new_apis[api_idx] != old_apis[api_idx]:
                            action_state = 1
                            found_diff = True
                            break

                    # api calls are same, no diff found, try to find in diff_sass_db
                    if not found_diff:
                        for kernel_name in api_db_writer.get_indexer().get_kernel_name_list_from_idx_tuple(new_apis):
                            if kernel_name in diff_db:
                                action_state = 1
                                break

            # write to new_api_db and binary if this is a brand new case
            try:

                # for "to_add" cases
                if action_state == 0:

                    # prepare to write to binary
                    append_to_flags_per_engine(flags_per_engine, config_str.split("|")[0])

                    # write to api db
                    api_db_writer.write(test_flags, config_str, api_reverse_idx_dict_from_query)

                # for "to_update" cases
                # write to new_api_db, binary and to_delete_idx_file if this case exists before but should be updated in persisten db
                # since this case exists in persisten db, the api_db_key will be indexed identically
                elif action_state == 1:

                    # prepare to write to binary
                    append_to_flags_per_engine(flags_per_engine, config_str.split("|")[0])

                    # write to api db
                    api_db_writer.write(test_flags, config_str, api_reverse_idx_dict_from_query)

                    # add all visited cases to `to_delete` file, one case per line
                    # format: api_db_key, aug_layer_name, dirty_bit (0 for keep in persistent db, 1 for delete)
                    to_delete_idx_file.write("%s;%s;%s\n" % (api_db_key, aug_layer_name, 1))

                # "to_ignore" and "to_delete" cases will be handled on final merging of artifacts
                # if this is a "to-ignore" case, we still need to add it to to_delete_idx_file but it is marked 0 (keep)
                # we add it to to_delete_idx_file because we have to know which cases have been covered in this run
                # so the cases that are not in to_delete_idx_file will be those "to_delete" cases
                else:
                    to_delete_idx_file.write("%s;%s;%s\n" % (api_db_key, aug_layer_name, 0))

            except:
                print("Failed to write case to file")


        for engine in sorted(flags_per_engine):
            for knob_str in flags_per_engine[engine]:
                pre_binary_file.write("%s;%s;%s;%s\n" % (test_flags.copy_without("backendQuery").get_descs_str(), 
                                                            "backendEngine:" + str(engine), 
                                                            knob_str.copy_without("backendEngine").get_descs_str(), 
                                                            aug_layer_name))

    # close old_api_db and diff_db
    diff_db_sqlite.close()

    indexer = api_db_writer.get_indexer()

    print("(%s) check persistent db time: %s" %
        (partition_idx, check_time))
    print("(%s) query persistent db time: %s" %
        (partition_idx, query_time))
    print("(%s) new api db size: %s" %
        (partition_idx, api_db_writer.get_db_size()))
    print("(%s) indexer size: %s" %
        (partition_idx, indexer.get_indexer_size()))
    print("(%s) worker time (%d tasks): %s" %
        (partition_idx, len(parsed_cudnnTest_backendQuery_output_list), (time.time() - start_time)))

    # close and save the api_db_writer and to_delte files
    api_db_writer.close_and_save_indexers()
    to_delete_idx_file.close()
    pre_binary_file.close()

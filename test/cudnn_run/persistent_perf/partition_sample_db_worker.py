#!/usr/bin/python3 -u

# The worker to partition given sample db into 256 partitions each

import sys
import argparse
import os
import time
import hashlib

# to import helpers from parent packages
sys.path.append(os.path.abspath('..'))

from heur_diff_utility import PeresistentSQLDB, print_used_mem

parser = argparse.ArgumentParser(
    description='worker script to parse query results and diff')
parser.add_argument('-args_file_path', '--args_file_path',
                    metavar='args_file_path',
                    dest='args_file_path',
                    required=True,
                    help='Sepecify the tmp file that stores a list of api db keys to diff.')
args = parser.parse_args()

start_time = time.time()

# ***********************************************
# prepare the data and extract basic infomation
# ***********************************************

# get partition idx from tmp file name
partition_idx = os.path.basename(args.args_file_path).split(".")[
    0].split("_")[-1]

# parse the partition info
input_sample_dbs = []
with open(args.args_file_path, "r") as tmp_file:
    input_sample_dbs = [section for section in tmp_file.read().split(",") if section != ""]


for input_sample_db_path in input_sample_dbs:
    
    # partition_idx -> rows in that partition
    partition_dict = {}
    print("[%s] [%s] start partition on %s" % (os.path.basename(__file__), print_used_mem(), input_sample_db_path))
    input_sample_db = PeresistentSQLDB(input_sample_db_path, "ConvData", load_to_mem=False, mode='r')

    # hash code -> db partitions
    hash_code_to_dbs = {}
    hex_chars = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('f')+1)]
    for first_hex_char in hex_chars:
        for second_hex_char in hex_chars:
            hash_code = first_hex_char + second_hex_char
            output_sample_partition_db_path = ".".join(input_sample_db_path.split(".")[0:-1]) + "_" + hash_code + ".sqlite3"
            output_sample_partition_db = PeresistentSQLDB(output_sample_partition_db_path, "ConvData", 
                                            elem_names=input_sample_db.get_elem_names(),
                                            type_names=input_sample_db.get_type_names(),
                                            load_to_mem=False, mode='n')
            hash_code_to_dbs[hash_code] = output_sample_partition_db

    to_retrieve = input_sample_db.get_elem_names()
    for row in input_sample_db.get_gen(retrieve_names=to_retrieve):
        if row["layer_name"] == None:
            continue
        layer_name = row["layer_name"]
        first_two_hash_code = hashlib.md5(layer_name.encode('utf-8')).hexdigest()[0:2]
        hash_code_to_dbs[first_two_hash_code].add(row)

    for hash_code in hash_code_to_dbs:
        hash_code_to_dbs[hash_code].close()
        
    input_sample_db.close()
#!/usr/bin/python3

# Merge the sample db partitions (256 each sample db) into persistent db partitions (256)
# e.g., persistent db partition 00 will merge partition 00 from sample db 0, 1, 2, 3, ...
# This script works as the dispatcher, assign the merge tasks to workers for them to work in parallel

import sys
import argparse
import os

sys.path.append(os.path.abspath('..'))

from heur_diff_utility import map_partition

parser = argparse.ArgumentParser(description='')
parser.add_argument('-partition_count', '--partition_count',
                    metavar='partition_count',
                    dest='partition_count',
                    required=True,
                    help='Specify the number of independent partitions we want to run.')
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
for arg in args.__dict__:
    print("\t%s = %s" % (arg, args.__dict__[arg]))

first_two_hash_codes = []
hex_chars = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('f')+1)]
for first_hex_char in hex_chars:
    for second_hex_char in hex_chars:
        hash_code = first_hex_char + second_hex_char
        first_two_hash_codes.append(hash_code)

common_args_list = ["-input_sample_db_dir"]
common_args_list += [args.input_sample_db_dir]
common_args_list += ["-persistent_db"]
common_args_list += [args.persistent_db]
common_args_list += ["-sampling_count"]
common_args_list += [args.sampling_count]
common_args_list += ["-new_api_db_path"]
common_args_list += [args.new_api_db_path]
print("[%s] start merging sample db partitions to persistent db" % (os.path.basename(__file__)))
map_partition(int(args.partition_count), "./persistent_perf/merge_sample_db_worker.py",
                "-args_file_path", first_two_hash_codes, common_args_list=common_args_list)
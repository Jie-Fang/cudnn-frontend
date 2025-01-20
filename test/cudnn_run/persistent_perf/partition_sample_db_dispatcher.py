#!/usr/bin/python3

# Partition each sample db into 256 partitions
# These sample db partitions will be merged to the corresponding persistent db partitions in the next stage
# This dispatcher will assign each worker a list of sample dbs to work on

import sys
import argparse
import os

sys.path.append(os.path.abspath('..'))

from heur_diff_utility import map_partition

def flatten_array_of_arrays(to_flatten):
    return [b for a in to_flatten for b in a]

def spreadsheet_post_argparse(parsed):
    '''Post process {parsed} command line argument:
    - Convert dest value to list of string if multiple values are allowed with comma separated value
    '''
    # -log_paths/--log-paths/-l
    # Support both format -l A,B and -l A -l B, and remove duplicates using OrderedDict
    parsed.input_sample_dbs = flatten_array_of_arrays(parsed.input_sample_dbs)

    return parsed

parser = argparse.ArgumentParser(description='')
parser.add_argument('-input_sample_dbs', '--input_sample_dbs',
                    metavar = 'LOG_FILE',
                    dest    = 'input_sample_dbs',
                    default = None,
                    required = True,
                    nargs   = '*',
                    action  = 'append',
                    help    = "Specify path(s) to sample sql dbs from cudnn_run.py.")
parser.add_argument('-partition_count', '--partition_count',
                    metavar='partition_count',
                    dest='partition_count',
                    required=True,
                    help='Specify the number of independent partitions we want to run.')
args = parser.parse_args()
args = spreadsheet_post_argparse(args)
for arg in args.__dict__:
    print("\t%s = %s" % (arg, args.__dict__[arg]))

print("[%s] start partition of sample dbs" % (os.path.basename(__file__)))

try:
    map_partition(int(args.partition_count), "./persistent_perf/partition_sample_db_worker.py",
                    "-args_file_path", args.input_sample_dbs, common_args_list=[])
except RuntimeError as e:
    print("Failed to partition te sample dbs")
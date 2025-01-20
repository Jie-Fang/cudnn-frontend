#!/usr/bin/python3

# Read backendQueries from logs written by cudnn_run.py and distribute the queries to workers
# to perform diffing (check if the case was sampled before or not)

# This dispatcher works in parallel with cudnn_run.py as follows:
#   1. cudnn_run.py performs backendQuery and write the queries into corresponding log files
#   2. the dispatcher monitors the log file and reads the queries to the buffer
#   3. whenever the buffer is filled (has `task_per_worker` * `partition_count` queries), the workers
#      will be launched in parallel (`partition_count` workers). 
#   4. the dispatcher will wait for all workers to complete before continuing reading the log file
#      Note: cudnn_run.py is still running in parallel with both dispatcher and workers
#   5. When all queries have been processed by the workers, the dispatcher will merge the artifacts
#      produced by the workers (new_api_db and pre_binary files), one per dispatcher (node)

import sys
import argparse
import re
import os
import collections
from sys import modules
import time
from types import prepare_class

# to import helpers from parent packages
sys.path.append(os.path.abspath('..'))

from heur_diff_utility import create_persistent_db_partition, map_partition, reverse_readline, print_used_mem
from scripts.helpers.cudnn_interface import LogExtractor
from scripts.helpers.parser_utility import LogFlagReader, LogParser, use_parser_on_output
from scripts.helpers.utility import get_unique, flags_from_descs_str
from multiprocessing import Pool, current_process, cpu_count


def is_cudnnTest_new_section_start(line):
    """check if this line is the first line of a cudnnTest section
    """

    pattern_cudnnTest_gpu_test = re.compile(
        r'^&&&& RUNNING cudnnTest -gpu[\d,]+$')

    if "&&&& RUNNING cudnnTest" in line:

        # pass initial test runs
        if line != "&&&& PASSED cudnnTest -g" and not pattern_cudnnTest_gpu_test.match(line):
            return True
    return False

def gen_query_task_from_log(log_path, start_line_no):
    """read query log file and generate cases one by one
    checkpointing: start reading from `start_line_no` inclusive
    """

    res_pattern = re.compile(r'\&\&\&\& (PASSED|WAIVED|FAILED) (.*)$')

    # do nothing when everything already parsed
    if start_line_no == -1:
        return

    is_in_cudnnTest_section = False
    line_no = 0

    with open(log_path, 'r') as log_file:
        log_file.seek(0)
        while True:
            line = log_file.readline()

            # wait until a line
            if not line:
                time.sleep(0.5)
                continue

            # if we come here successfully, it means this is a valid line
            if line_no < start_line_no:
                line_no += 1
                continue
            line_no += 1

            line = line.strip()

            if (is_cudnnTest_new_section_start(line)):
                task_content = (line + "\n")
                task_len = len(line)
                is_in_cudnnTest_section = True

            if is_in_cudnnTest_section and line != "":
                task_content += (line + "\n")
                task_len += len(line)
                if res_pattern.match(line):
                    is_in_cudnnTest_section = False
                    yield (task_content, task_len, line_no)

            if "Basic Sanity" in line:
                print("file end: ", log_path)
                break
    return

parser = argparse.ArgumentParser(
    description='Dispatch tasks of parsing the query and diffing.')
parser.add_argument('-partition_count', '--partition_count',
                    metavar='partition_count',
                    dest='partition_count',
                    required=True,
                    help='Specify the number of independent partitions we want to run.')
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
                    help='Specify the index of current job, which dispatcher I am')
parser.add_argument('-input_file', '--input_file',
                    metavar='input_file',
                    dest='input_file',
                    required=True,
                    help='Specify the log file to read')
parser.add_argument('-log_file', '--log_file',
                    metavar='log_file',
                    dest='log_file',
                    required=False,
                    help='Specify the path to log file of this job')
args = parser.parse_args()

args.partition_count = int(args.partition_count)

print("[%s] \nArguments" % os.path.basename(__file__))
for arg in args.__dict__:
    print("\t%s = %s" % (arg, args.__dict__[arg]))

log_dir_path = os.path.dirname(args.diff_sass_db_file)

# create an empty persistent fb if not exists
try:
    create_persistent_db_partition(args.persistent_db_path)
except:
    print("Unable to create persistent db partitions on %s" % (args.persistent_db_path))

# get parent dir (cudnn/scripts) path
cudnn_scripts_path = os.path.abspath(os.path.dirname(__file__))


# ***********************************************
# checkpointing: 
#   if log_file is provided, we will read log_file backwards
#   until a line_no is found or reaching the beginning of the file
# ***********************************************
start_line_no = 0
pattern_line_no = re.compile(r'^------- completed diffing line_no ([\d]+) -------$')
if args.log_file != None:
    if os.path.exists(args.log_file):
        with open(args.log_file, "r") as log:
            for line in reverse_readline(log):
                line = line.strip()

                # stop reading when we find `Restarts = 0`
                # we do this because the slurm log write in append mode, 
                # there could be old logs in the same log file
                if "Restarts = 0" == line:
                    break

                # also stop when the the diffing already completed
                if "------- completed diffing all -------" == line:
                    start_line_no = -1
                    break
                if pattern_line_no.match(line):
                    start_line_no = int(pattern_line_no.match(line).groups()[0])+1
                    break
    else:
        raise RuntimeError("checkpointing log file %s not found!" % (args.log_file))
print("Start reading from line_no %s" % (start_line_no))


# ***********************************************
# we launch `partition_count' partitions to parse the backend query results read from cudnn_run.py log
# ***********************************************

# read from input file
timeout = 600
start_time = time.time()
while not os.path.exists(args.input_file):
    time.sleep(5)
    print("waiting for input at ", args.input_file)
    if (time.time() - start_time) > timeout:
        raise RuntimeError("No input file generated at %s" % args.input_file)
print("found input at ", args.input_file)

# the tasks (queries) are accumulated to batches of `task_per_worker` * `partition_count`
# then dispatched to the workers
task_count = 0
task_list = []
task_per_worker = 10

# parser to parse cudnnTest outputs
query_parser = LogParser(LogExtractor.getExtractorList())
query_time = 0

for task_content, task_len, line_no in gen_query_task_from_log(args.input_file, start_line_no):
    task_count += 1
    if task_count % 100 == 0:
        print("[%s] got %d tasks from %s" % (print_used_mem(), task_count, args.input_file))
    task_list.append([task_content, task_len])

    # the tasks are dispatched to workers in batches
    if task_count % (task_per_worker * args.partition_count) == 0:

        # sort the tasks based on size from large to small -> balance the load of workers
        task_list.sort(key=lambda task: task[1], reverse=True)
        for partition_idx in range(args.partition_count):
            count = 0
            for i in range(partition_idx, len(task_list), args.partition_count):
                count += task_list[i][1]
            print("(%s) total task query len: %s" % (partition_idx, count))
        task_list = [task[0] for task in task_list]

        # launch `partition_count` worker instaces in parallel
        print("Launching %d parallel binary_gen_worker.py for %d tasks" %
              (args.partition_count, len(task_list)))
        common_args_list = ["-diff_sass_db_file"]
        common_args_list += [args.diff_sass_db_file]
        common_args_list += ["-persistent_db_path"]
        common_args_list += [args.persistent_db_path]
        common_args_list += ["-old_api_db_file"]
        common_args_list += [args.old_api_db_file]
        common_args_list += ["-job_idx"]
        common_args_list += [args.job_idx]
        try:
            map_partition(args.partition_count, "./persistent_perf/binary_gen_worker_txt.py",
                        "-args_file_path", task_list, common_args_list, unique_str=("job"+args.job_idx+"_"), args_seperator="\n\n\n")
        except:
            print("Failed to parse the query results and diff")
        task_list = []
        sys.stdout.flush()
        print("------- completed diffing line_no %s -------" % (line_no))

# handle last batch of tasks which may not be a full batch
if len(task_list) > 0:
    task_list.sort(key=lambda task: task[1], reverse=True)
    for partition_idx in range(args.partition_count):
        count = 0
        for i in range(partition_idx, len(task_list), args.partition_count):
            count += task_list[i][1]
        print("(%s) total task query len: %s" % (partition_idx, count))
    task_list = [task[0] for task in task_list]

    print("Launching %d parallel binary_gen_worker.py for %d tasks" %
          (args.partition_count, len(task_list)))
    common_args_list = ["-diff_sass_db_file"]
    common_args_list += [args.diff_sass_db_file]
    common_args_list += ["-persistent_db_path"]
    common_args_list += [args.persistent_db_path]
    common_args_list += ["-old_api_db_file"]
    common_args_list += [args.old_api_db_file]
    common_args_list += ["-job_idx"]
    common_args_list += [args.job_idx]
    try:
        map_partition(args.partition_count, "./persistent_perf/binary_gen_worker_txt.py",
                    "-args_file_path", task_list, common_args_list, unique_str=("job"+args.job_idx+"_"), args_seperator="\n\n\n")
    except:
        print("Failed to parse the query results and diff")
    sys.stdout.flush()
    print("------- completed diffing all -------")

print("[%s] total task parsed: %s" % (os.path.basename(__file__), task_count))
print("[%s] total persistent db query time: %s" % (os.path.basename(__file__), query_time))


# ***********************************************
# merge new_api_db and pre binary files
# ***********************************************

# merge new_api_db pairwise
start_time = time.time()
merge_tasks_count = args.partition_count
while merge_tasks_count > 1:

    # prepare pairs of merge db paths
    db_path_pairs = []
    for i in range(0, merge_tasks_count // 2):
        to_merge_1 = os.path.join(log_dir_path, "new_api_db_job" + str(args.job_idx) + "_part" + str(i) + ".sqlite3")
        if merge_tasks_count % 2 == 0:
            to_merge_2 = os.path.join(log_dir_path, "new_api_db_job" + str(
                args.job_idx) + "_part" + str(i+(merge_tasks_count//2)) + ".sqlite3")
        else:
            to_merge_2 = os.path.join(log_dir_path, "new_api_db_job" + str(
                args.job_idx) + "_part" + str(i+(merge_tasks_count//2)+1) + ".sqlite3")
        db_path_pairs.append(to_merge_1 + "," + to_merge_2)

    common_args_list = []
    try:
        map_partition(merge_tasks_count // 2, "./persistent_perf/merge_shelve_worker.py",
                    "-args_file_path", db_path_pairs, common_args_list, unique_str=("job"+args.job_idx+"_"), args_seperator="\n")
    except:
        print("Failed to merge api db")
    
    print("merge workers: ", merge_tasks_count // 2)

    if merge_tasks_count % 2 == 0:
        merge_tasks_count //= 2
    else:
        merge_tasks_count = (merge_tasks_count // 2) + 1

# rename final merged shelve file
old_partition_0_db_path = os.path.join(log_dir_path, "new_api_db_job" + str(args.job_idx) + "_part0.sqlite3")
new_api_db_path = os.path.join(log_dir_path, "new_api_db_job" + str(args.job_idx) + ".sqlite3")
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
merge_tasks_count = args.partition_count
while merge_tasks_count > 1:
    
    # prepare pairs of merge db paths
    binary_path_pairs = []
    for i in range(0, merge_tasks_count // 2):
        to_merge_1 = os.path.join(log_dir_path, "main_job" + str(args.job_idx) + "_part" + str(i) + ".txt")
        if merge_tasks_count % 2 == 0:
            to_merge_2 = os.path.join(log_dir_path, "main_job" + str(
                args.job_idx) + "_part" + str(i+(merge_tasks_count//2)) + ".txt")
        else:
            to_merge_2 = os.path.join(log_dir_path, "main_job" + str(
                args.job_idx) + "_part" + str(i+(merge_tasks_count//2)+1) + ".txt")
        binary_path_pairs.append(to_merge_1 + "," + to_merge_2)

    common_args_list = []
    try:
        map_partition(merge_tasks_count // 2, "./persistent_perf/merge_pre_binary_worker.py",
                    "-args_file_path", binary_path_pairs, common_args_list, unique_str=("job"+args.job_idx+"_"), args_seperator="\n")
    except:
        print("Failed to merge the main txt file")

    print("merge workers: ", merge_tasks_count // 2)

    if merge_tasks_count % 2 == 0:
        merge_tasks_count //= 2
    else:
        merge_tasks_count = (merge_tasks_count // 2) + 1

# rename final merged binary file
old_partition_0_binary_path = os.path.join(log_dir_path, "main_job" + str(args.job_idx) + "_part0.txt")
new_binary_path = os.path.join(log_dir_path, "main_job" + str(args.job_idx) + ".txt")
if os.path.exists(new_binary_path):
    os.remove(new_binary_path)
if os.path.exists(old_partition_0_binary_path):
    os.rename(old_partition_0_binary_path, new_binary_path)
    print("renamed %s to %s" % (old_partition_0_binary_path, new_binary_path))
else:
    raise RuntimeError("pre binary partition 0 does not exist on %s" % (old_partition_0_binary_path))

print("[%s] merge binary total time: %s seconds" % (os.path.basename(__file__), time.time() - start_time))

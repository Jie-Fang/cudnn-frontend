#!/usr/bin/env python

from __future__ import print_function
import argparse
import os
import re
import multiprocessing
import signal
import sys
import random
import subprocess
import time
import json

try:
    from StringIO import StringIO  # for Python 2
except ImportError:
    from io import StringIO  # for Python 3
import platform

from collections import deque
from copy import deepcopy
from datetime import datetime
from threading import Thread, Lock, Condition, Event, current_thread

from helpers.sys_config import augment_lib, print_lib
from helpers.sys_config import get_gpu_info, print_general_info, get_gpu_filter, get_nvidia_smi_cmd
from helpers.sys_config import redirect_glibc_backtraces, enable_logging
from helpers.label_parser import get_labels_from_file
from helpers.layer_parser import gen_layers_from_file, DuplicateFlagDetector
from helpers.binary_parser import gen_layers_from_binary_file, get_next_group_idx_from_log
from helpers.cudnn_interface import RunCache, LogExtractor, open_or_stdout
from helpers.utility import split_comma, flags_from_descs_str, get_modified_layers, modify_layer_backendQuery, split_space, split_and_strip
from helpers.parser_utility import LogFlagReader, LogParser, use_parser_on_output
from helpers.utility_py3 import *
from helpers.Logger import Logger
if not sys.platform.lower().startswith('qnx'):
    from helpers.SQLDB import SQLDB
    from persistent_perf.heur_diff_utility import get_last_cl, get_indexer_from_new_api_db, add_persistent_db_additional_content, add_persistent_db_additional_elem
from cudnnLogToTestCmd import logToTest_generate, logToTest_checkFlagSupport, logToTest_compareExecutionPath, stripLog
from helpers.cutlass_interface import cutlass_output_parse

# Note: this file is automatically formatted using autopep8==1.5.7


# *******************************************************************************
# * Argument parsing logic
# *******************************************************************************
def make_help(s, has_choices=False):
    result = s + " [default: %(default)s]"
    if (has_choices):
        result += " [choices: %(choices)s]"
    return result


return_modes = ('DEFAULT', 'FAILED', 'NONE')  # first item is default value
# known incompatible routines with -gpuRef
gpuref_incompatible = ('activationb', 'activationf')
# Note: all_dynamic_sublib must be in topological order. e.g. since _train sub-libraries include
#       their _infer counterparts, _train sub-libraries must come after their _infer counterparts.
#       cudnnTest_cnn_train is a superset of cudnnTest_ops_train, so cudnnTest_cnn_train must
#       come after cudnnTest_ops_train.
all_dynamic_sublib = [
    "cudnnTest_ops",
    "cudnnTest_cnn",
    "cudnnTest_adv",
]
all_static_sublib = [sublib + "_static" for sublib in all_dynamic_sublib]
all_test_list = []

# Get all arguments available in both composite file and CLI


def add_comp_args(arg_parser):
    arg_parser.add_argument(
        '-composite_file',
        '--composite-file',
        metavar='"str"',
        dest='composite_path',
        default=None,
        help=make_help(
            "Specify composite file (path to composite definitions); NOTE: Cannot be used in combination with label/layer flags"))

    arg_parser.add_argument(
        '-include_flags',
        '--include-flags',
        metavar='"str"',
        dest='include_flags_str',
        default=None,
        help=make_help(
            "Only allow cases within the given flags (example: \"-include_flags R: conv\""))

    arg_parser.add_argument(
        '-include_layer_name',
        '--include-layer-name',
        metavar='"str"',
        dest='include_layer_name',
        action='append',
        help=make_help(
            "Only allow cases where layer name matches the given regex; multiple instance are treated as OR"))

    arg_parser.add_argument(
        "-exclude_layer_name",
        "--exclude-layer-name",
        metavar='"str"',
        dest="exclude_layer_name",
        action="append",
        help="Skip cases where layer name matches the given regex; multiple instance are treated as AND")

    arg_parser.add_argument(
        '-exclude_flags',
        '--exclude-flags',
        metavar='FLAGS',
        dest='exclude_flags_str',
        default=None,
        help=make_help(
            "Specifically exclude case with give flags (example: \"--exclude-flags 'R:conv\""))

    arg_parser.add_argument(
        '-layer_file',
        '--layer-file',
        metavar='"path"',
        dest='layers_path',
        default=None,
        help=make_help(
            "Specify layers file (path to layer_definitions); defaults to \"default.layer\""))

    arg_parser.add_argument(
        '-label_file',
        '--label-file',
        metavar='"path"',
        dest='labels_path',
        default=None,
        help=make_help(
            "Specify labels file (path to layer_labels), can be \"None\" if no label is needed"))

    arg_parser.add_argument(
        '-binary_file',
        '--binary-file',
        metavar='"path"',
        dest='binary_path',
        default=None,
        help=make_help(
            "Specify binary file (path to binary), can be \"None\" if no binary is needed"))

    arg_parser.add_argument(
        '-global_flags',
        '--global-flags',
        '-g',
        metavar='"str"',
        dest='global_flags_str',
        default=None,
        help=make_help(
            "Specify flags to set for all layers (example: -global_flags \"Pin:h * Pcomp:h,s * Pout:h\""))

    arg_parser.add_argument(
        '-config',
        '--config',
        '-C',
        metavar='"str"',
        dest='filter_config_str',
        default=None,
        help=make_help(
            "Specify filters defining current system/test configuration"))

    arg_parser.add_argument(
        '-rand_sample_per_layer',
        '--rand-sample',
        metavar='N',
        dest='rand_sample_per_layer',
        type=int,
        default=None,
        help=make_help("Randomly sample count per layer"))


def cudnn_run_argparser():
    '''Parse command line argument for cudnn_run.py'''
    def arg_format(prog): return argparse.HelpFormatter(
        prog, max_help_position=100, width=100)

    parser = argparse.ArgumentParser(
        description='cuDNN layer tests', formatter_class=arg_format)

    parser._optionals.title = "Help Options"

    # General Options
    gen_args = parser.add_argument_group('General Options')

    gen_args.add_argument(
        '-dryrun',
        '--dryrun',
        '-n',
        '-n',
        action='store_const',
        const=True,
        default=False,
        help=make_help("Only print test names; do not execute"))

    gen_args.add_argument(
        '-testsList_batch_size',
        '--testslist-batch-size',
        metavar='N',
        dest='testsList_batch_size',
        type=int,
        default=1,
        help=make_help(
            "The number of tests to write together using -testsList (0 for auto batch size, 1 for no batching)"))

    gen_args.add_argument(
        '-batch_method',
        '--batch-method',
        metavar='N',
        type=int,
        dest='batch_method',
        default=0,
        help=make_help(
            "Specify batching method (0 for non-Rheur, 1 for RheurBackend, 2 for Rheur)"))

    gen_args.add_argument(
        '-threads',
        '--threads',
        metavar='N',
        type=int,
        default=1,
        help=make_help(
            "Specify the number of threads to run tests concurrently (0 for auto thread count)"))

    gen_args.add_argument(
        '-no_rerun',
        '--no-rerun',
        action='store_const',
        const=True,
        default=False,
        help=make_help("Don't rerun waived and failed tests in multithreaded or gpuRef runs and force print instead"))

    gen_args.add_argument(
        '-use_waive_lists',
        '--use-waive-lists',
        action='store_const',
        const=True,
        default=False,
        help=make_help("Enable the waived cases according to the labels of waive list"))

    gen_args.add_argument(
        '-check_waive_lists',
        '--check-waive-lists',
        action='store_const',
        const=True,
        default=False,
        help=make_help("Comparing the waive list with all level test cases, help to cleanup the waive list"))

    gen_args.add_argument(
        '-output_buffer_mult',
        '--output-buffer-mult',
        type=float,
        default=-1,
        help=make_help(
            "The multiple for output buffer in multithreaded mode. output_buffer_mult * threads " +
            "specifies the number of batches executed before outputting to console. " +
            "-1 (default) denotes fully buffered (i.e. output upon completed multithreaded execution). " +
            "Note that small multiples cause noticeable performance hit, as such the multiple " +
            "should be tuned based on time or memory constraints (perhaps start with 3 or so)."))

    gen_args.add_argument(
        '-return_mode',
        '--return-mode',
        default=return_modes[0],
        choices=(x.lower() for x in return_modes),
        help=make_help(
            "Specify batching return method (0 return on waived and failed, 1 return on failed, 2 no return)"))

    gen_args.add_argument(
        '-dump_to_log',
        '--dump_to_log',
        metavar='"str"',
        dest='dump_to_log_str',
        default=None,
        help=make_help("Specify log file"))

    gen_args.add_argument(
        '-checkpoint_log',
        '--checkpoint_log',
        action='store_const',
        const=True,
        default=False,
        help=make_help(
            "Re-read -dump-to-log to not re-run any cases previously run"))

    gen_args.add_argument(
        '-only_dump_to_log',
        '--only_dump_to_log',
        action='store_const',
        const=True,
        default=False,
        help=make_help(
            "If dump_to_log is enabled, disable printing to stdout/stderr."))

    gen_args.add_argument(
        '-device',
        '--device',
        '-d',
        metavar='n',
        dest='device',
        default='0',
        help=make_help(
            "Specify device indexes ('-d' flag), comma delimited, valid inputs: -d 0,1; -d all; -d 0"))

    gen_args.add_argument(
        '-sweep_heurgen',
        '--sweep-heurgen',
        action='store_const',
        const=True,
        default=False,
        help=make_help("Use backendQuery to sweep across all possible values"))

    gen_args.add_argument(
        '-skip_duplicates',
        '--skip-duplicates',
        action='store_const',
        const=True,
        default=False,
        help=make_help("Skip duplicate flags"))

    gen_args.add_argument(
        '-partition',
        '--partition',
        metavar="partIndex,partCount",
        dest='partition_str',
        default=None,
        help=make_help(
            "Partition layers and run only one partition (example: \"4,10\" will run partition #4 of 10 partitions of layers)"))

    gen_args.add_argument(
        '-API_log_test',
        '--API-log-test',
        action='store_const',
        const=True,
        default=False,
        help=make_help(
            "Enable execution comparison for API logging and test command generation, should run without randomization"))

    gen_args.add_argument(
        '-pre_flags',
        '--pre-flags',
        metavar='PREFLAGS',
        dest='pre_flags_str',
        default="",
        help=make_help("Specify pre_flags to place before cudnnTest"))

    gen_args.add_argument(
        '-default_ref',
        '--default-ref',
        action='store_true',
        help=make_help(
            "Use default reference defined in .layers if specified, else force all calls to include -gpuRef."))

    gen_args.add_argument(
        '-checkjit',
        '--checkjit',
        action='store_const',
        const=True,
        default=False,
        help=make_help(
            "check if JIT process is trigged, do not check the correctness of GPU results. Test fails only if JIT process is triggered."))

    gen_args.add_argument(
        '-randomize',
        '--randomize',
        action='store_const',
        const=True,
        default=False,
        help=make_help("Randomize order of tests"))

    gen_args.add_argument(
        '-insert_persistent_db_cols',
        '--insert_persistent_db_cols',
        action='store_const',
        const=True,
        default=False,
        help=make_help(
            "Add and insert persistent db specific cols, e.g., flag_idx, api_list, etc."))

    gen_args.add_argument(
        '-track_memory',
        '--track-memory',
        action='store_true',
        help=make_help("Track maximum memory usage when multithreading"))

    # layer generation arguments
    layer_gen_args = parser.add_argument_group('Layer Generation Options')

    add_comp_args(layer_gen_args)

    # Caching arguments
    cache_args = parser.add_argument_group('Caching Options')

    cache_args.add_argument(
        '-cache_path',
        metavar='"str"',
        dest='cache_path',
        default=None,
        help=make_help(
            "Specify cache path; default assumes no caching. A cache stores results of calls so that another run of cudnn_perf.py can reload the results."))

    cache_args.add_argument(
        '-cache_freq',
        metavar='N',
        dest='cache_freq',
        default=-1,
        type=int,
        help=make_help(
            "Specify interval to update cache; 2 means update after every other test call. The special case of -1 will only update cache after all calls are finished."))

    # Path arguments
    path_args = parser.add_argument_group('Path Options')

    path_args.add_argument(
        '-binpath',
        '--bin-path',
        metavar='"path"',
        dest='bin_path',
        default='./../../../bin/x86_64_Linux_release',
        help=make_help("Specify bin path (where binary is located)"))

    path_args.add_argument(
        '-bin_name',
        '--bin-name',
        metavar='"str"',
        dest='bin_name',
        default='cudnnTest',
        choices=[
            "cudnnTest", "cudnnTest_static",
            "cudnnTest_ops",
            "cudnnTest_cnn",
            "cudnnTest_adv",
            "cublasTest", "cutlass_profiler",
            "pycudnnTest.py",
        ],
        help=make_help("Specify which binary to run", True))

    path_args.add_argument(
        '-static_lib_prob',
        '--static-lib-prob',
        type=float,
        default=0.0,
        help=make_help("Probability of selecting cudnnTest_static for testing"))

    path_args.add_argument(
        '-sub_lib_prob',
        '--sub-lib-prob',
        type=float,
        default=0.0,
        help=make_help("Probability of selecting cudnnTest sub libraries for testing"))

    path_args.add_argument(
        '-dynamic_sub_lib_selection_mode',
        '--dynamic-sub-lib-selection-mode',
        type=str,
        default='uniform',
        choices=['host', 'uniform'],
        help=make_help(
            "Method to select the executing dynamic sub-library for routines that are included in multiple sub-libraries. " +
            "host selects the host sub-library, i.e. the sub-library that the routine is defined in, " +
            "with `host_prob` probability and evenly distributes (1 - `host_prob`) among the other sub-libraries; " +
            "uniform samples uniformly.", True))

    path_args.add_argument(
        '-static_sub_lib_selection_mode',
        '--static-sub-lib-selection-mode',
        type=str,
        default='host',
        choices=['host', 'uniform'],
        help=make_help("Same as -dynamic_sub_lib_selection_mode but for static sub-library", True))

    path_args.add_argument(
        '-host_prob',
        '--host-prob',
        type=float,
        default=1.0,
        help=make_help("Probability for -sub_lib_selection_mode host"))

    path_args.add_argument(
        '-random_bin_seed',
        '--random-bin-seed',
        type=int,
        default=None,
        help=make_help("Seed for random selection of static or sub libraries. None uses unix time."))

    path_args.add_argument(
        '-libpath',
        '--lib-path',
        '-L',
        metavar='"path"',
        dest='lib_path',
        default='./../../../bin/x86_64_Linux_release',
        help=make_help("Specify lib path (where libcudnn.so is located)"))

    path_args.add_argument(
        '--use-dir',
        action='store_true',
        default=False,
        help=make_help('whether to use the new labels/layers dirs hotness'))

    path_args.add_argument(
        '-sqlite_file',
        '--sqlite-file',
        metavar='"path"',
        dest='sqlite_path',
        default=None,
        help=make_help(
            "Specify file path to sqlite3 database to be generated. If checkpoint_log is not specified, this file will be reset on launch"))

    path_args.add_argument(
        '-new_api_db_path',
        '--new_api_db_path',
        metavar='"path"',
        dest='new_api_db_path',
        default=None,
        help=make_help("Specify file path to the new api sqlite db"))

    path_args.add_argument(
        '-explicit_waive_path',
        '--explicit-waive-path',
        metavar='"path"',
        dest='explicit_waive_path',
        default=None,
        help=make_help('Specify the explicit path to the waive files'))

    path_args.add_argument(
        '-io_timeout',
        '--io_timout',
        dest='io_timeout',
        type=int,
        default=600,
        help=make_help('Specify number of seconds before we log extended debugging information from a long running process'))

    path_args.add_argument(
        '-process_timeout',
        '--process_timeout',
        dest='process_timeout',
        type=int,
        default=3000,
        help=make_help('Specify number of seconds before we terminate a long running process'))

    default_debug_path = get_default_debug_path_str()
    path_args.add_argument(
        '-extended_debug_path',
        '--extended_debug_path',
        dest='extended_debug_path',
        default=default_debug_path,
        help=make_help('Specify location to log extended debugging information after an io_timeout. use \"stdout\" for stdout logging or empty string to disable. The default path is \'D:\tmp\' (Windows) or \'/data/tmp\', falling back to the current working directory or stdout as a last resort. Fallback if path does not exist and we do not have write access'))

    return parser

def get_default_debug_path_str():
    paths = [r'D:\tmp'] if sys.platform.startswith('win') else ['/data/tmp']
    paths.append(os.getcwd())

    for path in paths:
        if (os.path.exists(path) and os.access(path, os.W_OK)):
            return str(path)

    return 'stdout'

def get_available_mem_linux():
    """Use free command in linux to find the avail memory

    It is the last integer on second line of the output, in bytes.
    """
    check_mem = subprocess.Popen(['free','-b'], stdout=subprocess.PIPE)
    mem_str = check_mem.communicate()[0].decode()
    mem = int( mem_str.split('\n')[1].split()[-1] )
    return mem

def default_thread_count(gpu):
    """Determine the num of threads to use depends on the num of CPUs and GPU memory size
    """
    # With 4.0GiB of buffer, assume that average test takes 1.5GiB of GPU memory; can adjust in the future.
    print('cpu count: {}'.format(multiprocessing.cpu_count()))
    return min(multiprocessing.cpu_count(), max(1, (int(gpu.mem) - 4096) // 1536))


def default_batch_size(gpu, threads):
    # Inversely correlated with thread count for now; can consider OS/Arch-based value in the future.
    # Min 20 when using > 80 threads; max 100 when using single thread
    return 80 // threads + 20


def post_parse_args(parsedargs):
    '''Make some reasonble check of parsed arguments and throw exception if argument cannot be interpreted.'''
    layers_dir = labels_dir = scripts_dir = composite_dir = binary_dir = os.path.dirname(
        __file__)  # scripts_dir is where cudnn_run.py resides
    if parsedargs.use_dir:
        # Assume layer and label files are stored in scripts_dir/layers and scripts_dir/labels subdirectories, respectively
        layers_dir = os.path.join(scripts_dir, 'layers')
        labels_dir = os.path.join(scripts_dir, 'labels')

    if parsedargs.rand_sample_per_layer == None:
        parsedargs.rand_sample_per_layer = -1

    if parsedargs.binary_path != None:
        # only basename provided
        if os.path.basename(parsedargs.binary_path) == parsedargs.binary_path:
            test_path = os.path.join(binary_dir, parsedargs.binary_path)
            if os.path.isfile(test_path):
                parsedargs.binary_path = test_path
        # assert that the label file exists
        if not os.path.isfile(parsedargs.binary_path):
            raise IOError('binary file {} is not a valid path'.format(
                parsedargs.binary_path))

        # make sure `rand_sample_per_layer` is -1
        if parsedargs.rand_sample_per_layer != -1:
            raise ValueError(
                "[INVALID FLAGS] rand_sample_per_layer should be -1 when using binary file")

        # should not provide layer/label file or composite file
        if parsedargs.composite_path != None:
            raise ValueError(
                "[INVALID FLAGS] Composite file doesn't work with binary file")

        # should not provide layer/label file or composite file
        if parsedargs.layers_path != None or parsedargs.labels_path != None:
            raise ValueError(
                "[INVALID FLAGS] Layer/label file doesn't work with binary file")

    elif parsedargs.composite_path != None:
        # only basename provided
        if os.path.basename(parsedargs.composite_path) == parsedargs.composite_path:
            test_path = os.path.join(composite_dir, parsedargs.composite_path)
            if os.path.isfile(test_path):
                parsedargs.composite_path = test_path
        # assert that the label file exists
        if not os.path.isfile(parsedargs.composite_path):
            raise IOError('label file {} is not a valid path'.format(
                parsedargs.composite_path))

    else:
        if parsedargs.layers_path == None:
            parsedargs.layers_path = 'default.layer'
        if parsedargs.labels_path == None:
            parsedargs.labels_path = 'default.label'
        # Check --layer-file argument
        # only basename provided through argument
        if os.path.basename(parsedargs.layers_path) == parsedargs.layers_path:
            test_path = os.path.join(layers_dir, parsedargs.layers_path)
            if os.path.isfile(test_path):
                parsedargs.layers_path = test_path
        # assert that the layer file exists
        if not os.path.isfile(parsedargs.layers_path):
            raise IOError('layer file {} is not a valid path'.format(
                parsedargs.layers_path))

        # Check --label-file argument
        if parsedargs.labels_path != None:
            # only basename provided
            if os.path.basename(parsedargs.labels_path) == parsedargs.labels_path:
                test_path = os.path.join(labels_dir, parsedargs.labels_path)
                if os.path.isfile(test_path):
                    parsedargs.labels_path = test_path
            # assert that the label file exists
            if not os.path.isfile(parsedargs.labels_path):
                raise IOError('label file {} is not a valid path'.format(
                    parsedargs.labels_path))

    if parsedargs.sweep_heurgen:
        if parsedargs.dryrun:
            raise ValueError(
                "[INVALID FLAGS] Unable to run sweep_heurgen with dryrun")
        if parsedargs.batch_method != 1:
            raise ValueError(
                "[INVALID FLAGS] Unable to run sweep_heurgen without batch_method 1")
        if parsedargs.threads != 1:
            raise ValueError(
                "[NOT IMPLEMENTED] Unable to run sweep_heurgen without threads 1")

    if parsedargs.check_waive_lists and (not parsedargs.dryrun or "L4" not in parsedargs.filter_config_str):
        raise ValueError(
                "[INVALID FLAGS] Unable to run check_waive_lists without dryrun and L4 config")

    if parsedargs.batch_method and parsedargs.threads != 1:
        raise ValueError(
            "[INVALID FLAGS] Unable to run batch_method without threads 1")
    if parsedargs.API_log_test and parsedargs.threads != 1:
        raise ValueError(
            "[NOT IMPLEMENTED] Unable to run API_log_test without threads 1")
    if parsedargs.output_buffer_mult > 0 and parsedargs.threads == 1:
        raise ValueError(
            "[INVALID FLAGS] Unable to run output_buffer_mult with threads 1")
    if parsedargs.track_memory and parsedargs.threads == 1:
        raise ValueError(
            "[NOT IMPLEMENTED] Unable to run track_memory with threads 1")

    parsedargs.checkpointed_flags = None
    if parsedargs.dump_to_log_str:
        if parsedargs.checkpoint_log:
            parsedargs.checkpointed_flags = LogFlagReader(
                parsedargs.dump_to_log_str, False)
        else:
            if os.path.exists(parsedargs.dump_to_log_str):
                try:
                    os.remove(parsedargs.dump_to_log_str)
                except OSError as e:
                    print(("Unable to remove log file: %s - %s." %
                           (e.filename, e.strerror)))
        sys.stdout = Logger(parsedargs.dump_to_log_str,
                            parsedargs.checkpoint_log,
                            parsedargs.only_dump_to_log)
    else:
        sys.stdout = Logger(None, False, False)
    sys.stderr = sys.stdout

    parsedargs.return_mode = return_modes.index(parsedargs.return_mode.upper())

    # make sure new_api_db_path is provided when enabling insert_persistent_db_cols option
    if parsedargs.insert_persistent_db_cols:
        if parsedargs.new_api_db_path == None:
            raise ValueError(
                "[INVALID FLAGS] Unable to enable insert_persistent_db_cols without new_api_db_path specified")

    if parsedargs.bin_name != 'cudnnTest' and (parsedargs.static_lib_prob or parsedargs.sub_lib_prob):
        raise ValueError(
            "[INVALID FLAGS] Must run with bin_name=cudnnTest if enabling random static or sub-library testing")
    if (parsedargs.static_lib_prob or parsedargs.sub_lib_prob) and parsedargs.random_bin_seed is None:
        parsedargs.random_bin_seed = int(time.time())

    return parsedargs


# *******************************************************************************
# * Setup LD_LIBRARY_PATH (os-agnostic)
# *******************************************************************************
def lib_path_setup(parsed_args):
    # Add lib_path to OS-dependant library path
    if (parsed_args.lib_path != ''):
        augment_lib(parsed_args.lib_path)
    # Print current library info
    print("\n\nPrinting current LIBRARY path")
    print_lib()
    print("")
    # Redict glibc backtrace to retrieve errors from test calls
    redirect_glibc_backtraces()


# *******************************************************************************
# * Initiliaze sqlite DB & parser
# *******************************************************************************
def sqlite_setup(parsed_args):
    """set up SQLite: if persistent db is used, create/read idx_table and populate the
    flag indexer. Then if job SQLite is used, setup the log parser

    Returns:
        4-tuple: SQLDB object, LogParser object, FlagApiIndexer object, and a
        string of the last CL number. Each element can be None if not available.
    """
    sqlite_db = None
    sqlite_parser = None
    indexer = None
    last_cl = None

    if sys.platform.lower().startswith('qnx') and (parsed_args.insert_persistent_db_cols or parsed_args.sqlite_path != None or parsed_args.new_api_db_path != None):
        raise ValueError(
            "On QNX system, the following flags are not supported: sqlite_path, insert_persistent_db_cols, new_api_db_path")

    if parsed_args.insert_persistent_db_cols:
        # load indexer from new api db
        if parsed_args.new_api_db_path != None:
            indexer = get_indexer_from_new_api_db(parsed_args.new_api_db_path)
        # get last cl from file
        last_cl = get_last_cl(parsed_args.bin_path)
        if parsed_args.sqlite_path != None:
            sqlite_parser = LogParser(LogExtractor.getExtractorList())
            elem_names, type_names = add_persistent_db_additional_elem(
                sqlite_parser.get_compact_names(), sqlite_parser.get_type_names())
            sqlite_db = SQLDB(parsed_args.sqlite_path,
                              True, elem_names, type_names)
            if not parsed_args.checkpoint_log:
                sqlite_db.reset()
    else:
        if parsed_args.sqlite_path != None:
            sqlite_parser = LogParser(LogExtractor.getExtractorList())
            sqlite_db = SQLDB(parsed_args.sqlite_path, True,
                              sqlite_parser.get_compact_names(),
                              sqlite_parser.get_type_names())
            if not parsed_args.checkpoint_log:
                sqlite_db.reset()
    return sqlite_db, sqlite_parser, indexer, last_cl


# *******************************************************************************
# * GPU Detection
# *******************************************************************************
def detect_gpu(parsed_args, sqlite_db, sqlite_parser):
    """Identify the local GPU model

    Returns:
        3-tuple: A string of the GPU, a dict of the GPU attributes, and a string
        of the cudart version
    """
    gpu = None
    if parsed_args.bin_name != 'cutlass_profiler':
        if sqlite_db:
            sys.stdout.start_intercept()
        gpu = get_gpu_info(parsed_args.device,
                           parsed_args.bin_path, parsed_args.bin_name)
        if sqlite_db:
            all_parsed = use_parser_on_output(
                sys.stdout.end_intercept(), sqlite_parser)
            sqlite_db.add(all_parsed)
            sqlite_db.commit()

    print("")

    if sqlite_db:
        sys.stdout.start_intercept()
    gpu_test_res, id2gpu, cudart = print_general_info(parsed_args.bin_path,
                                                      parsed_args.bin_name)
    if sqlite_db:
        all_parsed = use_parser_on_output(sys.stdout.end_intercept(),
                                          sqlite_parser)

        sqlite_db.add(all_parsed)
        sqlite_db.commit()

    if parsed_args.device == 'all':
        parsed_args.device = ','.join([str(i) for i in xrange(len(id2gpu))])
        print("Running on all available devices:\n%s\n" % ('\n'.join(
            [d + ':' + id2gpu[d] for d in id2gpu])))
    else:
        for d in parsed_args.device.split(','):
            try:
                int(d)
            except ValueError:
                raise Exception("Invalid device id")
    return gpu, gpu_test_res, cudart


# *******************************************************************************
# * Parse Partition Flags
# *******************************************************************************
def get_partition_info(parsed_args):
    """Parse the partition provided in command line as a string "M,N", which
    is to mean part M of N partitions, with 0 <= M < N

    Returns:
        A tuple of two integers, (M, N)
    """
    partition_index = 0
    partition_count = 1
    if (parsed_args.partition_str):
        partition_config = parsed_args.partition_str.split(',')
        if len(partition_config) != 2:
            raise Exception("[PARTITION CONFIG]: Invalid config %s from \"%s\"" %
                            (partition_config, parsed_args.partition_str))
        partition_index = int(partition_config[0])
        partition_count = int(partition_config[1])
        if partition_count < 1:
            raise Exception(
                "[PARTITION CONFIG]: Invalid partition count (less than 1)")
        if partition_index < 0:
            raise Exception(
                "[PARTITION CONFIG]: Invalid partition index (less than 0)")
        if partition_index >= partition_count:
            raise Exception(
                "[PARTITION CONFIG]: Invalid partition index (greater than or equal to count)")
    return partition_index, partition_count


# *******************************************************************************
# * Layer Generation
# *******************************************************************************
def gen_layers_from_label_layer(
        layers_file_path,
        labels_file_path,
        filter_config,
        include_flags_str,
        include_layer_name,
        exclude_flags_str,
        exclude_layer_name,
        global_flags_str,
        dup_detector,
        rand_sample_per_layer):
    """Generate the label-substituted layer definitions
    """
    # Get labels
    if labels_file_path != "none":
        labels = get_labels_from_file(labels_file_path, filter_config)
    else:
        labels = None
    # Global flags (will take priority over layer/label definitions)
    global_flags_list = flags_from_descs_str(global_flags_str, labels)
    # Flags to include
    include_flags_list = flags_from_descs_str(include_flags_str, labels)
    # Flags to exclude
    exclude_flags_list = flags_from_descs_str(exclude_flags_str, labels)
    # Get layers
    return gen_layers_from_file(layers_file_path, include_flags_list, include_layer_name,
                                exclude_flags_list, exclude_layer_name, global_flags_list, labels,
                                dup_detector, rand_sample_per_layer)


def partition_and_randomize_layers(layer_gen, partition_index, partition_count, randomize):
    """Yield a subset of layer definitions based on the partition configuration.

    The layers may be shuffled with a fixed random seed for consistency across different
    partitions. Layers are assigned to partitions in a round-robin fashion.
    """
    LAYER_BUFFER_SIZE = max(100000, partition_count)

    done = False
    while not done:
        # Assemble buffer of layers
        layer_buffer = []
        try:
            while len(layer_buffer) < LAYER_BUFFER_SIZE:
                layer_buffer.append(next(layer_gen))
        except StopIteration:
            done = True
        # Shuffle buffer if randomize on
        if randomize:
            random.seed(123)
            for i in xrange(100):
                random.shuffle(layer_buffer)
        # Return portion of buffer
        for idx in xrange(partition_index, len(layer_buffer), partition_count):
            yield layer_buffer[idx]


def gen_partitioned_random_layers_from_label_layer(
        global_filter_config, layers_file_path, labels_file_path, include_flags_str,
        include_layer_name, exclude_flags_str, exclude_layer_name, global_flags_str,
        skip_duplicates, partition_index, partition_count, randomize, rand_sample_per_layer):
    """Read the layer file from disk and resolve labels, then return the subset
    of layer definitions according to the partition config.
    """
    dup_detector = None
    if skip_duplicates:
        dup_detector = DuplicateFlagDetector()
    layers = gen_layers_from_label_layer(
        layers_file_path,
        labels_file_path,
        global_filter_config,
        include_flags_str,
        include_layer_name,
        exclude_flags_str,
        exclude_layer_name,
        global_flags_str,
        dup_detector,
        rand_sample_per_layer)
    return partition_and_randomize_layers(layers, partition_index,
                                          partition_count, randomize)


def merge_comp_cli_flag(comp_flag_val, cli_flag_val, flag_name,
                        cur_line_debug_str):
    """Return either comp_flag_val or cli_flag_val depends on which one is defined.
    """
    if comp_flag_val != None and cli_flag_val != None:
        raise IOError(
            "[COMPOSITE_PARSER] Unable to provide CLI include_flags_str along with composite include_flags_str (%s)"
            % cur_line_debug_str)
    if comp_flag_val != None:
        return comp_flag_val
    return cli_flag_val


def get_and_check_path(abs_or_rel_path, rel_root, debug_str):
    """Return a well-constructed path of abs_or_rel_path relative to rel_root
    """
    path = abs_or_rel_path
    if not os.path.isabs(abs_or_rel_path):
        path = rel_root + "/" + abs_or_rel_path
    if not os.path.isfile(path):
        raise IOError(
            'ERORR: Path \"%s\" does not exist (%s)' % (path, debug_str))
    return path


def gen_layers_from_composite(global_filter_config, composite_file_path, include_flags_str,
                              include_layer_name, exclude_flags_str, exclude_layer_name,
                              global_flags_str, dup_detector, rand_sample_per_layer):
    """Read composite file and in turn, the specified layer and label files. Then
    yield one layer at a time.

    The actual layer is resolved using gen_layers_from_label_layer(), but a
    composite file can refer to another composite file. In that case, this
    gen_layers_from_composite() function may be called recursively.
    """
    # Create ArgumentParser object for composite/label/layer file arguments
    # Then use the parser to read the content of composite file, one line at a time
    comp_arg_parser = argparse.ArgumentParser()
    add_comp_args(comp_arg_parser)
    with open(composite_file_path, 'r') as composite_file:
        comp_dir = os.path.dirname(os.path.abspath(composite_file_path))
        abs_comp_path = os.path.abspath(composite_file_path)

        for line_idx, line in enumerate(composite_file):
            cur_line_debug_str = "%s:%d" % (abs_comp_path, line_idx)
            comp_args = comp_arg_parser.parse_args(split_space(line))
            # Merge include_flag_str
            cur_include_flag_str = merge_comp_cli_flag(
                comp_args.include_flags_str, include_flags_str,
                "include_flags_str", cur_line_debug_str)
            # Merge include_layer_name
            cur_include_layer_name = merge_comp_cli_flag(
                comp_args.include_layer_name, include_layer_name,
                "include_layer_name", cur_line_debug_str)
            # Merge exclude_flags_str
            cur_exclude_flag_str = merge_comp_cli_flag(
                comp_args.exclude_flags_str, exclude_flags_str,
                "exclude_flags_str", cur_line_debug_str)
            # Merge exclude_layer_name
            cur_exclude_layer_name = merge_comp_cli_flag(
                comp_args.exclude_layer_name, exclude_layer_name,
                "exclude_layer_name", cur_line_debug_str)
            # Merge global_flags_str
            cur_global_flags_str = comp_args.global_flags_str
            if cur_global_flags_str == None:
                cur_global_flags_str = global_flags_str
            else:
                if global_flags_str != None:
                    cur_global_flags_str = cur_global_flags_str + " * " + global_flags_str

            cur_filter_config = global_filter_config
            if comp_args.filter_config_str != None:
                cur_filter_config = cur_filter_config + split_and_strip(
                    comp_args.filter_config_str, ',')
            cur_rand_sample_per_layer = rand_sample_per_layer
            if comp_args.rand_sample_per_layer != None:
                if cur_rand_sample_per_layer == -1:
                    cur_rand_sample_per_layer = comp_args.rand_sample_per_layer
                else:
                    cur_rand_sample_per_layer = min(
                        cur_rand_sample_per_layer,
                        comp_args.rand_sample_per_layer)

            if comp_args.composite_path != None:
                if comp_args.layers_path != None:
                    raise IOError(
                        'Line in comp file with both "-composite_file" and "-layer_file" (%s)'
                        % cur_line_debug_str)
                if comp_args.labels_path != None:
                    raise IOError(
                        'Line in comp file with both "-composite_file" and "-label_file" (%s)'
                        % cur_line_debug_str)
                sub_comp_path = get_and_check_path(
                    comp_args.composite_path, comp_dir, cur_line_debug_str)
                for layer in gen_layers_from_composite(
                        global_filter_config, sub_comp_path, cur_include_flag_str,
                        cur_include_layer_name, cur_exclude_flag_str, cur_exclude_layer_name,
                        cur_global_flags_str, dup_detector, cur_rand_sample_per_layer):
                    yield layer
                continue

            if comp_args.layers_path == None:
                raise IOError('Line in comp file missing layer path (%s)' %
                              cur_line_debug_str)
            cur_layers_path = get_and_check_path(comp_args.layers_path,
                                                 comp_dir, cur_line_debug_str)
            if comp_args.labels_path == None:
                raise IOError('Line in comp file missing layer path (%s)' %
                              cur_line_debug_str)
            cur_labels_path = get_and_check_path(comp_args.labels_path,
                                                 comp_dir, cur_line_debug_str)
            for layer in gen_layers_from_label_layer(
                    cur_layers_path, cur_labels_path, cur_filter_config, cur_include_flag_str,
                    cur_include_layer_name, cur_exclude_flag_str, cur_exclude_layer_name,
                    cur_global_flags_str, dup_detector, cur_rand_sample_per_layer):
                yield layer


def gen_partitioned_random_layers_from_composite(
        global_filter_config, composite_file_path, include_flags_str, include_layer_name,
        exclude_flags_str, exclude_layer_name, global_flags_str, skip_duplicates, partition_index,
        partition_count, randomize, rand_sample_per_layer):
    """Yield a shuffled and partitioned layer from a composite file.

    This combines the gen_layers_from_composite() to decode a composite file
    and partition_and_randomize_layers() to shuffle and partition.
    """
    dup_detector = None
    if skip_duplicates:
        dup_detector = DuplicateFlagDetector()
    layers = gen_layers_from_composite(global_filter_config, composite_file_path,
                                       include_flags_str, include_layer_name, exclude_flags_str,
                                       exclude_layer_name, global_flags_str, dup_detector,
                                       rand_sample_per_layer)
    return partition_and_randomize_layers(layers, partition_index,
                                          partition_count, randomize)


def gen_partitioned_random_layers_from_binary(
        binary_file_path, global_flags_str, skip_duplicates,
        dump_to_log_str, checkpoint_log,
        partition_index, partition_count, randomize):
    """Layers from a binary file. Otherwise similar to
    gen_partitioned_random_layers_from_label_layer()
    """
    # where should we start to read and run tests
    checkpointed_restart_group_idx = 0
    if dump_to_log_str and checkpoint_log:
        checkpointed_restart_group_idx = get_next_group_idx_from_log(
            dump_to_log_str, partition_count)
    dup_detector = None
    if skip_duplicates:
        dup_detector = DuplicateFlagDetector()
    # Global flags (will take priority over layer/label definitions)
    global_flags_list = flags_from_descs_str(global_flags_str)
    layers = gen_layers_from_binary_file(
        binary_file_path, checkpointed_restart_group_idx, global_flags_list,
        dup_detector, partition_index, partition_count, randomize)
    return layers


def get_layer_generator(parsed_args, global_filter_config, partition_index, partition_count):
    """Based on what is provided from the command line as captured in parsed_args,
    call the appropriate function to yield the layers from composite file, layer
    file, or a binary file.
    """
    if parsed_args.binary_path != None:
        return gen_partitioned_random_layers_from_binary(
            parsed_args.binary_path,
            parsed_args.global_flags_str,
            parsed_args.skip_duplicates,
            parsed_args.dump_to_log_str,
            parsed_args.checkpoint_log,
            partition_index,
            partition_count,
            parsed_args.randomize)
    elif parsed_args.composite_path == None:
        return gen_partitioned_random_layers_from_label_layer(
            global_filter_config,
            parsed_args.layers_path,
            parsed_args.labels_path,
            parsed_args.include_flags_str,
            parsed_args.include_layer_name,
            parsed_args.exclude_flags_str,
            parsed_args.exclude_layer_name,
            parsed_args.global_flags_str,
            parsed_args.skip_duplicates,
            partition_index,
            partition_count,
            parsed_args.randomize,
            parsed_args.rand_sample_per_layer)
    elif parsed_args.composite_path != None:
        return gen_partitioned_random_layers_from_composite(
            global_filter_config,
            parsed_args.composite_path,
            parsed_args.include_flags_str,
            parsed_args.include_layer_name,
            parsed_args.exclude_flags_str,
            parsed_args.exclude_layer_name,
            parsed_args.global_flags_str,
            parsed_args.skip_duplicates,
            partition_index,
            partition_count,
            parsed_args.randomize,
            parsed_args.rand_sample_per_layer)


# *******************************************************************************
# * Routine identification
# *******************************************************************************
def get_routines(executable):
    """Run `cudnnTest -help' and capture the stdout to find the list of
    available routines
    """
    # Get a list of available routines in an executable
    #
    # Purposely not protecting this Popen with a try/catch block.
    # If this Popen fails, it is fatal and should interrupt cudnn_run.py
    # In the future we could handle exceptions here and raise our own up the callstack.
    proc = subprocess.Popen(
        [executable, '-help'], shell=False, stdout=subprocess.PIPE)
    out = proc.communicate()[0].decode()
    out = out[out.find('List of available routines:'):]
    out = out[:out.find('\n\n')]
    routines = [line[:line.find(':')].strip() for line in out.split('\n')[1:]]
    return routines


def get_routine_lib_dict(bin_path):
    """Get a list of all routines from cudnnTest and cudnnTest-{ops,cnn,adv}
    """
    # Get a dict mapping all available routines in cudnnTest to their respective sub-libraries
    all_routines = get_routines(bin_path + '/cudnnTest')
    sublib_routines = [get_routines(bin_path + '/' + sublib)
                       for sublib in all_dynamic_sublib]
    routine_dict = {}
    # Exploit the fact that routines in cudnnTest and sub-libraries are sorted the same way
    for routine in all_routines:
        sublib_list = []
        for sublib, sublib_routine in zip(all_dynamic_sublib, sublib_routines):
            if sublib_routine and sublib_routine[0] == routine:
                sublib_routine.pop(0)
                sublib_list.append(sublib)
        routine_dict[routine] = sublib_list
    return routine_dict


# *******************************************************************************
# * Queue definition for concurrency management
# *******************************************************************************
class Queue:
    def __init__(self, maxsize=0):
        self.maxsize = maxsize
        self.done_count = 0
        self.queue = deque()
        self.mutex = Lock()
        self.done_mutex = Lock()
        self.cv = Condition(self.mutex)
        self.done_cv = Condition(self.done_mutex)
        self.terminate = False

    def qsize(self):
        return len(self.queue)

    def empty(self):
        return self.qsize() == 0

    def put(self, item):
        with self.cv:
            while self.qsize() >= self.maxsize and not self.terminate:
                self.cv.wait()
            if not self.terminate:
                self.queue.append(item)
            self.cv.notify()

    def get(self):
        ret = None
        with self.cv:
            while self.qsize() == 0 and not self.terminate:
                self.cv.wait()
            if not self.terminate:
                ret = self.queue.popleft()
            self.cv.notify()
        return ret

    def task_done(self):
        with self.done_cv:
            self.done_count += 1
            self.done_cv.notify()

    def wait(self, n):
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        with self.done_cv:
            while self.done_count < n:
                self.done_cv.wait()
            self.done_count = 0

    def join(self):
        # Block until self.queue is empty
        with self.done_cv:
            while self.qsize():
                self.done_cv.wait()
            with self.cv:
                self.terminate = True
                self.queue.clear()
                self.cv.notify_all()


# *******************************************************************************
# * GPU memory tracker
# *******************************************************************************
def gpu_mem_used(smi_cmd):
    # Return the amount of used GPU memory
    #
    # Never let this Popen command fail cudnn_run.py
    # So catch all exceptions and return an error
    try:
        nvsmi = subprocess.Popen([smi_cmd, '-q'], stdout=subprocess.PIPE)
        nvsmi_str = nvsmi.communicate()[0].decode()
        mem_str = nvsmi_str[nvsmi_str.find('FB Memory Usage'):]
        used_str = mem_str[mem_str.find('Used'):]
        return int(used_str[used_str.find(':') + 1: used_str.find('MiB')])
    except Exception as e:
        print("SMI_CMD {} failed with {} in gpu_mem_used".format(smi_cmd, e))
        return -1


# *******************************************************************************
# * Test Execution
# *******************************************************************************
def random_choice(population, weights):
    # Reimplemented random.choices since random.choices is not available for python < 3.6
    num = random.random() * sum(weights)
    for i, w in enumerate(weights):
        num -= w
        if num < 0:
            return population[i]


def backdoor_gen(layer, run_result):
    if (run_result.parsed != None and run_result.parsed["query"] != None
            and run_result.return_code == 0):
        for engine_str in run_result.parsed["query"].flags_str.split('\\n'):
            engine_flags = flags_from_descs_str(engine_str)
            if len(engine_flags) != 1:
                raise Exception(
                    "[BackdoorGen] Cannot understand query string %s" % engine_str)
            engine_flags[0]['Dprint_dbg='] = ("8", )
            engine_layers = get_modified_layers(layer, engine_flags[0])
            for engine_layer in engine_layers:
                yield engine_layer


def produce_layers(parsed_args, cache, layers, backdoors=[], ref_change=set(), bin_names=None, multithread=False):
    # Yield batches, corresponding suggested_flags, and bin_name in layers
    if parsed_args.sub_lib_prob and bin_names is None:
        # maps routine to corresponding sub-libraries
        routine_lib_dict = get_routine_lib_dict(parsed_args.bin_path)
    testing_libs = [parsed_args.bin_name] + \
        (['cudnnTest_static'] if parsed_args.static_lib_prob else []) + \
        (all_dynamic_sublib if parsed_args.sub_lib_prob else []) + \
        (all_static_sublib if parsed_args.static_lib_prob and parsed_args.sub_lib_prob else [])
    # maps sub-libraries to a list of layers that should be run using the sub-library
    lib_batches = {lib: [] for lib in testing_libs}
    current_batch = lib_batches[parsed_args.bin_name]
    total_batch_count = parsed_args.testsList_batch_size

    for i, layer in enumerate(layers):
        while len(backdoors) and not multithread:
            for backdoor in backdoors[0]:
                current_batch.append(backdoor)
                if len(current_batch) == total_batch_count:
                    yield current_batch, cache.get_suggested_flags(current_batch), parsed_args.bin_name
                    del current_batch[:]
            backdoors.popleft()
        if parsed_args.sweep_heurgen:
            layer.flags['backendQuery'] = ('', )
        if not parsed_args.default_ref and \
                'gpuRef' not in layer.flags and \
                layer.flags['R'][0] not in gpuref_incompatible and \
                not ('backendEngine' in layer.flags and layer.flags['backendEngine'][0] == 'GroupedDirect'):
            layer.flags['gpuRef'] = ('', )
            ref_change.add(layer)

        if not parsed_args.static_lib_prob and not parsed_args.sub_lib_prob:
            current_batch.append(layer)
        elif bin_names is not None:
            lib_batches[bin_names[i]].append(layer)
        else:
            random.seed(parsed_args.random_bin_seed + i)
            use_static = random.random() < parsed_args.static_lib_prob
            use_sub = random.random() < parsed_args.sub_lib_prob
            if not use_static and not use_sub:
                current_batch.append(layer)
            elif use_static and not use_sub:
                lib_batches['cudnnTest_static'].append(layer)
            else:  # use_sub
                available_libs = routine_lib_dict[layer.flags['R'][0]]
                sub_lib_selection_mode = parsed_args.static_sub_lib_selection_mode if use_static \
                    else parsed_args.dynamic_sub_lib_selection_mode
                if len(available_libs) == 1:
                    sublib = available_libs[0]
                elif sub_lib_selection_mode == 'host':
                    if parsed_args.host_prob == 1.0:
                        sublib = available_libs[0]
                    else:
                        num_other_libs = len(available_libs) - 1
                        sublib = random_choice(
                            available_libs,
                            [parsed_args.host_prob] +
                            [(1 - parsed_args.host_prob) / num_other_libs] * num_other_libs)
                else:  # uniform
                    sublib = random.choice(available_libs)
                if use_static:
                    sublib = sublib + '_static'
                lib_batches[sublib].append(layer)

        if len(current_batch) == total_batch_count:
            # deepcopy since batches may not be immediately consumed in multithreaded mode
            yield deepcopy(current_batch), cache.get_suggested_flags(current_batch), parsed_args.bin_name
            del current_batch[:]

    for sublib in testing_libs:
        current_batch = lib_batches[sublib]
        while len(current_batch):
            yield deepcopy(current_batch[:total_batch_count]), \
                cache.get_suggested_flags(current_batch[:total_batch_count]), \
                sublib
            del current_batch[:total_batch_count]

def get_case_label(flags, waive_list_dict):
    test_flags_list = str(flags).split()
    label_flag = None
    filter_flags_list = []
    for item in test_flags_list:
        if not is_filtered_flag(item):
            filter_flags_list.append(item)

    # If a subset of the test case's flags contains a label in the waive_list_dict, return its label.
    for label in waive_list_dict.keys():
        test_list = waive_list_dict[label]
        for item in test_list:
            tmp_flags = item.split()
            tmp_flags = [flag for flag in tmp_flags if not is_filtered_flag(flag)]
            if set(filter_flags_list) == set(tmp_flags):
                label_flag = label
                return label_flag

    if "-RgraphRunner" in test_flags_list and not label_flag:
        label_flag = "PASS"
    return label_flag

def consume_layers(parsed_args, cache, layer, suggested_flags, bin_name,
                   stdout_log, thread_local_counts, waive_list_dict, devices, backdoors=[],
                   sqlite_db=None, sqlite_parser=None, indexer=None, last_cl=None,
                   batch_queue=None, nonpass_layers=[], multithread=False,
                   print_on_waive=True, print_on_fail=False):
    # Run layer with suggested flags and log to stdout_log, with test result stored in thread_local_counts
    if not multithread:
        assert stdout_log == sys.stdout

        if sqlite_db:
            sys.stdout.start_intercept()

        is_heurgen_query = parsed_args.sweep_heurgen and \
            'backendQuery' in layer.flags and layer.flags['backendQuery'] == (
                '', )
        if is_heurgen_query:
            base_layer = deepcopy(layer)
            del base_layer.flags['backendQuery']
            layer = modify_layer_backendQuery(layer)
    # else:
        # assert not print_on_waive and not print_on_fail

    print_to_log = lambda *args, **kwargs: print(
        file=stdout_log, *args, **kwargs)
    layer_flags_str = str(layer.flags)

    # bug 4720318: Wrap -jsonStr={...} with quotation marks to make it a single argument of testMain.cpp.
    def deal_str(s):
        if (' ' in s) or ("\"" in s):
            return "\'"+s+"\'"
        return s
    layer_flags_str = ' '.join([deal_str(layer_flag) for layer_flag in layer_flags_str.split()])

    test_name_str = '{} {}'.format(bin_name, layer_flags_str)

    if parsed_args.dryrun:
        print_to_log("Running test {} : '{}/{} {}'\n".format(
                     layer.test_name, parsed_args.bin_path, bin_name, layer_flags_str))
        thread_local_counts['dryrun'] += 1
        if parsed_args.check_waive_lists:
            all_test_list.append(layer_flags_str)
        return

    # Extended logging information for processes that timeout
    extended_logging = dict()
    extended_logging['io_timeout'] = parsed_args.io_timeout
    extended_logging['process_timeout'] = parsed_args.process_timeout
    extended_logging['debug_path'] = parsed_args.extended_debug_path


    # Obtain run from cache; will run test if not available in cache
    tmp_log = StringIO()
    test_results = cache.get(
        layer.flags,
        suggested_flags,
        parsed_args.bin_path,
        bin_name,
        device=devices,
        pre_flags_str=parsed_args.pre_flags_str,
        checkjit=parsed_args.checkjit,
        batch_method=parsed_args.batch_method,
        return_mode=parsed_args.return_mode,
        debug_log=tmp_log,
        stdout_log=stdout_log,
        print_on_waive=print_on_waive,
        print_on_fail=print_on_fail,
        extended_logging=extended_logging)
    if not multithread and is_heurgen_query:
        backdoors.append(backdoor_gen(base_layer, test_results))

    # Use return code to determine PASS/FAIL/WAIVE
    extra_status = ""
    test_layer_label = None
    if parsed_args.use_waive_lists and waive_list_dict != {}:
        test_layer_label = get_case_label(layer.flags, waive_list_dict)

    if test_results.return_code == 0:
        current_test_status = 'PASSED'
        thread_local_counts['passed'] += 1
        if test_layer_label == "FAIL" or test_layer_label == "WAIVE":
            # thread_local_counts['failed'] += 1
            thread_local_counts['xpass'] += 1
            extra_status = "XPASS({})".format(test_layer_label)
    elif (test_results.return_code == 2):
        current_test_status = 'WAIVED'
        if print_on_waive:
            thread_local_counts['waived'] += 1
            if test_layer_label == "PASS" or test_layer_label == "FAIL":
                #thread_local_counts['failed'] += 1
                thread_local_counts['xwaive'] += 1
                extra_status = "XWAIVE"
        else:
            nonpass_layers.append((layer, bin_name))
    else:
        current_test_status = 'FAILED'
        if print_on_fail:
            if test_layer_label == "FAIL":
                thread_local_counts['waived'] += 1
                thread_local_counts['xfail'] += 1
                extra_status = "XFAIL"
            else:
                thread_local_counts['failed'] += 1
        else:
            nonpass_layers.append((layer, bin_name))

    should_print = current_test_status == 'PASSED' or \
        (current_test_status == 'WAIVED' and print_on_waive) or \
        (current_test_status == 'FAILED' and print_on_fail)
    if should_print:
        print_to_log("&&&& RUNNING {}".format(test_name_str))
        print_to_log("Running test {} : '{}/{} {}'".format(
                     layer.test_name, parsed_args.bin_path, bin_name, layer_flags_str))
        print_to_log("Test Flags: {}".format(layer.flags.get_str(
            prefix='', delimiter=' * ', seperator=':', quotify_comma=True)))
        print_to_log("Layer Name: {}".format(layer.base_name))
        print_to_log("Unique Flags: {}".format(layer.test_diff_flags.get_str(
            prefix='', delimiter=' * ', seperator=':', quotify_comma=True)))
        # Print output if it exists
        if test_results.output is None:
            print_to_log("No output detected\n")
        else:
            # Only print API log when there is an error
            if parsed_args.API_log_test:
                if test_results.error_msg:
                    print_to_log(test_results.output)
                    print_to_log("")
                else:
                    print_to_log(stripLog(test_results.output, mode=1))
            else:
                print_to_log(test_results.output)
                if bin_name == 'cutlass_profiler':
                    cutlass_output_parse(parsed_args, test_results)
                print_to_log("")
        # Print error if it exists
        if test_results.error_msg:
            print_to_log("[TEST EXECUTION] Error Detected: {}".format(
                test_results.error_msg))

        if extra_status and extra_status != "XFAIL":
            print_to_log("&&&& {} {}".format(current_test_status, test_name_str))
            print_to_log("&&&& {} {}".format(extra_status, test_name_str))
        elif extra_status == "XFAIL":
            print_to_log("&&&& {} {}".format(extra_status, test_name_str))
        else:
            print_to_log("&&&& {} {}".format(current_test_status, test_name_str))

    if not multithread:
        if parsed_args.API_log_test and current_test_status == 'PASSED':
            if logToTest_checkFlagSupport(layer.flags) == True:
                print("@@@@ Original flags: {}".format(layer.flags))
                generated_flags = logToTest_generate(test_results, layer.flags)
                print("@@@@ Generated flags: {}".format(generated_flags))
                if (generated_flags != 'error') and (generated_flags !=
                                                     'waived'):
                    # don't run ref on second run, save time
                    generated_flags['T'] = ('1', )
                    API_rerun_results = cache.get(
                        generated_flags,
                        suggested_flags,
                        parsed_args.bin_path,
                        bin_name,
                        device=parsed_args.device,
                        return_mode=parsed_args.return_mode,
                        pre_flags_str=parsed_args.pre_flags_str,
                        stdout_log=stdout_log)
                    if (API_rerun_results.return_code == 0):
                        execPath_match = logToTest_compareExecutionPath(
                            test_results.output, API_rerun_results.output)
                        if execPath_match:
                            print(
                                "@@@@ API LOG TEST PASSED (ExecPath_match=%r): %s\n"
                                % (execPath_match, test_name_str))
                        else:
                            print(
                                "@@@@ API LOG TEST FAILED (EXECPATH MISMATCH): %s\n"
                                % (test_name_str))
                    else:
                        print("@@@@ API LOG TEST FAILED (RERUN FAIL): %s\n" %
                              (test_name_str))
                elif generated_flags == 'error':
                    print("@@@@ API LOG TEST FAILED (GEN FLAG ERROR): %s\n" %
                          (test_name_str))
                    print(test_results.output)
                    print("")
                elif generated_flags == 'waived':
                    print("@@@@ API LOG TEST WAIVED: %s\n" % (test_name_str))
            else:
                print(
                    "@@@@ API LOG TEST NOT SUPPORTED: %s\n" % (test_name_str))

        if sqlite_db:
            all_parsed = use_parser_on_output(sys.stdout.end_intercept(),
                                              sqlite_parser)
            # insert additional columns if it runs with persistent db
            if parsed_args.insert_persistent_db_cols:
                all_parsed = add_persistent_db_additional_content(
                    all_parsed, indexer, parsed_args.new_api_db_path, last_cl)
            sqlite_db.add(all_parsed)
            sqlite_db.commit()

        # log the idx of current case when using binary file
        if parsed_args.binary_path != None:
            print("Case Index: {}".format(layer.case_idx))

    print_to_log("")

    debug_str = tmp_log.getvalue()
    if should_print and len(debug_str) > 0:
        with open_or_stdout(parsed_args.extended_debug_path, 'DebugInfo.log') as debug_info_file:
            if debug_info_file is not None:
                print_to_log("Write debug info into " + getattr(debug_info_file, 'name', 'stdout'))
                debug_info_file.write("-- DEBUG INFO FOR ABOVE CASE --")
                debug_info_file.write("[DEBUG] " + debug_str.replace("\n", "\n[DEBUG] "))
                debug_info_file.write("-- END OF DEBUG INFO --")


def rerun_single_thread(parsed_args, nonpass_tests, ref_change, waive_list_dict, devices, counts, is_interrupted,
                        sqlite_db=None, sqlite_parser=None, indexer=None, last_cl=None):
    # Rerun all waived or failed tests in multithread or all failed tests in single thread if not -default_ref
    rerun_start, nonpass_layers = None, []
    if not is_interrupted and (parsed_args.threads != 1 or not parsed_args.default_ref):
        nonpass_layers = [
            layer[0] for batch_layer in nonpass_tests for layer in batch_layer]
        nonpass_bins = [
            layer[1] for batch_layer in nonpass_tests for layer in batch_layer]
        if len(nonpass_layers):
            rerun_start = time.time()
            # Revert changed -gpuRef settings
            if not parsed_args.default_ref:
                parsed_args.default_ref = True
                for layer in nonpass_layers:
                    if layer in ref_change:
                        del layer.flags['gpuRef']

            if parsed_args.threads != 1:
                print("&&&& {} tests waived or failed in multithread mode; rerunning in single thread\n"
                    .format(len(nonpass_layers)))
            else:
                print(
                    "&&&& {} tests failed without -default_ref; rerunning with original ref setting\n"
                    .format(len(nonpass_layers)))
            print("@@@@ Rerunning tests:")
            for layer in nonpass_layers:
                print("     " + layer.flags.get_descs_str())
            print("\n\n")

            cache = RunCache(
                batch_size=parsed_args.testsList_batch_size,
                max_cache=max(1000, parsed_args.testsList_batch_size))
            backdoors = deque()
            for batch, flags, bin_name in produce_layers(
                    parsed_args, cache, nonpass_layers, backdoors, bin_names=nonpass_bins):
                for layer in batch:
                    consume_layers(parsed_args, cache, layer, flags, bin_name,
                                   sys.stdout, counts, waive_list_dict, devices, backdoors,
                                   sqlite_db, sqlite_parser, indexer, last_cl,
                                   print_on_waive=True, print_on_fail=True)
            del nonpass_tests[:]
    return (0 if rerun_start is None else time.time() - rerun_start), len(nonpass_layers)


def produce_multithread(parsed_args, batch_queue, cache, layers,
                        stdout_logs, nonpass_tests, ref_change,
                        waive_list_dict, devices, counts, interrupt):
    if parsed_args.output_buffer_mult > 0:
        total_rerun_time, total_rerun_count = 0, 0
        buffer_size = int(parsed_args.output_buffer_mult * parsed_args.threads)
    for i, (batch, flags, bin_name) in enumerate(
            produce_layers(parsed_args, cache, layers, ref_change=ref_change, multithread=True)):
        stdout_logs.append(StringIO())
        nonpass_tests.append([])
        batch_queue.put((batch, flags, bin_name, stdout_logs[-1], nonpass_tests[-1]))
        # batch_queue.put((batch, flags, bin_name, sys.stdout, i))  # for debugging
        if parsed_args.output_buffer_mult > 0 and i % buffer_size == 0:
            print("@@@@ Dispatching batches\n")
        if parsed_args.output_buffer_mult > 0 and (i + 1) % buffer_size == 0:
            batch_queue.wait(buffer_size)
            for stdout_log in stdout_logs:
                print(stdout_log.getvalue())
            del stdout_logs[:]
            rerun_time, rerun_count = rerun_single_thread(parsed_args, nonpass_tests, ref_change,
                                                          waive_list_dict, devices, counts, interrupt.is_set())
            total_rerun_time += rerun_time
            total_rerun_count += rerun_count
    if parsed_args.output_buffer_mult > 0:
        batch_queue.wait((i + 1) % buffer_size)
        for stdout_log in stdout_logs:
            print(stdout_log.getvalue())
        del stdout_logs[:]
    return (total_rerun_time, total_rerun_count) if parsed_args.output_buffer_mult > 0 else (0, 0)


def consume_multithread(parsed_args, batch_queue, cache, idx, waive_list_dict, devices, per_thread_counts, smi_cmd):
    thread_local_counts = per_thread_counts[idx]
    while True:
        queue_item = batch_queue.get()
        if queue_item is None:
            break
        batch, flags, bin_name, stdout_log, nonpass_layers = queue_item
        print("@@@@ thread id ({}) processing batch:\n".format(current_thread().ident) + \
            "@@@@ run string:\n" + \
            "     " + "\n     ".join([layer.flags.get_descs_str() for layer in batch]) + "\n" + \
            "@@@@ run command:\n" + \
            "     " + "\n     ".join([layer.flags.get_str() for layer in batch]))
        for layer in batch:
            consume_layers(parsed_args, cache, layer, flags, bin_name,
                           stdout_log, thread_local_counts, waive_list_dict, devices,
                           batch_queue=batch_queue, nonpass_layers=nonpass_layers,
                           multithread=True, print_on_waive=parsed_args.no_rerun, print_on_fail=parsed_args.no_rerun)
        consumed_output = "@@@@ thread id ({}) completed batch:\n".format(current_thread().ident) + \
            "@@@@ run string:\n" + \
            "     " + "\n     ".join([layer.flags.get_descs_str() for layer in batch]) + "\n" + \
            "@@@@ run command:\n" + \
            "     " + "\n     ".join([layer.flags.get_str() for layer in batch]) + "\n"
        # never fail cudnn_run with a failed nvidia-smi command. Could just be a temporary OS error.
        # so catch all exceptions and print an error message
        try:
            print((consumed_output + subprocess.Popen([smi_cmd], stdout=subprocess.PIPE).communicate()[0].decode()) if smi_cmd else consumed_output)
        except Exception as e:
            print("SMI_CMD: {} failed with {}".format(smi_cmd, e))
        batch_queue.task_done()


def display_process_and_disk_info(filter_term):
    # never fail cudnn_run with a failed ps and df command. Could just be a temporary OS error.
    # so catch all exceptions and print an error message
    try:
        if platform.system() == "Linux" and platform.machine() == "x86_64":

            # Execute the ps and df commands and grep for the filter term
            result_string = "ps aux | grep " + str(filter_term) + ";df -h"
            result = subprocess.check_output(result_string, shell=True)
            return result
        elif platform.system() == "Windows":
            proc_name = filter_term + ".exe"
            result = subprocess.check_output('tasklist /FI "IMAGENAME eq {}"'.format(proc_name), shell=True)
            return result
        else:
            return "OS {} / machine {} does not support ps and df info".format(platform.system(), platform.machine())
    except Exception as e:
        return "display_process_and_disk_info() failed with {}".format(e)


# get configured lists from waive_list file which defined according to GPU and OS
def get_waive_list(gpu, cudart, explicit_waive_path, check_all=False):
    """
    Retrieves the waive list based on the provided arguments and updates it with global flags.
    Args:
        parsed_args (argparse.Namespace): The parsed command-line arguments.
        gpu (object): The GPU object.
        cudart (str): The CUDA runtime version.
        explicit_waive_path (str): The path to the explicit waive file.
        check_all (bool, optional): Whether to check all waive cases. Defaults to False.
    Returns:
        dict: The updated waive list, with keys 'XFAIL' and 'WAIVE'.
    """
    waive_test_dict = {}
    query_waive_dict = {}
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)),"waive_lists","test_db"))
    try:

        from test_db import query_waive_list
    except ImportError:
        print("Waive list feature unsupported because of missing test_db or dependent packages,such as yaml.")
        return waive_test_dict



    if gpu == None or cudart == None:
        print('[GPU DETECTION] Error Detected, Please check the output for "cudnnTest -g"')
        return waive_test_dict

    default_waive_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"waive_lists", "db")
    if explicit_waive_path is not None:
        waive_path = explicit_waive_path if os.path.isabs(explicit_waive_path) else os.path.join(os.path.dirname(os.path.abspath(__file__)), explicit_waive_path)
    else:
        waive_path = default_waive_path

    if not os.path.exists(waive_path):
        print('Waive file path (%s) does not exist' % waive_path)
        return waive_test_dict

    os_type = platform.system()
    if check_all == True:
        query_opts_dict = json.dumps({})
    else:
        gpu_cap = round(float(gpu.cap)/10,1)
        if os_type == "Linux":
            try:
                import distro
            except ImportError:
                if sys.version_info.major < 3:
                    distribute_name = platform.linux_distribution()[0].lower()
                else:
                    print("Waive list feature unsupported because of missing distro, please install it.")
                    return waive_test_dict
            else:
                distribute_name = distro.linux_distribution()[0].lower()
        else:
            distribute_name = None

        is_native_windows = True if os_type == "Windows" else False
        cpu_type = platform.processor() if os_type == "Linux" else "x86_64"

        cudart_version = ".".join(cudart.split("0")[:2])
        print("gpu_cap = {}, os_type = {}, distribute_name = {}, cpu_type = {}".format(gpu_cap,os_type,distribute_name,cpu_type))

        query_opts_dict = json.dumps({"compute_capability": str(gpu_cap),
                                      "cpu": cpu_type,
                                      "linux_distribution_name": distribute_name,
                                      "cuda_version": cudart_version,
                                      "is_native_windows":is_native_windows})

    query_label_list = ["FAIL","WAIVE"]
    print('Need to query context {} from waive path {}'.format(query_label_list, waive_path))
    try:
        query_waive_dict  = query_waive_list(query_label_list, waive_path, query_opts_dict)
    except Exception as err:
        print("Query waive list failed : {}".format(err))
        return waive_test_dict

    for item in list(query_waive_dict.keys()):
        waive_test_dict[item] = query_waive_dict[item]
    return waive_test_dict


def is_filtered_flag(flag):
    gpuRef_flags = ["-gpuRef", "-gpuRef1"]
    device_flag_pattern = re.compile(r"-d\d+\b")
    thread_terminator_pattern = re.compile(r"-threadTerminator\d+")

    return re.search(device_flag_pattern,flag) \
            or re.search(thread_terminator_pattern,flag) \
            or flag in gpuRef_flags


def get_unused_waive_tests(all_test_list, waive_list_dict):
    unused_tests_dict = {}
    if not waive_list_dict or not all_test_list:
        print("The queried waive list or obtained level tests list is empty")
        return unused_tests_dict

    filtered_all_tests_list = []
    for test in  all_test_list:
        tmp_flags = test.split()
        filtered_flags = []
        for flag in tmp_flags:
            if not is_filtered_flag(flag):
                filtered_flags.append(flag)

        filtered_all_tests_list.append(set(filtered_flags))  # [(), (), ..., ()]

    for label in waive_list_dict.keys():
        unused_tests_dict.setdefault(label, [])
        tmp_test_list = waive_list_dict[label]
        for test_case in tmp_test_list:
            test_flags = test_case.split()
            test_flags = [flag for flag in test_flags if not is_filtered_flag(flag)]
            if not set(test_flags) in filtered_all_tests_list:
                unused_tests_dict[label].append(test_case)

    return unused_tests_dict


# *******************************************************************************
# * Main
# *******************************************************************************
if __name__ == '__main__':
    parsed_args = post_parse_args(cudnn_run_argparser().parse_args())

    lib_path_setup(parsed_args)
    if parsed_args.API_log_test:
        enable_logging()
    sqlite_db, sqlite_parser, indexer, last_cl = sqlite_setup(parsed_args)
    if parsed_args.bin_name == 'cutlass_profiler':
        parsed_args.filter_config_str = 'FAST'
    global_filter_config = split_comma(parsed_args.filter_config_str)

    counts = {'passed': 0, 'waived': 0, 'failed': 0, 'xwaive': 0, 'xpass': 0, 'xfail': 0, 'dryrun': 0}
    gpu, gpu_test_res, cudart = detect_gpu(parsed_args, sqlite_db, sqlite_parser)
    counts['failed' if gpu is None else 'passed'] += 1
    counts['passed' if gpu_test_res else 'failed'] += 1
    global_filter_config.append(get_gpu_filter(gpu))
    devices = parsed_args.device.split(',')

    if parsed_args.dryrun:
        smi_cmd = None
        print("Here is dryrun. We won't run nvidia-smi.")
    else:
        smi_cmd = get_nvidia_smi_cmd()
    if parsed_args.testsList_batch_size == 0 or parsed_args.threads == 0:
        if parsed_args.threads == 0:
            parsed_args.threads = default_thread_count(gpu)

            # WAR for the OOM issues on TU102 and TU104 (Linux OS). Bug 4583648
            if ( gpu.cap=='75' ) and ( platform.system() == "Linux" ):
                # Assume that test may take 30GiB of GPU memory; can adjust in the future.
                parsed_args.threads = max(1, min( parsed_args.threads, get_available_mem_linux() // (1024**3) // 30 ))
                print('get_available_mem = ',get_available_mem_linux())

        if parsed_args.testsList_batch_size == 0:
            parsed_args.testsList_batch_size = default_batch_size(gpu, parsed_args.threads)
            is_MT_test = parsed_args.global_flags_str and 'MT:' in parsed_args.global_flags_str
            if is_MT_test:
                # For the MT test, the test cases are more time-consuming than the usual test.
                # Set the upper bound of batch_size as 30 to avoid the timeout issues. See bug 4334077.
                parsed_args.testsList_batch_size = min( parsed_args.testsList_batch_size , 30 )

    # Limit the output_buffer_mult on TU102 and TU104. WAR for tge bug 4583648
    if ( parsed_args.output_buffer_mult == -1) and ( gpu is not None ) and ( gpu.cap == '75' ):
        parsed_args.output_buffer_mult = 1000

    if parsed_args.global_flags_str and 'T:' in parsed_args.global_flags_str:
        # prevents unnecessary rerun in the case for example --threads 1 --global_flags "T:-1" with a waived or failed case
        # since the waive / failure is not dependent on the reference calculation
        parsed_args.default_ref = True
        print("\nDetected T in global flags. Since reference calculations do not take place, default_ref is turned on automatically.")

    print("\n\nPrinting all command line arguments")
    for arg, val in vars(parsed_args).items():
        print("\t{:25s} = {}".format(arg, val))
    if parsed_args.testsList_batch_size == 1 and len(devices) > 1:
        print(
            "Multi-gpu mode only available with batch size > 1, running on device {}".format(devices[0]))
    if parsed_args.global_flags_str and 'gpuRef:' in parsed_args.global_flags_str and 'gpuRef:0' not in parsed_args.global_flags_str:
        print("Detected gpuRef in global flags. Doing so may result in unintentional / undesirable test execution.\n" +
              "Please consider removing gpuRef in global flags as cudnn_run.py now defaults to gpuRef when possible in a safe manner.")

    waive_list_dict = {}
    if parsed_args.check_waive_lists or parsed_args.use_waive_lists:
        print("Waive list checking is enabled")
        waive_list_dict = get_waive_list(gpu, cudart, parsed_args.explicit_waive_path, parsed_args.check_waive_lists)

    print(waive_list_dict)
    partition_index, partition_count = get_partition_info(parsed_args)
    layers = get_layer_generator(
        parsed_args, global_filter_config, partition_index, partition_count)

    cache = RunCache(
        batch_size=parsed_args.testsList_batch_size,
        max_cache=max(1000, parsed_args.testsList_batch_size))
    backdoors = deque()
    batch_queue = Queue(maxsize=max(100000, parsed_args.testsList_batch_size))
    stdout_logs = []
    nonpass_tests = []
    per_thread_counts = [{'passed': 0, 'waived': 0, 'failed': 0, 'xpass': 0, 'xwaive': 0, 'xfail': 0, 'dryrun': 0}
                         for _ in range(parsed_args.threads)]
    ref_change = set()
    display_interval = 10
    mem_interval = 0.1
    sleep_interval = mem_interval if parsed_args.track_memory else display_interval
    display_idx = int(display_interval / sleep_interval) - 1
    max_mem = 0
    interrupt = Event()

    def catch_interrupt(_signum, _frame):
        interrupt.set()

    print("\n\n")
    start_time = time.time()

    total_rerun_time, total_rerun_count = 0, 0
    if parsed_args.threads == 1:
        for i, (batch, flags, bin_name) in enumerate(
                produce_layers(parsed_args, cache, layers, backdoors, ref_change)):
            nonpass_tests.append([])
            print("@@@@ processing batch:")
            print("@@@@ run string:")
            print("     " + "\n     ".join([layer.flags.get_descs_str() for layer in batch]))
            print("@@@@ run command:")
            print("     " + "\n     ".join([layer.flags.get_str() for layer in batch]))
            for layer in batch:
                consume_layers(parsed_args, cache, layer, flags, bin_name,
                               sys.stdout, counts, waive_list_dict, devices, backdoors,
                               sqlite_db, sqlite_parser, indexer, last_cl,
                               nonpass_layers=nonpass_tests[-1],
                               print_on_waive=True, print_on_fail=parsed_args.default_ref or parsed_args.no_rerun)
    else:
        signal.signal(signal.SIGINT, catch_interrupt)
        signal.signal(signal.SIGTERM, catch_interrupt)

        consumer_threads = [Thread(target=consume_multithread,
                                   args=(parsed_args, batch_queue, cache, i, waive_list_dict, devices, per_thread_counts, smi_cmd))
                            for i in xrange(parsed_args.threads)]
        for thread in consumer_threads:
            thread.setDaemon(True)
            thread.start()
        rerun_time, rerun_count = produce_multithread(parsed_args, batch_queue, cache,
                                                      layers, stdout_logs, nonpass_tests, ref_change,
                                                      waive_list_dict, devices, per_thread_counts[0], interrupt)
        total_rerun_time += rerun_time
        total_rerun_count += rerun_count

        last_ps_time = time.time()

        if parsed_args.output_buffer_mult == -1:
            i = display_idx
            while not (batch_queue.empty() or interrupt.is_set()):
                #Every 5 minutes print ps and df info
                if (time.time() - last_ps_time) > 300:
                    last_ps_time = time.time()
                    filter_term = "cudnnTest"
                    processes = display_process_and_disk_info(filter_term)
                    print("[PS.DEBUG] {}".format(processes))
                if i == display_idx:
                    print("@@@@ [{}] {} batches queued"
                        .format(datetime.now().isoformat(), batch_queue.qsize()))
                    i = 0
                else:
                    i += 1
                if parsed_args.track_memory and smi_cmd:
                    # gpu_mem_used will return -1 in case of error.
                    max_mem = max(max_mem, gpu_mem_used(smi_cmd))
                interrupt.wait(sleep_interval)
            if batch_queue.empty():
                print(
                    "@@@@ [{}] 0 batches queued\n".format(datetime.now().isoformat()))

        if batch_queue.empty():
            print("INFO cudnn_run.py shutting down because batch_queue is empty")
            batch_queue.join()
        else:
            print("INFO cudnn_run.py shutting down because interrupt is set")
            with batch_queue.cv:
                batch_queue.terminate = True
                batch_queue.queue.clear()
                batch_queue.cv.notify_all()
        print("INFO [{}] queue joined".format(datetime.now().isoformat()))
        for thread in consumer_threads:
            thread.join()
        print("INFO [{}] threads joined".format(datetime.now().isoformat()))
        for stdout_log in stdout_logs:
            print(stdout_log.getvalue())
        for thread_local_counts in per_thread_counts:
            for k in counts.keys():
                counts[k] += thread_local_counts[k]
    if parsed_args.no_rerun == False:
        print("INFO [{}] Re-Run".format(datetime.now().isoformat()))
        rerun_time, rerun_count = rerun_single_thread(parsed_args, nonpass_tests, ref_change,
                                                    waive_list_dict, devices, counts, interrupt.is_set(),
                                                    sqlite_db, sqlite_parser, indexer, last_cl)
        total_rerun_time += rerun_time
        total_rerun_count += rerun_count

    if sqlite_db:
        sqlite_db.close()
        sqlite_db = None
    print("INFO [{}] Check Waived lists".format(datetime.now().isoformat()))
    unused_test_dict = {}
    if parsed_args.check_waive_lists:
        if all_test_list == []:
            print("Test list is empty, skip checking waive list")
        else:
            unused_test_dict = get_unused_waive_tests(all_test_list, waive_list_dict)
    print("INFO [{}] Finished".format(datetime.now().isoformat()))
    end_time = time.time()
    time_elapsed = end_time - start_time

    sys.stdout.flush()
    print("")
    print("RESULT")
    print("{:14}: {}".format("Failures", counts['failed']))
    print("{:14}: {}".format("Successes", counts['passed']))
    if waive_list_dict != {}:
        print("({} : {})".format("xpass", counts['xpass']))
    print("{:14}: {}".format("Waived", counts['waived']))
    if waive_list_dict != {}:
        print("({} : {}, {} : {})".format("xfail", counts['xfail'], "xwaive", counts['xwaive']))

    if counts['dryrun'] > 0:
        print("{:14}: {}".format("Listed", counts['dryrun']))
    print("{:14}: {:4.2f}%".format(
        "Basic Sanity",
        ((100 * counts['passed']) / (counts['passed'] + counts['failed']))))
    print("{:14}: {:.2f} sec".format("Total Time", time_elapsed))
    if total_rerun_time:
        print("{:14}: {:.2f} sec / {} tests".format("Rerun Time", total_rerun_time, total_rerun_count))
    if parsed_args.track_memory:
        print("{:14}: {}MiB".format("Max Memory", max_mem))
    if parsed_args.check_waive_lists or parsed_args.use_waive_lists:
        if waive_list_dict != {}:
            print("{:14}: Take effect".format("Waived Check"))
        else:
            print("{:14}: Not take effect".format("Waived Check"))
    if unused_test_dict:
        for label in unused_test_dict.keys():
            print("{:14}: {}".format("Unused Tests", label))
            for line in unused_test_dict[label]:
                print("{}".format(line))
    sys.stdout.flush()

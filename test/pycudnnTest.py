#!/usr/bin/env python

import utils
utils.reportCurrentTime("Start")
import argparse
import os, sys
utils.reportCurrentTime("import_python_modules")
from json_graph_test import run_test_from_legacy_args
utils.reportCurrentTime("import_json_graph_test")

if __name__ == "__main__":
    pct_parser = argparse.ArgumentParser(prog='pycudnnTest')
    # TODO(@mbreughe): generalize this to a directory of python files
    # TODO(@mbreughe): legacy json tests specify through -jsonPath (see json_graph_test.py). Make this mutually exclusive to -RgraphRunner and -RgrStream
    pct_parser.add_argument('--testPath', default="json_graph_defs/graphTests.json", 
                        help="This can be a json file or python file with graph definitions. "
                        "e.g. json_graph_defs/graphTests.json, python_graph_defs/basic_tests.py")
    # TODO(@mbreughe): again this has no meaning in the graph runner modes
    pct_parser.add_argument('--testName', default=[], action="append", help="Test Name (multiple names are allowed and recommended for performance). Note: in python graph mode, no name means all tests in file are executed. ")
    pct_parser.add_argument('--verbose', '-v', action="store_true", default=False, help="Verbose output")
    pct_parser.add_argument('--vverbose', '-vv', action="store_true", default=False, help="Very verbose output")
    pct_parser.add_argument('--disable_cupti', action="store_true", default=False, help="Disable usage of cupti profiling (e.g., running through nsys is mutually exclusive)")
    # TODO(@mbreughe): no meaning in the graph runner modes
    pct_parser.add_argument('--threads', '-n', action="store", default=1, help="Number of threads to parallelize tests across.")
    pct_parser.add_argument('--R', '-R', choices=['graphRunner', "grStream"])
    # TODO(@mbreughe): option exclusive for grStream
    pct_parser.add_argument('--stream_start_line', '--start_line', '--stream_start', action="store", dest='start_line', default=1, type=int, help="In stream mode, which line should be our first test?")
    # TODO(@mbreughe): option exclusive for grStream
    pct_parser.add_argument('--stream_group_size', action="store", dest='num_lines', type=int, default=None, help="In stream mode, how many tests do we want to batch?")


    cmd = " ".join(sys.argv)
    print("Running: {}".format(cmd))

    args, unknown_args = pct_parser.parse_known_args()
    utils.reportCurrentTime("arg_parse_1")

    utils.DISABLE_CUPTI = args.disable_cupti

    base_path = os.path.dirname(os.path.abspath(__file__))
    json_graph_test = os.path.join(base_path, "json_graph_pytest_wrapper.py")
    python_graph_test = os.path.join(base_path, "python_graph_test.py")

    # Legacy style of calling cudnnTest (from e.g., cudnn_run.py)
    if args.R == 'graphRunner':
        run_test_from_legacy_args(args, unknown_args)
   
        utils.reportCurrentTime("done")
        sys.exit(0)

    elif args.R == 'grStream':
        import shlex
        error_count = 0
        line_count = 0

        # Process each line in stdin, and run as a graphRunner command
        for line in sys.stdin:
            line_count +=1
            # Start at the line number indicated by the user
            if line_count < args.start_line:
                continue

            # End at the line number indicated by the user
            if (args.num_lines is not None) and (line_count >= args.start_line + args.num_lines):
                break

            print("Running in stream mode: ", line.strip())
            args_stream, unknown_args_stream = pct_parser.parse_known_args(shlex.split(line))
            try:
                run_test_from_legacy_args(args_stream, unknown_args_stream)
            except Exception as e:
                print(e)
                error_count +=1
                print("ERROR: {}".format(line.strip()))

            utils.reportCurrentTime("done")
        
        if error_count > 0:
            print("ERROR: {} failed tests.".format(error_count))
            sys.exit(1)
        sys.exit(0)

    # Graphs defined in json file
    elif args.testPath.endswith(".json"):
        pytest_cmd = [json_graph_test]
    # Graphs defined in python file
    elif args.testPath.endswith(".py"):
        pytest_cmd = [python_graph_test]
    else:
        print("Unrecognized test file {}".format(args.testPath))

    for test_name in args.testName:
        pytest_cmd.extend(["--testName", test_name])

    pytest_cmd.extend(["--testPath", args.testPath])

    if args.vverbose:
        pytest_cmd.append("-v")
        pytest_cmd.append("-s")
    elif args.verbose:
        pytest_cmd.append("-v")

    if int(args.threads) > 1:
        pytest_cmd.extend(["-n", args.threads])

    import pytest
    sys.exit(pytest.main(pytest_cmd))
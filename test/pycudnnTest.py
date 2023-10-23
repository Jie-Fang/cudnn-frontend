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
    pct_parser.add_argument('--testPath', default="json_graph_defs/graphTests.json", 
                        help="This can be a json file or python file with graph definitions. "
                        "e.g. json_graph_defs/graphTests.json, python_graph_defs/basic_tests.py")
    pct_parser.add_argument('--testName', default=[], action="append", help="Test Name (multiple names are allowed and recommended for performance). Note: in python graph mode, no name means all tests in file are executed. ")
    pct_parser.add_argument('--verbose', '-v', action="store_true", default=False, help="Verbose output")
    pct_parser.add_argument('--vverbose', '-vv', action="store_true", default=False, help="Very verbose output")
    pct_parser.add_argument('--threads', '-n', action="store", default=1, help="Number of threads to parallelize tests across.")
    pct_parser.add_argument('--R', '-R', choices=['graphRunner', "grStream"])

    cmd = " ".join(sys.argv)
    print("Running: {}".format(cmd))

    args, unknown_args = pct_parser.parse_known_args()
    utils.reportCurrentTime("arg_parse_1")

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
        for line in sys.stdin:
            args_stream, unknown_args_stream = pct_parser.parse_known_args(shlex.split(line))
            run_test_from_legacy_args(args_stream, unknown_args_stream)
            utils.reportCurrentTime("done")

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
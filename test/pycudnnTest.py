import pytest
import argparse
import json
from json_graph_test import runTestFromJsonDefinition

from python_graph_defs.basic_tests import test_conv_relu

if __name__ == "__main__":
    pct_parser = argparse.ArgumentParser(prog='pyCudnnTest')
    # TODO(@mbreughe): generalize this to a directory of python files
    pct_parser.add_argument('--testPath', default="json_graph_defs/graphTests.json", 
                        help="This can be a json file or python file with graph definitions. "
                        "e.g. json_graph_defs/graphTests.json, python_graph_defs/basic_tests.py")
    # TODO(@mbreughe): allow selection of specfic tests
    pct_parser.add_argument('--testName', default="bla", help="test name")
    pct_parser.add_argument('--verbose', '-v', action="store_true", default=False, help="Verbose output")
    pct_parser.add_argument('--vverbose', '-vv', action="store_true", default=False, help="Very verbose output")

    args = pct_parser.parse_args()

    print(args)

    # Graphs defined in json file
    if args.testPath.endswith(".json"):
        pytest_cmd = ["json_graph_test.py", "--testName", args.testName, "--testPath", args.testPath]
    # Graphs defined in python file
    elif args.testPath.endswith(".py"):
        pytest_cmd = ["python_graph_test.py"]
        if args.testName is not None:
            pytest_cmd.append("-k {}".format(args.testName))
    else:
        print("Unrecognized test file {}".format(args.testPath))

    if args.vverbose:
        pytest_cmd.append("-v")
        pytest_cmd.append("-s")
    elif args.verbose:
        pytest_cmd.append("-v")

    pytest.main(pytest_cmd)
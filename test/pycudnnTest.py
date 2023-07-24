import pytest
import argparse
import json
from json_graph_test import runTestFromJsonDefinition

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='pyCudnnTest')
    # TODO(@mbreughe): generalize this to a directory of python files
    parser.add_argument('--testPath', default="json_graph_defs/graphTests.json", 
                        help="This can be a json file or python file with graph definitions. "
                        "e.g. json_graph_defs/graphTests.json, python_graph_defs/basic_tests.py")
    # TODO(@mbreughe): allow selection of specfic tests
    parser.add_argument('--testName', default=None)

    args = parser.parse_args()
    if args.testPath.endswith(".json"):
        fname = args.testPath
        test_name = "ConvRelu1"

        with open(fname) as ifh:
            json_tests = json.load(ifh)

        runTestFromJsonDefinition(json_tests[test_name])
    elif args.testPath.endswith(".py"):
        pytest.main([args.testPath])
    else:
        print("Unrecognized test file {}".format(args.testPath))
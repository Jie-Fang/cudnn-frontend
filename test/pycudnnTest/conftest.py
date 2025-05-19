import os, json
import importlib.util
import inspect


def pytest_addoption(parser):
    parser.addoption("--testName", action="append", default=[])
    parser.addoption("--testPath", action="store", default=None)
    parser.addoption("--testInput", action="store", default=None)
    parser.addoption(
        "--graph_category",
        action="store",
        default="kGemm",
        choices=["kGemm", "kMemBound"],
        help="Graph category type (kGemm or kMemBound)",
    )


def get_python_graph_defs(path, module_name):

    filename = os.path.join(path, "{}.py".format(module_name))
    if os.path.exists(filename):
        spec = importlib.util.spec_from_file_location(
            name=module_name, location=filename
        )
        my_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(my_module)
    else:
        raise FileNotFoundError(filename)

    test_funcs = []
    test_names = []
    for name, obj in inspect.getmembers(my_module):
        if inspect.isfunction(obj):
            test_funcs.append(obj)
            test_names.append(name)

    return (test_funcs, test_names)


# @param test_funcs: list of function pointers to python graph test definitions
# @param test_names: list of names associated with each item in test_funcs
# @param params_path: path to a directory that contains input parameters for each function in test_names
# @param wanted_tests: list of function names that the user requested to run. Empty list means all test will be run
# @return tuples: list of tuples of (function name, input param). -- function name will be repeated for each input param
# @return test_ids: list of unique test names (function name + input params) for each item in tuple
def createTestParamNameTuples(test_funcs, test_names, params_path, wanted_tests):
    tuples = []
    test_ids = []
    for test_name, func in zip(test_names, test_funcs):
        # If wanted_tests are specified, only look at tests we requested
        if len(wanted_tests) > 0 and not test_name in wanted_tests:
            continue

        filename = os.path.join(params_path, "{}.json".format(test_name))
        if os.path.exists(filename):
            params = []
            try:
                with open(filename) as ifh:
                    params = json.load(ifh)
            except Exception as e:
                print("Issue with input parameter file {}".format(filename))
                raise e

            cur_ids = ["{}[json({})]".format(test_name, i) for i in range(len(params))]

            for par in params:
                tuples.append((func, par))

            test_ids.extend(cur_ids)
        else:
            error = FileNotFoundError(filename)
            error.add_note(
                "Please make sure your test ({}) has an associated file with input params".format(
                    test_name
                )
            )
            raise error

    return tuples, test_ids


def pytest_generate_tests(metafunc):
    JSON_TEST_LIST_PARAM = "jparams"
    JSON_DICT_PATH = "json_dict"
    GRAPH_PYTHON_FPTR = "graph_builder_fptr"
    TEST_NAME_PARAM = "test_name"
    GRAPH_CATEGORY_PARAM = "graph_category"

    # Dynamically create tests for python_graph_test by identifying all test defs in a test directory
    if (
        metafunc.function.__name__ == "test_python_graph"
        or metafunc.function.__name__ == "test_negative_graph"
        or metafunc.function.__name__ == "test_tensor_ir_pygraph"
    ):
        # Find all the functions that define testgraphs in location testPath
        base_path = os.path.dirname(
            os.path.abspath(metafunc.config.getoption("testPath"))
        )
        filename = os.path.basename(metafunc.config.getoption("testPath")).split(".")[
            -2
        ]
        test_funcs, test_names = get_python_graph_defs(base_path, filename)

        params_path = metafunc.config.getoption("testInput")

        param_tuples, test_ids = createTestParamNameTuples(
            test_funcs, test_names, params_path, metafunc.config.getoption("testName")
        )

        metafunc.parametrize(
            GRAPH_PYTHON_FPTR + "," + JSON_TEST_LIST_PARAM, param_tuples, ids=test_ids
        )

        # Add graph_category parameter fixture for tensor_ir_pygraph
        if metafunc.function.__name__ == "test_tensor_ir_pygraph":
            metafunc.parametrize(
                GRAPH_CATEGORY_PARAM, [metafunc.config.getoption("graph_category")]
            )

    # Run a test from a json dictionary
    elif metafunc.function.__name__ == "test_json_graph":
        id = os.path.basename(metafunc.config.getoption("testPath"))
        # Using keyword indirect allows us to call the associated fixture
        metafunc.parametrize(
            JSON_DICT_PATH,
            [metafunc.config.getoption("testPath")],
            indirect=True,
            ids=[id],
        )
        metafunc.parametrize(TEST_NAME_PARAM, metafunc.config.getoption("testName"))

import os, json
import importlib.util
import inspect


def pytest_addoption(parser):
    parser.addoption("--testName", action="append", default=[])
    parser.addoption("--testPath", action="store", default=None)


def get_python_graph_defs(path, module_name):
    
    filename = os.path.join(path, "{}.py".format(module_name))
    if os.path.exists(filename):
        spec = importlib.util.spec_from_file_location(name=module_name, location=filename)
        my_module =  importlib.util.module_from_spec(spec)
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

def createTestParamNameTuples(test_funcs, test_names, params_path, wanted_tests):
    tuples = []
    test_ids = []
    for test_name, func in zip(test_names, test_funcs):
        # Only look at tests we requested
        if not test_name in wanted_tests:
            continue
        
        filename = os.path.join(params_path, "{}.json".format(test_name))
        if os.path.exists(filename):
            params = []
            with open(filename) as ifh:
                params = json.load(ifh)

            cur_ids = ["{}[json({})]".format(test_name, i) for i in range(len(params))]

            for par in params:
                tuples.append((func,par))

            test_ids.extend(cur_ids)

    return tuples, test_ids


def pytest_generate_tests(metafunc):
    JSON_TEST_LIST_PARAM = "jparams"
    JSON_DICT_PATH = "json_dict"
    TEST_NAME_PARAM = "test_name"

    # Dynamically create tests for python_graph_test by identifying all test defs in a test directory
    if metafunc.function.__name__ == "test_python_graph": 
        # Find all the functions that define testgraphs in location testPath
        base_path = os.path.dirname(os.path.abspath(metafunc.config.getoption("testPath")))
        filename = os.path.basename(metafunc.config.getoption("testPath")).split(".")[-2]
        test_funcs, test_names = get_python_graph_defs(base_path, filename)

        # TODO(@mbreughe): make this path a command line option
        params_path = os.path.join(base_path, "graph_input")

        # TODO(@mbreughe): Parsing all the associated json files may be time consuming
        # TODO: we actually don't need to load all of them, only the ones in the file specified. 
        # We could delay it for just-in-time discovery if there are too many tests
        param_tuples, test_ids = createTestParamNameTuples(test_funcs, test_names, params_path, metafunc.config.getoption("testName"))

        metafunc.parametrize(TEST_NAME_PARAM+","+JSON_TEST_LIST_PARAM, param_tuples, ids=test_ids)


    elif metafunc.function.__name__ == "test_json_graph":
        id=os.path.basename(metafunc.config.getoption("testPath"))
        # Using keyword indirect allows us to call the associated fixture
        metafunc.parametrize(JSON_DICT_PATH, [metafunc.config.getoption("testPath")], indirect=True, ids=[id])
        metafunc.parametrize(TEST_NAME_PARAM, metafunc.config.getoption("testName"))
        
        


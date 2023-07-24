import os, json
import importlib.util
import inspect



def get_python_graph_defs(path, module_name):
    
    filename = os.path.join(path, "{}.py".format(module_name))
    if os.path.exists(filename):
        spec = importlib.util.spec_from_file_location(name="basic_tests", location=filename)
        my_module =  importlib.util.module_from_spec(spec)
        loaded_m = spec.loader.exec_module(my_module)

    test_funcs = []
    test_names = []
    for name, obj in inspect.getmembers(my_module):
        if inspect.isfunction(obj):
            test_funcs.append(obj)
            test_names.append(name)

    return (test_funcs, test_names)


def pytest_generate_tests(metafunc):
    JSON_TEST_LIST_PARAM = "jparams"
    # For functions with "params" as argument
    if JSON_TEST_LIST_PARAM in metafunc.fixturenames:
        # Find the json with the coressponding name
        # TODO(@mbreughe): make the base path a variable
        base_path = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(base_path, "json", "{}.json".format(metafunc.function.__name__))
        if os.path.exists(filename):
            params = []
            with open(filename) as ifh:
                params = json.load(ifh)

            test_ids = ["json({})".format(i) for i in range(len(params))]

            metafunc.parametrize(JSON_TEST_LIST_PARAM, params, ids=test_ids)
        else:
            metafunc.parametrize(JSON_TEST_LIST_PARAM, [])

    MAGIC_STR = "test_name"
    if MAGIC_STR in metafunc.fixturenames: 
        base_path = os.path.dirname(os.path.abspath(__file__))
        module_path = os.path.join(base_path, "python_graph_defs")
        test_funcs, test_names = get_python_graph_defs(module_path, "basic_tests")

        metafunc.parametrize(MAGIC_STR, test_funcs, ids=test_names)


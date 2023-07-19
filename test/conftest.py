import os, json

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


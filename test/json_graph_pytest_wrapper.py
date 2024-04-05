import pytest
from json_graph_test import read_json_test_dict, run_test_from_json_definition, setup_test_graph_from_json

# A helper function to read json dictionaries
# @note: scope tells us that the dictionary is being loaded only once
@pytest.fixture(scope="module")
def json_dict(request):
    fname = request.param
    return read_json_test_dict(fname)

# Main entry point for json defined graphs
# @param json_dict: implicit call to a fixture using the json file name provided on the command line
# @param test_name: the specific test to be ran
def test_json_graph(json_dict, test_name):
    assert test_name in json_dict
    testGraph = setup_test_graph_from_json(json_dict[test_name], -1)
    run_test_from_json_definition(testGraph, json_dict[test_name])
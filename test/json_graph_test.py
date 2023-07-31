
import json
import os
import cudnn
import pytest

from test_graph import test_graph, operation

# A helper function to read json dictionaries
# @note: scope tells us that the dictionary is being loaded only once
@pytest.fixture(scope="module")
def json_dict(request):
    fname = request.param
    assert os.path.exists(fname)    

    with open(fname) as ifh:
        json_tests = json.load(ifh)

    return json_tests

# Main entry point for json defined graphs
# @param json_dict: implicit call to a fixture using the json file name provided on the command line
# @param test_name: the specific test to be ran
def test_json_graph(json_dict, test_name):
    assert test_name in json_dict
    run_test_from_json_definition(json_dict[test_name])

def run_test_from_json_definition(json_dict):
    testGraph = test_graph()

    jtensor_list = json_dict["tensors"]
    jnode_list = json_dict["nodes"]
    input_tensors = detect_input_tensors(jtensor_list, jnode_list)

    TGTensors = {}
    # Create a TestTensor for every input tensor
    for tensor in input_tensors:
        t = testGraph.tensor(**tensor)
        TGTensors[tensor["name"]] = t

    TGNodes = {}
    # Create all operation nodes (without input) and their associated output tensors
    for node in jnode_list:
        name = node["name"]
        # Create node without input and add to the graph
        operation = create_node(node)
        testGraph.nodes.append(operation)

        # Record it's output
        # TODO(@mbreughe): extend for operators with multiple nodes
        output_tensor = operation.output[0]
        # TODO(@mbreughe): generalize output name retrieval
        output_name = node["Y"]

        TGTensors[output_name] = output_tensor
        TGNodes[name] = operation

    # Finalize the connections by adding inputs to the operation nodes
    for name, node in TGNodes.items():
        jnode = None
        for n in jnode_list:
            if n["name"] == name:
                jnode = n
        node.set_kwargs(create_kwargs(jnode, TGTensors))

    # TODO(@mbreughe)
    # we can do a front-end test by using the json dimensions of the virtual tensors

    graph = testGraph.build_cudnn_graph()

    print(graph)

    # TODO(@mbreughe): read in rtol/atol from json
    testGraph.cudnn_execute_and_compare_to_reference(atol=1e-2,rtol=1e-2)
    
# TODO(@mbreughe): generalize
def create_node(node_params):
    name = node_params["name"]
    if node_params["operation"] == "conv":
        return test_graph.create_operation(cudnn.pygraph.conv, name)
    elif node_params["operation"] == "pointwise":
        return test_graph.create_operation(cudnn.pygraph.relu, name)

# TODO(@mbreughe): generalize
def create_kwargs(jnode, TGTensors):
    KEYS_TO_KEEP = ["dim", "dilation", "stride", "is_virtual", "name"]
    kwargs = {}
    for k in KEYS_TO_KEEP:
        if k in jnode:
            kwargs[k] = jnode[k]
    if jnode["operation"] == "conv":
        kwargs['padding'] = jnode['pad']
        kwargs['image'] = TGTensors[jnode['X']]
        kwargs['weight'] = TGTensors[jnode['W']]
    elif jnode["operation"] == "pointwise":
        kwargs['input'] = TGTensors[jnode['X']]
    
    return kwargs

# TODO(@mbreughe): generalize
def detect_input_tensors(tensors, nodes):
    output_tensors = []
    for node in nodes:
        if node["operation"] == "conv":
            output_tensors.append(node["Y"])
        elif node["operation"] == "pointwise":
            output_tensors.append(node["Y"])

    input_tensors = [tensor for tensor in tensors if not tensor["name"] in output_tensors]
    return input_tensors
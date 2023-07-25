
import torch, json
import os
import pycudnn
import pytest

from test_graph import TestGraph, Operation

# A helper function to read json dictionaries
@pytest.fixture
def json_dict(request):
    fname = request.param
    assert os.path.exists(fname)    

    with open(fname) as ifh:
        json_tests = json.load(ifh)

    return json_tests

# Main entry point: it will call the json_dict fixture,
# and use the test_name passed by the command line through conftest.py
def test_json_graph(json_dict, test_name):
    assert test_name in json_dict
    runTestFromJsonDefinition(json_dict[test_name])

def runTestFromJsonDefinition(json_dict):
    testGraph = TestGraph()

    jtensor_list = json_dict["tensors"]
    jnode_list = json_dict["nodes"]
    input_tensors = detectInputTensors(jtensor_list, jnode_list)

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
        operation = createNode(node)
        testGraph.nodes.append(operation)

        # Record it's output
        output_tensor = operation.output
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
        node.setKwargs(createKwargs(jnode, TGTensors))

    # TODO(@mbreughe)
    # we can do a front-end test by using the json dimensions of the virtual tensors

    graph = testGraph.buildPyCudnnGraph()

    print(graph)

    # TODO(@mbreughe): read in rtol/atol from json
    testGraph.cudnnExecuteAndCompareToReference(atol=1e-2,rtol=1e-2)
    
# TODO(@mbreughe): generalize
def createNode(node_params):
    name = node_params["name"]
    if node_params["operation"] == "conv":
        return TestGraph.createOperation(pycudnn.pygraph.conv, name)
    elif node_params["operation"] == "pointwise":
        return TestGraph.createOperation(pycudnn.pygraph.relu, name)

# TODO(@mbreughe): generalize
def createKwargs(jnode, TGTensors):
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
def detectInputTensors(tensors, nodes):
    output_tensors = []
    for node in nodes:
        if node["operation"] == "conv":
            output_tensors.append(node["Y"])
        elif node["operation"] == "pointwise":
            output_tensors.append(node["Y"])

    input_tensors = [tensor for tensor in tensors if not tensor["name"] in output_tensors]
    return input_tensors
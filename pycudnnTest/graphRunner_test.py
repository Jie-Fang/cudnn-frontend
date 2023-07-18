import pytest
import torch

from utils import TestGraph, TestTensor

# Dictionaries defined in test_conv_relu.json
def test_conv_relu(params):
    testGraph = TestGraph()
    X = testGraph.tensor(dim=params["in_dim"])
    W = testGraph.tensor(dim=params["filter_dim"])
    
    conv_out = testGraph.conv(image = X, weight = W, padding = params["padding"], stride = params["stride"], dilation = params["dilation"])
    Y = testGraph.relu(input = conv_out)

    graph, variant_pack, workspace = testGraph.buildPyCudnnGraph()

    # Front-end test: check shape inferencing
    assert params["expected_conv_out_dim"] == conv_out.pyCudnnTensor.get_dim()

    print(graph)
    
    graph.execute(variant_pack, workspace)
    Y_actual = testGraph.output_tensors[-1]

    Y_expected = testGraph.getReference()[-1]

    # Compare with reference
    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-2, rtol=1e-2)

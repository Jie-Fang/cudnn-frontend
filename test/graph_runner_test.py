import pytest
import torch

from test_graph import TestGraph, TestTensor

# Dictionaries defined in test_conv_relu.json
def test_conv_relu(jparams):
    testGraph = TestGraph()
    X = testGraph.tensor(dim=jparams["in_dim"])
    W = testGraph.tensor(dim=jparams["filter_dim"])
    
    conv_out = testGraph.conv(image = X, weight = W, padding = jparams["padding"], stride = jparams["stride"], dilation = jparams["dilation"])
    Y = testGraph.relu(input = conv_out)

    graph, variant_pack, workspace = testGraph.buildPyCudnnGraph()

    # Front-end test: check shape inferencing
    assert jparams["expected_conv_out_dim"] == conv_out.pyCudnnTensor.get_dim()

    print(graph)
    
    graph.execute(variant_pack, workspace)
    Y_actual = testGraph.output_tensors[-1]

    Y_expected = testGraph.getReference()[-1]

    # Compare with reference
    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-2, rtol=1e-2)

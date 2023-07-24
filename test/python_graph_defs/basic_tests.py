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

    graph = testGraph.buildPyCudnnGraph()

    # Front-end test: check shape inferencing
    assert jparams["expected_conv_out_dim"] == conv_out.pyCudnnTensor.get_dim()

    print(graph)

    # Note that the below can be replaced by testGraph.cudnnExecuteAndCompareToReference(atol=1e-2,rtol=1e-2)

    # Run the pycudnn graph
    workspace, variant_pack = testGraph.createWorkspaceAndVariantPack()
    testGraph.cudnn_graph.execute(variant_pack, workspace)

    # Run the reference
    ref_outputs = testGraph.calcReference()

    # Compare with reference
    for Y_expected, Y_actual in zip(ref_outputs, testGraph.getOutputs()):
        torch.testing.assert_close(Y_expected, Y_actual, atol=1e-2, rtol=1e-2)
import pytest
import torch

from utils import TestGraph, TestTensor, Convolution, ReLU

@pytest.mark.parametrize("in_dim, filter_dim, padding, stride, dilation, expected_conv_out_dim", [
    ([20, 40, 30, 40], [54, 40, 3, 4], [0,1], [2,3], [1,1], [20, 54, 14, 13]),
    ([16,128,256,256], [128,128,3,3], [1,1], [1,1], [1,1], [16, 128, 256, 256])
])
def test_conv_relu(in_dim, filter_dim, padding, stride, dilation, expected_conv_out_dim):
    testGraph = TestGraph()
    X = testGraph.addTensor(in_dim)
    W = testGraph.addTensor(filter_dim)
    
    conv_out = testGraph.addOperation(Convolution(image = X, weight = W, padding = padding, stride = stride, dilation = dilation))
    Y = testGraph.addOperation(ReLU(input = conv_out))

    graph, variant_pack, workspace = testGraph.buildPyCudnnGraph()

    # Front-end test: check shape inferencing
    assert expected_conv_out_dim == conv_out.pyCudnnTensor.get_dim()

    print(graph)
    
    graph.execute(variant_pack, workspace)
    Y_actual = testGraph.output_tensors[-1]

    Y_expected = testGraph.getReference()[-1]

    # Compare with reference
    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-2, rtol=1e-2)

def test_conv_relu_as_dict(params):
    testGraph = TestGraph()
    X = testGraph.addTensor(params["in_dim"])
    W = testGraph.addTensor(params["filter_dim"])
    
    conv_out = testGraph.addOperation(Convolution(image = X, weight = W, padding = params["padding"], stride = params["stride"], dilation = params["dilation"]))
    Y = testGraph.addOperation(ReLU(input = conv_out))

    graph, variant_pack, workspace = testGraph.buildPyCudnnGraph()

    # Front-end test: check shape inferencing
    assert params["expected_conv_out_dim"] == conv_out.pyCudnnTensor.get_dim()

    print(graph)
    
    graph.execute(variant_pack, workspace)
    Y_actual = testGraph.output_tensors[-1]

    Y_expected = testGraph.getReference()[-1]

    # Compare with reference
    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-2, rtol=1e-2)

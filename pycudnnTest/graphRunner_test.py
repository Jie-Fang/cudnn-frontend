import pytest
import torch

from utils import TestGraph, TestTensor, Convolution, ReLU

def test_conv_relu():
    padding = [0,1]
    stride = [2,3]
    dilation = [1,1]

    testGraph = TestGraph()
    X = testGraph.addTensor([20, 40, 30, 40])
    W = testGraph.addTensor([54, 40, 3, 4])
    
    conv_out = testGraph.addOperation(Convolution(image = X, weight = W, padding = padding, stride = stride, dilation = dilation))
    Y = testGraph.addOperation(ReLU(input = conv_out))

    graph, variant_pack, workspace = testGraph.buildPyCudnnGraph()

    print(graph)
    graph.execute(variant_pack, workspace)
    Y_actual = testGraph.output_tensors[-1]

    Y_expected = testGraph.getReference()[-1]
   
    # Compare
    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    test_conv_relu()
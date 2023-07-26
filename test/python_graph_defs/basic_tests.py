import pytest
import torch
import pycudnn

from test_graph import TestGraph

def test_conv_relu(jparams, testGraph):
    X = testGraph.tensor(dim=jparams["in_dim"], layout = "NHWC")
    W = testGraph.tensor(dim=jparams["filter_dim"], layout = "NHWC")
    
    conv_out = testGraph.conv(name = "conv", image = X, weight = W, padding = jparams["padding"], stride = jparams["stride"], dilation = jparams["dilation"])
    Y = testGraph.relu(input = conv_out)

def test_batchnorm(jparams, testGraph):
    testGraph.set_io_data_type(pycudnn.data_type.FLOAT)
    
    N, C, H, W = jparams["in_dim"]
    X = testGraph.tensor(dim=jparams["in_dim"], data_type=pycudnn.data_type.HALF) 
    X.layout="NHWC"
    scale = testGraph.tensor(dim=[1, C, 1, 1], data_type=pycudnn.data_type.FLOAT)
    bias = testGraph.tensor(dim=[1, C, 1, 1], data_type=pycudnn.data_type.FLOAT)
    in_running_mean = testGraph.tensor(dim=[1, C, 1, 1], data_type=pycudnn.data_type.FLOAT)
    in_running_var = testGraph.tensor(dim=[1, C, 1, 1], data_type=pycudnn.data_type.FLOAT)

    epsilon = testGraph.tensor_cpu_constant(1e-03, dim=[1,1,1,1], data_type=pycudnn.data_type.FLOAT)
    momentum = testGraph.tensor_cpu_constant(0.1, dim=[1,1,1,1], data_type=pycudnn.data_type.FLOAT)

    (Y, saved_mean, saved_inv_var, out_running_mean, out_running_var) = testGraph.batchnorm(name = "BN"
                        , norm_forward_phase = pycudnn.norm_forward_phase.TRAINING
                        , input = X
                        , scale = scale, bias = bias
                        , in_running_mean = in_running_mean, in_running_var = in_running_var
                        , epsilon = epsilon, momentum = momentum)
    
    # TODO: set_data_type. Also allow chaining by returning the tensor 
    Y.set_data_type(pycudnn.data_type.HALF)
 
import pytest
import torch

from test_graph import TestGraph

def test_conv_relu(jparams, testGraph):
    X = testGraph.tensor(dim=jparams["in_dim"])
    W = testGraph.tensor(dim=jparams["filter_dim"])
    
    conv_out = testGraph.conv(image = X, weight = W, padding = jparams["padding"], stride = jparams["stride"], dilation = jparams["dilation"])
    Y = testGraph.relu(input = conv_out)

def test_conv(jparams, testGraph):
    X = testGraph.tensor(dim=jparams["in_dim"])
    W = testGraph.tensor(dim=jparams["filter_dim"])
    
    conv_out = testGraph.conv(image = X, weight = W, padding = jparams["padding"], stride = jparams["stride"], dilation = jparams["dilation"])
 
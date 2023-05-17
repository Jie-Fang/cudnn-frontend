import pycudnn
import numpy as np
import cupy as cp

print("Example 1. Executing the conv + bias graph")

if pycudnn.is_cudnn_supported() == False:
    print("cudnn version is not supported")
    exit(0)

graph = pycudnn.pygraph("graph0", compute_data_type = pycudnn.data_type.FLOAT)

image  = graph.tensor(name = "image", dim = [4,16,56,56], data_type = pycudnn.data_type.HALF)
weight = graph.tensor(name = "weight", dim = [16,16,3,3], data_type = pycudnn.data_type.HALF)
bias   = graph.tensor(name = "bias", dim = [1,16,1,1], data_type = pycudnn.data_type.HALF)

response = graph.conv(name = "conv", image = image, weight = weight, padding = [1,1], stride = [1,1], dilation = [1,1], compute_type = pycudnn.data_type.FLOAT)
response.set_is_virtual(True).set_data_type(pycudnn.data_type.FLOAT)

output = graph.bias(name = "bias", input = response, bias = bias, compute_type = pycudnn.data_type.FLOAT)
output.set_data_type(pycudnn.data_type.HALF)

graph.build()

X_gpu = cp.full([4,16,56,56], 1, dtype=cp.half)
W_gpu = cp.full([16,16,3,3], 1, dtype=cp.half)
B_gpu = cp.full([1,16,1,1], 2, dtype=cp.half)
Y_gpu = cp.full([4,16,56,56], 0, dtype=cp.half)

graph.execute({image : X_gpu, weight : W_gpu, bias : B_gpu, output : Y_gpu})

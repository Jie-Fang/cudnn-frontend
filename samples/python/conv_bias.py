import pycudnn
import numpy as np
import cupy as cp

print("Example 1. Executing the conv + bias graph")

if pycudnn.get_cudnn_version() < 8500:
    print("cudnn version does not conv+bias fusion")
    exit(0)

graph = pycudnn.pygraph("graph0", io_data_type = pycudnn.data_type.HALF, intermediate_data_type = pycudnn.data_type.FLOAT, compute_data_type = pycudnn.data_type.FLOAT)

image  = graph.tensor(name = "image", dim = [4,16,56,56])
weight = graph.tensor(name = "weight", dim = [16,16,3,3])
bias   = graph.tensor(name = "bias", dim = [1,16,1,1])

response = graph.conv(name = "conv", image = image, weight = weight, padding = [1,1], stride = [1,1], dilation = [1,1])
response.set_is_virtual(True)

output = graph.bias(name = "bias", input = response, bias = bias)

graph.build()

X_gpu = cp.full([4,16,56,56], 1, dtype=cp.half)
W_gpu = cp.full([16,16,3,3], 1, dtype=cp.half)
B_gpu = cp.full([1,16,1,1], 2, dtype=cp.half)
Y_gpu = cp.full([4,16,56,56], 0, dtype=cp.half)

graph.execute({image : X_gpu, weight : W_gpu, bias : B_gpu, output : Y_gpu})

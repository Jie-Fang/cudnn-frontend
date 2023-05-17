import pycudnn
import numpy as np
import cupy as cp
import sys
print("Example 2. Executing the Matmul + bias + relu graph")

if pycudnn.is_cudnn_supported() == False:
    print("cudnn version is not supported")
    exit(0)

graph = pycudnn.pygraph("nvfuser")

image = graph.tensor(name = "image", dim = [4,16,56], data_type = pycudnn.data_type.HALF)
weight = graph.tensor(name = "weight", dim = [4,56,16], data_type = pycudnn.data_type.HALF)
bias = graph.tensor(name = "bias", dim = [4,16,16], data_type = pycudnn.data_type.HALF)

response = graph.matmul(name = "matmul", image = image, weight = weight, compute_type = pycudnn.data_type.FLOAT)
response.set_is_virtual(True).set_data_type(pycudnn.data_type.FLOAT)

output = graph.bias(name = "bias", input = response, bias = bias, compute_type = pycudnn.data_type.FLOAT)
output.set_is_virtual(True).set_data_type(pycudnn.data_type.FLOAT)

relu = graph.relu(name = "relu", input = output, compute_type = pycudnn.data_type.FLOAT)
relu.set_data_type(pycudnn.data_type.HALF)

graph.build()

X_cpu = np.full([4,16,56], 1, dtype=np.half)
W_cpu = np.full([4,56,16], 1, dtype=np.half)
B_cpu = np.full([4,16,16], 2, dtype=np.half)

X_gpu = cp.asarray(X_cpu)
W_gpu = cp.asarray(W_cpu)
B_gpu = cp.asarray(B_cpu)
Y_gpu = cp.full([4,16,16], 0, dtype=cp.half)

graph.execute({image : X_gpu, weight :  W_gpu, bias :  B_gpu, relu :  Y_gpu})

Y_actual = cp.asnumpy(Y_gpu)

Y_expected = np.matmul(X_cpu, W_cpu) + B_cpu
Y_expected[Y_expected < 0] = 0

np.testing.assert_allclose(Y_actual, Y_expected)

from cuda import cuda, cudart
import pycudnn
import numpy as np

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

h_X = np.full(4*16*56, 1).astype(dtype=np.half)
h_W = np.full(4*56*16, 1).astype(dtype=np.half)
h_B = np.full(4*16*16, 2).astype(dtype=np.half)
h_Y = np.full(4*16*16, 0).astype(dtype=np.half)

err, x_dptr = cudart.cudaMalloc(4*16*56)
err, w_dptr = cudart.cudaMalloc(4*56*16)
err, b_dptr = cudart.cudaMalloc(4*16*16)
err, y_dptr = cudart.cudaMalloc(4*16*16)

cuda.cuMemcpyHtoD(x_dptr, h_X, 4*16*56)
cuda.cuMemcpyHtoD(w_dptr, h_W, 4*56*16)
cuda.cuMemcpyHtoD(b_dptr, h_B, 4*16*16)
cuda.cuMemcpyHtoD(y_dptr, h_Y, 4*16*16)

graph.execute({image :x_dptr, weight : w_dptr, bias : b_dptr, relu : y_dptr})

cuda.cuMemcpyDtoH(h_Y, y_dptr, 4*16*16)

cudart.cudaFree(x_dptr)
cudart.cudaFree(w_dptr)
cudart.cudaFree(b_dptr)
cudart.cudaFree(y_dptr)
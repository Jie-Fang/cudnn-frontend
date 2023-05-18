from cuda import cuda, cudart
import pycudnn
import numpy as np

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

h_X = np.full(4*16*56*56, 1).astype(dtype=np.half)
h_W = np.full(16*16*3*3, 1).astype(dtype=np.half)
h_B = np.full(1*16*1*1, 2).astype(dtype=np.half)
h_Y = np.full(4*16*56*56, 0).astype(dtype=np.half)

err, x_dptr = cudart.cudaMalloc(4*16*56*56)
err, w_dptr = cudart.cudaMalloc(16*16*3*3)
err, b_dptr = cudart.cudaMalloc(1*16*1*1)
err, y_dptr = cudart.cudaMalloc(4*16*56*56)

cuda.cuMemcpyHtoD(x_dptr, h_X, 4*16*56*56)
cuda.cuMemcpyHtoD(w_dptr, h_W, 16*16*3*3)
cuda.cuMemcpyHtoD(b_dptr, h_B, 1*16*1*1)
cuda.cuMemcpyHtoD(y_dptr, h_Y, 4*16*56*56)

graph.execute({image :x_dptr, weight : w_dptr, bias : b_dptr, output : y_dptr})

cuda.cuMemcpyDtoH(h_Y, y_dptr, 4*16*56*56)

cudart.cudaFree(x_dptr)
cudart.cudaFree(w_dptr)
cudart.cudaFree(b_dptr)
cudart.cudaFree(y_dptr)
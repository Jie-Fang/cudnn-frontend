from cuda import cuda, cudart
import pycudnn
import numpy as np

if pycudnn.is_cudnn_supported() == False:
    exit(0)

graph = pycudnn.pygraph("nvfuser")

image = graph.insert_tensor(name = "image", dim = [4,16,56], data_type = pycudnn.data_type.HALF)
weight = graph.insert_tensor(name = "weight", dim = [4,56,16], data_type = pycudnn.data_type.HALF)
bias = graph.insert_tensor(name = "bias", dim = [4,16,16], data_type = pycudnn.data_type.HALF)

response = graph.insert_matmul(name = "matmul", image = image, weight = weight, compute_type = pycudnn.data_type.FLOAT)
response.set_is_virtual(True)

output = graph.insert_bias(name = "bias", input = response, bias = bias, compute_type = pycudnn.data_type.FLOAT)
output.set_is_virtual(True)

relu = graph.insert_relu(name = "relu", input = output, compute_type = pycudnn.data_type.FLOAT)

graph.build()

h_X = np.full(image.get_size(), 1).astype(dtype=np.half)
h_W = np.full(weight.get_size(), 1).astype(dtype=np.half)
h_B = np.full(bias.get_size(), 2).astype(dtype=np.half)
h_Y = np.full(relu.get_size(), 0).astype(dtype=np.half)

err, x_dptr = cudart.cudaMalloc(image.get_size())
err, w_dptr = cudart.cudaMalloc(weight.get_size())
err, b_dptr = cudart.cudaMalloc(bias.get_size())
err, y_dptr = cudart.cudaMalloc(output.get_size())

cuda.cuMemcpyHtoD(x_dptr, h_X, image.get_size())
cuda.cuMemcpyHtoD(w_dptr, h_W, weight.get_size())
cuda.cuMemcpyHtoD(b_dptr, h_B, bias.get_size())
cuda.cuMemcpyHtoD(y_dptr, h_Y, relu.get_size())

print("Executing the Matmul + bias + relu graph")
graph.execute({image :x_dptr, weight : w_dptr, bias : b_dptr, relu : y_dptr})

cuda.cuMemcpyDtoH(h_Y, y_dptr, relu.get_size())

cudart.cudaFree(x_dptr)
cudart.cudaFree(w_dptr)
cudart.cudaFree(b_dptr)
cudart.cudaFree(y_dptr)

# Sample that builds a Convolution Operatio

import cudnn_frontend_blocks as fe
from cuda import cuda, cudart

import numpy as np

x_dim = [16,32,56,56]
w_dim = [32,32,3,3]

ctx = fe.cuDNNFEContext()
print(ctx)

graph = fe.Graph("conv_graph", ctx)

conv0 = fe.convolution_node("conv0")
conv0.set_port_names([(fe.convolution_ports.X,"tensor0"), (fe.convolution_ports.W,"tensor1")])
conv0.set_padding([1,1])
conv0.set_stride([1,1])
conv0.set_dilation([1,1])

graph.add_node(conv0)

tensor0 = fe.tensor_properties("tensor0")
tensor0.set_dim(x_dim)
graph.add_tensor(tensor0)

tensor1 = fe.tensor_properties("tensor1")
tensor1.set_dim(w_dim)
graph.add_tensor(tensor1)

# Future might have a clone from method.
tensor2 = fe.tensor_properties("tensor2")
tensor2.set_dim(x_dim)
graph.add_tensor(tensor2)

pw0= fe.pointwise_node("pw0")
# pw0.set_inputs(["conv0::Y", "tensor2"])
pw0.set_port_names([(fe.pointwise_ports.X,"conv0::Y"), (fe.pointwise_ports.B,"tensor2")])
pw0.set_mode("Add")

graph.add_node(pw0)

graph.tensor_at("conv0::Y").set_is_virtual(True)

graph.infer_shapes()

graph.build()

# Allocate in numpy
h_X = np.full(tensor0.get_size(), 0.1).astype(dtype=np.half)
h_W = np.full(tensor0.get_size(), 1).astype(dtype=np.half)
h_B = np.full(tensor0.get_size(), 0).astype(dtype=np.half)
h_Y = np.full(tensor0.get_size(), 0).astype(dtype=np.half)

err, x_dptr = cudart.cudaMalloc(tensor0.get_size())
err, w_dptr = cudart.cudaMalloc(tensor1.get_size())
err, b_dptr = cudart.cudaMalloc(tensor2.get_size())
err, y_dptr = cudart.cudaMalloc(tensor2.get_size())

cuda.cuMemcpyHtoD(x_dptr, h_X, tensor0.get_size())
cuda.cuMemcpyHtoD(w_dptr, h_W, tensor1.get_size())
cuda.cuMemcpyHtoD(b_dptr, h_B, tensor2.get_size())
cuda.cuMemcpyHtoD(y_dptr, h_Y, tensor2.get_size())

graph.execute({"tensor0" :x_dptr, "tensor1" : w_dptr, "pw0::Y" : y_dptr, "tensor2" : b_dptr})

cuda.cuMemcpyDtoH(h_Y, y_dptr, tensor2.get_size())

cudart.cudaFree(x_dptr)
cudart.cudaFree(w_dptr)
cudart.cudaFree(b_dptr)
cudart.cudaFree(y_dptr)
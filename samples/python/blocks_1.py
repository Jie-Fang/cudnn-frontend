# Sample that builds a Convolution Operatio

import cudnn_frontend_blocks as fe

ctx = fe.cuDNNFEContext()
print(ctx)

graph = fe.Graph("conv_graph", ctx)

conv0 = fe.convolution_node("conv0")
# conv0.set_inputs(["tensor0", "tensor1"])
conv0.set_port_names([(fe.convolution_ports.X,"tensor0"), (fe.convolution_ports.W,"tensor1")])
conv0.set_padding([1,1])
conv0.set_stride([1,1])
conv0.set_dilation([1,1])

graph.add_node(conv0)

tensor0 = fe.tensor_properties("tensor0")
tensor0.set_dim([16,32,56,56])
graph.add_tensor(tensor0)

tensor1 = fe.tensor_properties("tensor1")
tensor1.set_dim([32,32,3,3])
graph.add_tensor(tensor1)

graph.infer_shapes()

graph.build()
import cudnn_frontend_blocks as fe

graph = fe.pygraph("nvfuser")

image = graph.add_tensor(name = "image", dim = [4,32,56,56])
weight = graph.add_tensor(name = "weight", dim = [32,32,3,3])
bias = graph.add_tensor(name = "bias", dim = [1,32,1,1])

response = graph.add_conv(name = "conv", image = image, weight = weight, padding = [1,1], stride = [1,1], dilation = [1,1])

output = graph.add_bias(name = "relu", input = response, bias = bias)

graph.build()
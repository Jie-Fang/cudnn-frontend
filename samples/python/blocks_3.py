import cudnn_frontend_blocks as fe

graph = fe.pygraph("nvfuser")

image = graph.add_tensor(name = "image", dim = [4,64,32,32])
weight = graph.add_tensor(name = "weight", dim = [32,64,3,3])

output = graph.add_conv(name = "conv", image = image, weight = weight, padding = [1,1], stride = [1,1], dilation = [1,1])


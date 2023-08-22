import cudnn

def test_conv_relu(jparams, testgraph):
    X = testgraph.tensor(dim=jparams["in_dim"], layout = "NHWC")
    W = testgraph.tensor(dim=jparams["filter_dim"], layout = "NHWC")
    
    conv_out = testgraph.conv_fprop(name = "conv", image = X, weight = W, padding = jparams["padding"], stride = jparams["stride"], dilation = jparams["dilation"])
    Y = testgraph.relu(input = conv_out)


#This test broke in merge commit 72b88c88f125382694a784bb4ce2535f62d09903
# def test_conv(jparams, testgraph):
#     X = testgraph.tensor(dim=jparams["in_dim"], data_type = cudnn.data_type.FLOAT, layout = "NHWC")
#     W = testgraph.tensor(dim=jparams["filter_dim"], data_type = cudnn.data_type.FLOAT, layout = "NHWC")
    
#     conv_out = testgraph.conv_fprop(name = "conv", image = X, weight = W, padding = jparams["padding"], stride = jparams["stride"], dilation = jparams["dilation"])

def test_conv_relu_bias_relu(jparams, testgraph):
    X = testgraph.tensor(dim=jparams["in_dim"], layout = "NHWC")
    W = testgraph.tensor(dim=jparams["filter_dim"], layout = "NHWC")
    bias = testgraph.tensor(dim=jparams["bias_dim"], layout = "NHWC")

    conv_out = testgraph.conv_fprop(name = "conv", image = X, weight = W, padding = jparams["padding"], stride = jparams["stride"], dilation = jparams["dilation"])
    relu_1 = testgraph.relu(input = conv_out)
    bias_out = testgraph.bias(name = "bias", input = relu_1, bias = bias)
    relu_output = testgraph.relu(input=bias_out)

def test_dgrad_add(jparams, testgraph):
    testgraph.set_compute_data_type(cudnn.data_type.FLOAT)
    testgraph.set_io_data_type(cudnn.data_type.FLOAT)
    wTensor = testgraph.tensor(dim=jparams["filter_dim"], layout = "NHWC", data_type=cudnn.data_type.FLOAT)
    dyTensor = testgraph.tensor(dim=jparams["conv_out_dim"], layout = "NHWC", data_type=cudnn.data_type.FLOAT)
    bTensor =  testgraph.tensor(dim=jparams["dx_dim"], layout = "NHWC", data_type=cudnn.data_type.FLOAT)

    dxTensor = testgraph.conv_dgrad(name="dgrad", loss=dyTensor, filter=wTensor, padding = jparams["padding"], stride = jparams["stride"], dilation = jparams["dilation"])
    afterAdd = testgraph.add(name="add", a=dxTensor, b=bTensor)

def test_batchnorm(jparams, testgraph):
    testgraph.set_io_data_type(cudnn.data_type.FLOAT)
    
    N, C, H, W = jparams["in_dim"]
    X = testgraph.tensor(dim=jparams["in_dim"], data_type=cudnn.data_type.HALF) 
    X.layout="NHWC"
    scale = testgraph.tensor(dim=[1, C, 1, 1], data_type=cudnn.data_type.FLOAT)
    bias = testgraph.tensor(dim=[1, C, 1, 1], data_type=cudnn.data_type.FLOAT)
    in_running_mean = testgraph.tensor(dim=[1, C, 1, 1], data_type=cudnn.data_type.FLOAT)
    in_running_var = testgraph.tensor(dim=[1, C, 1, 1], data_type=cudnn.data_type.FLOAT)

    epsilon = testgraph.tensor_cpu_constant(1e-03, dim=[1,1,1,1], data_type=cudnn.data_type.FLOAT)
    momentum = testgraph.tensor_cpu_constant(0.1, dim=[1,1,1,1], data_type=cudnn.data_type.FLOAT)

    (Y, saved_mean, saved_inv_var, out_running_mean, out_running_var) = testgraph.batchnorm(name = "BN"
                        , norm_forward_phase = cudnn.norm_forward_phase.TRAINING
                        , input = X
                        , scale = scale, bias = bias
                        , in_running_mean = in_running_mean, in_running_var = in_running_var
                        , epsilon = epsilon, momentum = momentum)
    
    # TODO: set_data_type. Also allow chaining by returning the tensor 
    Y.set_data_type(cudnn.data_type.HALF)

def test_gemm(jparams, testgraph):
    B, M, N, K = jparams["in_dim"]

    image = testgraph.tensor(name = "image", dim = [B, M, K], data_type = cudnn.data_type.HALF)
    weight = testgraph.tensor(name = "weight", dim = [B, K, N], data_type = cudnn.data_type.HALF)

    gemm_output = testgraph.matmul(name = "mb_matmul", A = image, B = weight, compute_data_type = cudnn.data_type.FLOAT)
    #gemm_output.set_stride([M*N, N, 1])

def test_gemm_relu(jparams, testgraph):
    B, M, N, K = jparams["in_dim"]

    image = testgraph.tensor(name = "image", dim = [B, M, K], data_type = cudnn.data_type.HALF)
    weight = testgraph.tensor(name = "weight", dim = [B, K, N], data_type = cudnn.data_type.HALF)

    gemm_output = testgraph.matmul(name = "mb_matmul", A = image, B = weight, compute_data_type = cudnn.data_type.FLOAT)
    # Make intermediate tensor output row-major:
    gemm_output.set_stride([M*N, N, 1])
    relu_output = testgraph.relu(input=gemm_output)

def test_gemm_bias_relu(jparams, testgraph):
    B, M, N, K = jparams["in_dim"]

    image = testgraph.tensor(name = "image", dim = [B, M, K], data_type = cudnn.data_type.HALF)
    weight = testgraph.tensor(name = "weight", dim = [B, K, N], data_type = cudnn.data_type.HALF)
    bias = testgraph.tensor(name = "weight", dim = [B, M, N], data_type = cudnn.data_type.HALF)

    gemm_output = testgraph.matmul(name = "mb_matmul", A = image, B = weight, compute_data_type = cudnn.data_type.FLOAT)
    # Make intermediate tensor output row-major:
    gemm_output.set_stride([M*N, N, 1])

    bias_out = testgraph.bias(name = "bias", input = gemm_output, bias = bias)
    relu_output = testgraph.relu(input=bias_out)

def test_conv_reduction(jparams, testgraph):
    N, K, C, H, W = 4, 32, 16, 64, 64
    R, S = 3, 3
    padding = stride = dilation = [1, 1]
    # N = jparams["in_dim"][0]
    # H = jparams["in_dim"][2]
    # W = jparams["in_dim"][3]
    X = testgraph.tensor(dim=[N,C,H,W], layout = "NHWC")
    Weight = testgraph.tensor(dim=[K,C,R,S], layout = "NHWC")
    Y0 = testgraph.conv_fprop(image = X, weight = Weight, padding = padding, stride = stride, dilation = dilation)
    
    Y = testgraph.reduction(input = Y0, mode = cudnn.reduction_mode.ADD)
    Y.set_dim([N,1,H,W])
    Y.set_data_type(cudnn.data_type.FLOAT)

    # N, K, C, H, W = 4, 32, 16, 64, 64
    # R, S = 3, 3
    # padding = stride = dilation = [1, 1]

    # # Reference
    # X_gpu = torch.randn(N, C, H, W, dtype=torch.float16, device='cuda').to(memory_format=torch.channels_last)
    # W_gpu = torch.randn(K, C, R, S, dtype=torch.float16, device='cuda').to(memory_format=torch.channels_last)
    # # Perform convolution using FP32 computation while input and filter remain in FP16
    # with torch.cuda.amp.autocast(dtype=torch.float32):
    #     conv_output = torch.nn.functional.conv2d(X_gpu, W_gpu, padding=padding, stride=stride, dilation=dilation)
    #     Y_expected = conv_output.sum(dim=1)

    # # Cudnn code
    # graph = cudnn.pygraph(io_data_type = cudnn.data_type.HALF, intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)
    # X = graph.tensor(name = "X", dim = X_gpu.size(), stride = X_gpu.stride(), data_type = convert_to_cudnn_type(X_gpu.dtype))
    # Weight = graph.tensor(name = "W", dim = W_gpu.size(), stride = W_gpu.stride(), data_type = convert_to_cudnn_type(W_gpu.dtype))

    # Y0 = graph.conv_fprop(image = X, weight = Weight, padding = padding, stride = stride, dilation = dilation)
    
    # Y = graph.reduction(input = Y0, mode = cudnn.reduction_mode.ADD)
    # Y.set_output(True).set_dim([N,1,H,W]).set_data_type(cudnn.data_type.FLOAT)
    

 
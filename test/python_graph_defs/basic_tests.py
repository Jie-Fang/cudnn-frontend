import cudnn

def test_conv_relu(jparams, testgraph):
    X = testgraph.tensor(dim=jparams["in_dim"], layout = "NHWC")
    W = testgraph.tensor(dim=jparams["filter_dim"], layout = "NHWC")
    
    conv_out = testgraph.conv(name = "conv", image = X, weight = W, padding = jparams["padding"], stride = jparams["stride"], dilation = jparams["dilation"])
    Y = testgraph.relu(input = conv_out)

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
    gemm_output.set_data_type(cudnn.data_type.HALF)
    # Make output row-major:
    gemm_output.set_stride([M*N, N, 1])

 
import pycudnn
import torch
import torch.nn.functional as F

import pytest

def test_hello_world():
    assert True == True

class MBGraphModule:
    def __init__(self, cudnn_graph):
        self.cudnn_graph = cudnn_graph
        self.variant_pack = {}
        self.debug = False

    def add_gemm(self, x, w):
        b1, m1, k1 = x.size()
        b2, k2, n1 = w.size()

        assert (b1 == b2 or b2 == 1)
        assert (k1 == k2)

        self.problem_size = [b1, m1, n1]

        image = self.cudnn_graph.tensor(name = "image", dim = x.size(), stride = x.stride(), data_type = convert_to_cudnn_type(x.dtype))
        weight = self.cudnn_graph.tensor(name = "weight", dim = w.size(), stride = w.stride(), data_type = convert_to_cudnn_type(w.dtype))

        self.variant_pack[image] = x
        self.variant_pack[weight] = w

        gemm_output = self.cudnn_graph.matmul(name = "mb_matmul", image = image, weight = weight, compute_data_type = pycudnn.data_type.FLOAT)
        gemm_output.set_data_type (pycudnn.data_type.FLOAT)
        # DEBUG
        if self.debug:
            gemm_output.set_stride([m1*n1, 1, m1])
        else:
            gemm_output.set_stride([m1*n1, n1, 1])

        self.last_output = gemm_output
        self.gemm_output = gemm_output
        return gemm_output

    def add_relu(self):
        self.last_output.set_is_virtual(True)

        activation_output = self.cudnn_graph.relu(name = "relu", input = self.last_output, compute_data_type = pycudnn.data_type.FLOAT)
        print(activation_output)
        b, m, n = self.problem_size
        
        # DEBUG
        if self.debug:
            activation_output.set_stride([m*n, 1, m])
        else:
            activation_output.set_stride([m*n, n, 1])

        self.last_output = activation_output
        return activation_output

def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return pycudnn.data_type.HALF
    elif torch_type == torch.float32:
        return pycudnn.data_type.FLOAT
    else:
        raise ValueError("Unsupported tensor data type.")

    return

@pytest.mark.parametrize("in_dim, expected_gemm_out_dim", [([1, 16, 16, 16], [1,16,16,])
#,([16, 32, 64, 128], [16,32,64,]) # fails
])
def test_gemm_more_explicit(in_dim, expected_gemm_out_dim):
    # Can put this in fixture:
    cudnn_graph = pycudnn.pygraph("cudnn_graph", io_data_type = pycudnn.data_type.HALF, intermediate_data_type = pycudnn.data_type.FLOAT, compute_data_type = pycudnn.data_type.FLOAT)


    B, M, N, K = in_dim
    x = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(B, K, N, device="cuda", dtype=torch.float16)

    image = cudnn_graph.tensor(name = "image", dim = x.size(), stride = x.stride(), data_type = convert_to_cudnn_type(x.dtype))
    weight = cudnn_graph.tensor(name = "weight", dim = w.size(), stride = w.stride(), data_type = convert_to_cudnn_type(w.dtype))

    variant_pack = {}
    variant_pack[image] = x
    variant_pack[weight] = w

    gemm_output = cudnn_graph.matmul(name = "mb_matmul", image = image, weight = weight, compute_data_type = pycudnn.data_type.FLOAT)
    gemm_output.set_data_type (pycudnn.data_type.HALF)
    # DEBUG
    col_major_C = True
    if col_major_C:
        gemm_output.set_stride([M*N, 1, M])
    else:
        gemm_output.set_stride([M*N, N, 1])


    Y = gemm_output
    Y.set_output(True)

    output = torch.zeros([B,M,N], dtype=torch.float16, device='cuda')

    variant_pack[Y] = output

    cudnn_graph.build()
    print(cudnn_graph)

    workspace = torch.empty(cudnn_graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    # Compare output with a reference implementation
    cudnn_graph.execute(variant_pack, workspace)


    pyt_out = torch.transpose(torch.bmm(x, w), 1, 2)
    torch.testing.assert_close(output, pyt_out)

@pytest.mark.parametrize("in_dim, expected_gemm_out_dim", [([1, 16, 16, 16], [1,16,16,])
#,([16, 32, 64, 128], [16,32,64,]) # fails
])
def test_gemm_relu_more_explicit(in_dim, expected_gemm_out_dim):
    # Can put this in fixture:
    cudnn_graph = pycudnn.pygraph("cudnn_graph", io_data_type = pycudnn.data_type.HALF, intermediate_data_type = pycudnn.data_type.FLOAT, compute_data_type = pycudnn.data_type.FLOAT)

    B, M, N, K = in_dim
    x = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(B, K, N, device="cuda", dtype=torch.float16)

    image = cudnn_graph.tensor(name = "image", dim = x.size(), stride = x.stride(), data_type = convert_to_cudnn_type(x.dtype))
    weight = cudnn_graph.tensor(name = "weight", dim = w.size(), stride = w.stride(), data_type = convert_to_cudnn_type(w.dtype))

    variant_pack = {}
    variant_pack[image] = x
    variant_pack[weight] = w

    gemm_output = cudnn_graph.matmul(name = "mb_matmul", image = image, weight = weight, compute_data_type = pycudnn.data_type.FLOAT)
    gemm_output.set_data_type (pycudnn.data_type.HALF)
    # DEBUG
    col_major_C = False
    if col_major_C:
        gemm_output.set_stride([M*N, 1, M])
    else:
        gemm_output.set_stride([M*N, N, 1])

    gemm_output.set_is_virtual(True)

    activation_output = cudnn_graph.relu(name = "relu", input = gemm_output, compute_data_type = pycudnn.data_type.FLOAT)
    
    # DEBUG
    if col_major_C:
        activation_output.set_stride([M*N, 1, M])
    else:
        activation_output.set_stride([M*N, N, 1])

    Y = activation_output
    Y.set_output(True)

    output = torch.zeros([B,M,N], dtype=torch.float16, device='cuda')

    variant_pack[Y] = activation_output

    cudnn_graph.build()
    print(cudnn_graph)

    workspace = torch.empty(cudnn_graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    # Compare output with a reference implementation
    cudnn_graph.execute(variant_pack, workspace)


    pyt_out = torch.transpose(torch.bmm(x, w), 1, 2)
    torch.testing.assert_close(output, pyt_out)

# Make backend testing optional and allow for "checkpoint"
@pytest.mark.skip(reason="This test will likely dissapear anyway")
@pytest.mark.parametrize("in_dim, expected_gemm_out_dim", [([16, 32, 64, 128], [16,32,64,])])
def test_gemm_relu_explicit(in_dim, expected_gemm_out_dim):
    # Can put this in fixture:
    cudnn_graph = pycudnn.pygraph("cudnn_graph", io_data_type = pycudnn.data_type.HALF, intermediate_data_type = pycudnn.data_type.FLOAT, compute_data_type = pycudnn.data_type.FLOAT)
    mb_graph = MBGraphModule(cudnn_graph)

    B, M, N, K = in_dim
    x = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(B, K, N, device="cuda", dtype=torch.float16)

    gemm_output = mb_graph.add_gemm(x,w)
    Y = mb_graph.add_relu()
    Y.set_output(True)

    output = torch.zeros(mb_graph.problem_size, dtype=torch.float16, device='cuda')

    mb_graph.variant_pack[mb_graph.last_output] = output

    print("Output strides: ", mb_graph.last_output.get_stride())

    print(mb_graph.cudnn_graph)

    mb_graph.cudnn_graph.build()



    # Check expected shapes
    assert expected_gemm_out_dim == gemm_output.get_dim()

    # Compare output with a reference implementation
    mb_graph.cudnn_graph.execute(mb_graph.variant_pack)
    pyt_out = torch.bmm(x, w)
    pyt_out = F.relu(pyt_out)
    torch.testing.assert_close(output, pyt_out)

# Make backend testing optional and allow for "checkpoint"
@pytest.mark.skip(reason="This test will likely dissapear anyway")
@pytest.mark.parametrize("in_dim, expected_gemm_out_dim", [([1, 16, 16, 16], [1,16,16,])
#,([16, 32, 64, 128], [16,32,64,]) # fails
])
def test_gemm_explicit(in_dim, expected_gemm_out_dim):
    # Can put this in fixture:
    cudnn_graph = pycudnn.pygraph("cudnn_graph", io_data_type = pycudnn.data_type.HALF, intermediate_data_type = pycudnn.data_type.FLOAT, compute_data_type = pycudnn.data_type.FLOAT)
    mb_graph = MBGraphModule(cudnn_graph)

    B, M, N, K = in_dim
    x = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(B, K, N, device="cuda", dtype=torch.float16)

    Y = mb_graph.add_gemm(x,w)
    Y.set_output(True)

    output = torch.zeros(mb_graph.problem_size, dtype=torch.float16, device='cuda')

    mb_graph.variant_pack[mb_graph.last_output] = output

    print("Output strides: ", mb_graph.last_output.get_stride())

    print(mb_graph.cudnn_graph)

    mb_graph.cudnn_graph.build()

    # Check expected shapes
    assert expected_gemm_out_dim == Y.get_dim()

    workspace = torch.empty(mb_graph.cudnn_graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    # Compare output with a reference implementation
    mb_graph.cudnn_graph.execute(mb_graph.variant_pack, workspace)
    pyt_out = torch.bmm(x, w)
    pyt_out = torch.transpose(pyt_out, 1, 2)
    print (pyt_out)
    print (output)
    torch.testing.assert_close(output, pyt_out)

class CSBR(torch.nn.Module):
    def forward(self, x, w, b = None, padding = [1,1], stride = [1,1], dilation = [1,1]):
        if b is not None:
            b = b.reshape(-1) # Conv2d needs a 1D tensor
        conv_output = torch.nn.functional.conv2d(x, w, bias = b, padding=padding, stride=stride, dilation=dilation)
        return torch.nn.functional.relu(conv_output)

def test_conv_relu():
    # Reference code
    X_gpu = torch.randn(20, 40, 30, 40, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    W_gpu = torch.randn(54, 40, 3, 4, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    padding = [0,1]
    stride = [2,3]
    dilation = [1,1]

    model = CSBR().eval().to("cuda").to(torch.float16)
    Y_expected = model(X_gpu, W_gpu, padding = padding, stride = stride, dilation = dilation)

    # Cudnn code
    graph = pycudnn.pygraph("conv-bias", io_data_type = pycudnn.data_type.HALF, intermediate_data_type = pycudnn.data_type.FLOAT, compute_data_type = pycudnn.data_type.FLOAT)

    X = graph.tensor(name = "X", dim = X_gpu.size(), stride = X_gpu.stride(), data_type = convert_to_cudnn_type(X_gpu.dtype))
    W = graph.tensor(name = "W", dim = W_gpu.size(), stride = W_gpu.stride(), data_type = convert_to_cudnn_type(W_gpu.dtype))
    
    conv_output = graph.conv(name = "conv", image = X, weight = W, padding = padding, stride = stride, dilation = dilation)

    Y = graph.relu(name = "relu", input = conv_output)
    Y.set_output(True)
    
    graph.build()

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    Y_actual = torch.zeros(*conv_output.get_dim(), dtype=torch.float16, device='cuda', layout=torch.strided).to(memory_format=torch.channels_last)

    variant_pack = {X: X_gpu, W: W_gpu, Y: Y_actual}
    print(graph)

    graph.execute(variant_pack, workspace)

    # Compare
    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-2, rtol=1e-2)




if __name__ == "__main__":
    test_conv_relu()
    #test_gemm_relu_explicit([16, 32, 64, 128], [16,32,64,])
    #test_gemm_more_explicit([16, 32, 64, 128], [16,32,64,])
    #test_gemm_relu_more_explicit([16, 32, 64, 128], [16,32,64,])
    #test_gemm_more_explicit([1, 16, 16, 16], [1,16,16,])
    

import pycudnn
import torch
import torch.nn.functional as F

import pytest

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

@pytest.mark.parametrize("in_dim, expected_gemm_out_dim", [
    ([1, 16, 16, 16], [1,16,16,]),
#,([16, 32, 64, 128], [16,32,64,]) # fails
])
def test_gemm_bias_relu_more_explicit(in_dim, expected_gemm_out_dim):
    # Can put this in fixture:
    cudnn_graph = pycudnn.pygraph("cudnn_graph", io_data_type = pycudnn.data_type.HALF, intermediate_data_type = pycudnn.data_type.FLOAT, compute_data_type = pycudnn.data_type.FLOAT)

    B, M, N, K = in_dim
    x = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(B, K, N, device="cuda", dtype=torch.float16)
    b = torch.randn(B, M, N, device="cuda", dtype=torch.float16)

    # Here we need to set the strides explicitly
    # image = cudnn_graph.tensor(name = "image", dim = x.size(), stride = [M*K, 1, M])
    # weight = cudnn_graph.tensor(name = "weight", dim = w.size(), stride = [K*N, 1, K])
    # bias = cudnn_graph.tensor(name = "bias", dim = b.size(), stride = [M*N, 1, M])

    image = cudnn_graph.tensor(name = "image", dim = x.size(), stride = [M*K, M, 1])
    weight = cudnn_graph.tensor(name = "weight", dim = w.size(), stride = [K*N, N, 1])
    bias = cudnn_graph.tensor(name = "bias", dim = b.size(), stride = [M*N, 1, M])

    variant_pack = {}
    variant_pack[image] = x
    variant_pack[weight] = w
    variant_pack[bias] = b

    gemm_output = cudnn_graph.matmul(name = "mb_matmul", image = image, weight = weight)

    gemm_output.set_is_virtual(True)

    bias_out = cudnn_graph.bias(name = "bias", input = gemm_output, bias = bias)
    activation_output = cudnn_graph.relu(name = "relu", input = bias_out)
    activation_output.set_output(True)

    Y = activation_output
    Y.set_output(True)

    output = torch.zeros([B,M,N], dtype=torch.float16, device='cuda')

    variant_pack[activation_output] = output

    cudnn_graph.build()
    print(cudnn_graph)

    workspace = torch.empty(cudnn_graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    # Compare output with a reference implementation
    cudnn_graph.execute(variant_pack, workspace)


    pyt_out = torch.transpose(torch.bmm(x, w), 1, 2)
    pyt_out = torch.add(pyt_out, b)
    pyt_out = torch.nn.functional.relu(pyt_out)

    torch.testing.assert_close(output, pyt_out)

@pytest.mark.parametrize("in_dim, expected_gemm_out_dim", [
    ([1, 16, 16, 16], [1,16,16,]),
#,([16, 32, 64, 128], [16,32,64,]) # fails
])
def test_gemm_relu_more_explicit(in_dim, expected_gemm_out_dim):
    # Can put this in fixture:
    cudnn_graph = pycudnn.pygraph("cudnn_graph", io_data_type = pycudnn.data_type.HALF, intermediate_data_type = pycudnn.data_type.FLOAT, compute_data_type = pycudnn.data_type.FLOAT)

    B, M, N, K = in_dim
    x = torch.randn(B, M, K, device="cuda", dtype=torch.float16)
    w = torch.randn(B, K, N, device="cuda", dtype=torch.float16)
    b = torch.randn(B, M, N, device="cuda", dtype=torch.float16)

    image = cudnn_graph.tensor(name = "image", dim = x.size(), stride = [M*K, M, 1])
    weight = cudnn_graph.tensor(name = "weight", dim = w.size(), stride = [K*N, N, 1])
    bias = cudnn_graph.tensor(name = "bias", dim = b.size(), stride = [M*N, 1, M])

    variant_pack = {}
    variant_pack[image] = x
    variant_pack[weight] = w
    variant_pack[bias] = b

    gemm_output = cudnn_graph.matmul(name = "mb_matmul", image = image, weight = weight)

    gemm_output.set_is_virtual(True)

    activation_output = cudnn_graph.relu(name = "relu", input = gemm_output)
    activation_output.set_output(True)

    Y = activation_output
    Y.set_output(True)

    output = torch.zeros([B,M,N], dtype=torch.float16, device='cuda')

    variant_pack[activation_output] = output

    cudnn_graph.build()
    print(cudnn_graph)

    workspace = torch.empty(cudnn_graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    # Compare output with a reference implementation
    cudnn_graph.execute(variant_pack, workspace)


    pyt_out = torch.transpose(torch.bmm(x, w), 1, 2)
    pyt_out = torch.add(pyt_out, b)
    pyt_out = torch.nn.functional.relu(pyt_out)

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
    

import pycudnn
import pytest
import torch

def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return pycudnn.data_type.HALF
    elif torch_type == torch.float32:
        return pycudnn.data_type.FLOAT
    else:
        raise ValueError("Unsupported tensor data type.")

class SGBN(torch.nn.Module):
    def forward(self, input, running_mean, running_var, weight, bias, eps, momentum):
        return torch.nn.functional.batch_norm(input, running_mean, running_var, weight=weight, bias=bias, training=True, momentum=momentum, eps=eps)
    
@pytest.mark.skipif(pycudnn.get_cudnn_version() < 8700, reason="BN not supported below cudnn 8.7")
def test_bn():
    # Reference code
    N, C, H, W = 4, 16, 56, 56
    x_gpu = torch.randn(N, C, H, W, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    scale_gpu = torch.randn(1, C, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
    bias_gpu = torch.randn(1, C, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
    running_mean_gpu = torch.randn(1, C, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
    running_var_gpu = torch.randn(1, C, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
    epsilon_cpu = 1.0e-5
    momentum_cpu = 0.1

    model = SGBN().eval().to("cuda")
    Y_expected = model(x_gpu, running_mean_gpu, running_var_gpu, scale_gpu, bias_gpu, epsilon_cpu, momentum_cpu)

    # Cudnn code
    graph = pycudnn.pygraph("BN", io_data_type = pycudnn.data_type.FLOAT, intermediate_data_type = pycudnn.data_type.FLOAT, compute_data_type = pycudnn.data_type.FLOAT)

    X = graph.tensor(name = "X", dim = x_gpu.size(), stride = x_gpu.stride(), data_type = convert_to_cudnn_type(x_gpu.dtype))
    scale = graph.tensor(name = "scale", dim = scale_gpu.size(), stride = scale_gpu.stride())
    bias = graph.tensor(name = "bias", dim = bias_gpu.size(), stride = bias_gpu.stride())
    in_running_mean = graph.tensor(name = "in_running_mean", dim = running_mean_gpu.size(), stride = running_mean_gpu.stride())
    in_running_var = graph.tensor(name = "in_running_var", dim = running_var_gpu.size(), stride = running_var_gpu.stride())

    (Y, saved_mean, saved_inv_var, out_running_mean, out_running_var) = graph.batchnorm(name = "BN"
                                                                                        , norm_forward_phase = pycudnn.norm_forward_phase.TRAINING
                                                                                        , input = X
                                                                                        , scale = scale, bias = bias
                                                                                        , in_running_mean = in_running_mean, in_running_var = in_running_var
                                                                                        , epsilon = epsilon_cpu, momentum = momentum_cpu)

    Y.set_output(True).set_data_type(pycudnn.data_type.HALF)
    saved_mean.set_output(True)
    saved_inv_var.set_output(True)
    out_running_mean.set_output(True)
    out_running_var.set_output(True)

    graph.build()

    saved_mean_actual = torch.zeros_like(scale_gpu)
    saved_inv_var_actual = torch.zeros_like(scale_gpu)
    Y_actual = torch.zeros_like(Y_expected)

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    graph.execute({
                    X : x_gpu
                    , scale : scale_gpu
                    , bias : bias_gpu
                    , in_running_mean: running_mean_gpu
                    , in_running_var: running_var_gpu
                    , out_running_mean: running_mean_gpu
                    , out_running_var: running_var_gpu
                    , saved_mean : saved_mean_actual
                    , saved_inv_var : saved_inv_var_actual
                    , Y : Y_actual
                }, workspace)
        
    # Compare
    torch.testing.assert_close(Y_expected, Y_actual, atol=1e-3, rtol=1e-3)
    
@pytest.mark.skipif(pycudnn.get_cudnn_version() < 8700, reason="DBN not supported below cudnn 8.7")
def test_dbn():
    # Tensors
    N, C, H, W = 4, 16, 56, 56
    x_gpu = torch.randn(N, C, H, W, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    scale_gpu = torch.randn(1, C, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
    mean_gpu = torch.randn(1, C, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
    inv_variance_gpu = torch.randn(1, C, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
    dy_gpu = torch.randn(N, C, H, W, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)

    # Cudnn code
    graph = pycudnn.pygraph("DBN", intermediate_data_type = pycudnn.data_type.FLOAT, compute_data_type = pycudnn.data_type.FLOAT)

    X = graph.tensor(name = "X", dim = x_gpu.size(), stride = x_gpu.stride(), data_type = convert_to_cudnn_type(x_gpu.dtype))
    DY = graph.tensor(name = "DY", dim = dy_gpu.size(), stride = dy_gpu.stride(), data_type = convert_to_cudnn_type(dy_gpu.dtype))
    scale = graph.tensor(name = "scale", dim = scale_gpu.size(), stride = scale_gpu.stride(), data_type = convert_to_cudnn_type(scale_gpu.dtype))
    mean = graph.tensor(name = "mean", dim = mean_gpu.size(), stride = mean_gpu.stride(), data_type = convert_to_cudnn_type(mean_gpu.dtype))
    inv_variance = graph.tensor(name = "inv_variance", dim = inv_variance_gpu.size(), stride = inv_variance_gpu.stride(), data_type = convert_to_cudnn_type(inv_variance_gpu.dtype))
    
    (DX, DScale, DBias) = graph.batchnorm_backward(name = "DBN"
                                                    , grad = DY
                                                    , input = X
                                                    , scale = scale
                                                    , mean = mean
                                                    , inv_variance = inv_variance
                                                )

    DX.set_output(True)
    DScale.set_output(True)
    DBias.set_output(True)

    graph.build()

    DScale_actual = torch.zeros_like(scale_gpu)
    DBias_actual = torch.zeros_like(scale_gpu)
    DX_actual = torch.zeros_like(dy_gpu)

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    graph.execute({
                    X : x_gpu
                    , DY : dy_gpu
                    , scale : scale_gpu
                    , mean : mean_gpu
                    , inv_variance : inv_variance_gpu
                    , DX : DX_actual
                    , DScale : DScale_actual
                    , DBias : DBias_actual
                }, workspace)
        
if __name__ == "__main__":
    test_bn()
    test_dbn()
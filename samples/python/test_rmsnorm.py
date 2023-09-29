import cudnn
import pytest
import torch
import itertools

import torch.nn as nn

def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.bool:
        return cudnn.data_type.BOOLEAN
    elif torch_type == torch.uint8:
        return cudnn.data_type.UINT8
    else:
        raise ValueError("Unsupported tensor data type.")

class RMSNorm(torch.nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return weight * x_normed + bias

embedding_dim_options = [768, 1024, 1280, 1600]
input_type_options = [torch.float16, torch.bfloat16]

all_options = [elem for elem in itertools.product(*[embedding_dim_options, input_type_options])]

@pytest.fixture(params=all_options)
def param_extract(request):
  return request.param

@pytest.mark.skipif(cudnn.backend_version() < 8906, reason="RmsNorm not supported below cudnn 8.9.6")
def test_rmsnorm(param_extract):
    # TODO(@barretw): ensure output is deterministic and reproducible
    torch.manual_seed(0)
    
    embedding_dim, input_type = param_extract
    
    if input_type == torch.bfloat16:
        atol, rtol = 0.125, 0.125
    else:
        atol, rtol = 0.1, 0.1

    batch_size, seq_size = 16, 128
    N,C,H,W = batch_size * seq_size, embedding_dim, 1, 1
    
    epsilon_value = 1e-3

    x_gpu = 2*torch.randn(N, C, H, W, requires_grad=True, device="cuda", dtype=input_type).to(memory_format=torch.channels_last) - 1.25
    scale_gpu = 3*torch.randn(1, C, H, W, requires_grad=True, device="cuda", dtype=input_type).to(memory_format=torch.channels_last) - 2.75
    bias_gpu = 4*torch.randn(1, C, H, W, requires_grad=True, device="cuda", dtype=input_type).to(memory_format=torch.channels_last)
    epsilon_cpu = torch.full((1, 1, 1, 1), epsilon_value, requires_grad=False, device="cpu", dtype=torch.float32)

    print("Running reference")

    model = RMSNorm(eps=epsilon_value, dim=(1,2,3))    
    Y_expected = model(x_gpu, scale_gpu, bias_gpu)
    inv_var_expected = torch.rsqrt(torch.var(x_gpu.to(torch.float32), dim=(1, 2, 3), keepdim=True) + epsilon_value)

    print("Building cudnn graph")

    graph = cudnn.pygraph(intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)

    X = graph.tensor_like(x_gpu.detach())
    scale = graph.tensor_like(scale_gpu.detach())
    bias = graph.tensor_like(bias_gpu.detach())
    epsilon = graph.tensor_like(epsilon_cpu)

    Y, inv_var = graph.rmsnorm(name = "RMS", 
                            norm_forward_phase = cudnn.norm_forward_phase.TRAINING,
                            input = X,
                            scale = scale, 
                            bias = bias,
                            epsilon = epsilon)
    
    Y.set_output(True).set_data_type(convert_to_cudnn_type(x_gpu.dtype))
    inv_var.set_output(True).set_data_type(convert_to_cudnn_type(inv_var_expected.dtype))

    graph.check_support()
    graph.build()
    
    Y_actual = torch.empty_like(x_gpu)
    inv_var_actual = torch.empty_like(inv_var_expected)
    
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    print("Executing cudnn graph")
    
    graph.execute({
                X : x_gpu.detach()
                , scale : scale_gpu.detach()
                , bias : bias_gpu.detach()
                , epsilon: epsilon_cpu
                , Y : Y_actual
                , inv_var: inv_var_actual
            }, workspace)
    
    print("Comparing with reference")
    torch.testing.assert_close(Y_expected, Y_actual, atol=atol, rtol=rtol)
    torch.testing.assert_close(inv_var_expected, inv_var_actual, atol=atol, rtol=rtol)
    print("Success!!")

    target = torch.randn_like(Y_expected)
    criterion = nn.MSELoss()
    loss = criterion(Y_expected, target)
    
    Y_expected.retain_grad()
    x_gpu.retain_grad()
    scale_gpu.retain_grad()
    bias_gpu.retain_grad()

    loss.backward()

    bwd_graph = cudnn.pygraph(intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)

    DY = bwd_graph.tensor_like(Y_expected.grad)
    X_bwd = bwd_graph.tensor_like(x_gpu.detach())
    scale_bwd = bwd_graph.tensor_like(scale_gpu.detach())
    inv_var_bwd = bwd_graph.tensor_like(inv_var_actual)
    epsilon_bwd = bwd_graph.tensor_like(epsilon_cpu)

    DX, Dscale, Dbias = bwd_graph.rmsnorm_backward(name = "DRMS", 
                            grad = DY,
                            input = X_bwd,
                            scale = scale_bwd, 
                            inv_variance = inv_var_bwd,
                            epsilon = epsilon_bwd)
    
    DX.set_output(True).set_data_type(convert_to_cudnn_type(x_gpu.dtype))
    Dscale.set_output(True).set_data_type(convert_to_cudnn_type(x_gpu.dtype))
    Dbias.set_output(True).set_data_type(convert_to_cudnn_type(x_gpu.dtype))
    
    bwd_graph.check_support()
    bwd_graph.build()
    
    DX_actual = torch.empty_like(x_gpu)
    DScale_actual = torch.empty_like(scale_gpu)
    Dbias_actual = torch.empty_like(bias_gpu)

    workspace = torch.empty(bwd_graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    print("Executing cudnn bwd_graph")
    
    bwd_graph.execute({
                X_bwd : x_gpu.detach()
                , scale_bwd : scale_gpu.detach()
                , DY : Y_expected.grad
                , inv_var_bwd: inv_var_actual
                , epsilon_bwd: epsilon_cpu
                , DX: DX_actual
                , Dscale: DScale_actual
                , Dbias: Dbias_actual
            }, workspace)

    print("Comparing with reference")
    torch.testing.assert_close(x_gpu.grad, DX_actual, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(scale_gpu.grad, DScale_actual, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(bias_gpu.grad, Dbias_actual, atol=5e-4, rtol=5e-4)
    print("Success!!")
    
if __name__ == "__main__":
    test_rmsnorm((1600, torch.bfloat16))
import cudnn
import pytest
import torch

def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.bool:
        return cudnn.data_type.BOOLEAN
    elif torch_type == torch.uint8:
        return cudnn.data_type.UINT8
    else:
        raise ValueError("Unsupported tensor data type.")
   
@pytest.mark.skipif(cudnn.get_cudnn_version() < 8905, reason="LN not supported below cudnn 8.9.5")
def test_ln():
    batch_size, seq_size, dim = 16, 128, 256

    x_gpu = torch.randn(batch_size * seq_size, dim, 1, 1, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
    scale_gpu = torch.randn(1, dim, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
    bias_gpu = torch.randn(1, dim, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
    epsilon_cpu = torch.full((1, 1, 1, 1), 1e-03, requires_grad=False, device="cpu", dtype=torch.float32)

    graph = cudnn.pygraph(io_data_type = cudnn.data_type.FLOAT, intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)

    X = graph.tensor(name = "X", dim = x_gpu.size(), stride = x_gpu.stride(), data_type = convert_to_cudnn_type(x_gpu.dtype))
    scale = graph.tensor(name = "scale", dim = scale_gpu.size(), stride = scale_gpu.stride())
    bias = graph.tensor(name = "bias", dim = bias_gpu.size(), stride = bias_gpu.stride())
    epsilon = graph.tensor(name = "epsilon", dim = epsilon_cpu.size(), stride = epsilon_cpu.stride(), is_pass_by_value = True)

    Y, mean, inv_var = graph.layernorm(name = "LN", 
                            norm_forward_phase = cudnn.norm_forward_phase.TRAINING,
                            input = X,
                            scale = scale, 
                            bias = bias,
                            epsilon = epsilon)
    
    Y.set_output(True).set_data_type(cudnn.data_type.HALF)
    mean.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    inv_var.set_output(True).set_data_type(cudnn.data_type.FLOAT)
    
    graph.check_support()
    graph.build()
    
    Y_actual = torch.zeros_like(x_gpu)
    saved_mean_actual    = torch.zeros(batch_size * seq_size, requires_grad=False, device="cuda", dtype=torch.float32)
    saved_inv_var_actual = torch.zeros(batch_size * seq_size, requires_grad=False, device="cuda", dtype=torch.float32)   

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    
    graph.execute({
                    X : x_gpu
                    , scale : scale_gpu
                    , bias : bias_gpu
                    , epsilon: epsilon_cpu
                    , mean : saved_mean_actual
                    , inv_var : saved_inv_var_actual
                    , Y : Y_actual
                }, workspace)
        
if __name__ == "__main__":
    test_ln()
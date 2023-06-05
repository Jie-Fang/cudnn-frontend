import pycudnn
import torch

def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return pycudnn.data_type.HALF
    elif torch_type == torch.float32:
        return pycudnn.data_type.FLOAT
    else:
        raise ValueError("Unsupported tensor data type.")

    return

class SGBN(torch.nn.Module):
    def forward(self, input, running_mean, running_var, weight, bias):
        return torch.nn.functional.batch_norm(input, running_mean, running_var, weight=weight, bias=bias, training=True, momentum=0.1, eps=1.0e-5)
    
graph = pycudnn.pygraph("BN", io_data_type = pycudnn.data_type.FLOAT, intermediate_data_type = pycudnn.data_type.FLOAT, compute_data_type = pycudnn.data_type.FLOAT)

x_gpu = torch.randn(4, 16, 56, 56, requires_grad=False, device="cuda", dtype=torch.float16).to(memory_format=torch.channels_last)
scale_gpu = torch.randn(1, 16, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
bias_gpu = torch.randn(1, 16, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
running_mean_gpu = torch.randn(1, 16, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
running_var_gpu = torch.randn(1, 16, 1, 1, requires_grad=False, device="cuda", dtype=torch.float32)
eps_cpu = torch.randn(1, 1, 1, 1, requires_grad=False, device="cpu", dtype=torch.float32).fill_(1.0e-5)
momentum_cpu = torch.randn(1, 1, 1, 1, requires_grad=False, device="cpu", dtype=torch.float32).fill_(1.0e-1)

model = SGBN().eval().to("cuda").to(torch.float16)
Y_expected = model(x_gpu, running_mean_gpu, running_var_gpu, scale_gpu, bias_gpu)

X = graph.tensor(name = "X", dim = x_gpu.size(), stride = x_gpu.stride(), data_type = convert_to_cudnn_type(x_gpu.dtype))
scale = graph.tensor(name = "scale")
bias = graph.tensor(name = "bias")
in_running_mean = graph.tensor(name = "in_running_mean")
in_running_var = graph.tensor(name = "in_running_var")
epsilon  = graph.tensor(name = "epsilon", is_pass_by_value = True)
exp_avg_factor = graph.tensor(name = "exp_avg_factor", is_pass_by_value = True)

(Y, saved_mean, saved_inv_var, out_running_mean, out_running_var) = graph.batchnorm(name = "BN", input = X, scale = scale, bias = bias, in_running_mean = in_running_mean, in_running_var = in_running_var, epsilon = epsilon, exp_avg_factor = exp_avg_factor)

Y.set_is_virtual(False).set_data_type(pycudnn.data_type.HALF)
saved_mean.set_is_virtual(False)
saved_inv_var.set_is_virtual(False)
out_running_mean.set_is_virtual(False)
out_running_var.set_is_virtual(False)

graph.build()

saved_mean_actual = torch.zeros_like(scale_gpu)
saved_inv_var_actual = torch.zeros_like(scale_gpu)
Y_actual = torch.zeros_like(Y_expected)

workspace = graph.tensor(name = "workspace")
workspacae_gpu = torch.empty(graph.get_workspace_size(), dtype=torch.uint8)

graph.execute({X : x_gpu
               , scale : scale_gpu
               , bias : bias_gpu
               , in_running_mean: running_mean_gpu
               , in_running_var: running_var_gpu
               , out_running_mean: running_mean_gpu
               , out_running_var: running_var_gpu
               , saved_mean : saved_mean_actual
               , saved_inv_var : saved_inv_var_actual
               , epsilon: eps_cpu
               , exp_avg_factor: momentum_cpu
               , Y : Y_actual
               , workspace: workspacae_gpu})

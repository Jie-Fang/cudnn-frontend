import pycudnn
import numpy as np
import cupy as cp

print("Example 3. Executing the BN graph")

if pycudnn.get_cudnn_version() < 8700:
    print("cudnn version does not support SGBN")
    exit(0)

graph = pycudnn.pygraph("BN", io_data_type = pycudnn.data_type.FLOAT, intermediate_data_type = pycudnn.data_type.FLOAT, compute_data_type = pycudnn.data_type.FLOAT)

X = graph.tensor(name = "X", dim = [4,16,56,56], data_type = pycudnn.data_type.HALF)
scale = graph.tensor(name = "scale", dim = [1,16,1,1])
bias = graph.tensor(name = "bias", dim = [1,16,1,1])
in_running_mean = graph.tensor(name = "in_running_mean", dim = [1,16,1,1])
in_running_var = graph.tensor(name = "in_running_var", dim = [1,16,1,1])
epsilon  = graph.tensor(name = "epsilon", dim = [1,1,1,1], is_pass_by_value = True)
exp_avg_factor = graph.tensor(name = "exp_avg_factor", dim = [1,1,1,1], is_pass_by_value = True)

(Y, saved_mean, saved_inv_var, out_running_mean, out_running_var) = graph.batchnorm(name = "BN", input = X, scale = scale, bias = bias, in_running_mean = in_running_mean, in_running_var = in_running_var, epsilon = epsilon, exp_avg_factor = exp_avg_factor)

Y.set_is_virtual(False).set_data_type(pycudnn.data_type.HALF)
saved_mean.set_is_virtual(False)
saved_inv_var.set_is_virtual(False)
out_running_mean.set_is_virtual(False)
out_running_var.set_is_virtual(False)

graph.build()

X_gpu = cp.full([4,16,56,56], 1, dtype=cp.half)
running_mean_gpu = cp.full([1,16,1,1], 0, dtype=cp.float32)
running_var_gpu = cp.full([1,16,1,1], 0, dtype=cp.float32)
scale_gpu = cp.full([1,16,1,1], 0, dtype=cp.float32)
bias_gpu = cp.full([1,16,1,1], 0, dtype=cp.float32)
saved_mean_gpu = cp.full([1,16,1,1], 0, dtype=cp.float32)
saved_inv_var_gpu = cp.full([1,16,1,1], 0, dtype=cp.float32)
epsilon_cpu = np.full(1, 0.01)
exp_avg_factor_cpu = np.full(1, 0.01)
Y_gpu = cp.full([4,16,56,56], 0, dtype=cp.half)

workspace = graph.tensor(name = "workspace");
workspacae_gpu = cp.empty(graph.get_workspace_size(), dtype=cp.uint8)

graph.execute({X : X_gpu, scale : scale_gpu, bias : bias_gpu, bias : bias_gpu, in_running_mean: running_mean_gpu, out_running_mean: running_mean_gpu, in_running_var: running_var_gpu, out_running_var: running_var_gpu, saved_mean : saved_mean_gpu, saved_inv_var : saved_inv_var_gpu, epsilon: epsilon_cpu, exp_avg_factor: exp_avg_factor_cpu, Y : Y_gpu, workspace: workspacae_gpu})

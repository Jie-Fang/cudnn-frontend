import cudnn

import torch
from torch.profiler import profile, record_function, ProfilerActivity


def test_gemm_silu_and_mul():

    # setup
    M = 64
    N = 64
    K = 64

    # cudnn graph
    handle = cudnn.create_handle()
    graph = cudnn.pygraph(
        handle=handle,
        name="cudnn_graph_0",
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    X_gpu = torch.randint(-8, 8, (1, M, K), requires_grad=False, device="cuda").to(
        dtype=torch.float8_e4m3fn
    )
    W_gpu = torch.randint(-8, 8, (2, K, N), requires_grad=False, device="cuda").to(
        dtype=torch.float8_e4m3fn
    )
    C_gpu = torch.zeros(1, M, N, requires_grad=False, device="cuda").to(
        dtype=torch.float
    )

    scale = 0.5
    X_DQ_cpu = torch.full((1, 1, 1), scale, dtype=torch.float32, device="cpu")
    W_DQ_cpu = torch.full((1, 1, 1), scale, dtype=torch.float32, device="cpu")
    C_Q_cpu = torch.full((1, 1, 1), scale, dtype=torch.float32, device="cpu")
    B_cpu = torch.full((1, 1, 1), 1, dtype=torch.int32, device="cpu")

    X = graph.tensor(
        name="X",
        dim=X_gpu.size(),
        stride=X_gpu.stride(),
        data_type=cudnn.data_type.FP8_E4M3,
    )
    W = graph.tensor(
        name="W",
        dim=W_gpu.size(),
        stride=W_gpu.stride(),
        data_type=cudnn.data_type.FP8_E4M3,
    )
    C0 = graph.matmul(X, W)

    X_DQ = graph.tensor(
        name="X_DQ",
        dim=X_DQ_cpu.size(),
        stride=X_DQ_cpu.stride(),
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
    )
    C1 = graph.mul(C0, X_DQ)

    W_DQ = graph.tensor(
        name="W_DQ",
        dim=W_DQ_cpu.size(),
        stride=W_DQ_cpu.stride(),
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
    )
    C2 = graph.mul(C1, W_DQ)

    C3 = graph.mul(graph.sigmoid(C2), C2)

    B_index = graph.gen_index(C3, 0)
    B_value = graph.tensor(
        name="B_cpu",
        dim=B_cpu.size(),
        stride=B_cpu.stride(),
        data_type=cudnn.data_type.INT32,
        is_pass_by_value=True,
    )
    Mask_upper = graph.cmp_lt(B_index, B_value)
    C_combined = graph.binary_select(C2, C3, Mask_upper)

    C = graph.reduction(C_combined, mode=cudnn.reduction_mode.MUL)
    C.set_dim([1, M, N]).set_stride([M * N, N, 1]).set_output(True).set_data_type(
        cudnn.data_type.FLOAT
    )

    # The output of reductino operation has to be fp32.
    # Plus, the data is in global memory so its not possible to fuse anything now.
    # C_Q = graph.tensor(
    #     name="C_Q",
    #     dim=C_Q_cpu.size(),
    #     stride=C_Q_cpu.stride(),
    #     data_type=cudnn.data_type.FLOAT,
    #     is_pass_by_value=True,
    # )
    # C_fp8 = graph.mul(C, C_Q)
    # C_fp8.set_output(True)

    graph.build([cudnn.heur_mode.A])
    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        graph.execute(
            {
                X: X_gpu,
                W: W_gpu,
                X_DQ: X_DQ_cpu,
                W_DQ: W_DQ_cpu,
                B_value: B_cpu,
                C: C_gpu,
            },
            workspace,
            handle=handle,
        )
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Compare
    torch.cuda.synchronize()

    cudnn.destroy_handle(handle)


def test_silu_and_mul_and_quantization():

    # setup
    M = 64
    N = 64

    # cudnn graph
    handle = cudnn.create_handle()
    graph = cudnn.pygraph(
        handle=handle,
        name="cudnn_graph_0",
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    C2a_gpu = torch.randint(-8, 8, (1, M, N), requires_grad=False, device="cuda").to(
        dtype=torch.float8_e4m3fn
    )
    C2b_gpu = torch.randint(-8, 8, (1, M, N), requires_grad=False, device="cuda").to(
        dtype=torch.float8_e4m3fn
    )
    C_gpu = torch.empty(1, M, N, requires_grad=False, device="cuda").to(
        dtype=torch.float8_e4m3fn
    )

    scale = 0.5
    C2_DQ_cpu = torch.full((1, 1, 1), scale, dtype=torch.float32, device="cpu")
    C_Q_cpu = torch.full((1, 1, 1), scale, dtype=torch.float32, device="cpu")

    C2a = graph.tensor(
        name="C2a",
        dim=C2a_gpu.size(),
        stride=C2a_gpu.stride(),
        data_type=cudnn.data_type.FP8_E4M3,
    )
    C2b = graph.tensor(
        name="C2b",
        dim=C2b_gpu.size(),
        stride=C2b_gpu.stride(),
        data_type=cudnn.data_type.FP8_E4M3,
    )

    C2_DQ = graph.tensor(
        name="C2_DQ",
        dim=C2_DQ_cpu.size(),
        stride=C2_DQ_cpu.stride(),
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
    )
    C2a_fp32 = graph.mul(C2a, C2_DQ)
    C2b_fp32 = graph.mul(C2b, C2_DQ)

    C3 = graph.mul(graph.sigmoid(C2b_fp32), C2b_fp32)

    C_fp32 = graph.mul(C2a_fp32, C3)
    C_Q = graph.tensor(
        name="C_Q",
        dim=C_Q_cpu.size(),
        stride=C_Q_cpu.stride(),
        data_type=cudnn.data_type.FLOAT,
        is_pass_by_value=True,
    )
    C_fp8 = graph.mul(C_fp32, C_Q)
    C_fp8.set_output(True).set_data_type(cudnn.data_type.FP8_E4M3)

    graph.build([cudnn.heur_mode.A])
    workspace = torch.empty(
        graph.get_workspace_size(), device="cuda", dtype=torch.uint8
    )

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        graph.execute(
            {
                C2a: C2a_gpu,
                C2b: C2b_gpu,
                C2_DQ: C2_DQ_cpu,
                C_Q: C_Q_cpu,
                C_fp8: C_gpu,
            },
            workspace,
            handle=handle,
        )
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Compare
    torch.cuda.synchronize()

    cudnn.destroy_handle(handle)


if __name__ == "__main__":
    test_silu_and_mul_and_quantization()
    test_gemm_silu_and_mul()

"""
Utilities and fixtures for GEMM SwiGLU tests.
Contains test configuration fixtures, tensor creation, and reference implementations.
"""

import torch
import pytest


# Create and permute tensor A/B/C
def create_and_permute_tensor(
    l, mode0, mode1, is_mode0_major, dtype, is_dynamic_layout=True
):
    # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
    # else: (l, mode0, mode1) -> (mode0, mode1, l)
    shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
    permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
    is_unsigned = dtype in {torch.uint8}
    min_val = 0 if is_unsigned else -2
    max_val = 4 if is_unsigned else 2

    torch_tensor = (
        torch.empty(shape, dtype=dtype)
        .uniform_(min_val, max_val)
        .permute(permute_order)
        .cuda()
    )

    return torch_tensor


def run_gemm_swiglu_ref(a_ref, b_ref, alpha):
    c_ref, glu_ref = None, None
    if a_ref.dtype in {torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2}:
        c_ref = alpha * torch.einsum("mkl,nkl->mnl", (a_ref).cpu(), (b_ref).cpu())
    else:
        c_ref = (alpha * torch.einsum("mkl,nkl->mnl", (a_ref), (b_ref))).cpu()

    group = 32
    n = b_ref.shape[0]
    assert n % group == 0, "N must be divisible by 32 for GLU block grouping"
    num_blocks = n // group
    assert (
        num_blocks % 2 == 0
    ), "Number of 32-col blocks must be even (pairs of input/gate)"

    cols = torch.arange(n, device=c_ref.device, dtype=torch.long)
    block_cols = cols.view(num_blocks, group)
    input_idx = block_cols[0::2].reshape(-1)
    gate_idx = block_cols[1::2].reshape(-1)
    glu_ref = c_ref.index_select(1, input_idx) * (
        c_ref.index_select(1, gate_idx) * torch.sigmoid(c_ref.index_select(1, gate_idx))
    )
    glu_ref = glu_ref.to(torch.float32)

    return c_ref, glu_ref


def check_ref_gemm_swiglu(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    glu: torch.Tensor,
    config: dict,
):
    if not config["skip_ref"]:
        a_ref = a.clone().to(torch.float32)
        b_ref = b.clone().to(torch.float32)
        c_ref, glu_ref = run_gemm_swiglu_ref(a_ref, b_ref, config["alpha"])

        torch.testing.assert_close(c.cpu(), c_ref.to(c.dtype), atol=0.01, rtol=1e-05)
        torch.testing.assert_close(
            glu.cpu(), glu_ref.to(glu.dtype), atol=0.01, rtol=1e-05
        )
    else:
        print(f"Skipping reference check for testcase with config: {config}")


def allocate_input_tensors(config):
    m = config["m"]
    n = config["n"]
    k = config["k"]
    l = config["l"]
    ab_dtype = config["ab_dtype"]
    a_major = config["a_major"]
    b_major = config["b_major"]

    a_tensor = create_and_permute_tensor(
        l, m, k, a_major == "m", ab_dtype, is_dynamic_layout=True
    )
    b_tensor = create_and_permute_tensor(
        l, n, k, b_major == "n", ab_dtype, is_dynamic_layout=True
    )

    return a_tensor, b_tensor


def allocate_output_tensors(config):
    m = config["m"]
    n = config["n"]
    l = config["l"]
    c_dtype = config["c_dtype"]
    glu_dtype = config["glu_dtype"]
    c_major = config["c_major"]

    c_tensor = create_and_permute_tensor(
        l, m, n, c_major == "m", c_dtype, is_dynamic_layout=True
    )
    glu_tensor = create_and_permute_tensor(
        l, m, n // 2, c_major == "m", glu_dtype, is_dynamic_layout=True
    )

    return c_tensor, glu_tensor


def get_dtype_map():
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
    }


@pytest.fixture
def test_config(request):
    """Extract test configuration from command line arguments."""
    dtype_map = get_dtype_map()
    mnkl_str = request.config.getoption("--gemm-swiglu-mnkl", default=None)
    ab_dtype_str = request.config.getoption("--gemm-swiglu-ab-dtype", default=None)
    c_dtype_str = request.config.getoption("--gemm-swiglu-c-dtype", default=None)
    glu_dtype_str = request.config.getoption("--gemm-swiglu-glu-dtype", default=None)
    acc_dtype_str = request.config.getoption("--gemm-swiglu-acc-dtype", default=None)
    a_major = request.config.getoption("--gemm-swiglu-a-major", default=None)
    b_major = request.config.getoption("--gemm-swiglu-b-major", default=None)
    c_major = request.config.getoption("--gemm-swiglu-c-major", default=None)
    use_2cta_instrs = request.config.getoption(
        "--gemm-swiglu-use-2cta-instrs", default=False
    )
    mma_tiler_str = request.config.getoption("--gemm-swiglu-mma-tiler", default=None)
    cluster_shape_str = request.config.getoption(
        "--gemm-swiglu-cluster-shape", default=None
    )
    alpha = request.config.getoption("--gemm-swiglu-alpha", default=None)
    skip_ref = request.config.getoption("--gemm-swiglu-skip-ref", default=False)

    if mnkl_str is not None:
        m, n, k, l = [int(x.strip()) for x in mnkl_str.split(",")]
    else:
        m, n, k, l = 256, 256, 512, 1

    if mma_tiler_str is not None:
        mma_tiler_m, mma_tiler_n = [int(x.strip()) for x in mma_tiler_str.split(",")]
    else:
        mma_tiler_m, mma_tiler_n = 128, 128

    if cluster_shape_str is not None:
        cluster_shape_m, cluster_shape_n = [
            int(x.strip()) for x in cluster_shape_str.split(",")
        ]
    else:
        cluster_shape_m, cluster_shape_n = 1, 1

    ab_dtype_str = "bfloat16" if ab_dtype_str is None else ab_dtype_str
    c_dtype_str = "bfloat16" if c_dtype_str is None else c_dtype_str
    glu_dtype_str = "float16" if glu_dtype_str is None else glu_dtype_str
    acc_dtype_str = "float32" if acc_dtype_str is None else acc_dtype_str
    a_major = "k" if a_major is None else a_major
    b_major = "k" if b_major is None else b_major
    c_major = "n" if c_major is None else c_major
    alpha = 1.0 if alpha is None else alpha

    ab_dtype = dtype_map[ab_dtype_str]
    c_dtype = dtype_map[c_dtype_str]
    glu_dtype = dtype_map[glu_dtype_str]
    acc_dtype = dtype_map[acc_dtype_str]

    return {
        "m": m,
        "n": n,
        "k": k,
        "l": l,
        "ab_dtype": ab_dtype,
        "c_dtype": c_dtype,
        "glu_dtype": glu_dtype,
        "acc_dtype": acc_dtype,
        "a_major": a_major,
        "b_major": b_major,
        "c_major": c_major,
        "use_2cta_instrs": use_2cta_instrs,
        "mma_tiler_mn": (mma_tiler_m, mma_tiler_n),
        "cluster_shape_mn": (cluster_shape_m, cluster_shape_n),
        "alpha": alpha,
        "skip_ref": skip_ref,
    }

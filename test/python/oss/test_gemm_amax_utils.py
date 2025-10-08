import torch
from cudnn.datatypes import _convert_to_cutlass_data_type
import pytest
from test_low_precision_matmul import _bfloat16_to_float4_e2m1fn_x2

# Attempt to import CUTLASS cute at module load time, but don't fail hard if unavailable
try:
    import cutlass.cute as cute

    @cute.jit
    def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        sf_ref_tensor: cute.Tensor,
        sf_mma_tensor: cute.Tensor,
    ):
        """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
        # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
        # group to ((32, 4, rest_m), (4, rest_k), l)
        import cutlass

        sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
        sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
        for i in cutlass.range(cute.size(sf_ref_tensor)):
            mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
            sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]

except Exception:
    cute = None
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L = None


@pytest.fixture
def test_config(request):
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "fp8_e4m3": torch.float8_e4m3fn,
        "fp8_e5m2": torch.float8_e5m2,
        "fp8_e8m0fnu": torch.float8_e8m0fnu,
        "fp4_e2m1fn_x2": torch.float4_e2m1fn_x2,
    }

    mnkl_str = request.config.getoption("--gemm-amax-mnkl", default=None)
    ab_dtype_str = request.config.getoption("--gemm-amax-ab-dtype", default=None)
    sf_dtype_str = request.config.getoption("--gemm-amax-sf-dtype", default=None)
    sf_vec_size = request.config.getoption("--gemm-amax-sf-vec-size", default=None)
    c_dtype_str = request.config.getoption("--gemm-amax-c-dtype", default=None)
    acc_dtype_str = request.config.getoption("--gemm-amax-acc-dtype", default=None)
    a_major = request.config.getoption("--gemm-amax-a-major", default=None)
    b_major = request.config.getoption("--gemm-amax-b-major", default=None)
    c_major = request.config.getoption("--gemm-amax-c-major", default=None)
    mma_tiler_str = request.config.getoption("--gemm-amax-mma-tiler", default=None)
    cluster_shape_str = request.config.getoption(
        "--gemm-amax-cluster-shape", default=None
    )
    skip_ref = request.config.getoption("--gemm-amax-skip-ref", default=False)

    if mnkl_str is not None:
        m, n, k, l = [int(x.strip()) for x in mnkl_str.split(",")]
    else:
        m, n, k, l = 512, 256, 256, 1

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

    ab_dtype_str = "fp8_e5m2" if ab_dtype_str is None else ab_dtype_str
    sf_dtype_str = "fp8_e8m0fnu" if sf_dtype_str is None else sf_dtype_str
    c_dtype_str = "float16" if c_dtype_str is None else c_dtype_str
    acc_dtype_str = "float32" if acc_dtype_str is None else acc_dtype_str
    a_major = "k" if a_major is None else a_major
    b_major = "k" if b_major is None else b_major
    c_major = "n" if c_major is None else c_major
    sf_vec_size = 32 if sf_vec_size is None else sf_vec_size

    ab_dtype = dtype_map[ab_dtype_str]
    sf_dtype = dtype_map[sf_dtype_str]
    c_dtype = dtype_map[c_dtype_str]
    acc_dtype = dtype_map[acc_dtype_str]

    return {
        "m": m,
        "n": n,
        "k": k,
        "l": l,
        "ab_dtype": ab_dtype,
        "sf_dtype": sf_dtype,
        "sf_vec_size": sf_vec_size,
        "c_dtype": c_dtype,
        "acc_dtype": acc_dtype,
        "a_major": a_major,
        "b_major": b_major,
        "c_major": c_major,
        "mma_tiler_mn": (mma_tiler_m, mma_tiler_n),
        "cluster_shape_mn": (cluster_shape_m, cluster_shape_n),
        "skip_ref": skip_ref,
    }


def allocate_input_tensors(config):
    m = config["m"]
    n = config["n"]
    k = config["k"]
    l = config["l"]
    ab_dtype = config["ab_dtype"]
    a_major = config["a_major"]
    b_major = config["b_major"]
    sf_vec_size = config["sf_vec_size"]
    sf_dtype = config["sf_dtype"]

    a_ref, a_tensor = _create_and_permute_tensor(l, m, k, a_major == "m", ab_dtype)
    b_ref, b_tensor = _create_and_permute_tensor(l, n, k, b_major == "n", ab_dtype)
    sfa_ref, sfa_tensor = create_scale_factor_tensor(l, m, k, sf_vec_size, sf_dtype)
    sfb_ref, sfb_tensor = create_scale_factor_tensor(l, n, k, sf_vec_size, sf_dtype)

    return a_tensor, a_ref, b_tensor, b_ref, sfa_tensor, sfa_ref, sfb_tensor, sfb_ref


def allocate_output_tensors(config):
    m = config["m"]
    n = config["n"]
    l = config["l"]
    c_dtype = config["c_dtype"]
    c_major = config["c_major"]

    _, c_tensor = _create_and_permute_tensor(l, m, n, c_major == "m", c_dtype)
    amax_tensor = torch.full(
        (1, 1, 1), -float("inf"), device="cuda", dtype=torch.float32
    )
    return c_tensor, amax_tensor


def check_ref_gemm_amax(a, b, sfa_ref, sfb_ref, c, amax, config):
    if config["skip_ref"]:
        print(
            f"Skipping reference check for testcase with config: {{'m': {config['m']}, 'n': {config['n']}, 'k': {config['k']}, 'l': {config['l']}}}"
        )
        return

    a_ref = a.float().cpu()
    b_ref = b.float().cpu()
    sfa_ref = sfa_ref.float().cpu()
    sfb_ref = sfb_ref.float().cpu()

    res_a = torch.einsum("mkl,mkl->mkl", a_ref, sfa_ref)
    res_b = torch.einsum("nkl,nkl->nkl", b_ref, sfb_ref)
    c_ref = torch.einsum("mkl,nkl->mnl", res_a, res_b)
    amax_ref = torch.amax(torch.abs(c_ref)).to(torch.float32).reshape(1, 1, 1)
    c_ref = c_ref.to(c.dtype)

    torch.testing.assert_close(c_ref, c.cpu(), atol=1e-01, rtol=1e-01)
    torch.testing.assert_close(amax_ref, amax.cpu(), atol=1e-01, rtol=1e-01)


def _create_and_permute_tensor(l_val, mode0, mode1, is_mode0_major, dtype):
    # is_mode0_major: (l, mode1, mode0) -> (mode0, mode1, l)
    # else: (l, mode0, mode1) -> (mode0, mode1, l)
    shape = (l_val, mode1, mode0) if is_mode0_major else (l_val, mode0, mode1)
    permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
    f32_tensor = torch.randn(shape, dtype=torch.float32)

    dtype_tensor = None
    ref_tensor = None
    if dtype is not torch.float4_e2m1fn_x2:
        dtype_tensor = f32_tensor.to(dtype).permute(permute_order).cuda()
        ref_tensor = dtype_tensor.to(torch.float32)
    else:
        dtype_tensor = (
            _bfloat16_to_float4_e2m1fn_x2(f32_tensor.to(torch.bfloat16))
            .permute(permute_order)
            .cuda()
        )
        ref_tensor = f32_tensor.permute(permute_order).cuda()

    return ref_tensor, dtype_tensor


# Create scale factor tensor SFA/SFB
def create_scale_factor_tensor(l, mn, k, sf_vec_size, dtype):
    from cutlass.cute.runtime import from_dlpack
    import cutlass.torch as cutlass_torch

    def ceil_div(a, b):
        return (a + b - 1) // b

    sf_k = ceil_div(k, sf_vec_size)
    ref_shape = (l, mn, sf_k)

    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        l,
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )

    ref_permute_order = (1, 2, 0)
    mma_permute_order = (3, 4, 1, 5, 2, 0)

    # Create f32 ref torch tensor (cpu)
    ref_f32_torch_tensor_cpu = (
        torch.empty(ref_shape, dtype=torch.float32)
        .uniform_(1, 3)
        .permute(ref_permute_order)
        .to(torch.int8)
        .to(torch.float32)
    )

    # Create f32 cute torch tensor (cpu)
    cute_f32_torch_tensor_cpu = torch.zeros(mma_shape, dtype=torch.float32).permute(
        mma_permute_order
    )

    # convert ref f32 tensor to cute f32 tensor
    try:
        cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
            from_dlpack(ref_f32_torch_tensor_cpu),
            from_dlpack(cute_f32_torch_tensor_cpu),
        )
    except Exception:
        pytest.skip(
            "CUTLASS is not installed; skipping GEMM Amax tests requiring CUTLASS."
        )

    # reshape makes memory contiguous
    ref_f32_torch_tensor_cpu = (
        ref_f32_torch_tensor_cpu.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(l, mn, sf_k, sf_vec_size)
        .reshape(l, mn, sf_k * sf_vec_size)
        .permute(*ref_permute_order)
    )
    ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu[:, :k, :]

    # TODO @mingyangw: I have not found a better way to do this, for some reason the behavior is different from just directly using the torch tensor
    # Create dtype cute torch tensor (cpu)
    _, cute_torch_tensor = cutlass_torch.cute_tensor_like(
        cute_f32_torch_tensor_cpu,
        _convert_to_cutlass_data_type(dtype),
        is_dynamic_layout=True,
        assumed_align=16,
    )
    # cute_torch_tensor_2 = cute_f32_torch_tensor_cpu.to(torch.float8_e8m0fnu)

    return ref_f32_torch_tensor_cpu.cuda(), cute_torch_tensor

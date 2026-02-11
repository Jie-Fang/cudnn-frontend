import cudnn
import pytest
import torch
import math
from enum import IntEnum
from looseversion import LooseVersion

from .helpers import fill_sparse_small_int
from .mxfp8_ref import compute_ref

# Try to import CUTLASS for scale factor conversion
try:
    import cutlass.cute as cute
    import cutlass
    from cutlass.cute.runtime import from_dlpack

    @cute.jit
    def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        sf_ref_tensor: cute.Tensor,
        sf_mma_tensor: cute.Tensor,
    ):
        """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
        sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
        sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
        for i in cutlass.range(cute.size(sf_ref_tensor)):
            mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
            sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]

    HAS_CUTLASS = True
except Exception:
    HAS_CUTLASS = False
    cute = None
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L = None

# fmt: off

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class GraphFwdUid(IntEnum):
    q = 0
    k = 1
    v = 2
    sf_q = 5
    sf_k = 6
    sf_v = 7
    o = 3
    stats = 4
    o_amax = 12


# Helper to compare tensors with detailed output
def compare_tensors(actual, expected, atol, rtol, tag, disp_elems=10):
    actual_f32 = actual.float()
    mismatches = torch.where(torch.isclose(actual_f32, expected, rtol=rtol, atol=atol, equal_nan=True) == False)
    mismatch_cnt = mismatches[0].numel()
    num_elements = torch.numel(actual)

    if mismatch_cnt != 0:
        percentage = 100 * mismatch_cnt / num_elements
        print(f"\nComparing '{tag}' using rtol={rtol:.4e}, atol={atol:.4e}")
        combined = torch.stack(mismatches, dim=-1).tolist()
        for i, index in enumerate(combined[:disp_elems]):
            idx = tuple(index)
            gpu_val = actual_f32[idx].item()
            ref_val = expected[idx].item()
            diff = gpu_val - ref_val
            print(f"  idx{index}: {tag}_gpu={gpu_val:+.6e}, {tag}_ref={ref_val:+.6e}, diff={diff:+.2e}")
        print(f"Total {mismatch_cnt:,} mismatches ({percentage:.1f}%) for '{tag}'")
    else:
        print(f"'{tag}' within tolerance (rtol={rtol}, atol={atol})")

    return mismatch_cnt

def compute_mxfp8_scale_dims(s, d, block_size=32):
    """
    Compute scale tensor dimensions for MXFP8.

    For Q/K: scale the d (hidden) dimension
    For V: scale the s (sequence) dimension (BMM2 contracts on s)

    F8_128x4 reordering requires:
    - Sequence dimension padded to multiple of 128
    - Scale dimension padded to multiple of 4
    """
    d_scale = ceil_div(d, block_size)
    s_scale = ceil_div(s, block_size)

    s_padded = ceil_div(s, 128) * 128
    d_scale_padded = ceil_div(d_scale, 4) * 4
    s_scale_padded = ceil_div(s_scale, 4) * 4
    d_padded = ceil_div(d, 128) * 128  # Must be multiple of 128 for F8_128x4

    return {
        "s_padded": s_padded,
        "d_scale": d_scale,
        "d_scale_padded": d_scale_padded,
        "s_scale": s_scale,
        "s_scale_padded": s_scale_padded,
        "d_padded": d_padded,
    }


def create_sf_layout_tensor(l, mn, nk, sf_vec_size):
    """Create scale factor tensor with F8_128x4 layout."""
    sf_k = ceil_div(nk, sf_vec_size)

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

    mma_permute_order = (3, 4, 1, 5, 2, 0)
    cute_f32_torch_tensor_cpu = torch.zeros(mma_shape, dtype=torch.float32).permute(mma_permute_order)

    return cute_f32_torch_tensor_cpu, sf_k


def create_scale_factor_tensor(l, mn, k, sf_vec_size, dtype=torch.float8_e8m0fnu):
    """
    Create scale factor tensor for SDPA with F8_128x4 reordering.

    Args:
        l: batch dimension (b * h for SDPA)
        mn: non-contracting dimension (s for Q/K, d for V)
        k: contracting dimension to be scaled (d for Q/K, s for V)
        sf_vec_size: block size (32 for MXFP8)
        dtype: output dtype (torch.float8_e8m0fnu)

    Returns:
        ref_tensor: reference tensor for computation [mn, sf_k, l] -> broadcast to [mn, k, l]
        cute_tensor: F8_128x4 reordered tensor for cuDNN
    """
    if not HAS_CUTLASS:
        pytest.skip("CUTLASS is not installed; skipping MXFP8 tests.")

    cute_f32_torch_tensor_cpu, sf_k = create_sf_layout_tensor(l, mn, k, sf_vec_size)
    ref_shape = (l, mn, sf_k)
    ref_permute_order = (1, 2, 0)

    # Create reference scale factors (small positive values for stability)
    ref_f32_torch_tensor_cpu = (
        torch.empty(ref_shape, dtype=torch.float32).uniform_(0.5, 2.0).permute(ref_permute_order).to(torch.int8).to(torch.float32)
    )

    # Convert ref f32 tensor to cute f32 tensor with F8_128x4 layout
    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        from_dlpack(ref_f32_torch_tensor_cpu),
        from_dlpack(cute_f32_torch_tensor_cpu),
    )

    # Expand scale factors to match the original k dimension
    ref_expanded = ref_f32_torch_tensor_cpu.permute(2, 0, 1).unsqueeze(-1).expand(l, mn, sf_k, sf_vec_size).reshape(l, mn, sf_k * sf_vec_size).permute(*ref_permute_order)
    ref_expanded = ref_expanded[:, :k, :]

    # Convert to E8M0 dtype
    cute_torch_tensor = cute_f32_torch_tensor_cpu.to(torch.float8_e8m0fnu).cuda()

    return ref_expanded.cuda(), cute_torch_tensor


def generate_graph_fwd(b, h_q, h_k, h_v,
                       s_qo, s_kv, d_qk, d_vo, attn_scale,
                       use_causal_mask,
                       block_size=32,
                       cudnn_itype=cudnn.data_type.FP8_E4M3,
                       cudnn_otype=cudnn.data_type.HALF):
    # Compute padded dimensions for F8_128x4 scale factors
    s_q_padded = ceil_div(s_qo, 128) * 128
    s_kv_padded = ceil_div(s_kv, 128) * 128
    d_qk_scale_padded = ceil_div(ceil_div(d_qk, block_size), 4) * 4
    d_vo_padded = ceil_div(d_vo, 128) * 128
    s_kv_scale_padded = ceil_div(ceil_div(s_kv, block_size), 4) * 4

    # Build graph
    graph = cudnn.pygraph(
        io_data_type=cudnn_itype,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT
    )

    # Q, K, V tensors with BHSD layout
    # Stride: (s * h * d, d, h * d, 1) for interleaved layout
    q = graph.tensor(
        uid=GraphFwdUid.q,
        dim=(b, h_q, s_qo, d_qk),
        stride=(s_qo * h_q * d_qk, d_qk, h_q * d_qk, 1),
        data_type=cudnn_itype
    )
    k = graph.tensor(
        uid=GraphFwdUid.k,
        dim=(b, h_k, s_kv, d_qk),
        stride=(s_kv * h_k * d_qk, d_qk, h_k * d_qk, 1),
        data_type=cudnn_itype
    )
    v = graph.tensor(
        uid=GraphFwdUid.v,
        dim=(b, h_v, s_kv, d_vo),
        stride=(s_kv * h_v * d_vo, d_vo, h_v * d_vo, 1),
        data_type=cudnn_itype
    )

    # Scale factor tensors (FP8_E8M0 with F8_128x4 reordering)
    # SF_Q: [B, H_q, S_q_padded, D_scale_padded], d_scale contiguous
    sf_q_dims = (b, h_q, s_q_padded, d_qk_scale_padded)
    sf_q_strides = (h_q * s_q_padded * d_qk_scale_padded, s_q_padded * d_qk_scale_padded, d_qk_scale_padded, 1)
    sf_q = graph.tensor(
        uid=GraphFwdUid.sf_q,
        dim=sf_q_dims,
        stride=sf_q_strides,
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4
    )

    # SF_K: [B, H_k, S_kv_padded, D_scale_padded], d_scale contiguous
    sf_k_dims = (b, h_k, s_kv_padded, d_qk_scale_padded)
    sf_k_strides = (h_k * s_kv_padded * d_qk_scale_padded, s_kv_padded * d_qk_scale_padded, d_qk_scale_padded, 1)
    sf_k = graph.tensor(
        uid=GraphFwdUid.sf_k,
        dim=sf_k_dims,
        stride=sf_k_strides,
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4
    )

    # SF_V: [B, H_v, S_scale_padded, D_v_padded], s_scale contiguous
    sf_v_dims = (b, h_v, s_kv_scale_padded, d_vo_padded)
    sf_v_strides = (h_v * s_kv_scale_padded * d_vo_padded, s_kv_scale_padded * d_vo_padded, 1, s_kv_scale_padded)
    sf_v = graph.tensor(
        uid=GraphFwdUid.sf_v,
        dim=sf_v_dims,
        stride=sf_v_strides,
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4
    )

    # Call MXFP8 SDPA
    o, stats, amax_o = graph.sdpa_mxfp8(
        q=q, k=k, v=v,
        descale_q=sf_q, descale_k=sf_k, descale_v=sf_v,
        attn_scale=attn_scale,
        use_causal_mask=use_causal_mask,
        generate_stats=True,
    )

    # Set output tensor properties
    o.set_uid(GraphFwdUid.o).set_output(True).set_dim((b, h_q, s_qo, d_vo)).set_stride((s_qo * h_q * d_vo, d_vo, h_q * d_vo, 1)).set_data_type(cudnn_otype)
    stats.set_uid(GraphFwdUid.stats).set_output(True).set_dim((b, h_q, s_qo, 1)).set_stride((h_q * s_qo, s_qo, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_o.set_uid(GraphFwdUid.o_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)

    return graph


def exec_sdpa_mxfp8(cfg, request, cudnn_handle):
    """Execute MXFP8 SDPA test."""
    if request.config.option.dryrun:
        pytest.skip("dry run mode")

    cudnn_version = LooseVersion(cudnn.backend_version_string())
    if cudnn_version < "9.21.0":
        pytest.skip("MXFP8 SDPA requires cuDNN 9.21.0 or higher")

    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("MXFP8 SDPA requires Blackwell or higher")

    if not HAS_CUTLASS:
        pytest.skip("CUTLASS is not installed; skipping MXFP8 tests.")

    # Extract config
    b = cfg.batches
    h_q, h_k, h_v = cfg.h_q, cfg.h_k, cfg.h_v
    s_qo, s_kv = cfg.s_q, cfg.s_kv
    d_qk, d_vo = cfg.d_qk, cfg.d_v
    block_size = 32

    attn_scale = 1.0 / math.sqrt(d_qk)
    use_causal_mask = getattr(cfg, 'use_causal_mask', False)

    # Get input/output types from config
    torch_itype = cfg.data_type if hasattr(cfg, 'data_type') and cfg.data_type else torch.float8_e4m3fn
    torch_otype = cfg.output_type if hasattr(cfg, 'output_type') and cfg.output_type else torch.bfloat16

    # Map torch types to cudnn types
    if torch_itype == torch.float8_e4m3fn:
        cudnn_itype = cudnn.data_type.FP8_E4M3
    elif torch_itype == torch.float8_e5m2:
        cudnn_itype = cudnn.data_type.FP8_E5M2
    else:
        pytest.skip(f"Unsupported input type: {torch_itype}")
    cudnn_otype = cudnn.data_type.HALF if torch_otype == torch.float16 else cudnn.data_type.BFLOAT16

    # Compute padded dimensions for F8_128x4 scale factors
    s_q_padded = ceil_div(s_qo, 128) * 128
    s_kv_padded = ceil_div(s_kv, 128) * 128
    d_qk_scale_padded = ceil_div(ceil_div(d_qk, block_size), 4) * 4
    d_vo_padded = ceil_div(d_vo, 128) * 128
    s_kv_scale_padded = ceil_div(ceil_div(s_kv, block_size), 4) * 4

    # Build forward graph
    try:
        graph_fwd = generate_graph_fwd(
            b, h_q, h_k, h_v,
            s_qo, s_kv, d_qk, d_vo, attn_scale,
            use_causal_mask, block_size,
            cudnn_itype, cudnn_otype
        )
        graph_fwd.validate()
        graph_fwd.build_operation_graph()
        graph_fwd.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph_fwd.check_support()
        graph_fwd.build_plans()
    except cudnn.cudnnGraphNotSupportedError as e:
        pytest.skip(f"MXFP8 SDPA not supported: {e}")
    except Exception as e:
        pytest.fail(f"Error building MXFP8 SDPA graph: {e}")

    # Create FP8 input tensors using sparse small integers for better low-precision testing
    rng_data = torch.Generator(device="cuda").manual_seed(cfg.rng_data_seed)
    q_f32 = torch.empty(b, h_q, s_qo, d_qk, dtype=torch.float32, device="cuda")
    fill_sparse_small_int(q_f32, rng_data, sparsity=0.8, abs_max=2)
    k_f32 = torch.empty(b, h_k, s_kv, d_qk, dtype=torch.float32, device="cuda")
    fill_sparse_small_int(k_f32, rng_data, sparsity=0.8, abs_max=2)
    v_f32 = torch.empty(b, h_v, s_kv, d_vo, dtype=torch.float32, device="cuda")
    fill_sparse_small_int(v_f32, rng_data, sparsity=0.8, abs_max=2)

    q_fp8 = q_f32.to(torch_itype)
    k_fp8 = k_f32.to(torch_itype)
    v_fp8 = v_f32.to(torch_itype)

    # Create scale factor tensors using CUTLASS cute DSL for proper F8_128x4 layout
    # create_scale_factor_tensor returns: (ref_expanded [mn, k, l], cute_tensor with F8_128x4 layout)

    # SF_Q: [s_q_padded, d_qk, b*h_q]
    # sf_q_ref_raw: [s_q_padded, d_qk, b*h_q]
    # sf_q_cute: [32, 4, s_q_padded / 128, 4, d_qk_scale_padded / 4, b*h_q]
    sf_q_ref_raw, sf_q_cute = create_scale_factor_tensor(
        l=b * h_q, mn=s_q_padded, k=d_qk, sf_vec_size=block_size,
    )

    # SF_K: [s_kv_padded, d_qk, b*h_k]
    # sf_k_ref_raw: [s_kv_padded, d_qk, b*h_k]
    # sf_k_cute: [32, 4, s_kv_padded / 128, 4, d_qk_scale_padded / 4, b*h_k]
    sf_k_ref_raw, sf_k_cute = create_scale_factor_tensor(
        l=b * h_k, mn=s_kv_padded, k=d_qk,
        sf_vec_size=block_size,
    )

    # SF_V: [d_vo_padded, s_kv, b*h_v] - s is scaled (contiguous in BMM2 contraction)
    # sf_v_ref_raw: [d_vo_padded, s_kv, b*h_v]
    # sf_v_cute: [32, 4, d_vo_padded / 128, 4, s_kv_scale_padded / 4, b*h_v]
    sf_v_ref_raw, sf_v_cute = create_scale_factor_tensor(
        l=b * h_v, mn=d_vo_padded, k=s_kv, sf_vec_size=block_size,
    )

    # Cute tensors are already in F8_128x4 layout, use directly
    sf_q_cudnn = sf_q_cute
    sf_k_cudnn = sf_k_cute
    sf_v_cudnn = sf_v_cute

    # Trim reference scale factors to actual dimensions for compute_ref
    sf_q_ref = sf_q_ref_raw[:s_qo, :d_qk, :]
    sf_k_ref = sf_k_ref_raw[:s_kv, :d_qk, :]
    sf_v_ref = sf_v_ref_raw[:d_vo, :s_kv, :]

    # Allocate output tensors
    o_gpu = torch.empty(b, h_q, s_qo, d_vo, dtype=torch_otype, device="cuda")
    stats_gpu = torch.empty(b, h_q, s_qo, 1, dtype=torch.float32, device="cuda")
    amax_o_gpu = torch.zeros(1, 1, 1, 1, dtype=torch.float32, device="cuda")

    # Build variant pack
    variant_pack = {
        int(GraphFwdUid.q): q_fp8,
        int(GraphFwdUid.k): k_fp8,
        int(GraphFwdUid.v): v_fp8,
        int(GraphFwdUid.sf_q): sf_q_cudnn,
        int(GraphFwdUid.sf_k): sf_k_cudnn,
        int(GraphFwdUid.sf_v): sf_v_cudnn,
        int(GraphFwdUid.o): o_gpu,
        int(GraphFwdUid.stats): stats_gpu,
        int(GraphFwdUid.o_amax): amax_o_gpu,
    }

    # Execute
    workspace = torch.empty(graph_fwd.get_workspace_size(), dtype=torch.uint8, device="cuda")
    graph_fwd.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    # Compute reference
    o_ref, stats_ref = compute_ref(q_fp8, k_fp8, v_fp8, sf_q_ref, sf_k_ref, sf_v_ref, attn_scale, use_causal_mask, torch_itype, torch_otype)

    # Compare output
    o_gpu_f32 = o_gpu.float()
    o_atol, o_rtol = 0.04, 0.20
    o_err = compare_tensors(o_gpu, o_ref, o_atol, o_rtol, "output")

    # Compare stats (logsumexp) - tight tolerance
    stats_atol, stats_rtol = 0.05, 0.05
    stats_err = compare_tensors(stats_gpu, stats_ref, stats_atol, stats_rtol, "stats")

    # Compare amax
    amax_ref = torch.amax(torch.abs(o_ref)).item()
    amax_gpu = amax_o_gpu.item()
    amax_diff = abs(amax_gpu - amax_ref)
    amax_atol = 0.02 * max(amax_ref, 1.0)  # 2% tolerance for FP8
    print(f"amax: gpu={amax_gpu:.6e}, ref={amax_ref:.6e}, diff={amax_diff:.2e}, tol={amax_atol:.2e}")

    # Assert all checks pass
    assert o_err == 0, f"Output mismatch: {o_err} elements differ"
    assert stats_err == 0, f"Stats mismatch: {stats_err} elements differ"
    # assert amax_diff < amax_atol, f"amax mismatch: diff={amax_diff:.4e} > tol={amax_atol:.4e}"

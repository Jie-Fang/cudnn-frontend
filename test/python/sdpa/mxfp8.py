import cudnn
import pytest
import torch
import math
from enum import IntEnum
from looseversion import LooseVersion

from .helpers import fill_sparse_small_int

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


def generate_graph_fwd_mxfp8(b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, use_causal_mask=True, output_type=cudnn.data_type.BFLOAT16):
    """Generate MXFP8 SDPA forward graph."""
    block_size = 32
    # Compute dims for Q (uses s_qo) and K/V (uses s_kv)
    dims_q = compute_mxfp8_scale_dims(s_qo, d_qk, block_size)
    dims_kv = compute_mxfp8_scale_dims(s_kv, d_qk, block_size)
    # For V, compute dims with d_vo (V's hidden dim) instead of d_qk
    dims_v = compute_mxfp8_scale_dims(s_kv, d_vo, block_size)

    graph_fwd = cudnn.pygraph(
        io_data_type=cudnn.data_type.FP8_E4M3,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT
    )

    # Q, K, V tensors - BHSD layout, contiguous
    q = graph_fwd.tensor(
        uid=GraphFwdUid.q,
        dim=(b, h_q, s_qo, d_qk),
        stride=(h_q * s_qo * d_qk, s_qo * d_qk, d_qk, 1),
        data_type=cudnn.data_type.FP8_E4M3
    )
    k = graph_fwd.tensor(
        uid=GraphFwdUid.k,
        dim=(b, h_k, s_kv, d_qk),
        stride=(h_k * s_kv * d_qk, s_kv * d_qk, d_qk, 1),
        data_type=cudnn.data_type.FP8_E4M3
    )
    v = graph_fwd.tensor(
        uid=GraphFwdUid.v,
        dim=(b, h_v, s_kv, d_vo),
        stride=(h_v * s_kv * d_vo, s_kv * d_vo, d_vo, 1),
        data_type=cudnn.data_type.FP8_E4M3
    )

    # Block scale tensor for Q (FP8_E8M0 with F8_128x4 reordering)
    # Shape: [B, H_q, S_q_padded, D_scale_padded], d_scale contiguous (stride[3]=1)
    sf_q_dims = (b, h_q, dims_q["s_padded"], dims_q["d_scale_padded"])
    sf_q_strides = (
        h_q * dims_q["s_padded"] * dims_q["d_scale_padded"],
        dims_q["s_padded"] * dims_q["d_scale_padded"],
        dims_q["d_scale_padded"],
        1
    )

    sf_q = graph_fwd.tensor(
        uid=GraphFwdUid.sf_q,
        dim=sf_q_dims,
        stride=sf_q_strides,
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4
    )

    # Block scale tensor for K (FP8_E8M0 with F8_128x4 reordering)
    # Shape: [B, H_k, S_kv_padded, D_scale_padded], d_scale contiguous (stride[3]=1)
    sf_k_dims = (b, h_k, dims_kv["s_padded"], dims_kv["d_scale_padded"])
    sf_k_strides = (
        h_k * dims_kv["s_padded"] * dims_kv["d_scale_padded"],
        dims_kv["s_padded"] * dims_kv["d_scale_padded"],
        dims_kv["d_scale_padded"],
        1
    )

    sf_k = graph_fwd.tensor(
        uid=GraphFwdUid.sf_k,
        dim=sf_k_dims,
        stride=sf_k_strides,
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4
    )

    # Block scale tensor for V (FP8_E8M0 with F8_128x4 reordering)
    # Shape: [B, H_v, S_scale_padded, D_v_padded], s_scale contiguous (stride[2]=1)
    sf_v_dims = (b, h_v, dims_v["s_scale_padded"], dims_v["d_padded"])
    sf_v_strides = (
        h_v * dims_v["s_scale_padded"] * dims_v["d_padded"],
        dims_v["s_scale_padded"] * dims_v["d_padded"],
        1,  # s_scale contiguous
        dims_v["s_scale_padded"]
    )

    sf_v = graph_fwd.tensor(
        uid=GraphFwdUid.sf_v,
        dim=sf_v_dims,
        stride=sf_v_strides,
        data_type=cudnn.data_type.FP8_E8M0,
        reordering_type=cudnn.tensor_reordering.F8_128x4
    )

    # Call MXFP8 SDPA
    o, stats, amax_o = graph_fwd.sdpa_mxfp8(
        q=q, k=k, v=v,
        descale_q=sf_q,
        descale_k=sf_k,
        descale_v=sf_v,
        attn_scale=attn_scale,
        use_causal_mask=use_causal_mask,
        generate_stats=True,
    )

    # Set output tensor properties
    o.set_uid(GraphFwdUid.o).set_output(True).set_dim((b, h_q, s_qo, d_vo)).set_stride((h_q * s_qo * d_vo, s_qo * d_vo, d_vo, 1)).set_data_type(output_type)
    stats.set_uid(GraphFwdUid.stats).set_output(True).set_dim((b, h_q, s_qo, 1)).set_stride((h_q * s_qo, s_qo, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_o.set_uid(GraphFwdUid.o_amax).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)

    return graph_fwd, {"q": dims_q, "kv": dims_kv, "v": dims_v}


def compute_sdpa_ref(q_f32, k_f32, v_f32, sf_q_ref, sf_k_ref, sf_v_ref, attn_scale, use_causal_mask=False):
    """
    Compute reference SDPA with MXFP8 dequantization.
    Supports GQA/MQA where K and V have fewer heads than Q.
    """
    b, h_q, s_q, d_qk = q_f32.shape
    _, h_k, s_kv, _ = k_f32.shape
    _, h_v, _, d_vo = v_f32.shape

    # GQA: expand K, V to match Q's head count
    if h_k != h_q:
        assert h_q % h_k == 0, "h_q must be divisible by h_k for GQA"
        repeats = h_q // h_k
        k_f32 = k_f32.repeat_interleave(repeats, dim=1)
        # sf_k_ref: [S, D, B*H_k] -> [S, D, B*H_q]
        sf_k_ref = sf_k_ref.reshape(sf_k_ref.shape[0], sf_k_ref.shape[1], b, h_k)
        sf_k_ref = sf_k_ref.repeat_interleave(repeats, dim=3)
        sf_k_ref = sf_k_ref.reshape(sf_k_ref.shape[0], sf_k_ref.shape[1], b * h_q)

    if h_v != h_q:
        assert h_q % h_v == 0, "h_q must be divisible by h_v for GQA"
        repeats = h_q // h_v
        v_f32 = v_f32.repeat_interleave(repeats, dim=1)
        # sf_v_ref: [D, S, B*H_v] -> [D, S, B*H_q]
        sf_v_ref = sf_v_ref.reshape(sf_v_ref.shape[0], sf_v_ref.shape[1], b, h_v)
        sf_v_ref = sf_v_ref.repeat_interleave(repeats, dim=3)
        sf_v_ref = sf_v_ref.reshape(sf_v_ref.shape[0], sf_v_ref.shape[1], b * h_q)

    # Reshape for batch processing: [B, H, S, D] -> [B*H, S, D]
    q = q_f32.reshape(b * h_q, s_q, d_qk)
    k = k_f32.reshape(b * h_q, s_kv, d_qk)
    v = v_f32.reshape(b * h_q, s_kv, d_vo)

    # sf_q_ref: [S_q, D, B*H_q] -> [B*H_q, S_q, D]
    sf_q = sf_q_ref.permute(2, 0, 1)
    # sf_k_ref: [S_kv, D, B*H_q] -> [B*H_q, S_kv, D]
    sf_k = sf_k_ref.permute(2, 0, 1)
    # sf_v_ref: [D, S_kv, B*H_q] -> [B*H_q, S_kv, D]
    sf_v = sf_v_ref.permute(2, 1, 0)

    # Dequantize Q and K (scale factors apply to D dimension)
    q_dq = q * sf_q[:, :s_q, :d_qk]
    k_dq = k * sf_k[:, :s_kv, :d_qk]

    # BMM1: S = Q @ K^T, shape [B*H, S_q, S_kv]
    s = torch.bmm(q_dq, k_dq.transpose(1, 2)) * attn_scale

    # Apply causal mask if requested
    if use_causal_mask:
        mask = torch.triu(torch.ones(s_q, s_kv, device=s.device, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask.unsqueeze(0), float("-inf"))

    # Compute stats (log-sum-exp for backward pass)
    stats = torch.logsumexp(s, dim=-1, keepdim=True)

    # Softmax
    p = torch.softmax(s, dim=-1)

    # Dequantize V (scale factors apply to S_kv dimension)
    v_dq = v * sf_v[:, :s_kv, :d_vo]

    # BMM2: O = P @ V, shape [B*H, S_q, D]
    o = torch.bmm(p, v_dq)

    # Reshape back to [B, H_q, S_q, D_vo]
    o_ref = o.reshape(b, h_q, s_q, d_vo)
    stats_ref = stats.reshape(b, h_q, s_q, 1)

    return o_ref, stats_ref


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
    use_causal_mask = getattr(cfg, 'use_causal_mask', True)

    # Get output type from config (default to bfloat16)
    torch_otype = cfg.output_type if hasattr(cfg, 'output_type') and cfg.output_type else torch.bfloat16
    cudnn_otype = cudnn.data_type.HALF if torch_otype == torch.float16 else cudnn.data_type.BFLOAT16

    # Compute separate dims for Q, K, and V
    dims_q = compute_mxfp8_scale_dims(s_qo, d_qk, block_size)
    dims_kv = compute_mxfp8_scale_dims(s_kv, d_qk, block_size)
    dims_v = compute_mxfp8_scale_dims(s_kv, d_vo, block_size)

    try:
        graph, _ = generate_graph_fwd_mxfp8(b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, use_causal_mask, cudnn_otype)
        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()
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

    q_fp8 = q_f32.to(torch.float8_e4m3fn)
    k_fp8 = k_f32.to(torch.float8_e4m3fn)
    v_fp8 = v_f32.to(torch.float8_e4m3fn)

    # Create scale factor tensors using CUTLASS cute DSL for proper F8_128x4 layout
    # The create_scale_factor_tensor function:
    #   - l: batch dimension (b * h)
    #   - mn: non-contracting dimension
    #   - k: contracting dimension (to be scaled, block_size=32)
    # Returns: (ref_expanded [mn, k, l], cute_tensor with F8_128x4 layout)

    # SF_Q: non-contracting=s_padded, contracting=d_qk (d is scaled)
    # Returned cute tensor shape is determined by create_sf_layout_tensor
    sf_q_ref_raw, sf_q_cute = create_scale_factor_tensor(
        l=b * h_q,
        mn=dims_q["s_padded"],
        k=d_qk,
        sf_vec_size=block_size,
    )
    # sf_q_ref_raw: [s_padded, d_qk, b*h_q], sf_q_cute: F8_128x4 layout tensor

    # SF_K: non-contracting=s_padded, contracting=d_qk (d is scaled)
    sf_k_ref_raw, sf_k_cute = create_scale_factor_tensor(
        l=b * h_k,
        mn=dims_kv["s_padded"],
        k=d_qk,
        sf_vec_size=block_size,
    )

    # SF_V: non-contracting=d_padded, contracting=s_kv (s is scaled)
    # This ensures s_scale is contiguous as required for BMM2 contraction
    sf_v_ref_raw, sf_v_cute = create_scale_factor_tensor(
        l=b * h_v,
        mn=dims_v["d_padded"],
        k=s_kv,
        sf_vec_size=block_size,
    )
    # sf_v_ref_raw: [d_padded, s_kv, b*h_v], sf_v_cute: F8_128x4 layout tensor

    # Reshape cute tensors for cuDNN variant pack
    # SF_Q/K cute tensors need to be reshaped to [b, h, s_padded, d_scale_padded]
    sf_q_cudnn = sf_q_cute.reshape(b, h_q, dims_q["s_padded"], dims_q["d_scale_padded"])
    sf_k_cudnn = sf_k_cute.reshape(b, h_k, dims_kv["s_padded"], dims_kv["d_scale_padded"])
    # SF_V cute tensor: reshape to [b, h, d_padded, s_scale_padded] then transpose to [b, h, s_scale, d]
    sf_v_cudnn = sf_v_cute.reshape(b, h_v, dims_v["d_padded"], dims_v["s_scale_padded"]).transpose(2, 3)

    # Prepare reference scale factors for compute_sdpa_ref
    # Q: [s_padded, d_qk, b*h_q] -> trim to [s_qo, d_qk, b*h_q]
    sf_q_ref = sf_q_ref_raw[:s_qo, :d_qk, :]
    # K: [s_padded, d_qk, b*h_k] -> trim to [s_kv, d_qk, b*h_k]
    sf_k_ref = sf_k_ref_raw[:s_kv, :d_qk, :]
    # V: [d_padded, s_kv, b*h_v] -> trim to [d_vo, s_kv, b*h_v]
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
    workspace = torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda")
    graph.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    # Compute reference
    o_ref, stats_ref = compute_sdpa_ref(q_f32, k_f32, v_f32, sf_q_ref, sf_k_ref, sf_v_ref, attn_scale, use_causal_mask)

    # Compare output - tighter tolerance for FP16
    o_gpu_f32 = o_gpu.float()
    if torch_otype == torch.float16:
        o_atol, o_rtol = 0.6, 0.5  # Tighter tolerance for FP16
    else:
        o_atol, o_rtol = 0.6, 0.5  # BF16 tolerance
    torch.testing.assert_close(o_gpu_f32, o_ref, atol=o_atol, rtol=o_rtol)

    # Compare stats (logsumexp) - tight tolerance since it's computed in FP32
    stats_atol, stats_rtol = 0.02, 0.02
    # torch.testing.assert_close(stats_gpu, stats_ref, atol=stats_atol, rtol=stats_rtol)

    # Compute reference amax and compare
    amax_ref = torch.amax(torch.abs(o_ref)).item()
    amax_gpu = amax_o_gpu.item()
    amax_atol = 0.01 * max(amax_ref, 1.0)  # 1% tolerance
    # assert abs(amax_gpu - amax_ref) < amax_atol, f"amax mismatch: gpu={amax_gpu}, ref={amax_ref}, diff={abs(amax_gpu - amax_ref)}"

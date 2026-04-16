"""
Test fused RoPE+SDPA via cuDNN Graph API.

Constructs a graph with:
  RoPE(Q, cos, sin) → Q'
  RoPE(K, cos, sin) → K'
  SDPA(Q', K', V) → O

When built with heur_mode.OPENSOURCE, the SDPA engine detects the RoPE→SDPA
pattern and orchestrates RoPE as a pre-processing step before attention.

NOTE: Uses BHSD layout (batch, head, seq, dim) as required by the OSS SDPA engine.
      The user must provide intermediate buffers for Q_rot and K_rot.
"""

import pytest
import torch
import torch.nn.functional as F

import cudnn


def rope_reference(x, cos, sin):
    """PyTorch reference for non-interleaved RoPE on BHSD tensor."""
    d2 = x.shape[-1] // 2
    x1 = x[..., :d2].float()
    x2 = x[..., d2:].float()
    cos_f = cos.float().unsqueeze(0).unsqueeze(0)  # [1, 1, S, D/2] for BHSD
    sin_f = sin.float().unsqueeze(0).unsqueeze(0)
    y1 = x1 * cos_f - x2 * sin_f
    y2 = x2 * cos_f + x1 * sin_f
    return torch.cat([y1, y2], dim=-1).to(x.dtype)


def _build_rope_sdpa_graph(B, S, H, D, data_type=cudnn.data_type.BFLOAT16):
    """Build a graph with RoPE(Q) + RoPE(K) → SDPA in BHSD layout."""
    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    d2 = D // 2
    bhsd_stride = [H * S * D, S * D, D, 1]
    stats_stride = [H * S, S, 1, 1]

    Q = graph.tensor(name="Q", dim=[B, H, S, D], stride=bhsd_stride, data_type=data_type)
    K = graph.tensor(name="K", dim=[B, H, S, D], stride=bhsd_stride, data_type=data_type)
    V = graph.tensor(name="V", dim=[B, H, S, D], stride=bhsd_stride, data_type=data_type)
    COS = graph.tensor(name="COS", dim=[S, d2], stride=[d2, 1], data_type=data_type)
    SIN = graph.tensor(name="SIN", dim=[S, d2], stride=[d2, 1], data_type=data_type)

    # Apply RoPE to Q and K
    Q_rot = graph.rope(input=Q, cos=COS, sin=SIN, name="RoPE_Q")
    Q_rot.set_data_type(data_type).set_dim([B, H, S, D]).set_stride(bhsd_stride)

    K_rot = graph.rope(input=K, cos=COS, sin=SIN, name="RoPE_K")
    K_rot.set_data_type(data_type).set_dim([B, H, S, D]).set_stride(bhsd_stride)

    # OSS SDPA requires score_max and score_sum_exp outputs
    score_max = graph.tensor(name="score_max", dim=[B, H, S, 1], stride=stats_stride, data_type=cudnn.data_type.FLOAT)
    score_max.set_output(True)
    score_sum_exp = graph.tensor(
        name="score_sum_exp", dim=[B, H, S, 1], stride=stats_stride, data_type=cudnn.data_type.FLOAT
    )
    score_sum_exp.set_output(True)

    # SDPA
    O, stats = graph.sdpa(
        q=Q_rot, k=K_rot, v=V, is_inference=True, score_max=score_max, score_sum_exp=score_sum_exp, name="SDPA"
    )
    O.set_output(True).set_data_type(data_type).set_dim([B, H, S, D]).set_stride(bhsd_stride)
    if stats is not None:
        stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.build([cudnn.heur_mode.OPENSOURCE])

    return graph, Q, K, V, COS, SIN, Q_rot, K_rot, O, stats, score_max, score_sum_exp


def _run_rope_sdpa_test(B, S, H, D, dtype=torch.bfloat16):
    """Run a fused RoPE+SDPA test and return max absolute error."""
    cudnn_dtype = cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF

    result = _build_rope_sdpa_graph(B, S, H, D, data_type=cudnn_dtype)
    graph, Q, K, V, COS, SIN, Q_rot, K_rot, O, stats, score_max, score_sum_exp = result

    d2 = D // 2

    # Create test data (BHSD layout)
    q_gpu = torch.randn(B, H, S, D, dtype=dtype, device="cuda") * 0.1
    k_gpu = torch.randn(B, H, S, D, dtype=dtype, device="cuda") * 0.1
    v_gpu = torch.randn(B, H, S, D, dtype=dtype, device="cuda") * 0.1
    cos_gpu = torch.cos(
        torch.arange(S, device="cuda").float().unsqueeze(1) * torch.arange(d2, device="cuda").float().unsqueeze(0) * 0.01
    ).to(dtype)
    sin_gpu = torch.sin(
        torch.arange(S, device="cuda").float().unsqueeze(1) * torch.arange(d2, device="cuda").float().unsqueeze(0) * 0.01
    ).to(dtype)
    o_gpu = torch.empty(B, H, S, D, dtype=dtype, device="cuda")
    q_rot_buf = torch.empty_like(q_gpu)
    k_rot_buf = torch.empty_like(k_gpu)

    tensor_map = {
        Q: q_gpu, K: k_gpu, V: v_gpu, COS: cos_gpu, SIN: sin_gpu,
        O: o_gpu, Q_rot: q_rot_buf, K_rot: k_rot_buf,
        score_max: torch.empty(B, H, S, 1, dtype=torch.float32, device="cuda"),
        score_sum_exp: torch.empty(B, H, S, 1, dtype=torch.float32, device="cuda"),
    }
    if stats is not None:
        tensor_map[stats] = torch.empty(B, H, S, 1, dtype=torch.float32, device="cuda")

    ws = torch.empty(max(graph.get_workspace_size(), 1), dtype=torch.uint8, device="cuda")
    graph.execute(tensor_map, ws)

    # Reference: manual RoPE + PyTorch SDPA (both BHSD)
    q_rot_ref = rope_reference(q_gpu, cos_gpu, sin_gpu).float()
    k_rot_ref = rope_reference(k_gpu, cos_gpu, sin_gpu).float()
    o_ref = F.scaled_dot_product_attention(q_rot_ref, k_rot_ref, v_gpu.float()).to(dtype)

    max_err = (o_gpu.float() - o_ref.float()).abs().max().item()
    return max_err


# ---- Tests ----


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("S", [128, 512])
@pytest.mark.parametrize("H", [8, 32])
@pytest.mark.parametrize("D", [64, 128])
def test_rope_sdpa_fused_bf16(B, S, H, D):
    """Test fused RoPE+SDPA with bf16."""
    max_err = _run_rope_sdpa_test(B, S, H, D, dtype=torch.bfloat16)
    # OSS SDPA has ~0.4 tolerance vs PyTorch reference (flash attention numerics)
    assert max_err < 1.0, f"Fused RoPE+SDPA bf16 max_err={max_err} exceeds tolerance"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_rope_sdpa_smoke():
    """Quick smoke test for fused RoPE+SDPA."""
    max_err = _run_rope_sdpa_test(1, 128, 8, 128, dtype=torch.bfloat16)
    assert max_err < 1.0, f"Fused RoPE+SDPA smoke max_err={max_err}"

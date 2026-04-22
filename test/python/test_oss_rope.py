"""
Test RoPE + SDPA via cuDNN Graph API.

Constructs a graph with RoPE nodes feeding into SDPA.
RoPE takes raw frequency angles (freqs) and computes sincosf internally.

User graph:
  q_rot = graph.rope(q, freqs)
  k_rot = graph.rope(k, freqs)
  o = graph.sdpa(q_rot, k_rot, v)
"""

import pytest
import torch
import torch.nn.functional as F

import cudnn


def rope_reference(x, freqs):
    """PyTorch reference for non-interleaved RoPE with raw freqs (BHSD layout).

    Args:
        x: [B, H, S, D] input tensor
        freqs: [S, 1, 1, D] raw angle values (float32)

    Returns:
        [B, H, S, D] rotated tensor
    """
    d = x.shape[-1]
    d2 = d // 2
    # freqs is [S, 1, 1, D], we need [S, D/2] for the cos/sin
    angles = freqs[:, 0, 0, :d2].float()  # [S, D/2]
    cos_vals = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # [1, 1, S, D/2]
    sin_vals = torch.sin(angles).unsqueeze(0).unsqueeze(0)

    x1 = x[..., :d2].float()
    x2 = x[..., d2:].float()
    y1 = x1 * cos_vals - x2 * sin_vals
    y2 = x2 * cos_vals + x1 * sin_vals
    return torch.cat([y1, y2], dim=-1).to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_rope_sdpa_smoke():
    """Smoke test: build a RoPE+SDPA graph and verify it constructs."""
    B, S, H, D = 1, 128, 8, 128

    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    bhsd_stride = [H * S * D, S * D, D, 1]

    Q = graph.tensor(name="Q", dim=[B, H, S, D], stride=bhsd_stride, data_type=cudnn.data_type.BFLOAT16)
    K = graph.tensor(name="K", dim=[B, H, S, D], stride=bhsd_stride, data_type=cudnn.data_type.BFLOAT16)
    V = graph.tensor(name="V", dim=[B, H, S, D], stride=bhsd_stride, data_type=cudnn.data_type.BFLOAT16)
    FREQS = graph.tensor(name="freqs", dim=[S, 1, 1, D], stride=[D, D, D, 1], data_type=cudnn.data_type.FLOAT)

    Q_rot = graph.rope(input=Q, freqs=FREQS, name="RoPE_Q")
    Q_rot.set_data_type(cudnn.data_type.BFLOAT16).set_dim([B, H, S, D]).set_stride(bhsd_stride)

    K_rot = graph.rope(input=K, freqs=FREQS, name="RoPE_K")
    K_rot.set_data_type(cudnn.data_type.BFLOAT16).set_dim([B, H, S, D]).set_stride(bhsd_stride)

    O, stats = graph.sdpa(q=Q_rot, k=K_rot, v=V, is_inference=True, name="SDPA")
    O.set_output(True).set_data_type(cudnn.data_type.BFLOAT16).set_dim([B, H, S, D]).set_stride(bhsd_stride)
    if stats is not None:
        stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    # This should build successfully with backend heuristics
    # (the backend detects RoPE+SDPA pattern and fuses them)
    graph.build([cudnn.heur_mode.HEURISTICS_CHOICE])
    print(f"Graph built successfully, workspace: {graph.get_workspace_size()}")

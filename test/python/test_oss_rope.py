"""
Test standalone RoPE OSS engine via cuDNN Graph API.

Constructs a graph with a single RoPE node and builds with
heur_mode.OPENSOURCE to dispatch to the NVRTC-compiled RoPE kernel.

Non-interleaved (halved) RoPE:
  y[..., :D/2] = x[..., :D/2] * cos - x[..., D/2:] * sin
  y[..., D/2:] = x[..., D/2:] * cos + x[..., :D/2] * sin
"""

import pytest
import torch

import cudnn


def rope_reference(x, cos, sin):
    """PyTorch reference for non-interleaved RoPE.

    Args:
        x: [B, S, H, D] input tensor
        cos: [S, D/2] cosine values
        sin: [S, D/2] sine values

    Returns:
        [B, S, H, D] rotated tensor
    """
    d2 = x.shape[-1] // 2
    x1 = x[..., :d2].float()
    x2 = x[..., d2:].float()
    cos_f = cos.float().unsqueeze(0).unsqueeze(2)  # [1, S, 1, D/2]
    sin_f = sin.float().unsqueeze(0).unsqueeze(2)  # [1, S, 1, D/2]
    y1 = x1 * cos_f - x2 * sin_f
    y2 = x2 * cos_f + x1 * sin_f
    return torch.cat([y1, y2], dim=-1).to(x.dtype)


def _build_rope_graph(B, S, H, D, data_type=cudnn.data_type.BFLOAT16):
    """Build a graph with a standalone RoPE node."""
    graph = cudnn.pygraph(
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )

    d2 = D // 2

    X = graph.tensor(
        name="X",
        dim=[B, S, H, D],
        stride=[S * H * D, H * D, D, 1],
        data_type=data_type,
    )

    COS = graph.tensor(
        name="COS",
        dim=[S, d2],
        stride=[d2, 1],
        data_type=data_type,
    )

    SIN = graph.tensor(
        name="SIN",
        dim=[S, d2],
        stride=[d2, 1],
        data_type=data_type,
    )

    Y = graph.rope(input=X, cos=COS, sin=SIN, name="RoPE")
    Y.set_output(True).set_data_type(data_type)

    graph.build([cudnn.heur_mode.OPENSOURCE])

    return graph, X, COS, SIN, Y


def _run_rope_test(B, S, H, D, dtype=torch.bfloat16):
    """Run a standalone RoPE test and return max absolute error."""
    cudnn_dtype = cudnn.data_type.BFLOAT16 if dtype == torch.bfloat16 else cudnn.data_type.HALF

    graph, X, COS, SIN, Y = _build_rope_graph(B, S, H, D, data_type=cudnn_dtype)

    d2 = D // 2

    # Create test data
    x_gpu = torch.randn(B, S, H, D, dtype=dtype, device="cuda")
    cos_gpu = torch.randn(S, d2, dtype=dtype, device="cuda")
    sin_gpu = torch.randn(S, d2, dtype=dtype, device="cuda")
    y_gpu = torch.empty_like(x_gpu)

    # Execute
    graph.execute(
        {X: x_gpu, COS: cos_gpu, SIN: sin_gpu, Y: y_gpu},
        torch.empty(graph.get_workspace_size(), dtype=torch.uint8, device="cuda"),
    )

    # Reference
    y_ref = rope_reference(x_gpu, cos_gpu, sin_gpu)

    max_err = (y_gpu.float() - y_ref.float()).abs().max().item()
    return max_err


# ---- Tests ----

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
@pytest.mark.parametrize("B", [1, 4])
@pytest.mark.parametrize("S", [1, 128, 2048])
@pytest.mark.parametrize("H", [8, 32])
@pytest.mark.parametrize("D", [64, 128])
def test_rope_standalone_bf16(B, S, H, D):
    """Test standalone RoPE with bf16."""
    max_err = _run_rope_test(B, S, H, D, dtype=torch.bfloat16)
    assert max_err < 1e-2, f"RoPE bf16 max_err={max_err} exceeds tolerance"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
@pytest.mark.parametrize("D", [64, 128, 256])
def test_rope_standalone_fp16(D):
    """Test standalone RoPE with fp16."""
    max_err = _run_rope_test(1, 128, 8, D, dtype=torch.float16)
    assert max_err < 1e-3, f"RoPE fp16 max_err={max_err} exceeds tolerance"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU")
def test_rope_smoke():
    """Quick smoke test."""
    max_err = _run_rope_test(2, 256, 32, 128, dtype=torch.bfloat16)
    assert max_err < 1e-2, f"RoPE smoke test max_err={max_err}"

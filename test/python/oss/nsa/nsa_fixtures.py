"""
Fixtures for NSA (Native Sparse Attention) tests.
Contains test configuration fixtures and related utilities.
"""

import pytest
import torch
import math


def get_dtype_map():
    """Get mapping from string dtype names to PyTorch dtypes."""
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }


@pytest.fixture
def test_config(request):
    """Extract test configuration from command line arguments."""
    dtype_map = get_dtype_map()

    b = request.config.getoption("--nsa-batch-size")
    s_q = request.config.getoption("--nsa-seq-len")
    h_q = request.config.getoption("--nsa-num-q-heads")
    h_kv = request.config.getoption("--nsa-num-kv-heads")
    d = request.config.getoption("--nsa-head-dim")
    d_v = request.config.getoption("--nsa-value-dim")
    block_size = request.config.getoption("--nsa-block-size")
    topk_size = request.config.getoption("--nsa-topk-size")
    dtype_str = request.config.getoption("--nsa-dtype")
    acc_dtype_str = request.config.getoption("--nsa-acc-dtype")
    skip_ref = request.config.getoption("--nsa-skip-ref")
    window_size = request.config.getoption("--nsa-window-size")
    layout = request.config.getoption("--nsa-layout")

    b = 2 if b is None else b
    s_q = 1024 if s_q is None else s_q
    h_q = 4 if h_q is None else h_q
    h_kv = 1 if h_kv is None else h_kv
    d = 128 if d is None else d
    d_v = 128 if d_v is None else d_v
    block_size = 64 if block_size is None else block_size
    topk_size = 16 if topk_size is None else topk_size
    dtype_str = "bfloat16" if dtype_str is None else dtype_str
    acc_dtype_str = "float32" if acc_dtype_str is None else acc_dtype_str
    window_size = 64 if window_size is None else window_size
    layout = "thd" if layout is None else layout
    assert layout in ["bshd", "thd"], "Layout must be 'bshd' or 'thd'"

    dtype = dtype_map[dtype_str]
    acc_dtype = dtype_map[acc_dtype_str]

    actual_s_q = torch.tensor([s_q] * b, dtype=torch.int32) if layout == "thd" else None
    topk_sizes = torch.tensor([topk_size] * b, dtype=torch.int32)
    softmax_scale = 1.0 / math.sqrt(d)

    return {
        "b": b,
        "s_q": s_q,
        "actual_s_q": actual_s_q,
        "h_q": h_q,
        "h_kv": h_kv,
        "d": d,
        "d_v": d_v,
        "block_size": block_size,
        "topk_sizes": topk_sizes,
        "dtype": dtype,
        "acc_dtype": acc_dtype,
        "softmax_scale": softmax_scale,
        "skip_ref": skip_ref,
        "window_size": window_size,
        "layout": layout,
    }

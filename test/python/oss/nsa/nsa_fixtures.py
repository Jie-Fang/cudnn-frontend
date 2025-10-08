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
    seq_len = request.config.getoption("--nsa-seq-len")
    h_q = request.config.getoption("--nsa-num-q-heads")
    h_kv = request.config.getoption("--nsa-num-kv-heads")
    d = request.config.getoption("--nsa-head-dim")
    d_v = request.config.getoption("--nsa-value-dim")
    block_size = request.config.getoption("--nsa-block-size")
    topk_size = request.config.getoption("--nsa-topk-size")
    dtype_str = request.config.getoption("--nsa-dtype")
    acc_dtype_str = request.config.getoption("--nsa-acc-dtype")
    skip_ref = request.config.getoption("--nsa-skip-ref")

    b = 2 if b is None else b
    seq_len = 1024 if seq_len is None else seq_len
    h_q = 4 if h_q is None else h_q
    h_kv = 1 if h_kv is None else h_kv
    d = 128 if d is None else d
    d_v = 128 if d_v is None else d_v
    block_size = 64 if block_size is None else block_size
    topk_size = 16 if topk_size is None else topk_size
    dtype_str = "bfloat16" if dtype_str is None else dtype_str
    acc_dtype_str = "float32" if acc_dtype_str is None else acc_dtype_str

    dtype = dtype_map[dtype_str]
    acc_dtype = dtype_map[acc_dtype_str]

    s_q = [seq_len] * b
    topk_sizes = [topk_size] * b
    softmax_scale = 1.0 / math.sqrt(d)

    return {
        "b": b,
        "s_q": s_q,
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
    }

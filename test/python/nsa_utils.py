"""
Utilities for NSA (Native Sparse Attention) tests.
Contains helper functions for environment checking and data generation.
"""

import torch


def _env_supported():
    """Check if the environment supports NSA tests."""
    try:
        import cutlass
    except ImportError:
        print("cutlass is not available")
        return False

    if not torch.cuda.is_available():
        print("CUDA is not available")
        return False

    major, _ = torch.cuda.get_device_capability()
    return major >= 9


def generate_block_indices(
    seq_lens: list[int], num_kv_heads: int, topk_sizes: list[int], block_size: int
):
    """
    Generate block indices and counts for sparse attention.

    Args:
        seq_lens: List of sequence lengths for each batch
        num_kv_heads: Number of key/value heads
        topk_sizes: List of top-k sizes for each batch
        block_size: Size of each block

    Returns:
        Tuple of (block_counts, block_indices) tensors on CUDA
    """
    total_seq_len = sum(seq_lens)
    max_topk_size = max(topk_sizes)
    block_counts = torch.zeros(total_seq_len, num_kv_heads, dtype=torch.int32)
    block_indices = torch.zeros(
        total_seq_len, num_kv_heads, max_topk_size, dtype=torch.int32
    )

    seq_len_offset = 0
    for i in range(len(seq_lens)):
        seq_len = seq_lens[i]
        topk_size = topk_sizes[i]
        max_index = seq_len // block_size
        for t in range(seq_len):
            for h in range(num_kv_heads):
                block_indices[seq_len_offset + t, h, :topk_size] = (
                    torch.randperm(max_index)[:topk_size].sort().values
                )
                block_counts[seq_len_offset + t, h] = topk_size
        seq_len_offset += seq_len

    return block_counts.cuda(), block_indices.cuda()


def init_input_tensors(test_config):
    b = test_config["b"]
    s_q = test_config["s_q"]
    h_q = test_config["h_q"]
    h_kv = test_config["h_kv"]
    d = test_config["d"]
    d_v = test_config["d_v"]
    block_size = test_config["block_size"]
    topk_sizes = test_config["topk_sizes"]
    dtype = test_config["dtype"]
    acc_dtype = test_config["acc_dtype"]
    softmax_scale = test_config["softmax_scale"]
    skip_ref = test_config["skip_ref"]
    seq_total_len = sum(s_q)

    # Q: (T, H_q, D)
    Q = torch.randn((seq_total_len, h_q, d), dtype=dtype).cuda()
    # K: (T, H_kv, D)
    K = torch.randn((seq_total_len, h_kv, d), dtype=dtype).cuda()
    # V: (T, H_kv, D_v)
    V = torch.randn((seq_total_len, h_kv, d_v), dtype=dtype).cuda()

    # block_counts: (T, H_kv), block_indices: (T, H_kv, max(topk_sizes))
    block_counts, block_indices = generate_block_indices(
        s_q, h_kv, topk_sizes, block_size
    )

    # seq_offsets: (B + 1)
    seq_offsets = torch.zeros(b + 1, dtype=torch.int32).cuda()
    for i in range(b):
        seq_offsets[i] = sum(s_q[:i])
    seq_offsets[len(s_q)] = seq_total_len
    max_length = max(s_q)

    return Q, K, V, block_counts, block_indices, seq_offsets, max_length


def allocate_output_tensors(test_config):
    b = test_config["b"]
    s_q = test_config["s_q"]
    h_q = test_config["h_q"]
    h_kv = test_config["h_kv"]
    d = test_config["d"]
    d_v = test_config["d_v"]
    block_size = test_config["block_size"]
    topk_sizes = test_config["topk_sizes"]
    dtype = test_config["dtype"]
    acc_dtype = test_config["acc_dtype"]
    skip_ref = test_config["skip_ref"]
    seq_total_len = sum(s_q)

    # O: (T, H_q, D_v)
    O = torch.zeros((seq_total_len, h_q, d_v), dtype=dtype).cuda()
    # L: (T, H_q)
    L = torch.zeros((seq_total_len, h_q), dtype=torch.float32).cuda()
    # M: (T, H_q)
    M = torch.zeros((seq_total_len, h_q), dtype=torch.float32).cuda()

    return O, L, M

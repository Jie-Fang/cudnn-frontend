"""
Utilities for NSA (Native Sparse Attention) tests.
Contains helper functions for environment checking and data generation.
"""

import torch
import traceback


def _env_supported(target_major=9.0):
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
    if major < target_major:
        print(f"NSA requires compute capability >= {target_major}, found {major}.x")
        return False

    try:
        from cudnn import NSA
    except ImportError as e:
        traceback.print_exc()
        return False

    return True


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
    actual_s_q = test_config["actual_s_q"]
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
    layout = test_config["layout"]

    Q, K, V, block_counts, block_indices, seq_offsets, max_length = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )  # TODO: remove seq_offsets
    if layout == "bshd":
        # bhsd logical layout, bshd stride
        Q = torch.randn(b, s_q, h_q, d, dtype=dtype).transpose(1, 2).cuda()
        K = torch.randn(b, s_q, h_kv, d, dtype=dtype).transpose(1, 2).cuda()
        V = torch.randn(b, s_q, h_kv, d_v, dtype=dtype).transpose(1, 2).cuda()

        block_counts, block_indices = None, None  # TODO
    elif layout == "thd":
        seq_total_len = actual_s_q.sum().item()

        # Q: (T, H_q, D)
        Q = torch.randn((seq_total_len, h_q, d), dtype=dtype).cuda()
        # K: (T, H_kv, D)
        K = torch.randn((seq_total_len, h_kv, d), dtype=dtype).cuda()
        # V: (T, H_kv, D_v)
        V = torch.randn((seq_total_len, h_kv, d_v), dtype=dtype).cuda()

        # block_counts: (T, H_kv), block_indices: (T, H_kv, max(topk_sizes))
        block_counts, block_indices = generate_block_indices(
            actual_s_q, h_kv, topk_sizes, block_size
        )

        # seq_offsets: (B + 1)
        seq_offsets = torch.zeros(b + 1, dtype=torch.int32).cuda()
        for i in range(b):
            seq_offsets[i] = actual_s_q[:i].sum().item()
        seq_offsets[len(actual_s_q)] = seq_total_len
        max_length = max(actual_s_q).item()

    actual_s_q = actual_s_q.cuda() if actual_s_q is not None else None
    seq_offsets = seq_offsets.cuda() if seq_offsets is not None else None
    return (
        Q,
        K,
        V,
        block_counts,
        block_indices,
        actual_s_q,
        seq_offsets,
        max_length,
    )


def allocate_output_tensors(test_config):
    b = test_config["b"]
    s_q = test_config["s_q"]
    actual_s_q = test_config["actual_s_q"]
    h_q = test_config["h_q"]
    h_kv = test_config["h_kv"]
    d = test_config["d"]
    d_v = test_config["d_v"]
    block_size = test_config["block_size"]
    topk_sizes = test_config["topk_sizes"]
    dtype = test_config["dtype"]
    acc_dtype = test_config["acc_dtype"]
    skip_ref = test_config["skip_ref"]
    layout = test_config["layout"]

    O, L, M = None, None, None
    if layout == "bshd":
        O = torch.empty(b, s_q, h_q, d_v, dtype=dtype).transpose(1, 2).cuda()
        L = torch.empty(b, s_q, h_q, 1, dtype=torch.float32).transpose(1, 2).cuda()
        M = torch.empty(b, s_q, h_q, 1, dtype=torch.float32).transpose(1, 2).cuda()
    elif layout == "thd":
        seq_total_len = actual_s_q.sum().item()

        # O: (T, H_q, D_v)
        O = torch.empty(seq_total_len, h_q, d_v, dtype=dtype).cuda()
        # L: (T, H_q, 1)
        L = torch.empty(seq_total_len, h_q, 1, dtype=torch.float32).cuda()
        # M: (T, H_q, 1)
        M = torch.empty(seq_total_len, h_q, 1, dtype=torch.float32).cuda()

    return O, L, M


def convert_thd_to_bshd(thd_tensor, seq_len: torch.Tensor, s: int):
    assert thd_tensor.dim() == 3
    t, h, d = thd_tensor.size()

    if seq_len.dim() == 1:
        seq_len = seq_len.view(-1, 1, 1, 1)
    assert seq_len.dim() == 4
    assert seq_len.size(1) == seq_len.size(2) == seq_len.size(3) == 1
    b = seq_len.size(0)
    seq_len = seq_len.flatten()

    bshd_tensor = torch.zeros(
        (b, s, h, d), dtype=thd_tensor.dtype, device=thd_tensor.device
    )

    cumulative_seq_len = torch.cumsum(seq_len, dim=0) - seq_len
    for bi in range(b):
        t_beg = cumulative_seq_len[bi]
        t_end = t_beg + seq_len[bi]
        bshd_tensor[bi, : seq_len[bi], :, :] = thd_tensor[t_beg:t_end, :, :]

    # Return a view with layout (b, h, s, d) while keeping strides as if (b, s, h, d)
    return bshd_tensor.permute(0, 2, 1, 3)


def convert_bshd_to_thd(bshd_tensor, seq_len: torch.Tensor, maxT: int):
    assert bshd_tensor.dim() == 4
    b, h, s, d = bshd_tensor.size()

    if seq_len.dim() == 1:
        seq_len = seq_len.view(-1, 1, 1, 1)
    assert seq_len.dim() == 4
    assert seq_len.size(1) == seq_len.size(2) == seq_len.size(3) == 1
    seq_len = seq_len.flatten()

    thd_tensor = torch.zeros(
        (maxT, h, d), dtype=bshd_tensor.dtype, device=bshd_tensor.device
    )

    # Interpret input as (b, s, h, d) in memory while keeping the (b, h, s, d) layout
    bshd_base = bshd_tensor.permute(0, 2, 1, 3)

    cumulative_seq_len = torch.cumsum(seq_len, dim=0) - seq_len
    for bi in range(b):
        t_beg = cumulative_seq_len[bi]
        t_end = t_beg + seq_len[bi]
        thd_tensor[t_beg:t_end, :, :] = bshd_base[bi, : seq_len[bi], :, :]

    return thd_tensor


def compute_exclusive_prefix_sum(tensor):
    assert list(tensor.size())[1:] == [1, 1, 1]
    # We need to provide a tuple of two tensors to torch.cat().
    return torch.cat(
        (
            torch.zeros(1, 1, 1, 1, dtype=tensor.dtype, device=tensor.device),
            torch.cumsum(tensor, dim=0),
        )
    )


def _generate_ragged_offset(test_config):
    if test_config["layout"] != "thd":
        return None, None, None, None, None

    h_q = test_config["h_q"]
    h_k = test_config["h_kv"]
    h_v = test_config["h_kv"]
    d_qk = test_config["d"]
    d_v = test_config["d_v"]
    seq_len_q = test_config["actual_s_q"]
    seq_len_kv = test_config["actual_s_q"]

    # Only for thd_thd_thd
    if seq_len_q.ndim == 1:
        seq_len_q = seq_len_q.view(-1, 1, 1, 1)
    if seq_len_kv.ndim == 1:
        seq_len_kv = seq_len_kv.view(-1, 1, 1, 1)

    q_ragged_offset = compute_exclusive_prefix_sum(seq_len_q) * h_q * d_qk
    k_ragged_offset = compute_exclusive_prefix_sum(seq_len_kv) * h_k * d_qk
    v_ragged_offset = compute_exclusive_prefix_sum(seq_len_kv) * h_v * d_v
    o_ragged_offset = compute_exclusive_prefix_sum(seq_len_q) * h_q * d_v
    stats_ragged_offset = compute_exclusive_prefix_sum(seq_len_q) * h_q

    # Convert to int64 for cuDNN 9.6.0
    q_ragged_offset = q_ragged_offset.to(dtype=torch.int64).cuda()
    k_ragged_offset = k_ragged_offset.to(dtype=torch.int64).cuda()
    v_ragged_offset = v_ragged_offset.to(dtype=torch.int64).cuda()
    o_ragged_offset = o_ragged_offset.to(dtype=torch.int64).cuda()
    stats_ragged_offset = stats_ragged_offset.to(dtype=torch.int64).cuda()

    return (
        q_ragged_offset,
        k_ragged_offset,
        v_ragged_offset,
        o_ragged_offset,
        stats_ragged_offset,
    )

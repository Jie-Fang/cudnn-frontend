"""
Reference implementations for NSA (Native Sparse Attention) tests.
Contains CPU/GPU reference implementations for verification.
"""

import torch


def run_ref_nsa_selection_attention(
    Q_in,
    K_in,
    V_in,
    O_out,
    L_out,
    M_out,
    seq_lens,
    block_indices,
    block_counts,
    block_size,
    softmax_scale,
    dtype=torch.float32,
):
    """
    Reference implementation of NSA selection attention.

    This is a CPU-based reference implementation for verifying the correctness
    of the CUDA NSA implementation.

    Args:
        Q_in: Query tensor of shape (T, H_q, D)
        K_in: Key tensor of shape (T, H_kv, D)
        V_in: Value tensor of shape (T, H_kv, D_v)
        O_out: Output tensor of shape (T, H_q, D_v)
        L_out: Log-sum-exp tensor of shape (T, H_q)
        M_out: Max values tensor of shape (T, H_q)
        seq_lens: List of sequence lengths for each batch
        block_indices: Block indices tensor
        block_counts: Block counts tensor
        block_size: Size of each block
        softmax_scale: Softmax scaling factor
        dtype: Data type for computation

    Returns:
        Tuple of (O_out, L_out, M_out) with updated values
    """
    # Q.shape: (T, H_q, D) -> (T, h_kv, g, D)
    # K.shape: (T, H_kv, D) -> (T, h_kv, 1, D)
    # V.shape: (T, H_kv, D_v) -> (T, h_kv, 1, D_v)
    # O.shape: (T, H_q, D_v) -> (T, h_kv, g, D_v)
    # L.shape: (T, H_q) -> (T, h_kv, g)
    # M.shape: (T, H_q) -> (T, h_kv, g)
    # seq_lens.shape: (batch_size)
    # block_indices.shape: (T, h_kv, topk_size)
    # block_counts.shape: (T, h_kv)

    t, h_q, d = Q_in.shape
    _, h_kv, d_v = V_in.shape

    head_num_kv = h_kv
    total_seq_len = t
    GQA_group_size = h_q // head_num_kv

    Q = Q_in.view(t, h_kv, GQA_group_size, d).to(dtype=dtype)
    K = K_in.view(t, h_kv, 1, d).to(dtype=dtype)
    V = V_in.view(t, h_kv, 1, d_v).to(dtype=dtype)
    O = O_out.view(t, h_kv, GQA_group_size, d_v).to(dtype=dtype)
    L = L_out.view(t, h_kv, GQA_group_size).to(dtype=torch.float32)
    M = M_out.view(t, h_kv, GQA_group_size).to(dtype=torch.float32)

    seq_offset = 0
    for seq_idx, seq_len in enumerate(seq_lens):
        seq_end = seq_offset + seq_len

        for h in range(h_kv):
            # Extract Q, K, V for current sequence and head
            q_seq = Q[seq_offset:seq_end, h, :, :]  # [seq_len, GQA_group_size, d]
            k_seq = K[seq_offset:seq_end, h, 0, :]  # [seq_len, d]
            v_seq = V[seq_offset:seq_end, h, 0, :]  # [seq_len, d_v]

            # Step 1: Compute full Q @ K^T attention matrix
            # q_seq: [seq_len, GQA_group_size, d] @ k_seq.T: [d, seq_len] -> [seq_len, GQA_group_size, seq_len]
            qk_scores = torch.matmul(q_seq, k_seq.transpose(-2, -1)) * softmax_scale

            # Step 2: Create block selection mask
            mask = torch.full(
                (seq_len, seq_len),
                float("-inf"),
                device=qk_scores.device,
                dtype=torch.float32,
            )
            seq_block_counts = block_counts[seq_offset:seq_end, h]  # [seq_len]
            seq_block_indices = block_indices[
                seq_offset:seq_end, h, :
            ]  # [seq_len, topk_size]
            topk_size = seq_block_indices.size(-1)
            block_range = torch.arange(topk_size, device=mask.device).unsqueeze(
                0
            )  # [1, topk_size]
            valid_mask = block_range < seq_block_counts.unsqueeze(
                1
            )  # [seq_len, topk_size]

            query_indices, block_indices_flat = torch.where(valid_mask)
            if len(query_indices) > 0:
                block_ids = seq_block_indices[query_indices, block_indices_flat]
                token_starts = block_ids * block_size
                token_ends = torch.clamp((block_ids + 1) * block_size, max=seq_len)

                block_sizes = token_ends - token_starts
                max_block_size = block_sizes.max().item() if len(block_sizes) > 0 else 0

                if max_block_size > 0:
                    offsets = torch.arange(
                        max_block_size, device=mask.device
                    )  # [max_block_size]

                    num_blocks = len(block_ids)
                    offsets_expanded = offsets.unsqueeze(0).expand(
                        num_blocks, -1
                    )  # [num_blocks, max_block_size]
                    block_sizes_expanded = block_sizes.unsqueeze(1)  # [num_blocks, 1]
                    token_starts_expanded = token_starts.unsqueeze(1)  # [num_blocks, 1]
                    query_indices_expanded = query_indices.unsqueeze(
                        1
                    )  # [num_blocks, 1]

                    position_valid = (
                        offsets_expanded < block_sizes_expanded
                    )  # [num_blocks, max_block_size]

                    token_positions = (
                        token_starts_expanded + offsets_expanded
                    )  # [num_blocks, max_block_size]

                    valid_positions = torch.where(position_valid)
                    if len(valid_positions[0]) > 0:
                        block_idx_flat = valid_positions[0]
                        offset_idx = valid_positions[1]

                        final_query_indices = query_indices_expanded[block_idx_flat, 0]
                        final_key_indices = token_positions[block_idx_flat, offset_idx]

                        mask[final_query_indices, final_key_indices] = 0.0

            # Step 3: Apply mask to attention scores
            qk_scores_fp32 = qk_scores.float() + mask.unsqueeze(
                1
            )  # [seq_len, 1, seq_len] -> [seq_len, GQA_group_size, seq_len]

            # Step 4: Compute softmax
            qk_max = torch.max(qk_scores_fp32, dim=-1, keepdim=True)[
                0
            ]  # [seq_len, GQA_group_size, 1]
            qk_exp = torch.exp(
                qk_scores_fp32 - qk_max
            )  # [seq_len, GQA_group_size, seq_len]
            qk_sum = torch.sum(
                qk_exp, dim=-1, keepdim=True
            )  # [seq_len, GQA_group_size, 1]
            attn_weights = qk_exp / qk_sum  # [seq_len, GQA_group_size, seq_len]

            # Step 5: Compute output O = attention_weights @ V
            # attn_weights: [seq_len, GQA_group_size, seq_len] @ v_seq: [seq_len, d_v] -> [seq_len, GQA_group_size, d_v]
            output = torch.matmul(
                attn_weights, v_seq.float()
            )  # [seq_len, GQA_group_size, d_v]

            # Store results
            O[seq_offset:seq_end, h, :, :] = output.to(dtype)

            # Store L (sum of exp) and M (max) statistics - reusing computed values
            # L should store the sum of exponentials (row_sum), not logsumexp, to match reference
            L[seq_offset:seq_end, h, :] = qk_sum.squeeze(
                -1
            )  # [seq_len, GQA_group_size]
            M[seq_offset:seq_end, h, :] = qk_max.squeeze(
                -1
            )  # [seq_len, GQA_group_size]

        seq_offset = seq_end

    return O.view(t, h_q, d_v), L.view(t, h_q), M.view(t, h_q)


def check_ref_nsa_selection_attention(
    Q,
    K,
    V,
    O,
    L,
    M,
    seq_lens,
    block_indices,
    block_counts,
    block_size,
    softmax_scale,
    dtype,
    skip_ref,
):
    if not skip_ref:
        O_ref = torch.zeros_like(O, dtype=torch.float32)
        L_ref = torch.zeros_like(L, dtype=torch.float32)
        M_ref = torch.zeros_like(M, dtype=torch.float32)
        O_ref, L_ref, M_ref = run_ref_nsa_selection_attention(
            Q,
            K,
            V,
            O_ref,
            L_ref,
            M_ref,
            seq_lens,
            block_indices,
            block_counts,
            block_size,
            softmax_scale,
            dtype=dtype,
        )

        torch.testing.assert_close(O, O_ref, atol=0.01, rtol=1e-05)
        # torch.testing.assert_close(L, L_ref, atol=0.01, rtol=1e-05)
        # torch.testing.assert_close(M, M_ref, atol=0.01, rtol=1e-05)
    else:
        print(
            f"Skipped reference computation for performance test with config: b={b}, seq_len={s_q}, h_q={h_q}, h_kv={h_kv}, d={d}"
        )

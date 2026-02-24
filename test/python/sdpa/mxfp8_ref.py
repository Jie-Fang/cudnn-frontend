import torch

# fmt: off

def compute_ref(q_fp8, k_fp8, v_fp8, sf_q_ref, sf_k_ref, sf_v_ref, attn_scale, use_causal_mask=False, torch_itype=torch.float8_e4m3fn, output_type=torch.bfloat16):
    """
    Compute reference SDPA with MXFP8 dequantization.
    Takes FP8 inputs and converts to FP32 to match cuDNN behavior.
    Supports GQA/MQA where K and V have fewer heads than Q.
    """
    # Convert FP8 to FP32 (matches cuDNN's input handling)
    q_f32 = q_fp8.float()
    k_f32 = k_fp8.float()
    v_f32 = v_fp8.float()

    b, h_q, s_q, d_qk = q_f32.shape
    _, h_k, s_kv, _ = k_f32.shape
    _, h_v, _, d_vo = v_f32.shape

    # GQA: expand K, V to match Q's head count
    if h_k != h_q:
        assert h_q % h_k == 0, "h_q must be divisible by h_k for GQA"
        repeats = h_q // h_k
        k_f32 = k_f32.repeat_interleave(repeats, dim=1)
        sf_k_ref = sf_k_ref.reshape(sf_k_ref.shape[0], sf_k_ref.shape[1], b, h_k)
        sf_k_ref = sf_k_ref.repeat_interleave(repeats, dim=3)
        sf_k_ref = sf_k_ref.reshape(sf_k_ref.shape[0], sf_k_ref.shape[1], b * h_q)

    if h_v != h_q:
        assert h_q % h_v == 0, "h_q must be divisible by h_v for GQA"
        repeats = h_q // h_v
        v_f32 = v_f32.repeat_interleave(repeats, dim=1)
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

    # Dequantize Q and K (scale factors apply to d_qk dimension)
    q_dq = q * sf_q
    k_dq = k * sf_k
    # Dequantize V (scale factors apply to s_kv dimension)
    v_dq = v * sf_v

    bias = torch.zeros((b * h_q, s_q, s_kv), dtype=torch.float32, device=q.device)
    if use_causal_mask:
        bias = torch.full((b * h_q, s_q, s_kv), float('-inf'), dtype=torch.float32, device=q.device)
        bias = torch.triu(bias, diagonal=1)

    block_size = 128
    num_blocks = (s_kv + block_size - 1) // block_size

    m_old = torch.full((b * h_q, s_q, 1), float('-inf'), dtype=torch.float32, device=q.device)
    l_old = torch.zeros((b * h_q, s_q, 1), dtype=torch.float32, device=q.device)
    o = torch.zeros((b * h_q, s_q, d_vo), dtype=torch.float32, device=q.device)

    for j in range(num_blocks):
        start_idx = j * block_size
        end_idx = min((j + 1) * block_size, s_kv)
        k_block = k_dq[:, start_idx:end_idx, :]
        v_block = v_dq[:, start_idx:end_idx, :]

        # Q (FP32) @ K^T (FP32) -> S (FP32)
        s_block = torch.einsum("bqd,bkd->bqk", q_dq.float(), k_block.float()) * attn_scale
        s_block = s_block + bias[:, :, start_idx:end_idx]

        m_block = s_block.max(dim=-1, keepdim=True).values
        m_new = torch.maximum(m_old, m_block)

        correction = torch.exp(m_old - m_new).nan_to_num()
        o = o * correction
        l_old = l_old * correction

        p_block = torch.exp(s_block - m_new).nan_to_num()
        l_new = l_old + p_block.sum(dim=-1, keepdim=True)

        # P (FP32) -> P (FP8)
        p_block_quant = p_block.to(torch_itype).float()

        o = o + torch.einsum("bqk,bkd->bqd", p_block_quant, v_block.float())
        m_old = m_new
        l_old = l_new

    o = o / l_old.clamp(min=1.0)

    # O (FP32) -> O (output)
    o_ref = o.reshape(b, h_q, s_q, d_vo).to(output_type).float()

    stats = m_old + torch.log(l_old)
    stats_ref = stats.reshape(b, h_q, s_q, 1).float()

    return o_ref, stats_ref

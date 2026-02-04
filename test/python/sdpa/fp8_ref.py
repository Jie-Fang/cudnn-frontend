import torch

# fmt: off


def compute_amax(q, k, v, attn_scale):
    """Compute amax values for o from float inputs."""
    _, _, h_q, _ = q.shape
    _, _, h_k, _ = k.shape
    _, _, h_v, _ = v.shape

    if h_q != h_k:
        k = k.repeat_interleave(h_q // h_k, dim=2)
    if h_q != h_v:
        v = v.repeat_interleave(h_q // h_v, dim=2)

    # Q @ K^T -> S
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * attn_scale
    p = s.softmax(dim=-1)

    # P @ V -> O
    o = torch.einsum("bhqk,bkhd->bqhd", p, v.float())

    o_amax = o.abs().max().item()

    return o_amax

def compute_backward_amax(q, k, v, o, dO, attn_scale):
    """Compute amax values for dP, dQ, dK, dV from float inputs."""
    _, _, h_q, _ = q.shape
    _, _, h_k, _ = k.shape
    _, _, h_v, _ = v.shape

    k_orig = k
    v_orig = v
    if h_q != h_k:
        k = k.repeat_interleave(h_q // h_k, dim=2)
    if h_q != h_v:
        v = v.repeat_interleave(h_q // h_v, dim=2)

    # Q @ K^T -> S
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * attn_scale
    p = s.softmax(dim=-1)

    # P @ dO -> dV
    dV = torch.einsum("bhqk,bqhd->bkhd", p, dO.float())

    # dO @ V -> dP
    dP = torch.einsum("bqhd,bkhd->bhqk", dO.float(), v.float())

    # dS computation
    D = (o.float() * dO.float()).sum(dim=-1, keepdim=True).transpose(1, 2)
    dS = p * (dP - D) * attn_scale

    # dS @ K -> dQ
    dQ = torch.einsum("bhqk,bkhd->bqhd", dS, k.float())

    # dS^T @ Q -> dK
    dK = torch.einsum("bhqk,bqhd->bkhd", dS, q.float())

    if h_q != h_k:
        dK = dK.reshape(dK.shape[0], dK.shape[1], h_k, h_q // h_k, dK.shape[3]).sum(dim=3)
    if h_q != h_v:
        dV = dV.reshape(dV.shape[0], dV.shape[1], h_v, h_q // h_v, dV.shape[3]).sum(dim=3)

    dP_amax = dP.abs().max().item()
    dQ_amax = dQ.abs().max().item()
    dK_amax = dK.abs().max().item()
    dV_amax = dV.abs().max().item()

    return dP_amax, dQ_amax, dK_amax, dV_amax

def compute_ref(q, k, v, attn_scale,
                q_descale, k_descale, v_descale,
                s_scale, s_descale, torch_itype,
                o_scale, torch_otype):
    """Compute forward pass reference with online softmax tiling."""
    b, s_q, h_q, d_qk = q.shape
    _, s_kv, h_k, _ = k.shape
    _, _, h_v, d_v = v.shape

    k_orig = k
    v_orig = v
    if h_q != h_k:
        k = k.repeat_interleave(h_q // h_k, dim=2)
    if h_q != h_v:
        v = v.repeat_interleave(h_q // h_v, dim=2)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    block_size = 128
    num_blocks = (s_kv + block_size - 1) // block_size

    m_old = torch.full((b, h_q, s_q, 1), float('-inf'), dtype=torch.float32, device=q.device)
    l_old = torch.zeros((b, h_q, s_q, 1), dtype=torch.float32, device=q.device)
    o = torch.zeros((b, h_q, s_q, d_v), dtype=torch.float32, device=q.device)

    for j in range(num_blocks):
        start_idx = j * block_size
        end_idx = min((j + 1) * block_size, s_kv)
        k_block = k[:, :, start_idx:end_idx, :]
        v_block = v[:, :, start_idx:end_idx, :]

        # Q (FP8) @ K^T (FP8) -> S (FP32)
        s_block = torch.einsum("bhqd,bhkd->bhqk", q.float(), k_block.float()) * q_descale * k_descale * attn_scale

        m_block = s_block.max(dim=-1, keepdim=True).values
        m_new = torch.maximum(m_old, m_block)

        correction = torch.exp(m_old - m_new)
        o = o * correction
        l_old = l_old * correction

        p_block = torch.exp(s_block - m_new)
        l_new = l_old + p_block.sum(dim=-1, keepdim=True)

        # P (FP32) -> P (FP8)
        p_block_quant = ((p_block * s_scale).to(torch_itype)).float()

        o = o + torch.einsum("bhqk,bhkd->bhqd", p_block_quant, v_block.float()) * v_descale * s_descale
        m_old = m_new
        l_old = l_new

    o = o / l_old
    o = o.transpose(1, 2)

    # O (FP32) -> O (FP8)
    o_quant = (o * o_scale).to(torch_otype)

    return o_quant

def compute_ref_backward(q, k, v, o, dO, attn_scale,
                         q_descale, k_descale, v_descale,
                         s_scale, s_descale, torch_itype,
                         o_descale, dO_descale,
                         dP_scale, dP_descale,
                         dQ_scale, dK_scale, dV_scale, torch_otype):
    """Compute backward pass reference."""
    _, _, h_q, _ = q.shape
    _, _, h_k, _ = k.shape
    _, _, h_v, _ = v.shape

    k_orig = k
    v_orig = v
    if h_q != h_k:
        k = k.repeat_interleave(h_q // h_k, dim=2)
    if h_q != h_v:
        v = v.repeat_interleave(h_q // h_v, dim=2)

    # Compute P from Q and K
    s = torch.einsum("bqhd,bkhd->bhqk", q.float(), k.float()) * q_descale * k_descale * attn_scale
    p = s.softmax(dim=-1)

    # P (FP32) -> P (FP8)
    p_quant = (p * s_scale).to(torch_itype)

    # P (FP8) @ dO (FP8) -> dV (FP32)
    dV = torch.einsum("bhqk,bqhd->bkhd", p_quant.float(), dO.float()) * s_descale * dO_descale

    # dO (FP8) @ V (FP8) -> dP (FP32)
    dP = torch.einsum("bqhd,bkhd->bhqk", dO.float(), v.float()) * dO_descale * v_descale

    # Compute dS
    o_float = o.float()
    dO_float = dO.float()
    D = (o_float * dO_float).sum(dim=-1, keepdim=True).transpose(1, 2) * o_descale * dO_descale
    dS = p * (dP - D) * attn_scale

    # dS (FP32) -> dS (FP8)
    dS_quant = ((dS * dP_scale).to(torch_itype)).float()

    # dS (FP8) @ K (FP8) -> dQ (FP32)
    dQ = torch.einsum("bhqk,bkhd->bqhd", dS_quant, k.float()) * k_descale * dP_descale

    # dS^T (FP8) @ Q (FP8) -> dK (FP32)
    dK = torch.einsum("bhqk,bqhd->bkhd", dS_quant, q.float()) * q_descale * dP_descale

    # Handle GQA reduction
    if h_q != h_k:
        dK = dK.reshape(dK.shape[0], dK.shape[1], h_k, h_q // h_k, dK.shape[3]).sum(dim=3)
    if h_q != h_v:
        dV = dV.reshape(dV.shape[0], dV.shape[1], h_v, h_q // h_v, dV.shape[3]).sum(dim=3)

    # dQ (FP32) -> dQ (FP8)
    dQ = (dQ * dQ_scale).to(torch_otype)
    # dK (FP32) -> dK (FP8)
    dK = (dK * dK_scale).to(torch_otype)
    # dV (FP32) -> dV (FP8)
    dV = (dV * dV_scale).to(torch_otype)

    return dQ, dK, dV

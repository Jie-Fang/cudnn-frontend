import cudnn
import pytest
import torch
import math

import itertools
import random

def convert_to_cudnn_type(torch_type):
    if torch_type == torch.float16:
        return cudnn.data_type.HALF
    elif torch_type == torch.bfloat16:
        return cudnn.data_type.BFLOAT16
    elif torch_type == torch.float32:
        return cudnn.data_type.FLOAT
    elif torch_type == torch.int32:
        return cudnn.data_type.INT32
    elif torch_type == torch.int64:
        return cudnn.data_type.INT64
    else:
        raise ValueError("Unsupported tensor data type.")

def make_tensor_attr(graph, torch_tensor, name="", dim=None, stride=None, is_pass_by_value=None):
    return graph.tensor(
        name=name,
        dim=dim if dim else torch_tensor.size(),
        stride=stride if stride else torch_tensor.stride(),
        data_type=convert_to_cudnn_type(torch_tensor.dtype),
        is_pass_by_value=is_pass_by_value,
    )

def compare_tensors(expected, actual, tensor_name, rtol=0.1, atol=0.1, fudge=1e-1, print_compare=False):
    assert expected.shape == actual.shape

    expected = expected.to(dtype=torch.float64, device="cuda").flatten()
    actual = actual.to(dtype=torch.float64, device="cuda").flatten()

    n_elem = torch.numel(expected)

    mae = (expected - actual).abs().mean().item()
    perr = ((expected - actual).abs().sum() / expected.abs().sum()).item()
    snr = (expected ** 2).mean().sqrt() / ((expected - actual) ** 2).mean().sqrt()
    snr_db = (10 * torch.log10(snr)).item()

    absolute_error = (expected - actual).abs()
    relative_error = relative_error = absolute_error / torch.where(expected.abs() < fudge, fudge, expected.abs())

    abs_error_indices = absolute_error > atol
    rel_error_indices = relative_error > rtol
    n_abs_errors = torch.sum(abs_error_indices)
    n_rel_errors = torch.sum(rel_error_indices)
    error_indices = torch.logical_and(abs_error_indices, rel_error_indices)
    n_errors = torch.sum(error_indices)

    if print_compare or n_errors != 0:
        print(f"========== {tensor_name} ==========")
        print(f"Absolute Tolerance = {atol}")
        print(f"Relative Tolerance = {rtol}")
        print(f"Number of elements = {n_elem}")
        print(f"Number of absolute errors = {n_abs_errors} ({n_abs_errors * 100 / n_elem:.2f}%)")
        print(f"Number of relative errors = {n_rel_errors} ({n_rel_errors * 100 / n_elem:.2f}%)")
        print(f"Number of errors (absolute and relative) = {n_errors} ({(n_errors * 100)/n_elem:.2f}%)")
        print(f"Maximum absolute error = {absolute_error.max():.4f}")
        print(f"Maximum relative error = {relative_error.max():.4f}")
        print(f"Mean average error = {mae:.4f}")
        print(f"Perr error = {perr:.4f} = 1/{1/perr:.2f}")
        print(f"Signal to noise ratio = {snr.item():.2f} = {snr_db:.2f}dB")
        print(f"Number of Nans = {(torch.isnan(actual)).sum().item()} ({(torch.isnan(actual)).sum().item() * 100 / n_elem:.2f}%)")
        print(f"Number of Zeros = {(n_elem - actual.nonzero().size(0))} ({(n_elem - actual.nonzero().size(0)) * 100 / n_elem:.2f}%)")
        print("===================================\n")

    return n_errors

def get_slopes(n_heads: int):
    """
    ## Get head-specific slope $m$ for each head

    * `n_heads` is the number of heads in the attention layer $n$

    The slope for first head is

    $$\frac{1}{2^{\frac{8}{n}}} = 2^{-\frac{8}{n}}$$

    The slopes for the rest of the heads are in a geometric series with a ratio same as above.

    For instance when the number of heads is $8$ the slopes are
    $$\frac{1}{2^1}, \frac{1}{2^2}, \dots, \frac{1}{2^8}$$
    """

    # Get the closest power of 2 to `n_heads`.
    # If `n_heads` is not a power of 2, then we first calculate slopes to the closest (smaller) power of 2,
    # and then add the remaining slopes.
    n = 2 ** math.floor(math.log2(n_heads))
    # $2^{-\frac{8}{n}}$
    m_0 = 2.0 ** (-8.0 / n)
    # $2^{-1\frac{8}{n}}, 2^{-2 \frac{8}{n}}, 2^{-3 \frac{8}{n}}, \dots$
    m = torch.pow(m_0, torch.arange(1, 1 + n))

    # If `n_heads` is not a power of 2, then we add the remaining slopes.
    # We calculate the remaining slopes for $n * 2$ (avoiding slopes added previously).
    # And pick the slopes upto `n_heads`.
    if n < n_heads:
        # $2^{-\frac{8}{2n}}$
        m_hat_0 = 2.0 ** (-4.0 / n)
        # $2^{-1\frac{8}{2n}}, 2^{-3 \frac{8}{2n}}, 2^{-5 \frac{8}{2n}}, \dots$
        # Note that we take steps by $2$ to avoid slopes added previously.
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        # Concatenate the slopes with the remaining slopes.
        m = torch.cat([m, m_hat])

    # Reshape the tensor to [1, num_heads, 1, 1]
    m = m.view(1, -1, 1, 1).to(device='cuda')

    return m

def compute_o_stats(q, k, v, is_causal=False, bias=None, is_alibi=False, attn_scale=1.0, device="cuda"):
    b, h, s_q, d = q.shape
    _, _, s_kv, _ = k.shape

    assert(k.shape == (b, h, s_kv, d))
    assert(v.shape == (b, h, s_kv, d))

    q = q.to(dtype=torch.float32, device=device)
    k = k.to(dtype=torch.float32, device=device)
    v = v.to(dtype=torch.float32, device=device)

    s = torch.einsum("bhqd,bhkd->bhqk", q, k) * attn_scale
    if bias is not None:
        s.add_(bias)
    if is_alibi:
        s.add_(((torch.arange(s_kv, dtype=torch.float32, device=device)) - torch.arange(s_q, dtype=torch.float32, device=device).view(-1, 1)) * get_slopes(h))
    if is_causal:
        causal_mask = torch.ones(s_q, s_kv, dtype=torch.bool).triu_(diagonal=1).cuda()
        s.masked_fill_(causal_mask, float("-inf"))
    p = torch.softmax(s, dim=-1)
    o = torch.einsum("bhqk,bhkd->bhqd", p, v)

    row_max = torch.amax(s, -1, True)
    row_exp = torch.exp(s - row_max)
    row_sum = torch.sum(row_exp, -1, True)
    stats = row_max + torch.log(row_sum)

    return o, stats

class ScaledDotProductAttentionPyT(torch.nn.Module):
    def __init__(self, is_causal=False, is_bias=False, is_alibi=False, attn_scale=1.0):
        super(ScaledDotProductAttentionPyT, self).__init__()
        self.is_bias = is_bias
        self.is_causal = is_causal
        self.is_alibi = is_alibi
        self.attn_scale = attn_scale

    def forward(self, q, k, v, bias=None):
        b, h, s_q, d = q.shape
        _, _, s_kv, _ = k.shape

        assert k.shape == (b, h, s_kv, d)
        assert v.shape == (b, h, s_kv, d)

        assert self.is_bias == (bias != None)

        s = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.attn_scale
        if self.is_bias:
            s.add_(bias)
        if self.is_alibi:
            s.add_(((torch.arange(s_kv, dtype=q.dtype)) - torch.arange(s_q, dtype=q.dtype).view(-1, 1)) * get_slopes(h))
        if self.is_causal:
            causal_mask = torch.ones(s_q, s_kv, dtype=torch.bool).triu_(diagonal=1).cuda()
            s.masked_fill_(causal_mask, float("-inf"))
        p = torch.softmax(s, dim=-1)
        o = torch.einsum("bhqk,bhkd->bhqd", p, v)
        return o

alibi_mask_options = [True, False]
padding_mask_options = [True, False]
causal_mask_options = [True, False]
layout_options      = ["non_interleaved", "bs3hd", "sbh3d"]
dropout             = [False]
is_infer_options    = [True, False]
bias                = [True, False]
input_type_options  = [torch.float16, torch.bfloat16]

all_options_forward = [elem for elem in itertools.product(*[
    alibi_mask_options,
    padding_mask_options,
    causal_mask_options,
    layout_options,
    dropout,
    is_infer_options,
    bias,
    input_type_options,
])]

@pytest.fixture(params=all_options_forward)
def param_extract(request):
  return request.param

@pytest.mark.skipif(cudnn.backend_version() < 8903, reason="requires cudnn 8.9 or higher")
def test_scale_dot_product_flash_attention(param_extract, print_compare=False):
    alibi_mask, padding_mask, causal_mask, layout, dropout_enable, is_infer, bias_enable, input_type = param_extract

    if alibi_mask and cudnn.backend_version() < 8904:
        pytest.skip("ALiBi mask is only supported 8.9.4 onwards.")

    if padding_mask and cudnn.backend_version() < 8903:
        pytest.skip("Padding mask is only supported 8.9.3 onwards.")

    s_q_choices = [256, 512, 1024, 2048]
    d_choices   = [64,128]

    b = 32
    h = 12
    s_q  = random.choice(s_q_choices)
    s_kv  = s_q
    d = random.choice(d_choices)

    print(f"{str(param_extract)} s={s_q} {d=}")

    attn_scale_val = 0.125

    if dropout_enable == False:
        dropout_prob = 1.0
    else:
        dropout_prob = 0.1

    shape_q = (b, h, s_q, d)
    shape_k = (b, h, d, s_kv)
    shape_v = (b, h, s_kv, d)

    stride_sbh3d = (3 * h * d, 3 * d, b * 3 * h * d, 1)
    stride_sbh3d_t = (3 * h * d, 3 * d, 1, b * 3 * h * d)
    stride_sbhd = (h * d, d, b * h * d, 1)

    stride_bs3hd = (s_q * 3 * h * d, d, 3 * h * d, 1)
    stride_bs3hd_t = (s_q * 3 * h * d, d, 1, 3 * h * d)
    stride_bshd = (s_q * h * d, d, h * d, 1)

    offset_multiple_sbh3d = d
    offset_multiple_bs3hd = h * d

    bias_gpu = torch.randn(b, 1, s_q, s_kv, requires_grad=False, device="cuda", dtype=input_type) if bias_enable else None

    if layout == 'sbh3d':
        stride_q = stride_sbh3d
        stride_k = stride_sbh3d_t
        stride_v = stride_sbh3d

        stride_o = stride_sbhd

        offset_q = offset_multiple_sbh3d * 0
        offset_k = offset_multiple_sbh3d * 1
        offset_v = offset_multiple_sbh3d * 2
    elif layout == 'bs3hd':
        stride_q = stride_bs3hd
        stride_k = stride_bs3hd_t
        stride_v = stride_bs3hd

        stride_o = stride_bshd

        offset_q = offset_multiple_bs3hd * 0
        offset_k = offset_multiple_bs3hd * 1
        offset_v = offset_multiple_bs3hd * 2
    elif layout == 'non_interleaved':
        stride_q = (1 * d * s_q *  h, 1 * d *  s_q, 1 * d, 1)
        stride_k = (1 * d * s_kv * h, 1 * d * s_kv, 1, 1 * d)
        stride_v = (1 * d * s_kv * h, 1 * d * s_kv, 1 * d, 1)

        stride_o = (d * s_q * h, d * s_q, d, 1)

        offset_q = 0
        offset_k = offset_q + b * d * s_q *  h
        offset_v = offset_k + b * d * s_kv * h
    else:
        assert False, "Layout should be either sbh3d or bs3hd or non_interleaved"

    qkv_gpu = 1 *  (torch.randn(b * s_q * 3 * h * d, dtype=input_type, device="cuda") - 0.5)

    q_gpu = torch.as_strided(qkv_gpu, shape_q, stride_q, storage_offset=offset_q)
    k_gpu = torch.as_strided(qkv_gpu, shape_k, stride_k, storage_offset=offset_k)
    v_gpu = torch.as_strided(qkv_gpu, shape_v, stride_v, storage_offset=offset_v)

    if padding_mask:
        seq_len_q_gpu = torch.full((b,1,1,1), s_q, dtype=torch.int32, device="cuda")
        seq_len_Kv_gpu = torch.full((b,1,1,1), s_kv, dtype=torch.int32, device="cuda")

    attn_scale_cpu = torch.full((1,1,1,1), attn_scale_val, dtype=torch.float32, device="cpu")

    seed_gpu = torch.full((1,1,1,1), 123456, dtype=torch.int64, device="cuda")
    offset_gpu = torch.full((1,1,1,1), 1, dtype=torch.int64, device="cuda")

    o_gpu = torch.zeros(b * s_q * h * d, dtype=input_type, device="cuda")
    stats_gpu = torch.zeros(b * h * s_q * 1, dtype=torch.float32, device="cuda")

    # cuDNN graph
    graph = cudnn.pygraph(io_data_type = convert_to_cudnn_type(input_type), intermediate_data_type = cudnn.data_type.FLOAT, compute_data_type = cudnn.data_type.FLOAT)
    q = make_tensor_attr(graph, q_gpu, "q")
    k = make_tensor_attr(graph, k_gpu, "k")
    v = make_tensor_attr(graph, v_gpu, "v")
    bias = make_tensor_attr(graph, bias_gpu, "bias") if bias_enable else None
    attn_scale = make_tensor_attr(graph, attn_scale_cpu, "attn_scale", is_pass_by_value=True)
    seed = make_tensor_attr(graph, seed_gpu, "seed")
    offset = make_tensor_attr(graph, offset_gpu, "attn_scale")
    dropout_tuple = (dropout_prob, seed, offset) if dropout_enable else None

    seq_len_q = None
    seq_len_kv = None
    if padding_mask:
        seq_len_q = make_tensor_attr(graph, seq_len_q_gpu, "seq_len_q")
        seq_len_kv = make_tensor_attr(graph, seq_len_Kv_gpu, "seq_len_kv")

    o, stats = graph.scaled_dot_product_flash_attention(name="scaled_dot_product_flash_attention",
                                                        q=q, k=k, v=v,
                                                        seq_len_q=seq_len_q, seq_len_kv=seq_len_kv,
                                                        is_inference=is_infer,
                                                        bias=bias,
                                                        dropout=dropout_tuple,
                                                        attn_scale=attn_scale,
                                                        use_alibi_mask=alibi_mask,
                                                        use_padding_mask=padding_mask,
                                                        use_causal_mask=causal_mask)

    o.set_output(True).set_stride(stride_o)

    if is_infer == False:
        stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    graph.check_support()
    graph.build()

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    variant_pack = {
        q: q_gpu,
        k: k_gpu,
        v: v_gpu,
        seed: seed_gpu,
        offset: offset_gpu,
        attn_scale: attn_scale_cpu,
        o: o_gpu,
        stats: stats_gpu
    }

    if bias_enable:
        variant_pack[bias] = bias_gpu

    if padding_mask:
        variant_pack[seq_len_q] = seq_len_q_gpu
        variant_pack[seq_len_kv] = seq_len_Kv_gpu

    graph.execute(variant_pack, workspace)


    q_ref = q_gpu.detach().clone().cuda().float()
    k_ref = k_gpu.permute(0, 1, 3, 2).detach().clone().cuda().float()
    v_ref = v_gpu.detach().clone().cuda().float()
    bias_ref = bias_gpu.detach().clone().cuda().float() if bias_enable else None

    if layout == 'sbh3d':
        o_gpu = o_gpu.view([s_q, b, h, d]).permute(1, 2, 0, 3)
    elif layout == 'bs3hd':
        o_gpu = o_gpu.view([b, s_q, h, d]).permute(0, 2, 1, 3)
    elif layout == 'non_interleaved':
        o_gpu = o_gpu.view([b, h, s_q, d])

    stats_gpu = stats_gpu.view(b, h, s_q, 1)

    # cpu reference
    o_ref, stats_ref = compute_o_stats(q_ref,
                                       k_ref,
                                       v_ref,
                                       is_causal=causal_mask,
                                       bias=bias_ref,
                                       is_alibi=alibi_mask,
                                       attn_scale=attn_scale_val)

    assert compare_tensors(o_ref, o_gpu, "O", print_compare=print_compare) == 0
    if is_infer == False:
        assert compare_tensors(stats_ref, stats_gpu, "stats", print_compare=print_compare) == 0

def test_scale_dot_product_flash_attention_backward(print_compare=False):
    is_causal = True
    layout = "naive"
    input_type = torch.float16

    s_q_choices = [256, 512, 1024, 2048]
    d_choices   = [64,128]

    b = 32
    h = 12
    s_q  = random.choice(s_q_choices)
    s_kv  = s_q
    d = random.choice(d_choices)

    print(f"s={s_q} {d=}")

    attn_scale_val = 0.125


    q = 1 * (torch.randn((b, h, s_q, d), dtype=input_type, device="cuda") - 0.5)
    k = 1 * (torch.randn((b, h, s_kv, d), dtype=input_type, device="cuda") - 0.5)
    v = 1 * (torch.randn((b, h, s_kv, d), dtype=input_type, device="cuda") - 0.5)
    dO = 1 * (torch.randn((b, h, s_q, d), dtype=input_type, device="cuda") - 0.5)

    o, stats = compute_o_stats(q, k, v, is_causal=is_causal, attn_scale=attn_scale_val)
    o = o.to(dtype=input_type).detach().clone()
    stats = stats.to(dtype=torch.float32).detach().clone()

    dQ = torch.empty((b, h, s_q, d), dtype=input_type, device="cuda")
    dK = torch.empty((b, h, s_kv, d), dtype=input_type, device="cuda")
    dV = torch.empty((b, h, s_kv, d), dtype=input_type, device="cuda")

    attn_scale_cpu = torch.full((1,1,1,1), attn_scale_val, dtype=torch.float32, device="cpu")

    # cuDNN graph
    graph = cudnn.pygraph(io_data_type=convert_to_cudnn_type(input_type), intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)
    q_attr = make_tensor_attr(graph, q, name="q")
    k_attr = make_tensor_attr(graph, k, dim=(b, h, d, s_kv), stride=(h * s_kv * d, s_kv * d, 1, d), name="k")
    v_attr = make_tensor_attr(graph, v, dim=(b, h, d, s_kv), stride=(h * s_kv * d, s_kv * d, 1, d), name="v")
    o_attr = make_tensor_attr(graph, o, name="o")
    dO_attr = make_tensor_attr(graph, dO, name="dO")
    stats_attr = make_tensor_attr(graph, stats, name="stats")
    attn_scale = make_tensor_attr(graph, attn_scale_cpu, is_pass_by_value=True, name="attn_scale")

    dQ_attr, dK_attr, dV_attr = graph.scaled_dot_product_flash_attention_backward(
        name="scaled_dot_product_flash_attention",
        q=q_attr,
        k=k_attr,
        v=v_attr,
        o=o_attr,
        dO=dO_attr,
        stats=stats_attr,
        use_causal_mask=True,
        attn_scale=attn_scale
    )

    dQ_attr.set_output(True).set_dim(dQ.size()).set_stride(dQ.stride())
    dK_attr.set_output(True).set_dim(dK.size()).set_stride(dK.stride())
    dV_attr.set_output(True).set_dim(dV.size()).set_stride(dV.stride())

    graph.check_support()
    graph.build()

    variant_pack = {
        q_attr: q,
        k_attr: k,
        v_attr: v,
        o_attr: o,
        dO_attr: dO,
        stats_attr: stats,

        dQ_attr: dQ,
        dK_attr: dK,
        dV_attr: dV,

        attn_scale: attn_scale_cpu
    }

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    graph.execute(variant_pack, workspace)

    # compare with autograd reference
    nn_ref = ScaledDotProductAttentionPyT(is_causal=is_causal, attn_scale=attn_scale_val).cuda().float()

    q_ref = q.detach().clone().cuda().float()
    q_ref.requires_grad = True
    k_ref = k.detach().clone().cuda().float()
    k_ref.requires_grad = True
    v_ref = v.detach().clone().cuda().float()
    v_ref.requires_grad = True
    do_ref = dO.detach().clone().cuda().float()

    o_ref = nn_ref(q_ref, k_ref, v_ref)

    dq_ref, dk_ref, dv_ref = torch.autograd.grad(outputs=o_ref, inputs=(q_ref, k_ref, v_ref), grad_outputs=do_ref)

    assert compare_tensors(dq_ref, dQ, "dQ", print_compare=print_compare) == 0
    assert compare_tensors(dk_ref, dK, "dK", print_compare=print_compare) == 0
    assert compare_tensors(dv_ref, dV, "dV", print_compare=print_compare) == 0

if __name__ == "__main__":
    # print("============= running forward tests ===========")
    # # alibi_mask, padding_mask, causal_mask, layout, dropout_enable, is_infer, bias_enable, input_type
    # test_scale_dot_product_flash_attention((False, False, False, "bs3hd", False, False, True, torch.float16), print_compare=True)
    # for option in all_options_forward:
    #     test_scale_dot_product_flash_attention(option)

    print("============= running backwards tests ===========")
    test_scale_dot_product_flash_attention_backward(print_compare=True)

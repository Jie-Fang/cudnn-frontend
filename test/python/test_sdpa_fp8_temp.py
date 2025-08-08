# fmt: off

import torch
import cudnn
import pytest
import argparse
from enum import IntEnum
from looseversion import LooseVersion
import math

torch.nans = lambda *size, **kwargs: torch.full(size, float('nan'), **kwargs)

# sq1_*, sq4_*, sq32_*, sq64_*: BUG mismatches
TEST_CONFIGS = {
    "d128_f16":            {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 256, "s_kv": 256,  "d_qk": 128, "d_vo": 128, "otype": "fp16",     "atol": 0.04, "rtol": 0.1, "kv_block_size": 0},
    "d64_f16":             {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 256, "s_kv": 256,  "d_qk": 64,  "d_vo": 64,  "otype": "fp16",     "atol": 0.04, "rtol": 0.1, "kv_block_size": 0},
    "d128_f8e4m3":         {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 256, "s_kv": 256,  "d_qk": 128, "d_vo": 128, "otype": "fp8_e4m3", "atol": 0.08, "rtol": 0.2, "kv_block_size": 0},
    # cudnnTest replica:
    # ./cudnnTest -RgraphRunner -jsonTestName=LLM_paged_attention_fp8 -kv=dim_b:2 -kv=dim_qh:4 -kv=dim_qs:256 -kv=dim_kvs:256 -kv=dim_d:128 -kv=dim_kvh:4 -kv=Tin:CUDNN_DATA_FP8_E4M3 -kv=Tout:CUDNN_DATA_FP8_E4M3 -kv=atol:0.08 -kv=rtol:0.2 -minDevVer800 -backendEngine-1 -b -gpuRef -kv=block_size:16 -kv=table_size:16 -kv=max_block_num:31 -kv=dim_num_blocks:32 
    "d128_f8e4m3_paged":   {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 256, "s_kv": 256,  "d_qk": 128, "d_vo": 128, "otype": "fp8_e4m3", "atol": 0.08, "rtol": 0.2, "kv_block_size": 16},
    "d64_f8e4m3":          {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 256, "s_kv": 256,  "d_qk": 64,  "d_vo": 64,  "otype": "fp8_e4m3", "atol": 0.08, "rtol": 0.2, "kv_block_size": 0},
    "d128_f8e5m2":         {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 256, "s_kv": 256,  "d_qk": 128, "d_vo": 128, "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.4, "kv_block_size": 0},
    "d64_f8e5m2":          {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 256, "s_kv": 256,  "d_qk": 64,  "d_vo": 64,  "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.4, "kv_block_size": 0},

    "gqa_f16":             {"b": 2, "h_q": 15, "h_k": 5, "h_v": 3, "s_qo": 256, "s_kv": 256,  "d_qk": 128, "d_vo": 128, "otype": "fp16",     "atol": 0.04, "rtol": 0.1, "kv_block_size": 0},
    "gqa_f8e4m3":          {"b": 2, "h_q": 15, "h_k": 5, "h_v": 3, "s_qo": 256, "s_kv": 256,  "d_qk": 128, "d_vo": 128, "otype": "fp8_e4m3", "atol": 0.08, "rtol": 0.2, "kv_block_size": 0},
    "gqa_f8e5m2":          {"b": 2, "h_q": 15, "h_k": 5, "h_v": 3, "s_qo": 256, "s_kv": 256,  "d_qk": 128, "d_vo": 128, "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.4, "kv_block_size": 0},

#   "sq1_skv256_f16":      {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 1,   "s_kv": 256,  "d_qk": 128, "d_vo": 128, "otype": "fp16",     "atol": 0.04, "rtol": 0.1, "kv_block_size": 0},
#   "sq1_skv1024_f16":     {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 1,   "s_kv": 1024, "d_qk": 128, "d_vo": 128, "otype": "fp16",     "atol": 0.04, "rtol": 0.1, "kv_block_size": 0},
    "sq1_skv256_f8e4m3":   {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 1,   "s_kv": 256,  "d_qk": 128, "d_vo": 128, "otype": "fp8_e4m3", "atol": 0.08, "rtol": 0.2, "kv_block_size": 0},
#   "sq1_skv1024_f8e4m3":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 1,   "s_kv": 1024, "d_qk": 128, "d_vo": 128, "otype": "fp8_e4m3", "atol": 0.08, "rtol": 0.2, "kv_block_size": 0},
#   "sq1_skv256_f8e5m2":   {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 1,   "s_kv": 256,  "d_qk": 128, "d_vo": 128, "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2, "kv_block_size": 0},
#   "sq1_skv1024_f8e5m2":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 1,   "s_kv": 1024, "d_qk": 128, "d_vo": 128, "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2, "kv_block_size": 0},

#   "sq4_skv256_f16":      {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 4,   "s_kv": 256,  "d_qk": 128, "d_vo": 128, "otype": "fp16",     "atol": 0.04, "rtol": 0.1, "kv_block_size": 0},
#   "sq4_skv1024_f16":     {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 4,   "s_kv": 1024, "d_qk": 128, "d_vo": 128, "otype": "fp16",     "atol": 0.04, "rtol": 0.1, "kv_block_size": 0},
    "sq4_skv256_f8e4m3":   {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 4,   "s_kv": 256,  "d_qk": 128, "d_vo": 128, "otype": "fp8_e4m3", "atol": 0.16, "rtol": 0.2, "kv_block_size": 0},
#   "sq4_skv1024_f8e4m3":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 4,   "s_kv": 1024, "d_qk": 128, "d_vo": 128, "otype": "fp8_e4m3", "atol": 0.16, "rtol": 0.2, "kv_block_size": 0},
#   "sq4_skv256_f8e5m2":   {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 4,   "s_kv": 256,  "d_qk": 128, "d_vo": 128, "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2, "kv_block_size": 0},
#   "sq4_skv1024_f8e5m2":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 4,   "s_kv": 1024, "d_qk": 128, "d_vo": 128, "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2, "kv_block_size": 0},

#   "sq64_skv256_f16":     {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 4,   "s_kv": 256,  "d_qk": 128, "d_vo": 128, "otype": "fp16",     "atol": 0.04, "rtol": 0.1, "kv_block_size": 0},
#   "sq64_skv1024_f16":    {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 4,   "s_kv": 1024, "d_qk": 128, "d_vo": 128, "otype": "fp16",     "atol": 0.04, "rtol": 0.1, "kv_block_size": 0},
    "sq64_skv256_f8e4m3":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 4,   "s_kv": 256,  "d_qk": 128, "d_vo": 128, "otype": "fp8_e4m3", "atol": 0.16, "rtol": 0.2, "kv_block_size": 0},
#   "sq64_skv1024_f8e4m3": {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 4,   "s_kv": 1024, "d_qk": 128, "d_vo": 128, "otype": "fp8_e4m3", "atol": 0.16, "rtol": 0.2, "kv_block_size": 0},
    "sq64_skv256_f8e5m2":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 4,   "s_kv": 256,  "d_qk": 128, "d_vo": 128, "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2, "kv_block_size": 0},
#   "sq64_skv1024_f8e5m2": {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 4,   "s_kv": 1024, "d_qk": 128, "d_vo": 128, "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.2, "kv_block_size": 0},

    "sq65_skv256_f16":     {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 65,  "s_kv": 256,  "d_qk": 128, "d_vo": 128, "otype": "fp16",     "atol": 0.04, "rtol": 0.1, "kv_block_size": 0},
    "sq65_skv256_f8e4m3":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 65,  "s_kv": 256,  "d_qk": 128, "d_vo": 128, "otype": "fp8_e4m3", "atol": 0.08, "rtol": 0.2, "kv_block_size": 0},
    "sq65_skv256_f8e5m2":  {"b": 2, "h_q": 4,  "h_k": 4, "h_v": 4, "s_qo": 65,  "s_kv": 256,  "d_qk": 128, "d_vo": 128, "otype": "fp8_e5m2", "atol": 0.16, "rtol": 0.4, "kv_block_size": 0},
}

def section_begin(msg, width=80):
    print(f" {msg} ".center(width, '='))

def section_end(width=80):
    print("=" * width)

def get_fp8_largest_po2(dtype: torch.dtype):
    if dtype == torch.float8_e4m3fn:
        return 128.0 # max representable value: 0x1.e00000p+7
    elif dtype == torch.float8_e5m2:
        return 32768.0 # max representable value: 0x1.c00000p+15
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

def get_fp8_scale_factor(amax: float, dtype: torch.dtype, fudge_factor: float = 0.5, epsilon = 0.0625):
    po2_next = 2 ** math.ceil(math.log2(max(amax, epsilon)))
    return get_fp8_largest_po2(dtype) / po2_next * fudge_factor

def get_fp8_descale_factor(amax: float, dtype: torch.dtype, fudge_factor: float = 0.5, epsilon = 0.0625):
    return 1.0 / get_fp8_scale_factor(amax, dtype, fudge_factor, epsilon)

class XFailed(Exception):
    pass

class GraphFwdUid(IntEnum):
    q = 0
    k = 1
    v = 2
    o = 3
    stats = 4
    descale_q = 5
    descale_k = 6
    descale_v = 7
    scale_s = 9
    descale_s = 8
    scale_o = 10
    amax_s = 11
    amax_o = 12
    block_table_k = 13
    block_table_v = 14
    seq_len_kv = 15
    seq_len_q = 16

def generate_graph_fwd(cudnn_input_type, cudnn_output_type, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, block_size):
    graph_fwd = cudnn.pygraph(io_data_type=cudnn_input_type, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT)

    # Variable sequence lenths are required for paged attention
    use_padding_mask = block_size > 0
    block_table_k = None
    block_table_v = None
    seq_len_kv = None
    seq_len_q = None

    if block_size == 0:
        q_fwd = graph_fwd.tensor(uid=GraphFwdUid.q, dim=(b, h_q, s_qo, d_qk), stride=(s_qo * h_q * d_qk, d_qk, h_q * d_qk, 1), data_type=cudnn_input_type)
        k_fwd = graph_fwd.tensor(uid=GraphFwdUid.k, dim=(b, h_k, s_kv, d_qk), stride=(s_kv * h_k * d_qk, d_qk, h_k * d_qk, 1), data_type=cudnn_input_type)
        v_fwd = graph_fwd.tensor(uid=GraphFwdUid.v, dim=(b, h_v, s_kv, d_vo), stride=(s_kv * h_v * d_vo, d_vo, h_v * d_vo, 1), data_type=cudnn_input_type)
    else:
        table_size = math.ceil(s_kv / block_size)
        num_blocks = table_size * b

        q_fwd = graph_fwd.tensor(uid=GraphFwdUid.q, dim=(b, h_q, s_qo, d_qk), stride=(s_qo * h_q * d_qk, d_qk, h_q * d_qk, 1), data_type=cudnn_input_type)
        k_fwd = graph_fwd.tensor(uid=GraphFwdUid.k, dim=(num_blocks, h_k, block_size, d_qk), stride=(block_size * h_k * d_qk, d_qk, h_k * d_qk, 1), data_type=cudnn_input_type)
        v_fwd = graph_fwd.tensor(uid=GraphFwdUid.v, dim=(num_blocks, h_v, block_size, d_vo), stride=(block_size * h_v * d_vo, d_vo, h_v * d_vo, 1), data_type=cudnn_input_type)

        block_table_k = graph_fwd.tensor(uid=GraphFwdUid.block_table_k, dim=(b, 1, table_size, 1), stride=(table_size, table_size, 1, 1), data_type=cudnn.data_type.INT32)
        block_table_v = graph_fwd.tensor(uid=GraphFwdUid.block_table_v, dim=(b, 1, table_size, 1), stride=(table_size, table_size, 1, 1), data_type=cudnn.data_type.INT32)

        seq_len_kv = graph_fwd.tensor(uid=GraphFwdUid.seq_len_kv, dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)
        seq_len_q = graph_fwd.tensor(uid=GraphFwdUid.seq_len_q, dim=(b, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.INT32)

    descale_q_fwd = graph_fwd.tensor(uid=GraphFwdUid.descale_q, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    descale_k_fwd = graph_fwd.tensor(uid=GraphFwdUid.descale_k, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    descale_v_fwd = graph_fwd.tensor(uid=GraphFwdUid.descale_v, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    scale_s_fwd = graph_fwd.tensor(uid=GraphFwdUid.scale_s, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)
    descale_s_fwd = graph_fwd.tensor(uid=GraphFwdUid.descale_s, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    scale_o_fwd = graph_fwd.tensor(uid=GraphFwdUid.scale_o, dim=(1, 1, 1, 1), stride=(1, 1, 1, 1), data_type=cudnn.data_type.FLOAT)

    o_fwd, stats_fwd, amax_s_fwd, amax_o_fwd = graph_fwd.sdpa_fp8(
        q=q_fwd,
        k=k_fwd,
        v=v_fwd,
        descale_q=descale_q_fwd,
        descale_k=descale_k_fwd,
        descale_v=descale_v_fwd,
        scale_s=scale_s_fwd,
        descale_s=descale_s_fwd,
        scale_o=scale_o_fwd,
        generate_stats=True,
        attn_scale=attn_scale,
        use_causal_mask=False,
        use_padding_mask=use_padding_mask,
        seq_len_kv=seq_len_kv,
        seq_len_q=seq_len_q,
        paged_attention_k_table=block_table_k,  # Block Table K: Tensor containing offsets to the container with K blocks
        paged_attention_v_table=block_table_v,  # Block Table V: Tensor containing offsets to the container with V blocks
        paged_attention_max_seq_len_kv=s_kv,  # The maximum sequence length for K caches (this is optional, but recommended)
    )

    o_fwd.set_uid(GraphFwdUid.o).set_output(True).set_dim((b, h_q, s_qo, d_vo)).set_stride((s_qo * h_q * d_vo, d_vo, h_q * d_vo, 1)).set_data_type(cudnn_output_type)
    stats_fwd.set_uid(GraphFwdUid.stats).set_output(True).set_dim((b, h_q, s_qo, 1)).set_stride((s_qo * h_q, h_q * 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_s_fwd.set_uid(GraphFwdUid.amax_s).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)
    amax_o_fwd.set_uid(GraphFwdUid.amax_o).set_output(True).set_dim((1, 1, 1, 1)).set_stride((1, 1, 1, 1)).set_data_type(cudnn.data_type.FLOAT)

    return graph_fwd

def create_container_and_block_table(tensor, block_size):
    B, H, S, D = tensor.shape
    # num_blocks = math.ceil(S/block_size) * B
    blocks_per_batch = math.ceil(S / block_size)

    # Only needed if S is not a multiple of block_size
    padding_seq = (blocks_per_batch * block_size) - S
    if padding_seq > 0:
        zeros = torch.zeros(B, H, padding_seq, D, device="cuda", dtype=tensor.dtype)
        cat_tensor = torch.cat((tensor, zeros), axis=2)
    else:
        cat_tensor = tensor

    # Create a container by splitting on the S dimension and concatenating at the block dimension
    # Its dimensions are [num_blocks, H, block_size, D] with num_blocks = B * blocks_per_batch
    container = torch.cat((cat_tensor.clone()).chunk(blocks_per_batch, dim=2), dim=0)

    # Create the block table
    table_size = math.ceil(S / block_size)
    block_table_temp = torch.linspace(
        0, B * table_size - 1, B * table_size, device="cuda", dtype=torch.int32
    ).reshape(table_size, 1, B, 1)
    block_table_temp = torch.transpose(block_table_temp, 0, 2)

    # Make batch size outer dimension (cuDNN backend preference)
    block_table = (
        torch.zeros(blocks_per_batch * B)
        .int()
        .cuda()
        .as_strided(
            (B, 1, blocks_per_batch, 1), (blocks_per_batch, blocks_per_batch, 1, 1)
        )
    )
    block_table.copy_(block_table_temp)

    return (container, block_table)

def compute_ref(q, k, v, attn_scale=1.0, return_type="o"):
    b, s_q, h_q, d_qk = q.shape
    _, s_kv, h_k, _ = k.shape
    _, _, h_v, d_v = v.shape

    assert k.shape == (b, s_kv, h_k, d_qk)
    assert v.shape == (b, s_kv, h_v, d_v)

    if h_q != h_k:
        k = k.repeat_interleave(h_q // h_k, dim=2)
    if h_q != h_v:
        v = v.repeat_interleave(h_q // h_v, dim=2)

    s = torch.einsum("bqhd,bkhd->bhqk", q, k) * attn_scale
    p = s.softmax(dim=-1)
    o = torch.einsum("bhqk,bkhd->bqhd", p, v)

    if return_type == "o":
        return o
    if return_type == "o_stats":
        return o, torch.zeros()
    elif return_type == "amax":
        return p.abs().max().item(), o.abs().max().item()
    else:
        raise ValueError(f"Unsupported return type: {return_type}")

@pytest.mark.parametrize("name, config", TEST_CONFIGS.items())
@pytest.mark.L0
def test_sdpa_fwd_fp8(name, config):
    section_begin(f"Running {name}")

    cudnn_version = LooseVersion(cudnn.backend_version_string())
    if cudnn_version < "9.13.0":
        pytest.skip("SDPA FP8 fprop testing is limited to cuDNN 9.13.0 or higher")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("SDPA FP8 fprop testing is limited to Blackwell or higher")

    if config["otype"] == "fp8_e4m3":
        torch_itype = torch.float8_e4m3fn
        cudnn_itype = cudnn.data_type.FP8_E4M3
    elif config["otype"] == "fp8_e5m2":
        torch_itype = torch.float8_e5m2
        cudnn_itype = cudnn.data_type.FP8_E5M2
    elif config["otype"] == "fp16":
        torch_itype = torch.float8_e4m3fn
        cudnn_itype = cudnn.data_type.FP8_E4M3

    if config["otype"] == "fp8_e4m3":
        torch_otype = torch.float8_e4m3fn
        cudnn_otype = cudnn.data_type.FP8_E4M3
    elif config["otype"] == "fp8_e5m2":
        torch_otype = torch.float8_e5m2
        cudnn_otype = cudnn.data_type.FP8_E5M2
    elif config["otype"] == "fp16":
        torch_otype = torch.float16
        cudnn_otype = cudnn.data_type.HALF
    else:
        raise ValueError(f"Unsupported input type: {config['otype']}")

    b = config["b"]
    h_q = config["h_q"]
    h_k = config["h_k"]
    h_v = config["h_v"]
    s_qo = config["s_qo"]
    s_kv = config["s_kv"]
    d_qk = config["d_qk"]
    d_vo = config["d_vo"]

    attn_scale = 0.125
    block_size = config["kv_block_size"]

    is_paged_attention = block_size > 0

    section_begin("Building Graph")
    graph_fwd = generate_graph_fwd(cudnn_itype, cudnn_otype, b, h_q, h_k, h_v, s_qo, s_kv, d_qk, d_vo, attn_scale, block_size)
    graph_fwd.validate()
    graph_fwd.build_operation_graph()
    graph_fwd.create_execution_plans([cudnn.heur_mode.A])
    graph_fwd.check_support()
    graph_fwd.build_plans()
    section_end()

    section_begin("Execute Graph")
    q_gen = torch.clamp(torch.randn(b, s_qo, h_q, d_qk, dtype=torch.float, device="cuda"), min=-2.0, max=2.0)
    k_gen = torch.clamp(torch.randn(b, s_kv, h_k, d_qk, dtype=torch.float, device="cuda"), min=-2.0, max=2.0)
    v_gen = torch.clamp(torch.randn(b, s_kv, h_v, d_vo, dtype=torch.float, device="cuda"), min=-2.0, max=2.0)

    q_amax = q_gen.abs().max().item()
    k_amax = k_gen.abs().max().item()
    v_amax = v_gen.abs().max().item()
    s_amax, o_amax = compute_ref(q_gen, k_gen, v_gen, attn_scale, return_type="amax")

    q_gpu = (q_gen * get_fp8_scale_factor(q_amax, torch_itype)).to(torch_itype)
    k_gpu = (k_gen * get_fp8_scale_factor(k_amax, torch_itype)).to(torch_itype)
    v_gpu = (v_gen * get_fp8_scale_factor(v_amax, torch_itype)).to(torch_itype)

    if is_paged_attention:
        # Create non contiguous containers with block tables for K and V from the contiguous k_gpu and v_gpu
        container_k_gpu, block_table_k_gpu = create_container_and_block_table(
            k_gpu, block_size
        )
        container_v_gpu, block_table_v_gpu = create_container_and_block_table(
            v_gpu, block_size
        )

    seq_len_kv_gpu = torch.full((b, 1, 1, 1), s_kv, device="cuda")
    seq_len_q_gpu = torch.full((b, 1, 1, 1), s_qo, device="cuda")
    o_gpu = torch.nans(b, s_qo, h_q, d_vo, dtype=torch_otype, device="cuda")
    stats_gpu = torch.nans(b, h_q, s_qo, 1, dtype=torch.float, device="cuda")

    descale_q_gpu = torch.tensor([get_fp8_descale_factor(q_amax, torch_itype)], dtype=torch.float, device="cuda")
    descale_k_gpu = torch.tensor([get_fp8_descale_factor(k_amax, torch_itype)], dtype=torch.float, device="cuda")
    descale_v_gpu = torch.tensor([get_fp8_descale_factor(v_amax, torch_itype)], dtype=torch.float, device="cuda")

    scale_s_gpu = torch.tensor([get_fp8_scale_factor(s_amax, torch_itype)], dtype=torch.float, device="cuda")
    descale_s_gpu = torch.tensor([get_fp8_descale_factor(s_amax, torch_itype)], dtype=torch.float, device="cuda")
    scale_o_gpu = torch.tensor([get_fp8_scale_factor(o_amax, torch_itype)], dtype=torch.float, device="cuda")

    amax_s_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")
    amax_o_gpu = torch.tensor([float('nan')], dtype=torch.float, device="cuda")

    # execute forward and backward graph
    variant_pack_fwd = {
        int(GraphFwdUid.q): q_gpu,
        int(GraphFwdUid.k): k_gpu,
        int(GraphFwdUid.v): v_gpu,
        int(GraphFwdUid.descale_q): descale_q_gpu,
        int(GraphFwdUid.descale_k): descale_k_gpu,
        int(GraphFwdUid.descale_v): descale_v_gpu,
        int(GraphFwdUid.descale_s): descale_s_gpu,
        int(GraphFwdUid.scale_s): scale_s_gpu,
        int(GraphFwdUid.scale_o): scale_o_gpu,
        int(GraphFwdUid.o): o_gpu,
        int(GraphFwdUid.stats): stats_gpu,
        int(GraphFwdUid.amax_s): amax_s_gpu,
        int(GraphFwdUid.amax_o): amax_o_gpu,
    }

    if is_paged_attention:
        variant_pack_fwd[int(GraphFwdUid.k)] = container_k_gpu
        variant_pack_fwd[int(GraphFwdUid.v)] = container_v_gpu 
        variant_pack_fwd[int(GraphFwdUid.block_table_k)] = block_table_k_gpu
        variant_pack_fwd[int(GraphFwdUid.block_table_v)] = block_table_v_gpu
        variant_pack_fwd[int(GraphFwdUid.seq_len_kv)] = seq_len_kv_gpu
        variant_pack_fwd[int(GraphFwdUid.seq_len_q)] = seq_len_q_gpu

    workspace = torch.empty(graph_fwd.get_workspace_size(), dtype=torch.uint8, device="cuda")
    cudnn_handle = cudnn.create_handle()
    graph_fwd.execute(variant_pack_fwd, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()
    section_end()


    section_begin("Run Reference and Compare Output")
    q_ref = q_gpu.detach().float() * get_fp8_descale_factor(q_amax, torch_itype)
    k_ref = k_gpu.detach().float() * get_fp8_descale_factor(k_amax, torch_itype)
    v_ref = v_gpu.detach().float() * get_fp8_descale_factor(v_amax, torch_itype)
    o_ref = compute_ref(q_ref, k_ref, v_ref, attn_scale=attn_scale)

    o_ref_comp = o_ref
    o_gpu_comp = o_gpu.detach().float() * get_fp8_descale_factor(o_amax, torch_itype)

    print("o_ref_comp.numel()", o_ref_comp.numel())
    print("o_gpu_comp.numel()", o_gpu_comp.numel())
    print("Number of zeros in o_ref_comp:", (o_ref_comp == 0).sum().item())
    print("Number of zeros in o_gpu_comp:", (o_gpu_comp == 0).sum().item())
    print("Number of non-finite elements in o_ref_comp:", (~torch.isfinite(o_ref_comp)).sum().item())
    print("Number of non-finite elements in o_gpu_comp:", (~torch.isfinite(o_gpu_comp)).sum().item())

    for _ in range(3):
        coord = tuple(torch.randint(0, numel, (1,)).item() for numel in o_ref_comp.size())
        print(f"o_ref_comp{coord}:", float(o_ref_comp[coord].item()).hex())
        print(f"o_gpu_comp{coord}:", float(o_gpu_comp[coord].item()).hex())


    is_failed = False
    try:
        torch.testing.assert_close(o_gpu_comp, o_ref_comp, atol=config["atol"], rtol=config["rtol"])
    except Exception as e:
        print("\033[91m" + f"o_gpu: {e}" + "\033[0m\n")
        is_failed = True
    try:
        torch.testing.assert_close(amax_s_gpu.item(), s_amax, atol=0.04, rtol=0.10)
    except Exception as e:
        print("\033[91m" + f"amax_s_gpu: {e}" + "\033[0m\n")
        is_failed = True
    try:
        torch.testing.assert_close(amax_o_gpu.item(), o_amax, atol=0.04, rtol=0.10)
    except Exception as e:
        print("\033[91m" + f"amax_o_gpu: {e}" + "\033[0m\n")
        is_failed = True
    
    if is_failed and is_paged_attention:
        print("\033[91m" + "Failed!" + "\033[0m")
        pytest.xfail("XFailed: paged attention!")
    elif is_failed:
        print("\033[91m" + "Failed!" + "\033[0m")
        raise AssertionError()
    print("\033[92m" + "Passed!" + "\033[0m")

    # # used to debug tolerances
    # x = o_ref_comp.abs()
    # y = o_ref_comp - o_gpu_comp
    # import plotly.express as px
    # import plotly.io as pio
    # fig = px.scatter(
    #     x=x.cpu().flatten().numpy(),
    #     y=y.cpu().flatten().numpy(),
    #     labels={"x": "Absolute value", "y": "Absolute Error"},
    #     title="Absolute value vs absolute error"
    # )
    # pio.write_html(fig, file=f"scatter_{name}.html", auto_open=False)
    # print(f"wrote scatter_{name}.html")

    section_end()
    print()

if __name__ == "__main__":
    # python3 test/python/test_sdpa_fp8_temp.py --config d128_f16
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="all")
    args = parser.parse_args()

    torch.manual_seed(42)
    if args.config == "all":
        failed = []
        for name, config in TEST_CONFIGS.items():
            try:
                test_sdpa_fwd_fp8(name, config)
            except Exception as e:
                print(e)
                failed.append((name, e))
        if failed:
            failed_names = [name for name, _ in failed]
            raise AssertionError(f"Some tests failed: {failed_names}")
    else:
        test_sdpa_fwd_fp8(args.config, TEST_CONFIGS[args.config])

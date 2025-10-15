import torch

import pytest
from test_utils import torch_fork_set_rng

from oss.nsa.nsa_fixtures import test_config
from oss.nsa.nsa_utils import (
    _env_supported,
    init_input_tensors,
    allocate_output_tensors,
)
from oss.nsa.nsa_reference import check_ref_nsa_compression_attention
from cuda.bindings import driver as cuda


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_nsa_compression_compile_execute(test_config):
    if not _env_supported(target_major=10.0):
        pytest.skip("Environment not supported")
    from cudnn import NSA

    Q, K, V, _, _, _, seq_offsets, _ = init_input_tensors(test_config)
    O, LSE, _ = allocate_output_tensors(test_config)

    # Compression Kernel requires LSE to be strided in this order
    if test_config["layout"] == "bshd":
        LSE = LSE.contiguous()
    elif test_config["layout"] == "thd":
        LSE = LSE.permute(2, 1, 0).contiguous().permute(2, 1, 0)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    comp_attn = NSA.CompressionAttention(
        sample_q=Q,
        sample_k=K,
        sample_v=V,
        sample_o=O,
        sample_lse=LSE,
        sample_cum_seqlen_q=seq_offsets,
        sample_cum_seqlen_k=seq_offsets,
        mma_tiler_mn=(128, 128),
        qk_acc_dtype=torch.float32,
        pv_acc_dtype=torch.float32,
        is_persistent=False,
        scale_q=1.0,
        scale_k=1.0,
        scale_v=1.0,
        inv_scale_o=1.0,
        scale_softmax=test_config["softmax_scale"],
    )

    assert comp_attn.check_support() is True
    comp_attn.compile(current_stream=stream)
    comp_attn.execute(
        q_tensor=Q,
        k_tensor=K,
        v_tensor=V,
        o_tensor=O,
        lse_tensor=LSE,
        cum_seqlen_q_tensor=seq_offsets,
        cum_seqlen_k_tensor=seq_offsets,
        scale_softmax=test_config["softmax_scale"],
        current_stream=stream,
    )

    check_ref_nsa_compression_attention(
        Q,
        K,
        V,
        O,
        LSE,
        scale_output=1.0,
        atol=2e-3,
        rtol=2e-3,
        test_config=test_config,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_nsa_compression_wrapper(test_config):
    if not _env_supported(target_major=9.0):
        pytest.skip("Environment not supported")
    from cudnn import NSA

    Q, K, V, _, _, _, seq_offsets, _ = init_input_tensors(test_config)

    O, LSE = NSA.compression_attention_wrapper(
        q_tensor=Q,
        k_tensor=K,
        v_tensor=V,
        cum_seqlen_q_tensor=seq_offsets,
        cum_seqlen_k_tensor=seq_offsets,
        enable_lse=True,
        mma_tiler_mn=(128, 128),
        o_dtype=test_config["dtype"],
        qk_acc_dtype=torch.float32,
        pv_acc_dtype=torch.float32,
        is_persistent=False,
        scale_q=1.0,
        scale_k=1.0,
        scale_v=1.0,
        inv_scale_o=1.0,
        scale_softmax=test_config["softmax_scale"],
    )

    check_ref_nsa_compression_attention(
        Q,
        K,
        V,
        O,
        LSE,
        scale_output=1.0,
        atol=2e-3,
        rtol=2e-3,
        test_config=test_config,
    )

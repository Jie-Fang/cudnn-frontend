import torch
import cudnn

import pytest
from test_utils import torch_fork_set_rng

from fe_api.nsa.nsa_utils import (
    nsa_init,
    allocate_input_tensors,
    allocate_output_tensors,
    with_nsa_swa_params,
    generate_ragged_offset,
)

from fe_api.nsa.nsa_reference import check_ref_nsa_swa


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_nsa_swa_params
def test_nsa_swa_compile_execute(
    layout,
    dtype,
    acc_dtype,
    window_size,
    scale_softmax,
    request,
):
    try:
        from cudnn import NSA
    except ImportError as e:
        pytest.skip(
            "Environment not supported: cudnn optional dependencies not installed"
        )
    cfg = nsa_init(
        request=request,
        layout=layout,
        dtype=dtype,
        acc_dtype=acc_dtype,
        scale_softmax=scale_softmax,
        window_size=window_size,
    )

    (
        Q,
        K,
        V,
        _,
        actual_s_q,
        actual_s_kv,
        cum_seqlen_q,
        cum_seqlen_kv,
        max_s_q,
        max_s_kv,
    ) = allocate_input_tensors(cfg)
    (
        q_ragged_offset_tensor,
        k_ragged_offset_tensor,
        v_ragged_offset_tensor,
        o_ragged_offset_tensor,
        stats_ragged_offset_tensor,
    ) = generate_ragged_offset(cfg)
    O, Stats, _, _, _ = allocate_output_tensors(cfg)
    cudnn_handle = cudnn.create_handle()

    swa = NSA.SlidingWindowAttention(
        sample_q=Q,
        sample_k=K,
        sample_v=V,
        sample_o=O,
        sample_stats=Stats,
        sample_seq_len_q=actual_s_q,
        sample_seq_len_kv=actual_s_kv,
        sample_q_ragged_offset=q_ragged_offset_tensor,
        sample_k_ragged_offset=k_ragged_offset_tensor,
        sample_v_ragged_offset=v_ragged_offset_tensor,
        sample_o_ragged_offset=o_ragged_offset_tensor,
        sample_stats_ragged_offset=stats_ragged_offset_tensor,
        max_seq_len_q=max_s_q,
        max_seq_len_kv=max_s_kv,
        left_bound=cfg["window_size"],
        right_bound=0,
        is_infer=False,
        attn_scale=cfg["scale_softmax"],
        intermediate_data_type=cfg["acc_dtype"],
        compute_data_type=cfg["acc_dtype"],
        cudnn_handle=cudnn_handle,
    )

    assert swa.check_support() is True
    swa.compile()
    swa.execute(
        q_tensor=Q,
        k_tensor=K,
        v_tensor=V,
        seq_len_q_tensor=actual_s_q,
        seq_len_kv_tensor=actual_s_kv,
        q_ragged_offset_tensor=q_ragged_offset_tensor,
        k_ragged_offset_tensor=k_ragged_offset_tensor,
        v_ragged_offset_tensor=v_ragged_offset_tensor,
        o_ragged_offset_tensor=o_ragged_offset_tensor,
        stats_ragged_offset_tensor=stats_ragged_offset_tensor,
        o_tensor=O,
        stats_tensor=Stats,
    )

    check_ref_nsa_swa(
        Q,
        K,
        V,
        O,
        Stats,
        actual_s_q,
        actual_s_kv,
        max_s_q,
        max_s_kv,
        cfg,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
@with_nsa_swa_params
def test_nsa_swa_wrapper(
    layout,
    dtype,
    acc_dtype,
    window_size,
    scale_softmax,
    request,
):
    try:
        from cudnn import NSA
    except ImportError as e:
        pytest.skip(
            "Environment not supported: cudnn optional dependencies not installed"
        )
    cfg = nsa_init(
        request=request,
        layout=layout,
        dtype=dtype,
        acc_dtype=acc_dtype,
        scale_softmax=scale_softmax,
        window_size=window_size,
    )

    (
        Q,
        K,
        V,
        _,
        actual_s_q,
        actual_s_kv,
        cum_seqlen_q,
        cum_seqlen_kv,
        max_s_q,
        max_s_kv,
    ) = allocate_input_tensors(cfg)
    (
        q_ragged_offset_tensor,
        k_ragged_offset_tensor,
        v_ragged_offset_tensor,
        o_ragged_offset_tensor,
        stats_ragged_offset_tensor,
    ) = generate_ragged_offset(cfg)
    cudnn_handle = cudnn.create_handle()

    O, Stats = NSA.sliding_window_attention_wrapper(
        q_tensor=Q,
        k_tensor=K,
        v_tensor=V,
        seq_len_q_tensor=actual_s_q,
        seq_len_kv_tensor=actual_s_kv,
        q_ragged_offset_tensor=q_ragged_offset_tensor,
        k_ragged_offset_tensor=k_ragged_offset_tensor,
        v_ragged_offset_tensor=v_ragged_offset_tensor,
        o_ragged_offset_tensor=o_ragged_offset_tensor,
        stats_ragged_offset_tensor=stats_ragged_offset_tensor,
        left_bound=cfg["window_size"],
        right_bound=0,
        is_infer=False,
        attn_scale=cfg["scale_softmax"],
        intermediate_data_type=cfg["acc_dtype"],
        compute_data_type=cfg["acc_dtype"],
        cudnn_handle=cudnn_handle,
    )

    check_ref_nsa_swa(
        Q,
        K,
        V,
        O,
        Stats,
        actual_s_q,
        actual_s_kv,
        max_s_q,
        max_s_kv,
        cfg,
    )

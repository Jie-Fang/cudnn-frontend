import torch
import cudnn

import pytest
from test_utils import torch_fork_set_rng

from oss.nsa.nsa_fixtures import test_config
from oss.nsa.nsa_utils import (
    _env_supported,
    init_input_tensors,
    allocate_output_tensors,
    _generate_ragged_offset,
)

from oss.nsa.nsa_reference import check_ref_nsa_swa


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_nsa_swa_compile_execute(test_config):
    if not _env_supported():
        pytest.skip("Environment not supported")
    from cudnn import NSA

    Q, K, V, _, _, actual_s_q, _, max_length = init_input_tensors(test_config)
    (
        q_ragged_offset_tensor,
        k_ragged_offset_tensor,
        v_ragged_offset_tensor,
        o_ragged_offset_tensor,
        stats_ragged_offset_tensor,
    ) = _generate_ragged_offset(test_config)

    O, Stats, _ = allocate_output_tensors(test_config)
    cudnn_handle = cudnn.create_handle()

    swa = NSA.SlidingWindowAttention(
        sample_q=Q,
        sample_k=K,
        sample_v=V,
        sample_o=O,
        sample_stats=Stats,
        sample_seq_len_q=actual_s_q,
        sample_seq_len_kv=actual_s_q,
        sample_q_ragged_offset=q_ragged_offset_tensor,
        sample_k_ragged_offset=k_ragged_offset_tensor,
        sample_v_ragged_offset=v_ragged_offset_tensor,
        sample_o_ragged_offset=o_ragged_offset_tensor,
        sample_stats_ragged_offset=stats_ragged_offset_tensor,
        max_seq_len_q=max_length,
        max_seq_len_kv=max_length,
        left_bound=test_config["window_size"],
        right_bound=0,
        is_infer=False,
        attn_scale=test_config["softmax_scale"],
        intermediate_data_type=test_config["acc_dtype"],
        compute_data_type=test_config["acc_dtype"],
        cudnn_handle=cudnn_handle,
    )

    assert swa.check_support() is True
    swa.compile()
    swa.execute(
        q_tensor=Q,
        k_tensor=K,
        v_tensor=V,
        seq_len_q_tensor=actual_s_q,
        seq_len_kv_tensor=actual_s_q,
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
        actual_s_q,
        max_length,
        max_length,
        test_config,
    )


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_nsa_swa_wrapper(test_config):
    if not _env_supported():
        pytest.skip("Environment not supported")
    from cudnn import NSA

    Q, K, V, _, _, actual_s_q, _, max_length = init_input_tensors(test_config)
    (
        q_ragged_offset_tensor,
        k_ragged_offset_tensor,
        v_ragged_offset_tensor,
        o_ragged_offset_tensor,
        stats_ragged_offset_tensor,
    ) = _generate_ragged_offset(test_config)
    cudnn_handle = cudnn.create_handle()

    O, Stats = NSA.sliding_window_attention_wrapper(
        q_tensor=Q,
        k_tensor=K,
        v_tensor=V,
        seq_len_q_tensor=actual_s_q,
        seq_len_kv_tensor=actual_s_q,
        q_ragged_offset_tensor=q_ragged_offset_tensor,
        k_ragged_offset_tensor=k_ragged_offset_tensor,
        v_ragged_offset_tensor=v_ragged_offset_tensor,
        o_ragged_offset_tensor=o_ragged_offset_tensor,
        stats_ragged_offset_tensor=stats_ragged_offset_tensor,
        left_bound=test_config["window_size"],
        right_bound=0,
        is_infer=False,
        attn_scale=test_config["softmax_scale"],
        intermediate_data_type=test_config["acc_dtype"],
        compute_data_type=test_config["acc_dtype"],
        cudnn_handle=cudnn_handle,
    )

    check_ref_nsa_swa(
        Q,
        K,
        V,
        O,
        Stats,
        actual_s_q,
        actual_s_q,
        max_length,
        max_length,
        test_config,
    )

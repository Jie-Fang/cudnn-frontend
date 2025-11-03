import torch

import pytest
from test_utils import torch_fork_set_rng

from fe_api.nsa.nsa_fixtures import test_config
from fe_api.nsa.nsa_utils import (
    _env_supported,
    init_input_tensors,
    allocate_output_tensors,
)
from fe_api.nsa.nsa_reference import check_ref_nsa_selection_attention

"""
SelectionAttention API with explicitset_params, compile, and execute paths. 
Use this method when running one static configuration for each FmhaCute object.
"""


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_nsa_selection_compile_execute(test_config):
    if not _env_supported():
        pytest.skip("Environment not supported")
    from cudnn import NSA

    assert (
        test_config["layout"] == "thd"
    ), "bshd layout for selection attention not yet implemented"

    Q, K, V, block_counts, block_indices, _, seq_offsets, max_length = (
        init_input_tensors(test_config)
    )

    O, L, M = allocate_output_tensors(test_config)

    selection_attention = NSA.SelectionAttention(
        sample_q=Q,
        sample_k=K,
        sample_v=V,
        sample_o=O,
        sample_l=L,
        sample_m=M,
        sample_block_indices=block_indices,
        sample_block_counts=block_counts,
        sample_seq_offsets=seq_offsets,
        acc_dtype=test_config["acc_dtype"],
        max_s=max_length,
        block_size=test_config["block_size"],
        scale_softmax=test_config["softmax_scale"],
    )

    assert selection_attention.check_support() is True

    selection_attention.compile()

    selection_attention.execute(
        q_tensor=Q,
        k_tensor=K,
        v_tensor=V,
        o_tensor=O,
        l_tensor=L,
        m_tensor=M,
        block_indices_tensor=block_indices,
        block_counts_tensor=block_counts,
        seq_offsets_tensor=seq_offsets,
        scale_softmax=test_config["softmax_scale"],
    )

    check_ref_nsa_selection_attention(
        Q,
        K,
        V,
        O,
        L,
        M,
        block_indices,
        block_counts,
        test_config,
    )


"""
SelectionAttention API with selection_attention_wrapper:
Use the wrapper to directly call SelectionAttention without explicit setup and compilation.
"""


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_nsa_selection_wrapper(test_config):
    if not _env_supported():
        pytest.skip("Environment not supported")
    from cudnn import NSA

    assert (
        test_config["layout"] == "thd"
    ), "bshd layout for selection attention not yet implemented"

    Q, K, V, block_counts, block_indices, _, seq_offsets, max_length = (
        init_input_tensors(test_config)
    )

    O, L, M = NSA.selection_attention_wrapper(
        q_tensor=Q,
        k_tensor=K,
        v_tensor=V,
        block_indices_tensor=block_indices,
        block_counts_tensor=block_counts,
        seq_offsets_tensor=seq_offsets,
        block_size=test_config["block_size"],
        scale_softmax=test_config["softmax_scale"],
        acc_dtype=test_config["acc_dtype"],
    )

    check_ref_nsa_selection_attention(
        Q,
        K,
        V,
        O,
        L,
        M,
        block_indices,
        block_counts,
        test_config,
    )

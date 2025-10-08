import torch

import pytest

pytest_plugins = ("oss.test_gemm_amax_utils",)

from test_utils import torch_fork_set_rng
from cuda.bindings import driver as cuda

"""
GemmAmax API with explicit set_params, compile, and execute paths. 
Use this method when running one static configuration for each GemmAmax object.
"""


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_gemm_amax_compile_execute(test_config):
    try:
        from cudnn import GemmAmax
        from oss.test_gemm_amax_utils import (
            allocate_input_tensors,
            allocate_output_tensors,
            check_ref_gemm_amax,
        )
    except ImportError as e:
        pytest.skip(
            "Environment not supported: cudnn optional dependencies not installed"
        )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a_torch, a_ref, b_torch, b_ref, sfa_torch, sfa_ref, sfb_torch, sfb_ref = (
        allocate_input_tensors(test_config)
    )
    c_torch, amax_torch = allocate_output_tensors(test_config)

    gemm = GemmAmax(
        sample_a=a_torch,
        sample_b=b_torch,
        sample_sfa=sfa_torch,
        sample_sfb=sfb_torch,
        sample_c=c_torch,
        sample_amax=amax_torch,
        acc_dtype=test_config["acc_dtype"],
        mma_tiler_mn=test_config["mma_tiler_mn"],
        cluster_shape_mn=test_config["cluster_shape_mn"],
        sf_vec_size=test_config["sf_vec_size"],
    )
    assert gemm.check_support(), "Unsupported testcase"
    gemm.compile(current_stream=stream)
    gemm.execute(
        a_tensor=a_torch,
        b_tensor=b_torch,
        sfa_tensor=sfa_torch,
        sfb_tensor=sfb_torch,
        c_tensor=c_torch,
        amax_tensor=amax_torch,
        current_stream=stream,
    )

    check_ref_gemm_amax(
        a_ref, b_ref, sfa_ref, sfb_ref, c_torch, amax_torch, test_config
    )


"""
GemmAmax API with gemm_amax_wrapper:
Use the wrapper to directly call GemmAmax without explicit setup and compilation.
"""


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_gemm_amax_wrapper(test_config):
    try:
        from cudnn import gemm_amax_wrapper
        from oss.test_gemm_amax_utils import (
            allocate_input_tensors,
            allocate_output_tensors,
            check_ref_gemm_amax,
        )
    except ImportError as e:
        pytest.skip(
            "Environment not supported: cudnn optional dependencies not installed"
        )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a_torch, a_ref, b_torch, b_ref, sfa_torch, sfa_ref, sfb_torch, sfb_ref = (
        allocate_input_tensors(test_config)
    )

    c_torch, amax_torch = gemm_amax_wrapper(
        a_tensor=a_torch,
        b_tensor=b_torch,
        sfa_tensor=sfa_torch,
        sfb_tensor=sfb_torch,
        c_major=test_config["c_major"],
        c_dtype=test_config["c_dtype"],
        acc_dtype=test_config["acc_dtype"],
        mma_tiler_mn=test_config["mma_tiler_mn"],
        cluster_shape_mn=test_config["cluster_shape_mn"],
        sf_vec_size=test_config["sf_vec_size"],
        stream=stream,
    )

    check_ref_gemm_amax(
        a_ref, b_ref, sfa_ref, sfb_ref, c_torch, amax_torch, test_config
    )

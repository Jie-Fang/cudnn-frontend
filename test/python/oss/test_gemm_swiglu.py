import torch

import pytest
from test_utils import torch_fork_set_rng
from oss.test_gemm_swiglu_utils import (
    allocate_input_tensors,
    allocate_output_tensors,
    check_ref_gemm_swiglu,
    test_config,
)
from cuda.bindings import driver as cuda

"""
GemmSwiglu API with explicit set_params, compile, and execute paths. 
Use this method when running one static configuration for each GemmSwiglu object.
"""


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_gemm_swiglu(test_config):
    try:
        from cudnn import GemmSwiglu
    except ImportError:
        pytest.skip(
            "Environment not supported: cudnn optional dependencies not installed"
        )
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip(
            f"Environment not supported: requires compute capability >= 10, found {major}"
        )

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    a_torch, b_torch = allocate_input_tensors(test_config)
    c_torch, glu_torch = allocate_output_tensors(test_config)

    gemm_swiglu = GemmSwiglu(
        sample_a=a_torch,
        sample_b=b_torch,
        sample_c=c_torch,
        sample_glu=glu_torch,
        alpha=test_config["alpha"],
        acc_dtype=test_config["acc_dtype"],
        use_2cta_instrs=test_config["use_2cta_instrs"],
        mma_tiler_mn=test_config["mma_tiler_mn"],
        cluster_shape_mn=test_config["cluster_shape_mn"],
    )
    assert gemm_swiglu.check_support(), "Unsupported testcase"
    gemm_swiglu.compile(current_stream=stream)
    gemm_swiglu.execute(
        a_tensor=a_torch,
        b_tensor=b_torch,
        c_tensor=c_torch,
        glu_tensor=glu_torch,
        alpha=test_config["alpha"],
        current_stream=stream,
    )

    check_ref_gemm_swiglu(a_torch, b_torch, c_torch, glu_torch, config=test_config)


"""
GemmSwiglu API with gemm_swiglu_wrapper:
Use the wrapper to directly call GemmSwiglu without explicit setup and compilation.
"""


@pytest.mark.L0
@torch_fork_set_rng(seed=0)
def test_gemm_swiglu_wrapper(test_config):
    try:
        from cudnn import gemm_swiglu_wrapper
    except ImportError:
        pytest.skip(
            "Environment not supported: cudnn optional dependencies not installed"
        )
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        pytest.skip(
            f"Environment not supported: requires compute capability >= 10, found {major}"
        )

    a_torch, b_torch = allocate_input_tensors(test_config)

    c_torch, glu_torch = gemm_swiglu_wrapper(
        a_tensor=a_torch,
        b_tensor=b_torch,
        alpha=test_config["alpha"],
        c_major=test_config["c_major"],
        c_dtype=test_config["c_dtype"],
        glu_dtype=test_config["glu_dtype"],
        acc_dtype=test_config["acc_dtype"],
        use_2cta_instrs=test_config["use_2cta_instrs"],
        mma_tiler_mn=test_config["mma_tiler_mn"],
        cluster_shape_mn=test_config["cluster_shape_mn"],
        stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )

    check_ref_gemm_swiglu(a_torch, b_torch, c_torch, glu_torch, config=test_config)

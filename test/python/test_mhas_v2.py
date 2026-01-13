"""
This script tests cuDNN front-end attention.
The recommended way to run tests:
> pytest -vv -s -rA test_mhas_v2.py
"""

import cudnn
import pytest
import random
import torch
import sys
from datetime import datetime
from dataclasses import asdict

from sdpa.random_config import (
    ExecConfig,
    generate_test_seeds,
    RandomizationContext,
    RandomBatchSize,
    RandomBlockSize,
    RandomSequenceLength,
    RandomHiddenDimSize,
    RandomHeadGenerator,
    RandomChoice,
    SlidingWindowMaskGenerator,
)
from sdpa.fp16 import exec_sdpa
from sdpa.fp8 import exec_sdpa_fp8
from sdpa.blocked import fetch_blocked_tests, show_blocked_tests

# fmt: off

if __name__ == "__main__":
    print("This is pytest script.")
    sys.exit(0)

class SDPATestConfig:
    __slots__ = ['gpu_arch', 'gpu_info', 'cudnn_ver', 'blocked_tests', 'implementation', 'cfg']

    def __init__(self, *, gpu_arch, gpu_info, cudnn_ver, blocked_tests, implementation):
        assert type(gpu_arch) == type(gpu_info) == type(cudnn_ver) == str, "expecting strings as arguments"
        assert isinstance(blocked_tests, list), "argument 'blocked_tests' must be list"

        # Initialize all attributes to None.
        for k in self.__slots__:
            setattr(self, k, None)

        self.gpu_arch      = gpu_arch
        self.gpu_info      = gpu_info
        self.cudnn_ver     = cudnn_ver
        self.blocked_tests = blocked_tests

        self.implementation = implementation

        self.cfg = ExecConfig()


    def showConfig(self, test_no, request, reg_run=True):
        if request.config.option.dryrun == 0 or request.config.option.dryrun == 1:
            if request.config.option.dryrun == 0:
                print("\n" + "=" * 90)
            else:
                print("\n" + "=" * 40 + "Dry-RUN" + "=" * 40)
            print(f"#### Test #{test_no[0]} of {test_no[1]} at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n")
            print(f"test_name        = {request.node.name}")
            # print(f"geom_seed        = {self.geom_seed}")
            # print(f"data_seed        = {self.data_seed}")
            print(f"platform_info    = {self.gpu_arch} ({self.gpu_info}), cudnn_ver={self.cudnn_ver}")
            print(f"rng_data_seed    = {self.cfg.rng_data_seed}")
            # print(f"head_group       = {self.cfg.head_group}")
            # print(f"layout           = {self.in_layout}->{self.out_layout}")
            print(f"basic_dims       = [b={self.cfg.batches}, h_q={self.cfg.h_q}, h_k={self.cfg.h_k}, h_v={self.cfg.h_v}, d_qk={self.cfg.d_qk}, d_v={self.cfg.d_v}, s_q={self.cfg.s_q}, s_kv={self.cfg.s_kv}]")
            print(f"shape_q(b,h,s,d) = {self.cfg.shape_q}, strides={self.cfg.stride_q}, elems={self.cfg.elems_q}")
            print(f"shape_k(b,h,s,d) = {self.cfg.shape_k}, strides={self.cfg.stride_k}, elems={self.cfg.elems_k}")
            print(f"shape_v(b,h,s,d) = {self.cfg.shape_v}, strides={self.cfg.stride_v}, elems={self.cfg.elems_v}")
            print(f"shape_o(b,h,s,d) = {self.cfg.shape_o}, strides={self.cfg.stride_o}, elems={self.cfg.elems_o}")
            
            print(f"is_infer         = {self.cfg.is_infer}")
            print(f"is_padding       = {self.cfg.is_padding} ({'ragged' if self.cfg.is_ragged else 'no ragged'})")
            print(f"is_alibi         = {self.cfg.is_alibi}")
            print(f"is_paged         = {self.cfg.is_paged} (block_size={self.cfg.block_size})")
            print(f"is_bias          = {self.cfg.is_bias}")
            print(f"is_block_mask    = {self.cfg.is_block_mask}")
            print(f"is_dropout       = {self.cfg.is_dropout}")
            if self.cfg.is_infer == False:
                print(f"is_determin      = {self.cfg.is_determin}")
            print(f"diag_align       = {self.cfg.diag_align}")
            print(f"left_bound       = {self.cfg.left_bound}", '(NO BOUND)' if self.cfg.left_bound is None else '')
            print(f"right_bound      = {self.cfg.right_bound}", '(NO BOUND)' if self.cfg.right_bound is None else '')
            # print(f"seq_len_q        = {self.seq_len_q}")
            # print(f"seq_len_kv       = {self.seq_len_kv}")
            print(f"data_type        = {self.cfg.data_type}")
            if self.cfg.output_type and self.cfg.output_type != self.cfg.data_type:
                print(f"output_type      = {self.cfg.output_type}")
            print(f"implementation   = {self.cfg.implementation.name}")
            if reg_run:
                # Convert enums to integers and handle torch dtypes for proper serialization
                cfg_dict = asdict(self.cfg)
                # Convert enum values to integers
                if cfg_dict.get('diag_align') is not None:
                    cfg_dict['diag_align'] = cfg_dict['diag_align'].value
                if cfg_dict.get('implementation') is not None:
                    cfg_dict['implementation'] = cfg_dict['implementation'].name
                # Convert torch dtype to string
                if cfg_dict.get('data_type') is not None:
                    cfg_dict['data_type'] = str(cfg_dict['data_type'])
                print(f"repro_cmd        = pytest -vv -s -rA {request.module.__file__}::test_repro --repro \"{repr(cfg_dict)}\"")
        elif request.config.option.dryrun == 2:
            print(f"\npytest -vv -s -rA {request.module.__file__}::{request.node.name} --geom_seed {self.geom_seed} --data_seed {self.data_seed}")
        elif request.config.option.dryrun == 3:
            print(f"repro_cmd        = pytest -vv -s -rA {request.module.__file__}::{request.node.name} --geom_seed {self.geom_seed} --data_seed {self.data_seed}")

        else:
            assert False, "wrong --dryrun command line option"

        # Make sure to flush everything out.
        print(" ", flush=True)


    def avoid_invalid_configs(self, avoid_invalid_configs):
        if avoid_invalid_configs == avoid_invalid_configs.ALWAYS:
            # LIMIT: always is_determin=True in inference.
            if self.is_infer:
                self.is_determin = True

            # LIMIT: Paged attention only in inference.
            if not self.is_infer:
                self.is_paged = False

            # LIMIT: Paged caches can only be used in combination with padding mask (variable sequence length).
            if self.is_paged and not self.is_padding:
                self.is_paged = False

            # LIMIT: Paged caches cannot be used with ragged offsets (packed variable sequence lengths).
            if self.is_paged and self.is_ragged:
                self.is_paged = False
        
            # LIMIT: left and right bounds are only supported with is_dropout=False, is_bias=False.
            if self.left_bound is not None and self.right_bound is not None:
                self.is_dropout = False
                self.is_bias = False

            # LIMIT: when alibi mask is used, diagonal_band_right_bound needs to be exactly 0.
            if self.is_alibi and self.right_bound != 0:
                self.is_alibi = False

            # LIMIT: bottom right causal mask is only supported with is_bias=False, is_alibi=False, is_dropout=False.
            if self.diag_align == self.diag_align.BOTTOM_RIGHT and (self.left_bound is not None or self.right_bound is not None):
                self.is_bias    = False
                self.is_alibi   = False
                self.is_dropout = False

            # LIMIT: Left or right bounds are only supported with is_dropout=False, is_bias=False.
            if self.left_bound is not None or self.right_bound is not None:
                self.is_dropout = False
                self.is_bias    = False

            # LIMIT: Left bound (a.k.a sliding window) does not support s_q > s_kv
            if self.left_bound is not None and self.s_q.val > self.s_kv.val:
                self.left_bound = None

            # LIMIT: Bottom right causal mask does not support s_q > s_kv. 
            if self.s_q.val > self.s_kv.val and self.diag_align == self.diag_align.BOTTOM_RIGHT and self.right_bound is not None:
                self.right_bound = None
            
            if not self.is_infer:
                self.is_block_mask = False


@pytest.fixture(scope="package")
def env_info(request):
    assert torch.cuda.is_available(), "no CUDA device"

    gpu_type = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()
    device   = torch.device('cuda:0')
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count

    gpu_arch     = f"SM_{gpu_type[0]}{gpu_type[1]}"
    gpu_info     = f"{sm_count} SM-s, {gpu_name}"
    cudnn_ver    = str(torch.backends.cudnn.version())

    blocked_tests = fetch_blocked_tests(gpu_arch, cudnn_ver)
    show_blocked_tests(blocked_tests, gpu_arch, cudnn_ver)

    return {"gpu_arch": gpu_arch, "gpu_info": gpu_info, "cudnn_ver": cudnn_ver, "blocked_tests": blocked_tests}

# These options are common to all test lists
data_type_options      = {torch.float16 : 1, torch.bfloat16 : 2}
diag_alignment_options = [cudnn.diagonal_alignment.TOP_LEFT, cudnn.diagonal_alignment.BOTTOM_RIGHT]
implementation_options = [cudnn.attention_implementation.AUTO, cudnn.attention_implementation.COMPOSITE, cudnn.attention_implementation.UNIFIED]
implementation_names   = ['cudnn.attention_implementation.AUTO', 'cudnn.attention_implementation.COMPOSITE', 'cudnn.attention_implementation.UNIFIED']

# # ==================================
# # L0 fprop tests
# # ==================================
@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_fwd_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=32, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_fwd_unified_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),  # Modified from non-unified test
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),  # Modified from non-unified test
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)
    test.cfg.implementation = getattr(cudnn.attention_implementation, request.config.getoption("--implementation") or "", cudnn.attention_implementation.UNIFIED)

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


# # ==================================
# # L0 bprop tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=256, rng_seed=844), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_bwd_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=8, max=16),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=192, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":5, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 4, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
        is_deterministic=RandomChoice({True : 3, False : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)

    test.cfg.is_infer = False
    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


# # ==================================
# # L0 fprop tests with s_q=1
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=111), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_sq1_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":100, "s_q=s_kv":1, "s_q=random":0}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 0, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=32, rng_seed=111), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_sq1_unified_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":100, "s_q=s_kv":1, "s_q=random":0}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),  # Modified from non-unified test
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 0, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)
    test.cfg.implementation = getattr(cudnn.attention_implementation, request.config.getoption("--implementation") or "", cudnn.attention_implementation.UNIFIED)

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


# # =====================================================
# # L0 lean attention, s_kv=513..2048
# # =====================================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=222), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_lean_attn_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1, s_kv_min=513, s_kv_max=2048, s_q_distribution={"s_q=1":100, "s_q=s_kv":0, "s_q=random":0}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=222), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_lean_attn_unified_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=32),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1, s_kv_min=513, s_kv_max=2048, s_q_distribution={"s_q=1":100, "s_q=s_kv":0, "s_q=random":0}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=32, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),  # Modified from non-unified test
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)
    test.cfg.implementation = getattr(cudnn.attention_implementation, request.config.getoption("--implementation") or "", cudnn.attention_implementation.UNIFIED)

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)

# # ==================================
# # L0 ragged tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_fwd_ragged_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 1, "padded" : 0, "full" : 0}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_fwd_ragged_unified_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),  # Modified from non-unified test
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),  # Modified from non-unified test
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 1, "padded" : 0, "full" : 0}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)
    test.cfg.implementation = getattr(cudnn.attention_implementation, request.config.getoption("--implementation") or "", cudnn.attention_implementation.UNIFIED)

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=256, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_bwd_ragged_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=8, max=16),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=192, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":5, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 1, "padded" : 0, "full" : 0}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
        is_deterministic=RandomChoice({True : 3, False : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)

    test.cfg.is_infer = False
    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


# # ==================================
# # L0 paged tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fwd_paged_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=64, s_kv_min=1, s_kv_max=512, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(64,64), (128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(causal=10, left_window_only=5, right_window_only=5, band_around_diag=10, no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 0}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
        block_size=RandomBlockSize(min=1, max=1024, with_high_probability=[1,32,128]),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)
        test.cfg.is_paged = True
        test.cfg.implementation=cudnn.attention_implementation.COMPOSITE  # FIXNOW

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fwd_paged_unified_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=64, s_kv_min=1, s_kv_max=512, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),  # Modified from non-unified test
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),  # Modified from non-unified test
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 1, "full" : 0}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
        block_size=RandomBlockSize(min=1, max=1024, with_high_probability=[1,32,128]),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)
        test.cfg.is_paged = True
    test.cfg.implementation = getattr(cudnn.attention_implementation, request.config.getoption("--implementation") or "", cudnn.attention_implementation.UNIFIED)

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=32, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_random_fwd_unified_block_mask_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    print(f"test: {test} hash {abs(hash(test_no))}")

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    # Create the randomization context within the test
    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=8, with_high_probability=[1,4]),
        s_q_s_kv = RandomSequenceLength(s_q_min=1, s_q_max=1024, s_kv_min=1, s_kv_max=1024, s_q_distribution={"s_q=1":0, "s_q=s_kv":5, "s_q=random":10}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=1, d_qk_max=128, d_v_min=1, d_v_max=128, head_dim_distribution={"d_qk=d_v":1, "d_qk=random":1}, with_high_probability=[(128,128), (192,128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float16 : 1, torch.bfloat16 : 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT : 1, cudnn.diagonal_alignment.BOTTOM_RIGHT : 0}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged" : 0, "padded" : 0, "full" : 1}),
        stats_layout=RandomChoice({"ragged" : 0, "full" : 0, "disabled" : 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)
        test.cfg.is_block_mask = True
    test.cfg.implementation = cudnn.attention_implementation.UNIFIED

    test.showConfig(test_no, request, reg_run=True)

    exec_sdpa(test.cfg, request, cudnn_handle)


# # ==================================
# # L0 FP8 fprop tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=128, rng_seed=999), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fp8_fwd_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=4, with_high_probability=[1, 2]),
        s_q_s_kv=RandomSequenceLength(s_q_min=1, s_q_max=256, s_kv_min=64, s_kv_max=1024, s_q_distribution={"s_q=1": 3, "s_q=s_kv": 5, "s_q=random": 2}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=64, d_qk_max=192, d_v_min=64, d_v_max=128, head_dim_distribution={"d_qk=d_v": 2, "d_qk=random": 1}, with_high_probability=[(64, 64), (128, 128), (192, 128)]),
        head_count=RandomHeadGenerator(min=1, max=16, head_group_options=(1, 5, 2)),
        data_type=RandomChoice({torch.float8_e4m3fn: 2, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float8_e5m2: 1, torch.float16: 2}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT: 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged": 0, "padded": 0, "full": 1}),
        stats_layout=RandomChoice({"disabled": 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)
    test.showConfig(test_no, request, reg_run=True)

    if request.node.name in test.blocked_tests:
        pytest.skip(f"blocked test: {request.node.name}")
    exec_sdpa_fp8(test.cfg, request, cudnn_handle)


# # ==================================
# # L0 FP8 bprop tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=64, rng_seed=998), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fp8_bwd_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=4, with_high_probability=[1, 2]),
        s_q_s_kv=RandomSequenceLength(s_q_min=64, s_q_max=256, s_kv_min=64, s_kv_max=256, s_q_distribution={"s_q=1": 0, "s_q=s_kv": 5, "s_q=random": 5}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=64, d_qk_max=128, d_v_min=64, d_v_max=128, head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 0}, with_high_probability=[(64, 64), (128, 128)]),
        head_count=RandomHeadGenerator(min=1, max=8, head_group_options=(1, 4, 1)),
        data_type=RandomChoice({torch.float8_e4m3fn: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float16: 1}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT: 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged": 0, "padded": 0, "full": 1}),
        stats_layout=RandomChoice({"disabled": 1}),
        is_deterministic=RandomChoice({True: 1, False: 1}),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)

    test.cfg.is_infer = False
    test.showConfig(test_no, request, reg_run=True)

    if request.node.name in test.blocked_tests:
        pytest.skip(f"blocked test: {request.node.name}")
    exec_sdpa_fp8(test.cfg, request, cudnn_handle)


# # ==================================
# # L0 FP8 paged attention tests
# # ==================================

@pytest.mark.parametrize("test_no", generate_test_seeds(num_tests=32, rng_seed=997), ids=lambda p: f"test{p[0]}")
@pytest.mark.L0
def test_sdpa_fp8_fwd_paged_L0(env_info, test_no, request, cudnn_handle):

    test = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)

    geom_seed = abs(hash(test_no))
    data_seed = test_no[2]

    rng = random.Random(geom_seed)

    with RandomizationContext(
        batches=RandomBatchSize(min=1, max=4, with_high_probability=[1, 2]),
        s_q_s_kv=RandomSequenceLength(s_q_min=64, s_q_max=256, s_kv_min=64, s_kv_max=512, s_q_distribution={"s_q=1": 0, "s_q=s_kv": 5, "s_q=random": 5}),
        d_qk_d_v=RandomHiddenDimSize(d_qk_min=64, d_qk_max=128, d_v_min=64, d_v_max=128, head_dim_distribution={"d_qk=d_v": 1, "d_qk=random": 0}, with_high_probability=[(64, 64), (128, 128)]),
        head_count=RandomHeadGenerator(min=1, max=4, head_group_options=(1, 2, 0)),
        data_type=RandomChoice({torch.float8_e4m3fn: 2, torch.float8_e5m2: 1}),
        output_type=RandomChoice({torch.float8_e4m3fn: 1, torch.float8_e5m2: 1, torch.float16: 1}),
        with_sliding_mask=SlidingWindowMaskGenerator(no_mask=10),
        diag_align=RandomChoice({cudnn.diagonal_alignment.TOP_LEFT: 1}),
        is_q_ragged_or_padded_or_full=RandomChoice({"ragged": 0, "padded": 1, "full": 0}),
        stats_layout=RandomChoice({"disabled": 1}),
        block_size=RandomBlockSize(min=16, max=128, with_high_probability=[16, 32, 64]),
    ) as randomization_ctx:
        test.cfg = randomization_ctx(rng, data_seed)
        test.cfg.is_paged = True
    test.showConfig(test_no, request, reg_run=True)

    if request.node.name in test.blocked_tests:
        pytest.skip(f"blocked test: {request.node.name}")
    exec_sdpa_fp8(test.cfg, request, cudnn_handle)


# # ===================
# # Single repro test
# # ===================

@pytest.mark.skipif("not config.getoption('--repro')", reason="used with '--repro' only")
@pytest.mark.L0
@pytest.mark.L1
@pytest.mark.L2
@pytest.mark.L3
@pytest.mark.L4
def test_repro(env_info, request, cudnn_handle):
    repro_str = request.config.getoption("--repro")
    cfg = SDPATestConfig(**env_info, implementation=cudnn.attention_implementation.AUTO)
    print(f"repro_str: {repro_str}")

    # Parse the dictionary string and reconstruct the ExecConfig object
    import ast
    repro_dict = ast.literal_eval(repro_str)

    # Convert integer enum values back to enum objects
    if 'diag_align' in repro_dict and repro_dict['diag_align'] is not None:
        repro_dict['diag_align'] = cudnn.diagonal_alignment(repro_dict['diag_align'])
    if 'implementation' in repro_dict and repro_dict['implementation'] is not None:
        repro_dict['implementation'] = getattr(cudnn.attention_implementation, repro_dict['implementation'])
    # Convert string dtype back to torch dtype
    if 'data_type' in repro_dict and repro_dict['data_type'] is not None:
        if 'torch.float16' in repro_dict['data_type']:
            repro_dict['data_type'] = torch.float16
        elif 'torch.bfloat16' in repro_dict['data_type']:
            repro_dict['data_type'] = torch.bfloat16
        elif 'torch.float32' in repro_dict['data_type']:
            repro_dict['data_type'] = torch.float32

    cfg.cfg = ExecConfig(**repro_dict)
    print(f"cfg.cfg: {cfg.cfg}")

    cfg.showConfig((1,1), request, False)
    exec_sdpa(cfg.cfg, request, cudnn_handle)

"""
This script tests cuDNN front-end attention.
The recommended way to run tests:
> pytest -vv -s -rA test_mhas_v2.py
"""

import cudnn
import pytest
import random
import torch
import math
import os
import sys
from looseversion import LooseVersion
from datetime import datetime
from enum import IntEnum

# Invalid left/right attention bound (negative values may be used in the future).
INVALID_BOUND = 99999

# fmt: off

if __name__ == "__main__":
    print("This is pytest script.")
    sys.exit(0)

data_type_options      = [torch.float16, torch.bfloat16]
head_group_options     = ["MHA", "GQA", "MQA"]
random_layout_options  = ["edge_random", "inner_random"]
diag_alignment_options = [cudnn.diagonal_alignment.TOP_LEFT, cudnn.diagonal_alignment.BOTTOM_RIGHT]
diag_alignment_names   = ['cudnn.diagonal_alignment.TOP_LEFT', 'cudnn.diagonal_alignment.BOTTOM_RIGHT']

def tlist(*, num_tests, rng_seed):
    assert num_tests >= 1 and type(num_tests) == int, "wrong input"
    rng = random.Random(rng_seed)
    return [(i+1, num_tests, rng.randint(65536, 2147483647)) for i in range(num_tests)]

def tname_hash(tname):
    tname = tname[tname.find('[')+1 : tname.find(']')]
    assert len(tname) > 0, "empty test name"
    hash = 0
    for chr in tname:
        hash = (hash * 65599 + ord(chr)) & (2**31 - 1)
    return hash

def get_strides_from_indices(shape, indices = [0, 1, 2, 3], gaps = [0, 0, 0, 0], rng_geom = None):
    assert len(shape) == len(gaps) == 4 and sorted(indices) == [0, 1, 2, 3], "wrong input"
    strides = [0, 0, 0, 0]
    curr_stride = 1
    j = indices[3]
    strides[j] = curr_stride
    for i in range(3, 0, -1):
        j = indices[i]
        curr_stride = (shape[j] + gaps[j]) * curr_stride
        j = indices[i-1]
        strides[j] = curr_stride

        # Corrupt strides intentionally for dim=1. When computing offsets,
        # the index used with this stride should always be zero.
        if rng_geom is not None and shape[j] == 1:
            strides[j] = rng_geom.choice([0, 3331333, 99990001])

    total_size = shape[j] * curr_stride
    return tuple(strides), tuple(gaps), total_size

def get_strides_from_layout(shape, layout, gaps = [0, 0, 0, 0], rng_geom = None):
     assert ''.join(sorted(layout)) == 'bdhs', f"wrong layout '{layout}'"
     indices = ['bhsd'.index(ch) for ch in layout]
     return get_strides_from_indices(shape, indices, gaps, rng_geom)

def get_layout_name(string, indices):
    assert len(string) == 4 and sorted(indices) == [0, 1, 2, 3], "wrong input"
    chars = [string[i] for i in indices]
    return ''.join(chars)

def get_all_divisers(num):
    assert num >= 1 and type(num) == int, "wrong input"
    divisors = [1]
    for x in range(2, num):
        if num % x == 0:
            divisors.append(x)
    divisors.append(num)
    return divisors

def get_powers_of_two(lo, hi):
    assert type(lo) == int and type(hi) == int, "wrong input type"
    assert lo > 0 and hi > 0 and lo <= hi, "wrong input values"
    powers_of_two = []
    val = (1 << lo.bit_length()) if (lo & (lo - 1)) != 0 else lo
    while val <= hi:
        powers_of_two.append(val)
        val *= 2
    assert len(powers_of_two) > 0, "empty return list"
    return powers_of_two

def get_multiples_of(val, lo, hi):
    assert type(val) == int and type(lo) == int and type(hi) == int, "wrong input type"
    assert val > 0 and lo > 0 and hi > 0 and lo <= hi, "wrong input values"
    multiples = []
    iter = int((lo+val-1)/val) * val
    while iter <= hi:
        multiples.append(iter)
        iter += val
    assert len(multiples) > 0, "empty return list"
    return multiples

def round_down(value, roundTo):
    assert type(value) == type(roundTo) == int and (roundTo & (roundTo - 1)) == 0, "wrong input"
    return value - (value % roundTo)

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
        assert False, "unsupported tensor data type"

def approx_equal(actual, expected, sepbuf, rawbuf, rtol, atol, tag, disp_elems):
    mismatches = torch.where(torch.isclose(actual.float(), expected, rtol=rtol, atol=atol) == False)
    mismatch_cnt = mismatches[0].numel()
    num_elements = torch.numel(actual)
    if mismatch_cnt != 0:
        if disp_elems > 0:
            print(f"Comparing '{tag}' using rtol={rtol:.4e}, atol={atol:.4e}")
            combined = torch.stack(mismatches, dim=-1).tolist()
            count = 0
            for index in combined:
                print(f"idx{index}: {tag}_gpu={actual[tuple(index)]:+.6e}, {tag}_ref={expected[tuple(index)]:+.6e}")
                count += 1
                if count >= disp_elems:
                    break
            print(f"%%%% Total {mismatch_cnt} mismatches in {num_elements} elements when validating '{tag}' (first {count} mismatches displayed)")
        else:
            print(f"%%%% Total {mismatch_cnt} mismatches in {num_elements} elements when validating '{tag}' results")
    else:
        print(f"%%%% Numerical divergence of '{tag}' within limits")

    # Check if areas before and after the tensor were overwritten (treated as one numerical mismatch).
    if sepbuf is not None and not torch.all(torch.isnan(sepbuf)).item():
        print(f"%%%% Buffer '{tag}' overwritten outside its boundaries")
        print(sepbuf)
        mismatch_cnt += 1

    # Check if unused elements of the tensor were overwritten (treated as one numerical mismatch).
    # Note that this check destroys computed data (overwrites them with NaN-s).
    if rawbuf is not None:
        actual.fill_(float('nan'))
        if not torch.all(torch.isnan(rawbuf)).item():
            print(f"%%%% Unused gaps of '{tag}' tensor were overwritten")
            mismatch_cnt += 1

    return mismatch_cnt

def alloc_tensor(shape, data_type, *, elems=None, strides=None, rng=None, mean=0.0, std=1.0, margins=512):
    # Arguments elems/strides must be both specified or both None.
    if elems is None and strides is None:
        if hasattr(shape, '__iter__'):
            strides = []
            prod = 1
            for dim in reversed(shape):
                strides.insert(0, prod)
                prod *= int(dim)
            elems = prod
        else:
            elems = int(shape)
            strides = (1,)
            shape = (shape,)
    else:
        assert elems is not None and strides is not None, "wrong input"

    assert margins >= 0 and type(margins) == int, "wrong input"

    rawbuf = torch.empty(elems+2*margins, dtype=data_type, device="cuda")
    if torch.is_floating_point(rawbuf):
        rawbuf.fill_(float('nan'))
    else:
        rawbuf.fill_(-1)

    tensor = torch.as_strided(rawbuf, shape, strides, storage_offset=margins)
    sepbuf = (torch.as_strided(rawbuf, (2, margins), (elems+margins, 1), storage_offset=0) if margins > 0 else None)

    # Use this initialization for floating point types only.
    if rng is not None:
        tensor.normal_(mean=mean, std=std, generator=rng)

    # Not returning the raw buffer, if the data tensor has no gaps between valid elements.
    # If there are unused gaps, then we want to check that those gaps were not overwritten.
    if math.prod(shape) == elems:
        rawbuf = None

    return tensor, sepbuf, rawbuf

def bool_cli_option(org_val, request, cli_opt):
    bool_map = {"False": False, "True": True}
    str_val = request.config.getoption(cli_opt)
    val = bool_map.get(str_val)
    return val if type(val) == bool else org_val

def int_cli_option(org_val, request, cli_opt):
    val = request.config.getoption(cli_opt)
    return val if type(val) == int else org_val

def diag_cli_option(org_val, request, cli_opt):
    diag_map = {False : cudnn.diagonal_alignment.TOP_LEFT, True : cudnn.diagonal_alignment.BOTTOM_RIGHT}
    val = request.config.getoption(cli_opt)
    return diag_map.get(bool(val)) if type(val) == int else org_val

def fetch_blocked_tests(file_path):
    blocked_map = {}
    try:
        line_number = None
        with open(file_path, 'r') as file:
            for line_number, line_buf in enumerate(file, 1):
                line_buf = line_buf.split('#', 1)[0]  # remove comments
                line_buf = "".join(line_buf.split())  # remove whitespaces
                if line_buf:
                    test,sms,libs = (line_buf+"::").split(':')[:3]
                    if not test:
                        raise ValueError("missing test name")
                    if test in blocked_map:
                        raise ValueError("duplicate test name")
                    sms  = sms.split(',') if sms else None
                    libs = libs.split(',') if libs else None
                    blocked_map[test] = (sms, libs)
    except Exception as e:
        blocked_map = {}
        if line_number != None:
            print(f"\n\nWARNING: {e} in {file_path}:{line_number}")
        else:
            print(f"\n\nWARNING: {e}")
    return blocked_map

def show_blocked_tests(blocked_map):
    print("\n\nBlocked tests:")
    if blocked_map:
        for test, values in blocked_map.items():
            blocked_sms  = ",".join(map(str, values[0])) if values[0] != None else ""
            blocked_libs = ",".join(map(str, values[1])) if values[1] != None else ""
            print(f"{test} : {blocked_sms} : {blocked_libs}")
    else:
        print("[empty]")

def is_test_blocked(test, gpu_arch, cudnn_ver, blocked_map):
    assert type(test) == type(gpu_arch) == type(cudnn_ver) == str, "expecting strings"
    values = blocked_map.get(test)
    if values is not None:
        blocked_sms, blocked_libs = values
        if (blocked_sms == None or gpu_arch in blocked_sms) and (blocked_libs == None or cudnn_ver in blocked_libs):
            return True
    return False

def truncated_list(beg, end, arr):
    if len(arr) >= beg + 3 + end:
        hi = max(arr)
        lo = min(arr)
        s = [*arr[:beg], '...', *arr[beg:][-end:]]
        s = '['+', '.join(map(str, s))+'], min='+str(lo)+', max='+str(hi)
    else:
        s = '['+', '.join(map(str, arr))+']'
    return s

class knobNAR(IntEnum):
    NEVER  = 0
    ALWAYS = 1
    RANDOM = 2

class knobNA(IntEnum):
    NEVER  = 0
    ALWAYS = 1

class testConfig:
    # To prevent creation of misspelled variables, listing all local variables of the class.
    __slots__ = ['rng_geom', 'geom_seed', 'rng_data', 'data_seed', 'gpu_arch', 'gpu_info', 'cudnn_ver', 'blocked_map',
                 'min_batches', 'max_batches', 'min_s_q', 'max_s_q', 'min_s_kv', 'max_s_kv', 'min_d_qk', 'max_d_qk', 
                 'min_d_v', 'max_d_v', 'min_h_qkv', 'max_h_qkv', 'min_blk_sz', 'max_blk_sz', 'head_group', 
                 'diag_align', 'left_bound', 'right_bound', 'is_infer', 'is_alibi', 'is_paged', 'is_bias', 
                 'is_dropout', 'is_padding', 'is_ragged', 'is_determin', 'data_type', 'batches', 'd_qk', 
                 'd_v', 's_q', 's_kv', 'h_q', 'h_k', 'h_v', 'block_size', 'in_layout', 'out_layout', 
                 'shape_q', 'gaps_q', 'stride_q', 'elems_q', 'shape_k', 'gaps_k', 'stride_k', 'elems_k', 
                 'shape_v', 'gaps_v', 'stride_v', 'elems_v', 'shape_o', 'gaps_o', 'stride_o', 'elems_o',
                 'seq_len_q', 'seq_len_kv']

    def __init__(self, *, gpu_arch, gpu_info, cudnn_ver, blocked_map):
        self.gpu_arch    = str(gpu_arch)
        self.gpu_info    = str(gpu_info)
        self.cudnn_ver   = str(cudnn_ver)
        self.blocked_map = blocked_map

        self.rng_geom    = random.Random()
        self.geom_seed   = None

        self.rng_data    = torch.Generator(device="cuda")
        self.data_seed   = None

        self.min_batches = self.max_batches = 1
        self.min_s_q     = self.max_s_q     = 1
        self.min_s_kv    = self.max_s_kv    = 1
        self.min_d_qk    = self.max_d_qk    = 1
        self.min_d_v     = self.max_d_v     = 1
        self.min_h_qkv   = self.max_h_qkv   = 1
        self.min_blk_sz  = self.max_blk_sz  = 1

        self.head_group  = None
        self.data_type   = None

        self.diag_align  = None
        self.left_bound  = None
        self.right_bound = None

        self.is_alibi    = None
        self.is_infer    = None
        self.is_paged    = None
        self.is_bias     = None
        self.is_dropout  = None
        self.is_determin = None

        self.is_padding  = None
        self.is_ragged   = None
        self.seq_len_q   = None
        self.seq_len_kv  = None

        self.batches     = None
        self.d_qk        = None
        self.d_v         = None
        self.s_q         = None
        self.s_kv        = None
        self.h_q         = None
        self.h_k         = None
        self.h_v         = None
        self.block_size  = None
        self.in_layout   = None
        self.out_layout  = None

        self.shape_q     = None
        self.gaps_q      = None
        self.stride_q    = None
        self.elems_q     = None

        self.shape_k     = None
        self.gaps_k      = None
        self.stride_k    = None
        self.elems_k     = None

        self.shape_v     = None
        self.gaps_v      = None
        self.stride_v    = None
        self.elems_v     = None

        self.shape_o     = None
        self.gaps_o      = None
        self.stride_o    = None
        self.elems_o     = None

    def config_str(self):
        banned = ("max_", "min_", "gpu_", "rng_", "blocked_map", "cudnn_ver", "shape_", "stride_", "elems_")
        stg = ""
        for k in self.__slots__:
            if k.startswith(banned):
                continue
            v = getattr(self, k)
            if type(v) == str:
                assert len(v) > 0, f"ERROR: empty string in {k}='{v}'"
                stg += f"{k}='{v}':"
            elif type(v) == cudnn._compiled_module.diagonal_alignment:
                stg += f"{k}={diag_alignment_names[int(v)]}:"
            else:
                assert v != None, f"ERROR: invalid value in '{k}={v}'"
                stg += f"{k}={v}:"
        stg = "".join(stg.split())  # remove whitespaces
        stg = stg[:-1]  # remove last ':' character
        return stg

    def load_config(self, code):
        print(f"\nLoading config: '{code}'")
        code = "".join(code.split())  # remove whitespaces
        for assign in filter(None, code.split(":")):
            code_to_run = "self." + assign
            try:
                exec(code_to_run)
            except Exception as e:
                assert False, f"ERROR: {e} in '{assign}'"

        banned = ("max_", "min_", "gpu_", "rng_", "blocked_map", "cudnn_ver", "shape_", "stride_", "elems_")
        for k in self.__slots__:
            if k.startswith(banned):
                continue
            v = getattr(self, k)
            assert v != None, f"ERROR: config value '{k}' not set"

        self.rng_geom.seed(self.geom_seed)  # not used in repro
        self.rng_data.manual_seed(self.data_seed)

        self.shape_q = (self.batches, self.h_q, self.s_q, self.d_qk)
        self.shape_k = (self.batches, self.h_k, self.s_kv, self.d_qk)
        self.shape_v = (self.batches, self.h_v, self.s_kv, self.d_v)
        self.shape_o = (self.batches, self.h_q, self.s_q, self.d_v)

        assert all(x > 0 and type(x) == int for x in self.shape_q), f"wrong shape_q(b,h,s,d)={self.shape_q}"
        assert all(x > 0 and type(x) == int for x in self.shape_k), f"wrong shape_k(b,h,s,d)={self.shape_k}"
        assert all(x > 0 and type(x) == int for x in self.shape_v), f"wrong shape_v(b,h,s,d)={self.shape_v}"
        assert all(x > 0 and type(x) == int for x in self.shape_o), f"wrong shape_o(b,h,s,d)={self.shape_o}"

        layout_q, layout_k, layout_v = (self.in_layout+'_'+'_').split('_')[:3]

        self.stride_q, _, self.elems_q = get_strides_from_layout(self.shape_q, layout_q, self.gaps_q)
        self.stride_k, _, self.elems_k = get_strides_from_layout(self.shape_k, layout_k, self.gaps_k)
        self.stride_v, _, self.elems_v = get_strides_from_layout(self.shape_v, layout_v, self.gaps_v)
        self.stride_o, _, self.elems_o = get_strides_from_layout(self.shape_o, self.out_layout, self.gaps_o)

    # Keyword-only arguments after "self".
    def setBatches(self, *, min_batches=1, max_batches=8):
        assert type(min_batches) == int and type(max_batches) == int, "wrong arg types"
        assert min_batches > 0 and max_batches > 0 and min_batches <= max_batches, "invalid range"
        self.min_batches = min_batches
        self.max_batches = max_batches

    # Keyword-only arguments after "self".
    def setSequences(self, *, min_s_q=1, max_s_q=16, min_s_kv=1, max_s_kv=16):
        arg_list = [min_s_q, max_s_q, min_s_kv, max_s_kv]
        assert all(isinstance(x, int) and (x > 0) for x in arg_list), "all args must be int and positive" 
        assert min_s_q <= max_s_q and min_s_kv <= max_s_kv, "invalid range"
        self.min_s_q  = min_s_q
        self.max_s_q  = max_s_q
        self.min_s_kv = min_s_kv
        self.max_s_kv = max_s_kv

    # Keyword-only arguments after "self".
    def setVectors(self, *, min_d_qk=1, max_d_qk=64, min_d_v=1, max_d_v=64):
        arg_list = [min_d_qk, max_d_qk, min_d_v, max_d_v]
        assert all(isinstance(x, int) and (x > 0) for x in arg_list), "all args must be int and positive" 
        assert min_d_qk <= max_d_qk and min_d_v <= max_d_v, "invalid range"
        self.min_d_qk = min_d_qk
        self.max_d_qk = max_d_qk
        self.min_d_v  = min_d_v
        self.max_d_v  = max_d_v

    # Keyword-only arguments after "self".
    def setHeads(self, *, min_h_qkv=1, max_h_qkv=16):
        assert type(min_h_qkv) == int and type(max_h_qkv) == int, "wrong arg types"
        assert min_h_qkv > 0 and max_h_qkv > 0 and min_h_qkv <= max_h_qkv, "invalid range"
        self.min_h_qkv = min_h_qkv
        self.max_h_qkv = max_h_qkv

    # Keyword-only arguments after "self".
    def setBlockSize(self, *, min_blk_sz=1, max_blk_sz=128):
        assert type(min_blk_sz) == int and type(max_blk_sz) == int, "wrong arg types"
        assert min_blk_sz > 0 and max_blk_sz > 0 and min_blk_sz <= max_blk_sz, "invalid range"
        assert max_blk_sz >= (1 << min_blk_sz.bit_length()), "no power of 2 value in range"
        self.min_blk_sz  = min_blk_sz
        self.max_blk_sz  = max_blk_sz

    def draw(self):
        return self.rng_geom.random()

    def showConfig(self, test_no, request, reg_run=True):
        if request.config.option.dryrun == 0 or request.config.option.dryrun == 1:
            if request.config.option.dryrun == 0:
                print("\n" + "=" * 90)
            else:
                print("\n" + "=" * 40 + "Dry-RUN" + "=" * 40)
            print(f"#### Test #{test_no[0]} of {test_no[1]} at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "\n")
            print(f"test_name        = {request.node.name}")
            print(f"geom_seed        = {self.geom_seed}")
            print(f"data_seed        = {self.data_seed}")
            print(f"platform_info    = {self.gpu_arch} ({self.gpu_info}), cudnn_ver={self.cudnn_ver}")
            print(f"head_group       = {self.head_group}")
            print(f"layout           = {self.in_layout}->{self.out_layout}")
            print(f"shape_q(b,h,s,d) = {self.shape_q}, strides={self.stride_q}, gaps={self.gaps_q}, elems={self.elems_q:,}")
            print(f"shape_k(b,h,s,d) = {self.shape_k}, strides={self.stride_k}, gaps={self.gaps_k}, elems={self.elems_k:,}")
            print(f"shape_v(b,h,s,d) = {self.shape_v}, strides={self.stride_v}, gaps={self.gaps_v}, elems={self.elems_v:,}")
            print(f"shape_o(b,h,s,d) = {self.shape_o}, strides={self.stride_o}, gaps={self.gaps_o}, elems={self.elems_o:,}")
            print(f"is_infer         = {self.is_infer}")
            print(f"is_padding       = {self.is_padding}")
            print(f"is_ragged        = {self.is_ragged}")
            print(f"is_alibi         = {self.is_alibi}")
            print(f"is_paged         = {self.is_paged} (block_size={self.block_size})")
            print(f"is_bias          = {self.is_bias}")
            print(f"is_dropout       = {self.is_dropout}")
            print(f"is_determin      = {self.is_determin}")
            print(f"diag_align       = {diag_alignment_names[int(self.diag_align)]} ({int(self.diag_align)})")
            print(f"left_bound       = {self.left_bound}", '(NO BOUND)' if self.left_bound == INVALID_BOUND else '')
            print(f"right_bound      = {self.right_bound}", '(NO BOUND)' if self.right_bound == INVALID_BOUND else '')
            print(f"seq_len_q        = {truncated_list(20, 3, self.seq_len_q)}")
            print(f"seq_len_kv       = {truncated_list(20, 3, self.seq_len_kv)}")
            print(f"data_type        = {self.data_type}")
            if reg_run:
                print(f"repro_cmd        = pytest -vv -s -rA {request.module.__file__}::{request.node.name} --geom_seed {self.geom_seed} --data_seed {self.data_seed}")
        elif request.config.option.dryrun == 2:
            print(f"\npytest -vv -s -rA {request.module.__file__}::{request.node.name} --geom_seed {self.geom_seed} --data_seed {self.data_seed}")
        elif request.config.option.dryrun == 3:
            print(f"\npytest -vv -s -rA {request.module.__file__}::test_repro --repro \"{self.config_str()}\"")
        else:
            assert False, "wrong --dryrun command line option"

        # Make sure to flush everything out.
        print(" ", flush=True)

    def random_layout(self, test_no, is_infer, data_type, head_group, layout_type, knob_generate_ragged_tests, knob_avoid_invalid_configs, request):
        assert type(knob_generate_ragged_tests) == knobNAR, "knob 'generate_ragged_tests' must have type knobNAR"
        assert type(knob_avoid_invalid_configs) == knobNA, "knob 'avoid_invalid_configs' must have type knobNA"
        assert data_type in data_type_options, "wrong data type"
        assert head_group in head_group_options, "wrong head group"
        assert layout_type in random_layout_options, "wrong layout type"

        # Get the initial seed from the 'test_no' sequence. Add to it the test name hash to generate unique RNG seed.
        self.geom_seed = test_no[2] + tname_hash(request.node.name)
        self.data_seed = test_no[2]

        # Overwrite RNG seeds from the command line.
        self.geom_seed = int_cli_option(self.geom_seed, request, "--geom_seed")
        self.data_seed = int_cli_option(self.data_seed, request, "--data_seed")

        self.rng_geom.seed(self.geom_seed)
        self.rng_data.manual_seed(self.data_seed)

        self.head_group   = head_group
        self.data_type    = data_type
        self.is_infer     = is_infer

        self.is_alibi     = self.rng_geom.choice([True, False])
        self.is_paged     = self.rng_geom.choice([True, False])
        self.is_bias      = self.rng_geom.choice([True, False])
        self.is_dropout   = self.rng_geom.choice([True, False])
        self.is_determin  = self.rng_geom.choice([True, False])
        self.is_padding   = self.rng_geom.choice([True, False])
        self.is_ragged    = self.rng_geom.choice([True, False])

        if knob_generate_ragged_tests == knob_generate_ragged_tests.ALWAYS:
           self.is_padding = True
           self.is_ragged  = True
        elif knob_generate_ragged_tests == knob_generate_ragged_tests.NEVER:
           self.is_ragged  = False

        if knob_avoid_invalid_configs == knob_avoid_invalid_configs.ALWAYS:
            # LIMIT: always is_determin=True in inference.
            if self.is_infer:
                self.is_determin = True

            # LIMIT: Paged attention only in inference.
            if not self.is_infer:
                self.is_paged = False

            # LIMIT: Ragged tensor always with variable sequence length (is_padding=False).
            if self.is_ragged and not self.is_padding:
                self.is_padding = True

            # LIMIT: Paged caches can only be used in combination with padding mask (variable sequence length).
            if self.is_paged and not self.is_padding:
                self.is_paged = False

            # LIMIT: Paged caches cannot be used with ragged offsets (packed variable sequence lengths).
            if self.is_paged and self.is_ragged:
                self.is_paged = False

        # Block size for paged attention in fprop (must be power of 2 and minimum 1).
        if self.is_infer and self.is_paged:
            self.block_size = self.rng_geom.choice(get_powers_of_two(self.min_blk_sz, self.max_blk_sz))
        else:
            self.block_size = 0

        # Overwrite all boolean varaibles and block_size from the command line.
        self.is_infer    = bool_cli_option(self.is_infer, request, "--mha_is_infer")
        self.is_alibi    = bool_cli_option(self.is_alibi, request, "--mha_is_alibi")
        self.is_bias     = bool_cli_option(self.is_bias, request, "--mha_is_bias")
        self.is_dropout  = bool_cli_option(self.is_dropout, request, "--mha_is_dropout")
        self.is_determin = bool_cli_option(self.is_determin, request, "--mha_is_determin")
        self.is_padding  = bool_cli_option(self.is_padding, request, "--mha_is_padding")
        self.is_ragged   = bool_cli_option(self.is_ragged, request, "--mha_is_ragged")
        self.is_paged    = bool_cli_option(self.is_paged, request, "--mha_is_paged")
        self.block_size  = int_cli_option(self.block_size, request, "--mha_block_size")

        if layout_type == "edge_random":
            self.batches = self.max_batches
            self.s_q = self.max_s_q
            self.s_kv = self.max_s_kv
        elif layout_type == "inner_random":
            self.batches = self.rng_geom.randint(1, self.max_batches)

            # Force singleton dim with small probability.
            self.s_q = 1 if (self.draw() < 0.05) else self.rng_geom.randint(1, self.max_s_q)
            self.s_kv = 1 if (self.draw() < 0.05) else self.rng_geom.randint(1, self.max_s_kv)

            if (self.draw() < 0.707 and self.s_q <= self.max_s_kv):
                self.s_kv = self.s_q
        else:
            assert False, "wrong layout type"

        # BUG: ragged, variable sequence length tests fail with batch size one, https://nvbugs/5335066
        if self.is_ragged and self.batches == 1:
            self.batches = 2

        # Overwrite batches, s_q, s_kv from the command line.
        self.batches = int_cli_option(self.batches, request, "--mha_batches")
        self.s_q     = int_cli_option(self.s_q, request, "--mha_s_q")
        self.s_kv    = int_cli_option(self.s_kv, request, "--mha_s_kv")

        # To avoid 'diag_align' being None we always assign TOP_LEFT or BOTTOM_RIGHT.
        self.diag_align = self.rng_geom.choice(diag_alignment_options)

        # The left_bound must be >= 1 or None.
        if self.draw() < 0.75:
            self.left_bound = self.rng_geom.randint(1, max(1, self.s_kv//2))
        else:
            self.left_bound = INVALID_BOUND

        # The right_bound must be >= 0 or None; right_bound=0 is a very common case.
        draw = self.draw()
        if draw < 0.5:
            self.right_bound = self.rng_geom.randint(1, max(1, self.s_kv//2))
        elif draw < 0.75:
            self.right_bound = 0
        else:
            self.right_bound = INVALID_BOUND

        # TODO: remove this workaround for bug https://nvbugs/5279917.
        # if self.diag_align == self.diag_align.BOTTOM_RIGHT and self.right_bound != INVALID_BOUND and not self.is_infer:
            self.right_bound = INVALID_BOUND

        # Handle command line options to overwrite diagonal alignment, left bound, and righ tbound.
        self.diag_align  = diag_cli_option(self.diag_align, request, "--mha_diag_align")
        self.left_bound  = int_cli_option(self.left_bound, request, "--mha_left_bound")
        self.right_bound = int_cli_option(self.right_bound, request, "--mha_right_bound")

        if knob_avoid_invalid_configs == knob_avoid_invalid_configs.ALWAYS:
            # LIMIT: left and right bounds are only supported with is_dropout=False, is_bias=False.
            if self.left_bound != INVALID_BOUND and self.right_bound != INVALID_BOUND:
                self.is_dropout = False
                self.is_bias = False

            # LIMIT: when alibi mask is used, diagonal_band_right_bound needs to be exactly 0 (not INVALID_BOUND).
            if self.is_alibi and self.right_bound != 0:
                self.is_alibi = False

            # LIMIT: bottom right causal mask is only supported with is_bias=False, is_alibi=False, is_dropout=False.
            if self.diag_align == self.diag_align.BOTTOM_RIGHT and (self.left_bound != INVALID_BOUND or self.right_bound != INVALID_BOUND):
                self.is_bias    = False
                self.is_alibi   = False
                self.is_dropout = False

            # LIMIT: Left or right bounds are only supported with is_dropout=False, is_bias=False.
            if self.left_bound != INVALID_BOUND or self.right_bound != INVALID_BOUND:
                self.is_dropout = False
                self.is_bias    = False

            # LIMIT: Left bound (a.k.a sliding window) does not support s_q > s_kv
            if self.left_bound != INVALID_BOUND and self.s_q > self.s_kv:
                self.left_bound = INVALID_BOUND

            # LIMIT: Bottom right causal mask does not support s_q > s_kv. 
            if self.s_q > self.s_kv and self.diag_align == self.diag_align.BOTTOM_RIGHT and self.right_bound != INVALID_BOUND:
                self.right_bound = INVALID_BOUND

        # Make sure all Q,K,V vectors in a tensor are aliagned to 16 bytes.
        # For dense tensors, vectors should be divisible into 16B chunks.
        tmp = torch.tensor([1.0], dtype=data_type)
        elem_align = int(16 / tmp.element_size())

        assert self.max_d_qk >= elem_align, "Value max_d_qk too small"
        assert self.max_d_v >= elem_align, "Value max_d_v too small"

        if layout_type == "edge_random":
            self.d_qk = self.max_d_qk
            self.d_v  = self.max_d_v
        elif layout_type == "inner_random":
            self.d_qk = self.rng_geom.randint(elem_align, self.max_d_qk)
            self.d_qk = round_down(self.d_qk, elem_align)
            if (self.draw() < 0.5 and self.d_qk <= self.max_d_v):
                self.d_v = self.d_qk
            else:
                self.d_v = self.rng_geom.randint(elem_align, self.max_d_v)
                self.d_v = round_down(self.d_v, elem_align)
        else:
            assert False, "wrong layout type"

        # Overwrite d_qk, d_v from command line arguments.
        self.d_qk = int_cli_option(self.d_qk, request, "--mha_d_qk")
        self.d_v  = int_cli_option(self.d_v, request, "--mha_d_v")

        if (layout_type == "edge_random"):
            self.h_q = self.max_h_qkv
            if self.head_group in ("MHA", "GQA"):
                self.h_k = self.h_q
                self.h_v = self.h_q
            elif self.head_group == "MQA":
                self.h_k = 1
                self.h_v = 1
            else:
                assert False, f"wrong attention flavor '{self.head_group}'"
        elif (layout_type == "inner_random"):
            self.h_q = self.rng_geom.randint(1, self.max_h_qkv)
            if self.head_group == "MHA":
                self.h_k = self.h_q
                self.h_v = self.h_q
            elif self.head_group == "GQA":
                h_kv_sizes = get_all_divisers(self.h_q)
                self.h_k = self.rng_geom.choice(h_kv_sizes)
                self.h_v = self.rng_geom.choice(h_kv_sizes)
            elif self.head_group == "MQA":
                self.h_k = 1
                self.h_v = 1
            else:
                assert False, f"wrong attention flavor '{self.head_group}'"
        else:
            assert False, "wrong layout type"

        # Overwrite h_q, h_k, h_v from command line arguments.
        self.h_q = int_cli_option(self.h_q, request, "--mha_h_q")
        self.h_k = int_cli_option(self.h_k, request, "--mha_h_k")
        self.h_v = int_cli_option(self.h_v, request, "--mha_h_v")

        # Using the 'bhsd' order for shape.
        self.shape_q = (self.batches, self.h_q, self.s_q, self.d_qk)
        self.shape_k = (self.batches, self.h_k, self.s_kv, self.d_qk)
        self.shape_v = (self.batches, self.h_v, self.s_kv, self.d_v)
        self.shape_o = (self.batches, self.h_q, self.s_q, self.d_v)

        if self.is_ragged:
            self.in_layout  = "bshd_bshd_bshd"
            self.out_layout = "bshd"

            layout_q, layout_k, layout_v = (self.in_layout+'_'+'_').split('_')[:3]

            # Regular gaps are zero in thd format.
            (self.stride_q, self.gaps_q, self.elems_q) = get_strides_from_layout(self.shape_q, layout_q)
            (self.stride_k, self.gaps_k, self.elems_k) = get_strides_from_layout(self.shape_k, layout_k)
            (self.stride_v, self.gaps_v, self.elems_v) = get_strides_from_layout(self.shape_v, layout_v)
            (self.stride_o, self.gaps_o, self.elems_o) = get_strides_from_layout(self.shape_o, self.out_layout)
        else:
            # Q strides, permute first three dimensions in the original layout 'bhsd'.
            base_indices = [0, 1, 2]
            self.rng_geom.shuffle(base_indices)
            base_indices.append(3)
            gaps = [0, 0, 0, 0]
            if (self.draw() < 0.5):
                gaps = [self.rng_geom.randint(0, 8) for _ in range(3)]
                gaps.append(elem_align * self.rng_geom.randint(0, 2))
            (self.stride_q, self.gaps_q, self.elems_q) = get_strides_from_indices(self.shape_q, base_indices, gaps, self.rng_geom)
            self.in_layout = get_layout_name("bhsd", base_indices) + '_'

            # For K strides, decide with some probability if a new layout should be used.
            indices = base_indices
            if (self.draw() < 0.707):
                indices = [0, 1, 2]
                self.rng_geom.shuffle(indices)
                indices.append(3)
            gaps = [0, 0, 0, 0]
            if (self.draw() < 0.5):
                gaps = [self.rng_geom.randint(0, 8) for _ in range(3)]
                gaps.append(elem_align * self.rng_geom.randint(0, 2))
            (self.stride_k, self.gaps_k, self.elems_k) = get_strides_from_indices(self.shape_k, indices, gaps, self.rng_geom)
            self.in_layout += get_layout_name("bhsd", indices) + '_'

            # For V strides, decide with some probability if a new layout should be used.
            indices = base_indices
            if (self.draw() < 0.707):
                indices = [0, 1, 2]
                self.rng_geom.shuffle(indices)
                indices.append(3)
            gaps = [0, 0, 0, 0]
            if (self.draw() < 0.5):
                gaps = [self.rng_geom.randint(0, 8) for _ in range(3)]
                gaps.append(elem_align * self.rng_geom.randint(0, 2))
            (self.stride_v, self.gaps_v, self.elems_v) = get_strides_from_indices(self.shape_v, indices, gaps, self.rng_geom)
            self.in_layout += get_layout_name("bhsd", indices)

            # Q, K, V buffers are not interleaved.
            # For O strides, decide with some probability if a new layout should be used
            indices = base_indices
            if (self.draw() < 0.5):
                indices = [0, 1, 2]
                self.rng_geom.shuffle(indices)
                indices.append(3)
            gaps = [0, 0, 0, 0]
            if (self.draw() < 0.5):
                gaps = [self.rng_geom.randint(0, 8) for _ in range(3)]
                gaps.append(elem_align * self.rng_geom.randint(0, 2))
            (self.stride_o, self.gaps_o, self.elems_o) = get_strides_from_indices(self.shape_o, indices, gaps, self.rng_geom)
            self.out_layout = get_layout_name("bhsd", indices)

        self.seq_len_q  = []
        self.seq_len_kv = []
        if self.is_padding:
            for _ in range(self.batches):
                self.seq_len_q.append(self.rng_geom.randint(1, self.s_q))
                self.seq_len_kv.append(self.rng_geom.randint(1, self.s_kv))

            if self.draw() < 0.1 and self.s_q == self.s_kv:
                # Force seq_len_q and seq_len_kv to be the same in small number of tests.
                self.seq_len_kv = self.seq_len_q.copy()
            elif self.left_bound != INVALID_BOUND or (self.right_bound != INVALID_BOUND and self.diag_align == self.diag_align.BOTTOM_RIGHT):
                # TODO: fix the reference model and remove this work-around.
                # Force seq_len_q to be less or equal seq_len_kv.
                self.seq_len_q = [min(x, y) for x, y in zip(self.seq_len_q, self.seq_len_kv)]

        self.showConfig(test_no, request)

def compute_ref(
    q,
    k,
    v,
    attn_scale=1.0,
    bias=None,
    is_alibi=False,
    padding=None,
    diag_align=cudnn.diagonal_alignment.TOP_LEFT,
    left_bound=INVALID_BOUND,
    right_bound=INVALID_BOUND,
    dropout_prob=0.0,
    dropout_mask=None,
    generate_stats=False,
    device="cuda",
):
    b, h_q, s_q, d_qk = q.shape
    _, h_k, s_kv, _ = k.shape
    _, h_v, _, d_v = v.shape

    assert k.shape == (b, h_k, s_kv, d_qk)
    assert v.shape == (b, h_v, s_kv, d_v)

    # use float32 datatype and math for reference computation
    q = q.to(dtype=torch.float32, device=device)
    k = k.to(dtype=torch.float32, device=device)
    v = v.to(dtype=torch.float32, device=device)

    # expand tensors for GQA and MQA
    if h_q != h_k:
        assert h_q % h_k == 0
        k = k.unsqueeze(2)
        k = k.expand(-1, -1, h_q // h_k, -1, -1)
        k = k.reshape(k.size(0), -1, k.size(3), k.size(4))
    if h_q != h_v:
        assert h_q % h_v == 0
        v = v.unsqueeze(2)
        v = v.expand(-1, -1, h_q // h_v, -1, -1)
        v = v.reshape(v.size(0), -1, v.size(3), v.size(4))

    if left_bound != INVALID_BOUND:
        swa_mask_zero = torch.ones(1, 1, s_q, 1, dtype=torch.bool, device=device)
        swa_mask_zero[:, :, s_kv + left_bound - 1 :, :] = False
        q = q * swa_mask_zero

    # generate masks to compute reference values for padding mask (also called variable sequence length)
    if padding is not None:
        q_mask = torch.zeros(b, 1, s_q, 1, dtype=torch.bool, device=device)
        k_mask = torch.zeros(b, 1, s_kv, 1, dtype=torch.bool, device=device)
        v_mask = torch.zeros(b, 1, s_kv, 1, dtype=torch.bool, device=device)
        s_mask = torch.zeros(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
        p_mask = torch.zeros(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
        seq_len_q, seq_len_kv = padding
        for i, (m, n) in enumerate(zip(seq_len_q, seq_len_kv)):
            q_mask[i, :, m:, :] = True
            k_mask[i, :, n:, :] = True
            v_mask[i, :, n:, :] = True
            s_mask[i, :, :, n:] = True
            p_mask[i, :, m:, :] = True

        q = q.masked_fill(q_mask, 0.0)
        k = k.masked_fill(k_mask, 0.0)
        v = v.masked_fill(v_mask, 0.0)

    s = torch.einsum("bhqd,bhkd->bhqk", q, k) * attn_scale

    # Attention masks are applied in the following order:
    # - Bias mask
    # - Alibi mask
    # - Padding mask
    # - Causal mask
    if bias is not None:
        s = s + bias
    if is_alibi:
        index_row = torch.arange(s_q, dtype=torch.float32, device=device).view(-1, 1)
        index_col = torch.arange(s_kv, dtype=torch.float32, device=device)
        distance = index_col - index_row

        # Get the closest power of 2 to `n_heads`.
        # If `n_heads` is not a power of 2, then we first calculate slopes to the closest (smaller) power of 2,
        # and then add the remaining slopes.
        n = 2 ** math.floor(math.log2(h_q))
        m_0 = 2.0 ** (-8.0 / n)
        m = torch.pow(m_0, torch.arange(1, 1 + n))

        # If `n_heads` is not a power of 2, then we add the remaining slopes.
        # We calculate the remaining slopes for $n * 2$ (avoiding slopes added previously).
        # And pick the slopes upto `n_heads`.
        if n < h_q:
            m_hat_0 = 2.0 ** (-4.0 / n)
            m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (h_q - n), 2))
            # Concatenate the slopes with the remaining slopes.
            m = torch.cat([m, m_hat])

        # Reshape the tensor to [1, num_heads, 1, 1]
        m = m.view(1, -1, 1, 1).to(device=device)

        alibi_mask = distance.to(dtype=torch.float32) * m
        s = s + alibi_mask

    if padding is not None:
        s = s.masked_fill(s_mask, float("-inf"))

    if diag_align == diag_align.TOP_LEFT and right_bound != INVALID_BOUND:
        causal_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
        causal_mask.triu_(diagonal=1 + right_bound)
        s = s.masked_fill(causal_mask, float("-inf"))
    elif (diag_align == diag_align.BOTTOM_RIGHT and right_bound != INVALID_BOUND):
        causal_mask_bottom_right = None
        if padding:
            causal_mask_bottom_right = torch.ones(
                b, 1, s_q, s_kv, dtype=torch.bool, device=device
            )
            seq_len_q, seq_len_kv = padding
            for i in range(b):
                causal_mask_bottom_right[i, :, :, :].triu_(
                    diagonal=seq_len_kv[i] - seq_len_q[i] + 1 + right_bound
                )
        else:
            causal_mask_bottom_right = torch.ones(
                s_q, s_kv, dtype=torch.bool, device=device
            )
            causal_mask_bottom_right.triu_(diagonal=s_kv - s_q + 1 + right_bound)
        s = s.masked_fill(causal_mask_bottom_right, float("-inf"))

    if left_bound != INVALID_BOUND:
        assert diag_align is not None
        if diag_align == diag_align.TOP_LEFT:
            swa_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
            swa_mask.tril_(diagonal=-1 * left_bound)
        elif diag_align == diag_align.BOTTOM_RIGHT:
            # BRCM + SWA for variable sequence lengths
            if padding:
                swa_mask = torch.ones(b, 1, s_q, s_kv, dtype=torch.bool, device=device)
                seq_len_q, seq_len_kv = padding
                for i in range(b):
                    swa_mask[i, :, :, :].tril_(
                        diagonal=seq_len_kv[i] - seq_len_q[i] - left_bound
                    )
            # BRCM + SWA for fixed sequence lengths
            else:
                swa_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
                swa_mask.tril_(diagonal=-1 * left_bound + (s_kv - s_q))
        swa_mask &= swa_mask_zero.view(s_q, 1)
        s = s.masked_fill(swa_mask, float("-inf"))

    p = torch.softmax(s, dim=-1)

    if left_bound != INVALID_BOUND:
        p = p * swa_mask_zero
    if padding is not None:
        p = p.masked_fill(p_mask, 0.0)

    # apply dropout mask over softmax outputs
    if dropout_prob != 0.0:
        assert dropout_mask != None, "PyTorch reference must have dropout_mask for dropout"
        p = (p * dropout_mask) / (1 - dropout_prob)

    o = torch.einsum("bhqk,bhkd->bhqd", p, v)

    # softmax stats is used for backwards computation
    if generate_stats:
        # amax (NOT absolute max) is used here to evenly distribute gradient
        row_max = torch.amax(s, -1, True)
        row_exp = torch.exp(s - row_max)
        row_sum = torch.sum(row_exp, -1, True)
        stats = row_max + torch.log(row_sum)
        return o, stats

    return o

# Compute the exclusive prefix sum for ragged sequence dimension
# input tensor has shape (B, 1, 1, 1)
# output tensor has shape (B+1, 1, 1, 1)
# example input seq_len: [2, 4, 1, 6] (along the B dimension)
# example output ragged_offset: [0, 2, 6, 7, 13] (along the B dimension)

def compute_exclusive_prefix_sum(tensor):
    assert list(tensor.size())[1:]==[1,1,1]
    # We need to provide a tuple of two tensors to torch.cat().
    return torch.cat((torch.zeros(1, 1, 1, 1, dtype=tensor.dtype, device=tensor.device), torch.cumsum(tensor, dim=0)))

def generate_ragged_offset(h_q, h_k, h_v, d_qk, d_v, seq_len_q, seq_len_kv):
    # Only for thd_thd_thd
    q_ragged_offset = compute_exclusive_prefix_sum(seq_len_q) * h_q * d_qk
    k_ragged_offset = compute_exclusive_prefix_sum(seq_len_kv) * h_k * d_qk
    v_ragged_offset = compute_exclusive_prefix_sum(seq_len_kv) * h_v * d_v
    o_ragged_offset = compute_exclusive_prefix_sum(seq_len_q) * h_q * d_v

    # Convert to int64 for cuDNN 9.6.0
    q_ragged_offset = q_ragged_offset.to(dtype=torch.int64)
    k_ragged_offset = k_ragged_offset.to(dtype=torch.int64)
    v_ragged_offset = v_ragged_offset.to(dtype=torch.int64)
    o_ragged_offset = o_ragged_offset.to(dtype=torch.int64)

    return q_ragged_offset, k_ragged_offset, v_ragged_offset, o_ragged_offset


def convert_ragged_to_uniform(ragged_tensor, seq_len):
    # limitations:
    # 1. tensor is bhsd dim order and bshd stride order (may be interleaved)
    # 2. ragged tensor is packed and in-order, therefore
    #    ragged offset is monatomically increasing
    assert ragged_tensor.dim() == 4
    b, h, s, d = ragged_tensor.size()
    b_stride, h_stride, s_stride, d_stride = ragged_tensor.stride()
    assert b_stride >= s_stride >= h_stride >= d_stride
    assert seq_len.dim() == 4 and (b, 1, 1, 1) == seq_len.size()

    # ragged offset is given in 4D, convert to 1D locally
    seq_len = seq_len.flatten()

    # convert bhsd to bshd and flatten
    uniform_tensor = torch.zeros(b, s, h, d).to(
        dtype=ragged_tensor.dtype, device=ragged_tensor.device
    )
    ragged_tensor_thd = torch.einsum("bhsd->bshd", ragged_tensor).reshape(b * s, h, d)

    # copy
    t = 0
    for b, s in enumerate(seq_len):
        uniform_tensor[b, 0:s, :, :] = ragged_tensor_thd[t : t + s, :, :]
        t += s

    # convert back to bshd to bhsd
    uniform_tensor = torch.einsum("bshd->bhsd", uniform_tensor)
    return uniform_tensor

def create_container_and_page_table(tensor, block_size):
    B, H, S, D = tensor.shape
    # num_blocks = math.ceil(S/block_size) * B
    blocks_per_batch = math.ceil(S/block_size)

    padding_seq = (blocks_per_batch * block_size) - S
    if padding_seq > 0:
        zeros = torch.zeros(B,H,padding_seq,D, device='cuda', dtype=tensor.dtype)
        cat_tensor = torch.cat((tensor, zeros), axis = 2)
    else:
        cat_tensor = tensor

    reshaped = torch.cat((cat_tensor.clone()).chunk(blocks_per_batch, dim=2), dim=0)

    table_size = math.ceil(S/block_size)
    page_table = torch.linspace(0, B*table_size-1, B*table_size, device='cuda', dtype=torch.int32).reshape(table_size,1,B,1)
    page_table = torch.transpose(page_table,0,2)

    return(reshaped, page_table)


def exec_sdpa(cfg, request, cudnn_handle):
    # Do not run any test when --dryrn option is provided.
    if request.config.option.dryrun:
        pytest.skip("dry run mode")

    # Check if the test is temporarily blocked.
    if is_test_blocked(request.node.name, cfg.gpu_arch, cfg.cudnn_ver, cfg.blocked_map):
        print(f"\nWARNING: test '{request.node.name}' is blocked on {cfg.gpu_arch} and cuDNN {cfg.cudnn_ver}")
        pytest.skip("test blocked")

    head_group   = cfg.head_group
    data_type    = cfg.data_type
    rng_data     = cfg.rng_data

    is_alibi     = cfg.is_alibi
    is_infer     = cfg.is_infer
    is_paged     = cfg.is_paged
    is_bias      = cfg.is_bias
    is_padding   = cfg.is_padding
    is_ragged    = cfg.is_ragged
    is_dropout   = cfg.is_dropout
    is_determin  = cfg.is_determin

    diag_align   = cfg.diag_align
    left_bound   = cfg.left_bound
    right_bound  = cfg.right_bound

    batches      = cfg.batches
    d_qk         = cfg.d_qk
    d_v          = cfg.d_v
    s_q          = cfg.s_q
    s_kv         = cfg.s_kv
    h_q          = cfg.h_q
    h_k          = cfg.h_k
    h_v          = cfg.h_v
    block_size   = cfg.block_size
    in_layout    = cfg.in_layout
    out_layout   = cfg.out_layout

    shape_q      = cfg.shape_q
    stride_q     = cfg.stride_q
    elems_q      = cfg.elems_q

    shape_k      = cfg.shape_k
    stride_k     = cfg.stride_k
    elems_k      = cfg.elems_k

    shape_v      = cfg.shape_v
    stride_v     = cfg.stride_v
    elems_v      = cfg.elems_v

    shape_o      = cfg.shape_o
    stride_o     = cfg.stride_o
    elems_o      = cfg.elems_o

    seq_len_q    = cfg.seq_len_q
    seq_len_kv   = cfg.seq_len_kv

    # ============================
    # Basic parameter check.
    # ============================

    if not all((x > 0 and type(x) == int) for x in (batches, d_qk, d_v, s_q, s_kv, h_q, h_k, h_v)):
       assert False, "tensor dimensions must be integer and positive"

    if head_group == "MHA":
        assert h_q == h_k == h_v, f"invalid heads, h_q={h_q}, h_k={h_k}, h_v={h_v} for '{head_group}' attention"
    elif head_group == "GQA":
        assert h_q % h_k == 0 and h_q % h_v == 0, f"invalid heads, h_q={h_q}, h_k={h_k}, h_v={h_v} for '{head_group}' attention"
    elif head_group == "MQA":
        assert h_k == 1 and h_v == 1, f"invalid heads, h_q={h_q}, h_k={h_k}, h_v={h_v} for '{head_group}' attention"
    else:
        assert False, f"invalid '{head_group}' attention flavor"

    assert shape_q == (batches, h_q, s_q, d_qk), f"wrong shape_q={shape_q}"
    assert shape_k == (batches, h_k, s_kv, d_qk), f"wrong shape_k={shape_k}"
    assert shape_v == (batches, h_v, s_kv, d_v), f"wrong shape_v={shape_v}"
    assert shape_o == (batches, h_q, s_q, d_v), f"wrong shape_o={shape_o}"

    if not is_infer:
        assert is_paged == False and block_size == 0, "paged attention not allowed in backward pass"

    if is_ragged:
        assert is_padding == True, "is_ragged=True and is_padding=False not allowed"
        assert in_layout == "bshd_bshd_bshd", f"wrong input layout '{in_layout}' for ragged variable seq len"
        assert out_layout == "bshd", f"wrong output layout '{out_layout}' for ragged variable seq len"

    assert isinstance(seq_len_q, (list, tuple)), "input 'seq_len_q' must be list or tuple"
    if is_padding:
        assert len(seq_len_q) == batches, f"wrong 'seq_len_q' length, expecting {batches}"
    else:
        assert len(seq_len_q) == 0, f"wrong 'seq_len_q' length, expecting 0"

    assert isinstance(seq_len_kv, (list, tuple)), "input 'seq_len_kv' must be list or tuple"
    if is_padding:
        assert len(seq_len_kv) == batches, f"wrong 'seq_len_kv' length, expecting {batches}"
    else:
        assert len(seq_len_kv) == 0, f"wrong 'seq_len_kv' length, expecting 0"

    assert all(x >= 0 and type(x) == int for x in seq_len_q), f"wrong seq_len_q={seq_len_q}"
    assert all(x >= 0 and type(x) == int for x in seq_len_kv), f"wrong seq_len_kv={seq_len_kv}"

    cudnn_version = LooseVersion(cudnn.backend_version_string())
    if cudnn_version < "9.10.0":
        print("@@@@ Overall result: WAIVED, test_mhas_v2.py supports cudnn 9.10.0 or higher.")
        pytest.skip("test_mhas_v2.py requires cudnn 9.10.0 or higher")

    if s_q == s_kv == 1:
        print("@@@@ Overall result: WAIVED, skipping known issue of s_q == s_kv == 1.")
        pytest.skip("skipping known issue of s_q == s_kv == 1")

    qkv_num_elems = elems_q + elems_k + elems_v

    (q_gpu, _, _) = alloc_tensor(shape_q, data_type, elems=elems_q, strides=stride_q, rng=rng_data, mean=-0.5, std=1.0)
    (k_gpu, _, _) = alloc_tensor(shape_k, data_type, elems=elems_k, strides=stride_k, rng=rng_data, mean=-0.5, std=1.0)
    (v_gpu, _, _) = alloc_tensor(shape_v, data_type, elems=elems_v, strides=stride_v, rng=rng_data, mean=-0.5, std=1.0)
    (bias_gpu, _, _) = (alloc_tensor((1, h_q, s_q, s_kv), data_type, rng=rng_data, mean=0.0, std=1.0) if is_bias else (None, None, None))

    if not is_infer:
        (dQ_gpu, dQ_sep, dQ_raw) = alloc_tensor(shape_q, data_type, elems=elems_q, strides=stride_q)
        (dK_gpu, dK_sep, dK_raw) = alloc_tensor(shape_k, data_type, elems=elems_k, strides=stride_k)
        (dV_gpu, dV_sep, dV_raw) = alloc_tensor(shape_v, data_type, elems=elems_v, strides=stride_v)
        (dBias_gpu, dBias_sep, dBias_raw) = (alloc_tensor((1, h_q, s_q, s_kv), data_type) if is_bias else (None, None, None))
        (dO_gpu, dO_sep, dO_raw) = alloc_tensor(shape_o, data_type, elems=elems_o, strides=stride_o, rng=rng_data, mean=0.0, std=0.1)

    # Sequence lenghts for gpu, must be a four dimensional tensor.
    seq_len_q_gpu = seq_len_kv_gpu = None
    if len(seq_len_q) > 0:
        seq_len_q_gpu = torch.tensor(seq_len_q, dtype=torch.int32, device="cuda")
        seq_len_q_gpu = seq_len_q_gpu[:, None, None, None]  # batches x 1 x 1 x 1
    if len(seq_len_kv) > 0:
        seq_len_kv_gpu = torch.tensor(seq_len_kv, dtype=torch.int32, device="cuda")
        seq_len_kv_gpu = seq_len_kv_gpu[:, None, None, None]  # batches x 1 x 1 x 1

    # maxT = next_multiple_of_64(sum(seq_len))
    max_t_q = ((torch.sum(seq_len_q_gpu).item() + 63) // 64) * 64 if is_ragged else None
    max_t_kv = ((torch.sum(seq_len_kv_gpu).item() + 63) // 64) * 64 if is_ragged else None

    if is_dropout:
        seed_gpu = torch.full((1, 1, 1, 1), 123456, dtype=torch.int64, device="cuda")
        offset_gpu = torch.full((1, 1, 1, 1), 789, dtype=torch.int64, device="cuda")

    rng_dump_gpu = torch.zeros((batches, h_q, s_q, s_kv), dtype=torch.float32, device="cuda") if is_dropout else None

    if is_ragged:
       q_ragged_offset_gpu, k_ragged_offset_gpu, v_ragged_offset_gpu, o_ragged_offset_gpu = generate_ragged_offset(h_q, h_k, h_v, d_qk, d_v, seq_len_q_gpu, seq_len_kv_gpu)

    (o_gpu, o_sep, o_raw) = alloc_tensor(shape_o, data_type, elems=elems_o, strides=stride_o)
    (stats_gpu, stats_sep, stats_raw) = (alloc_tensor((batches, h_q, s_q, 1), torch.float32) if not is_infer else (None, None, None))

    container_k_gpu = None
    container_v_gpu = None
    page_table_k_gpu = None
    page_table_v_gpu = None
    if is_paged:
        container_k_gpu, page_table_k_gpu = create_container_and_page_table(k_gpu, block_size)
        container_v_gpu, page_table_v_gpu = create_container_and_page_table(v_gpu, block_size)

    stream = torch.cuda.current_stream().cuda_stream
    cudnn.set_stream(handle=cudnn_handle, stream=stream)

    # Forward cuDNN graph
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(data_type),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=cudnn_handle,
    )

    q = graph.tensor_like(q_gpu)
    k = graph.tensor_like(k_gpu) if not is_paged else graph.tensor_like(container_k_gpu)
    v = graph.tensor_like(v_gpu) if not is_paged else graph.tensor_like(container_v_gpu)

    page_table_k = graph.tensor_like(page_table_k_gpu) if is_paged else None
    page_table_v = graph.tensor_like(page_table_v_gpu) if is_paged else None

    bias = graph.tensor_like(bias_gpu) if is_bias else None

    seq_len_q = graph.tensor_like(seq_len_q_gpu) if is_padding else None
    seq_len_kv = graph.tensor_like(seq_len_kv_gpu) if is_padding else None

    dropout_prob = 0.1 if is_dropout else 0.0

    if is_dropout:
        seed = graph.tensor_like(seed_gpu)
        offset = graph.tensor_like(offset_gpu)
        dropout_tuple = (dropout_prob, seed, offset)

    rng_dump = graph.tensor_like(rng_dump_gpu) if is_dropout else None

    q_ragged_offset = graph.tensor_like(q_ragged_offset_gpu) if is_ragged else None
    k_ragged_offset = graph.tensor_like(k_ragged_offset_gpu) if is_ragged else None
    v_ragged_offset = graph.tensor_like(v_ragged_offset_gpu) if is_ragged else None
    o_ragged_offset = graph.tensor_like(o_ragged_offset_gpu) if is_ragged else None

    if is_ragged:
        q.set_ragged_offset(q_ragged_offset)
        k.set_ragged_offset(k_ragged_offset)
        v.set_ragged_offset(v_ragged_offset)

    attn_scale = 0.125
    
    o, stats = graph.sdpa(
        name="sdpa_forward",
        q=q,
        k=k,
        v=v,
        generate_stats=not is_infer,
        attn_scale=attn_scale,
        bias=bias,
        use_alibi_mask=is_alibi,
        use_padding_mask=is_padding,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        diagonal_band_left_bound=left_bound if left_bound != INVALID_BOUND else None,
        diagonal_band_right_bound=right_bound if right_bound != INVALID_BOUND else None,
        diagonal_alignment=diag_align,
        dropout=dropout_tuple if is_dropout else None,
        rng_dump=rng_dump,
        paged_attention_k_table=page_table_k,
        paged_attention_v_table=page_table_v,
        paged_attention_max_seq_len_kv=s_kv if is_paged else None
    )

    o.set_output(True).set_dim(shape_o).set_stride(stride_o)
    if is_ragged:
        o.set_ragged_offset(o_ragged_offset)

    if is_infer == False:
        stats.set_output(True).set_data_type(cudnn.data_type.FLOAT)

    try:
        graph.validate()
    except cudnn.cudnnGraphNotSupportedError as e:
        print(f"@@@@ Overall result: WAIVED, not supported forward graph. {e}")
        pytest.xfail("not supported forward graph")
    except Exception as e:
        print(f"@@@@ Overall result: FAILED, unexpected '{e.__class__.__name__}' exception during forward graph validate. {e}")
        pytest.fail("unexpected exception during forward graph validate", pytrace=False)

    try:
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()
    except cudnn.cudnnGraphNotSupportedError as e:
        print(f"@@@@ Overall result: WAIVED, not supported forward graph after validate. {e}")
        pytest.xfail("not supported forward graph after validate")
    except Exception as e:
        print(f"@@@@ Overall result: FAILED, unexpected '{e.__class__.__name__}' exception after forward validate. {e}")
        pytest.fail("unexpected exception after forward validate", pytrace=False)

    variant_pack = {
        q: q_gpu,
        k: k_gpu if not is_paged else container_k_gpu,
        v: v_gpu if not is_paged else container_v_gpu,
        bias: bias_gpu,
        seq_len_q: seq_len_q_gpu,
        seq_len_kv: seq_len_kv_gpu,
        q_ragged_offset: q_ragged_offset_gpu if is_ragged else None,
        k_ragged_offset: k_ragged_offset_gpu if is_ragged else None,
        v_ragged_offset: v_ragged_offset_gpu if is_ragged else None,
        o_ragged_offset: o_ragged_offset_gpu if is_ragged else None,
        o: o_gpu,
        stats: stats_gpu,
        rng_dump: rng_dump_gpu,
        page_table_k: page_table_k_gpu,
        page_table_v: page_table_v_gpu
    }

    if is_dropout:
        variant_pack[seed] = seed_gpu
        variant_pack[offset] = offset_gpu

    # Allocate workspace for the forward call.
    (workspace, ws_sep, _) = alloc_tensor(graph.get_workspace_size(), torch.uint8)

    # Display available memory.
    # torch.cuda.empty_cache()
    # free_mem, total_mem = torch.cuda.mem_get_info()
    # print(f"Free GPU memory (before forward): {free_mem / (1024**3):.4f} GB of {total_mem / (1024**3):.4f} GB")

    # Execute forward cuDNN graph.
    graph.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    if ws_sep is not None and not torch.all(ws_sep==-1).item():
        print("@@@@ Overall result: FAILED, forward workspace overwritten outside its boundaries.")
        print(ws_sep)
        pytest.fail("forward workspace overwritten outside boundaries", pytrace=False)

    if not is_infer:
        if cudnn_version < "8.9.6" and is_padding:
            # zero out padded region of the output and stats
            for i, m in enumerate(seq_len_q_gpu):
                o_gpu[i, :, m:, :] = 0
                stats_gpu[i, :, m:, :] = 0

        stream = torch.cuda.current_stream().cuda_stream  #2
        cudnn.set_stream(handle=cudnn_handle, stream=stream)
        sm_version = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]

        # Backward cuDNN graph
        graph = cudnn.pygraph(
            io_data_type=convert_to_cudnn_type(data_type),
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=cudnn_handle,
            sm_version = sm_version
        )

        q = graph.tensor_like(q_gpu)
        k = graph.tensor_like(k_gpu)
        v = graph.tensor_like(v_gpu)
        o = graph.tensor_like(o_gpu)
        dO = graph.tensor_like(dO_gpu)
        stats = graph.tensor_like(stats_gpu)

        bias = graph.tensor_like(bias_gpu) if is_bias else None
        dBias = (graph.tensor_like(dBias_gpu).set_stride((h_q * s_q * s_kv, s_q * s_kv, s_kv, 1)) if is_bias else None)

        seq_len_q = graph.tensor_like(seq_len_q_gpu) if is_padding else None
        seq_len_kv = graph.tensor_like(seq_len_kv_gpu) if is_padding else None

        if is_dropout:
            seed = graph.tensor_like(seed_gpu)
            offset = graph.tensor_like(offset_gpu)
            dropout_tuple = (dropout_prob, seed, offset)

        q_ragged_offset = graph.tensor_like(q_ragged_offset_gpu) if is_ragged else None
        k_ragged_offset = graph.tensor_like(k_ragged_offset_gpu) if is_ragged else None
        v_ragged_offset = graph.tensor_like(v_ragged_offset_gpu) if is_ragged else None
        o_ragged_offset = graph.tensor_like(o_ragged_offset_gpu) if is_ragged else None

        if is_ragged:
            q.set_ragged_offset(q_ragged_offset)
            k.set_ragged_offset(k_ragged_offset)
            v.set_ragged_offset(v_ragged_offset)
            o.set_ragged_offset(o_ragged_offset)
            dO.set_ragged_offset(o_ragged_offset)

        dQ, dK, dV = graph.sdpa_backward(
            name="sdpa_backward",
            q=q,
            k=k,
            v=v,
            o=o,
            dO=dO,
            stats=stats,
            attn_scale=attn_scale,
            bias=bias,
            dBias=dBias,
            use_alibi_mask=is_alibi,
            use_padding_mask=is_padding,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            max_total_seq_len_q=max_t_q,
            max_total_seq_len_kv=max_t_kv,
            diagonal_band_left_bound=left_bound if left_bound != INVALID_BOUND else None,
            diagonal_band_right_bound=right_bound if right_bound != INVALID_BOUND else None,
            diagonal_alignment=diag_align,
            dropout=dropout_tuple if is_dropout else None,
            use_deterministic_algorithm=is_determin,
        )

        dQ.set_output(True).set_dim(dQ_gpu.size()).set_stride(dQ_gpu.stride())
        dK.set_output(True).set_dim(dK_gpu.size()).set_stride(dK_gpu.stride())
        dV.set_output(True).set_dim(dV_gpu.size()).set_stride(dV_gpu.stride())
        if is_ragged:
            dQ.set_ragged_offset(q_ragged_offset)
            dK.set_ragged_offset(k_ragged_offset)
            dV.set_ragged_offset(v_ragged_offset)

        try:
            graph.validate()
        except cudnn.cudnnGraphNotSupportedError as e:
            print(f"@@@@ Overall result: WAIVED, not supported backward graph. {e}")
            pytest.xfail("not supported backward graph")
        except Exception as e:
            print(f"@@@@ Overall result: FAILED, unexpected '{e.__class__.__name__}' exception during backward graph validate. {e}")
            pytest.fail("unexpected exception during backward graph validate", pytrace=False)

        try:
            graph.build_operation_graph()
            graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            graph.check_support()
            graph.build_plans()
        except cudnn.cudnnGraphNotSupportedError as e:
            print(f"@@@@ Overall result: WAIVED, not supported backward graph after validate. {e}")
            pytest.xfail("not supported backward graph after validate")
        except Exception as e:
            print(f"@@@@ Overall result: FAILED, unexpected '{e.__class__.__name__}' exception after backward validate. {e}")
            pytest.fail("unexpected exception after backward validate", pytrace=False)

        variant_pack = {
            q: q_gpu,
            k: k_gpu,
            v: v_gpu,
            o: o_gpu,
            dO: dO_gpu,
            stats: stats_gpu,
            dQ: dQ_gpu,
            dK: dK_gpu,
            dV: dV_gpu,
            bias: bias_gpu,
            dBias: dBias_gpu,
            seq_len_q: seq_len_q_gpu,
            seq_len_kv: seq_len_kv_gpu,
            q_ragged_offset: q_ragged_offset_gpu if is_ragged else None,
            k_ragged_offset: k_ragged_offset_gpu if is_ragged else None,
            v_ragged_offset: v_ragged_offset_gpu if is_ragged else None,
            o_ragged_offset: o_ragged_offset_gpu if is_ragged else None,
        }

        if is_dropout:
            variant_pack[seed] = seed_gpu
            variant_pack[offset] = offset_gpu

        # Allocate workspace for the backward call.
        (workspace, ws_sep, _) = alloc_tensor(graph.get_workspace_size(), torch.uint8)

        # Display available memory.
        # torch.cuda.empty_cache()
        # free_mem, total_mem = torch.cuda.mem_get_info()
        # print(f"Free GPU memory (before backward): {free_mem / (1024**3):.4f} GB of {total_mem / (1024**3):.4f} GB")

        # Execute backward cuDNN graph.
        graph.execute(variant_pack, workspace, handle=cudnn_handle)
        torch.cuda.synchronize()

        if ws_sep is not None and not torch.all(ws_sep==-1).item():
            print("@@@@ Overall result: FAILED, backward workspace overwritten outside its boundaries.")
            print(ws_sep)
            pytest.fail("backward workspace overwritten outside boundaries", pytrace=False)

    bias_ref = None
    rng_dump_ref = None

    if not is_infer:
        # Using torch autograd reference in the backward pass.
        q_ref  = q_gpu.detach().float().requires_grad_()
        k_ref  = k_gpu.detach().float().requires_grad_()
        v_ref  = v_gpu.detach().float().requires_grad_()
        dO_ref = dO_gpu.detach().float()
        if is_ragged:
            dO_ref = convert_ragged_to_uniform(dO_ref, seq_len_q_gpu.detach())
        if is_bias:
            bias_ref = bias_gpu.detach().float().requires_grad_()
    else:
        # No autograd in the forward pass.
        q_ref  = q_gpu.detach().float()
        k_ref  = k_gpu.detach().float()
        v_ref  = v_gpu.detach().float()
        dO_ref = None
        if is_bias:
            bias_ref = bias_gpu.detach().float()

    if is_ragged:
        q_ref  = convert_ragged_to_uniform(q_ref, seq_len_q_gpu.detach())
        k_ref  = convert_ragged_to_uniform(k_ref, seq_len_kv_gpu.detach())
        v_ref  = convert_ragged_to_uniform(v_ref, seq_len_kv_gpu.detach())

    if is_padding:
        seq_len_q_ref = seq_len_q_gpu.detach().flatten()
        seq_len_kv_ref = seq_len_kv_gpu.detach().flatten()

    if is_dropout:
        rng_dump_ref = rng_dump_gpu.detach().float()

    # Compute forward reference output.
    ret = compute_ref(
        q_ref,
        k_ref,
        v_ref,
        attn_scale=attn_scale,
        bias=bias_ref,
        is_alibi=is_alibi,
        padding=(seq_len_q_ref, seq_len_kv_ref) if is_padding else None,
        left_bound=left_bound,
        right_bound=right_bound,
        diag_align=diag_align,
        dropout_prob=dropout_prob,
        dropout_mask=rng_dump_ref,
        generate_stats=(is_infer == False),
    )

    if not is_infer:
        o_ref, stats_ref = ret
    else:
        o_ref = ret

    if is_ragged:
        o_gpu = convert_ragged_to_uniform(o_gpu, seq_len_q_gpu.detach())

    err_count = 0

    if is_padding:
        # zero out padded region of the output for comparison
        for i, m in enumerate(seq_len_q_ref):
            o_ref[i, :, m:, :] = 0
            o_gpu[i, :, m:, :] = 0
            if is_infer == False:
                stats_ref[i, :, m:, :] = 0
                stats_gpu[i, :, m:, :] = 0

    diffs = int_cli_option(10, request, "--diffs")

    err_count += approx_equal(o_gpu, o_ref, o_sep, o_raw, atol=2e-2, rtol=2e-2, tag="o", disp_elems=diffs)

    if not is_infer:
        err_count += approx_equal(stats_gpu, stats_ref, stats_sep, stats_raw, atol=2e-2, rtol=2e-2, tag="stats", disp_elems=diffs)

        inputs_ref = [q_ref, k_ref, v_ref]
        if is_bias:
            inputs_ref.append(bias_ref)

        [dQ_ref, dK_ref, dV_ref, *opt_refs] = list(
            torch.autograd.grad(outputs=o_ref, inputs=inputs_ref, grad_outputs=dO_ref)
        )

        if is_bias:
            dBias_ref = opt_refs.pop(0)

        if is_ragged:
            dQ_gpu = convert_ragged_to_uniform(dQ_gpu, seq_len_q_gpu.detach())
            dK_gpu = convert_ragged_to_uniform(dK_gpu, seq_len_kv_gpu.detach())
            dV_gpu = convert_ragged_to_uniform(dV_gpu, seq_len_kv_gpu.detach())

        if is_padding:
            # zero out padded region of the output for comparison
            for i, (m, n) in enumerate(zip(seq_len_q_ref, seq_len_kv_ref)):
                dQ_ref[i, :, m:, :] = 0
                dQ_gpu[i, :, m:, :] = 0
                dK_ref[i, :, n:, :] = 0
                dK_gpu[i, :, n:, :] = 0
                dV_ref[i, :, n:, :] = 0
                dV_gpu[i, :, n:, :] = 0

        torch.cuda.synchronize()

        err_count += approx_equal(dQ_gpu, dQ_ref, dQ_sep, dQ_raw, atol=2e-2, rtol=2e-2, tag="dQ", disp_elems=diffs)
        err_count += approx_equal(dK_gpu, dK_ref, dK_sep, dK_raw, atol=2e-2 if data_type != torch.bfloat16 else 7e-2, rtol=2e-2, tag="dK", disp_elems=diffs)
        err_count += approx_equal(dV_gpu, dV_ref, dV_sep, dV_raw, atol=2e-2 if data_type != torch.bfloat16 else 7e-2, rtol=2e-2, tag="dV", disp_elems=diffs)
        if is_bias:
            err_count += approx_equal(dBias_gpu, dBias_ref, dBias_sep, dBias_raw, atol=2e-2, rtol=2e-2, tag="dBias", disp_elems=diffs)

    if err_count != 0:
        print("@@@@ Overall result: FAILED, disallowed mismatches")
        pytest.fail("disallowed mismatches", pytrace=False)
    else:
        print("@@@@ Overall result: PASSED, everything looks good!")

@pytest.fixture(scope="package")
def env_info(request):
    assert torch.cuda.is_available(), "no CUDA device"

    gpu_type = torch.cuda.get_device_capability()
    gpu_name = torch.cuda.get_device_name()
    sm_count = torch.cuda.get_device_properties().multi_processor_count

    gpu_arch     = f"SM_{gpu_type[0]}{gpu_type[1]}"
    gpu_info     = f"{sm_count} SM-s, {gpu_name}"
    cudnn_ver    = torch.backends.cudnn.version()
    blocked_file = str(request.path)
    blocked_file = blocked_file[:-3] + ".block"
    blocked_map  = fetch_blocked_tests(blocked_file)

    show_blocked_tests(blocked_map)

    return {"gpu_arch": gpu_arch, "gpu_info": gpu_info, "cudnn_ver": cudnn_ver, "blocked_map": blocked_map}


# ==================================
# L0 fprop tests
# ==================================

@pytest.mark.parametrize("test_no", tlist(num_tests=64, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.parametrize("data_type", data_type_options, ids=lambda p: str(p))
@pytest.mark.parametrize("layout", random_layout_options)
@pytest.mark.parametrize("head_group", head_group_options)
@pytest.mark.parametrize("is_infer", [True], ids=lambda p: "FWD" if p else "BWD")
@pytest.mark.L0
def test_sdpa_random_fwd(env_info, test_no, data_type, is_infer, head_group, layout, request, cudnn_handle):
    cfg = testConfig(**env_info)
    cfg.setBatches(max_batches=8)
    cfg.setSequences(max_s_q=1024, max_s_kv=1024)
    cfg.setVectors(max_d_v=128, max_d_qk=128)
    cfg.setHeads(max_h_qkv=8)
    cfg.setBlockSize(max_blk_sz=256)
    cfg.random_layout(test_no, is_infer, data_type, head_group, layout, knobNAR.NEVER, knobNA.ALWAYS, request)
    exec_sdpa(cfg, request, cudnn_handle)

# ==================================
# L0 bprop tests
# ==================================

@pytest.mark.parametrize("test_no", tlist(num_tests=64, rng_seed=123), ids=lambda p: f"test{p[0]}")
@pytest.mark.parametrize("data_type", data_type_options, ids=lambda p: str(p))
@pytest.mark.parametrize("layout", random_layout_options)
@pytest.mark.parametrize("head_group", head_group_options)
@pytest.mark.parametrize("is_infer", [False], ids=lambda p: "FWD" if p else "BWD")
@pytest.mark.L0
def test_sdpa_random_bwd(env_info, test_no, data_type, is_infer, head_group, layout, request, cudnn_handle):
    cfg = testConfig(**env_info)
    cfg.setBatches(max_batches=8)
    cfg.setSequences(max_s_q=512, max_s_kv=512)
    cfg.setVectors(max_d_v=160, max_d_qk=160)
    cfg.setHeads(max_h_qkv=8)
    cfg.setBlockSize(max_blk_sz=256)
    cfg.random_layout(test_no, is_infer, data_type, head_group, layout, knobNAR.NEVER, knobNA.ALWAYS, request)
    exec_sdpa(cfg, request, cudnn_handle)

# ==================================
# L0 fprop tests with s_q=1
# ==================================

@pytest.mark.parametrize("test_no", tlist(num_tests=16, rng_seed=741), ids=lambda p: f"test{p[0]}")
@pytest.mark.parametrize("data_type", data_type_options, ids=lambda p: str(p))
@pytest.mark.parametrize("layout", random_layout_options)
@pytest.mark.parametrize("head_group", head_group_options)
@pytest.mark.parametrize("is_infer", [True], ids=lambda p: "FWD_SQ1" if p else "BWD_SQ1")
@pytest.mark.L0
def test_sdpa_random_sq1(env_info, test_no, data_type, is_infer, head_group, layout, request, cudnn_handle):
    cfg = testConfig(**env_info)
    cfg.setBatches(max_batches=32)
    cfg.setSequences(max_s_q=1, max_s_kv=512)
    cfg.setVectors(max_d_v=128, max_d_qk=128)
    cfg.setHeads(max_h_qkv=32)
    cfg.setBlockSize(max_blk_sz=256)
    cfg.random_layout(test_no, is_infer, data_type, head_group, layout, knobNAR.NEVER, knobNA.ALWAYS, request)
    exec_sdpa(cfg, request, cudnn_handle)

# ==================================
# L1 bprop ragged tests
# ==================================

@pytest.mark.parametrize("test_no", tlist(num_tests=16, rng_seed=555), ids=lambda p: f"test{p[0]}")
@pytest.mark.parametrize("data_type", data_type_options, ids=lambda p: str(p))
@pytest.mark.parametrize("layout", random_layout_options)
@pytest.mark.parametrize("head_group", head_group_options)
@pytest.mark.parametrize("is_infer", [False], ids=lambda p: "FWD_RAGGED_" if p else "BWD_RAGGED_")
@pytest.mark.L1
def test_sdpa_random_bwd_ragged(env_info, test_no, data_type, is_infer, head_group, layout, request, cudnn_handle):
    cfg = testConfig(**env_info)
    cfg.setBatches(max_batches=8)
    cfg.setSequences(max_s_q=512, max_s_kv=512)
    cfg.setVectors(max_d_v=128, max_d_qk=128)
    cfg.setHeads(max_h_qkv=8)
    cfg.setBlockSize(max_blk_sz=256)
    cfg.random_layout(test_no, is_infer, data_type, head_group, layout, knobNAR.ALWAYS, knobNA.ALWAYS, request)
    exec_sdpa(cfg, request, cudnn_handle)

# ===================
# Single repro test
# ===================

@pytest.mark.skipif("not config.getoption('--repro')", reason="used with '--repro' only")
@pytest.mark.L0
@pytest.mark.L1
@pytest.mark.L2
@pytest.mark.L3
@pytest.mark.L4
def test_repro(env_info, request, cudnn_handle):
    repro_str = request.config.getoption("--repro")
    cfg = testConfig(**env_info)
    cfg.load_config(repro_str)
    cfg.showConfig((1,1), request, False)
    exec_sdpa(cfg, request, cudnn_handle)

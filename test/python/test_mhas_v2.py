"""
This test harness allows for testing the various options of the attention operator. See example usage under "main" below.

The full documentation on the attention operator can be found in: https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md#scaled-dot-product-attention

Notebooks that demonstrate the attention operator can be found here:
- Introductory example: https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/python/50_scaled_dot_product_attention.ipynb
- Example with paged caches: https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/python/samples/python/52_scaled_dot_product_attention_with_paged_caches.ipynb
- Work in progress

The recommended way to run those tests:
> pytest -vv -s -rA --tb=short this_file.py
"""

import cudnn
import pytest
import random
import torch
import math
import os
import sys
from looseversion import LooseVersion

# fmt: off

if __name__ == "__main__":
    print("This is pytest script.")
    sys.exit(0)

data_type_options      = [torch.float16, torch.bfloat16]
head_group_options     = ["MHA", "GQA", "MQA"]
fixed_layout_options   = ["bshd_bshd_bshd", "bs3hd", "sbh3d"]
random_layout_options  = ["edge_random", "inner_random"]

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

def get_layout_strides(shape, indices = [0, 1, 2, 3], gaps = [0, 0, 0, 0]):
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
    total_size = shape[j] * curr_stride
    return tuple(strides), tuple(gaps), total_size

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
            print(f"Total {mismatch_cnt} mismatches in {num_elements} elements when validating '{tag}' (first {count} mismatches displayed)")
        else:
            print(f"Total {mismatch_cnt} mismatches in {num_elements} elements when validating '{tag}' results")
    else:
        print(f"Numerical divergence of '{tag}' is within limits")

    # Check if areas before and after the tensor were overwritten (treated as one numerical mismatch).
    if sepbuf is not None and not torch.all(torch.isnan(sepbuf)).item():
        print(f"ERROR: buffer '{tag}' overwritten outside its boundaries")
        mismatch_cnt += 1

    # Check if unused elements of the tensor were overwritten (treated as one numerical mismatch).
    # Note that this check destroys computed data (overwrites them with NaN-s).
    if rawbuf is not None:
        actual.fill_(float('nan'))
        if not torch.all(torch.isnan(rawbuf)).item():
            print(f"ERROR: unused gaps of '{tag}' tensor were overwritten")
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
    bool_map = {"True": True, "False": False}
    str_val = request.config.getoption(cli_opt)
    val = bool_map.get(str_val)
    return val if type(val) == bool else org_val

class testConfig:
    # To prevent creation of misspelled variables, listing all local variables of the class.
    __slots__ = ['rng_geom', 'geom_seed', 'rng_data', 'data_seed', 'gpu_info', 'min_batches', 'max_batches', 
                 'min_s_q', 'max_s_q', 'min_s_kv', 'max_s_kv', 'min_d_qk', 'max_d_qk', 'min_d_v', 'max_d_v', 
                 'min_h_qkv', 'max_h_qkv', 'min_blk_sz', 'max_blk_sz', 'head_group', 'is_infer', 'is_causal', 
                 'is_alibi', 'is_paged', 'is_bias', 'is_padding', 'is_causal_br', 'is_sliding_w', 'is_dropout', 
                 'is_ragged', 'is_determin', 'data_type', 'batches', 'd_qk', 'd_v', 's_q', 's_kv', 
                 'h_q', 'h_k', 'h_v', 'block_size', 'in_layout', 'out_layout', 'shape_q', 'gaps_q', 
                 'stride_q', 'offset_q', 'elems_q', 'shape_k', 'gaps_k', 'stride_k', 'offset_k', 'elems_k', 
                 'shape_v', 'gaps_v', 'stride_v', 'offset_v', 'elems_v', 'shape_o', 'gaps_o', 'stride_o', 'elems_o']

    def __init__(self):
        assert torch.cuda.is_available(), "no CUDA device"

        self.rng_geom    = random.Random()
        self.geom_seed   = None

        self.rng_data    = torch.Generator(device="cuda")
        self.data_seed   = None

        try:
            gpu_name = torch.cuda.get_device_name()
            gpu_type = torch.cuda.get_device_capability()
            sm_count = torch.cuda.get_device_properties().multi_processor_count
            self.gpu_info = f"SM_{gpu_type[0]}{gpu_type[1]} ({sm_count} SM-s, {gpu_name})"
        except Exception as e:
            self.gpu_info = f"{e}"

        self.min_batches = self.max_batches = 1
        self.min_s_q     = self.max_s_q     = 1
        self.min_s_kv    = self.max_s_kv    = 1
        self.min_d_qk    = self.max_d_qk    = 1
        self.min_d_v     = self.max_d_v     = 1
        self.min_h_qkv   = self.max_h_qkv   = 1
        self.min_blk_sz  = self.max_blk_sz  = 1

        self.head_group   = None
        self.is_causal    = None
        self.is_alibi     = None
        self.is_infer     = None
        self.is_paged     = None
        self.is_bias      = None
        self.is_padding   = None
        self.is_causal_br = None
        self.is_sliding_w = None
        self.is_dropout   = None
        self.is_ragged    = None
        self.is_determin  = None
        self.data_type    = None

        self.batches    = None
        self.d_qk       = None
        self.d_v        = None
        self.s_q        = None
        self.s_kv       = None
        self.h_q        = None
        self.h_k        = None
        self.h_v        = None
        self.block_size = None
        self.in_layout  = None
        self.out_layout = None

        self.shape_q    = None
        self.gaps_q     = None
        self.stride_q   = None
        self.offset_q   = None
        self.elems_q    = None

        self.shape_k    = None
        self.gaps_k     = None
        self.stride_k   = None
        self.offset_k   = None
        self.elems_k    = None

        self.shape_v    = None
        self.gaps_v     = None
        self.stride_v   = None
        self.offset_v   = None
        self.elems_v    = None

        self.shape_o    = None
        self.gaps_o     = None
        self.stride_o   = None
        self.elems_o    = None

    def config_str(self):
        banned = ("max_", "min_", "gpu_", "rng_")
        stg = ""
        for k in self.__slots__:
            if k.startswith(banned):
                continue
            v = getattr(self, k)
            if type(v) == str:
                assert len(v) > 0, f"ERROR: empty string in {k}='{v}'"
                stg += f"{k}='{v}':"
            else:
                assert v != None, f"ERROR: invalid value in '{k}={v}'"
                stg += f"{k}={v}:"
        stg = "".join(stg.split())  # remove whitespaces
        stg = stg[:-1]  # remove last ':' character
        return stg

    def load_config(self, code):
        print(f"Loading config: '{code}'")
        code = "".join(code.split())  # remove whitespaces
        for assign in filter(None, code.split(":")):
            code_to_run = "self." + assign
            try:
                exec(code_to_run)
            except Exception as e:
                assert False, f"ERROR: {e} in '{assign}'"
        banned = ("max_", "min_", "gpu_", "rng_")
        for k in self.__slots__:
            if k.startswith(banned):
                continue
            v = getattr(self, k)
            if v == None and not (k.startswith("max") or k.startswith("min")):
                assert False, f"ERROR: config value '{k}' not set"

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

    def showConfig(self, test_no, request, reg_run=True):
        if request.config.option.dryrun == 0:
            print(f"\nTest #{test_no[0]} of {test_no[1]}")
            print(f"test_name    = {request.node.name}")
            print(f"geom_seed    = {self.geom_seed}")
            print(f"data_seed    = {self.data_seed}")
            print(f"gpu_info     = {self.gpu_info}")
            print(f"head_group   = {self.head_group}")
            print(f"layout       = {self.in_layout}->{self.out_layout}")
            print(f"batches      = {self.batches}")
            print(f"d_qk         = {self.d_qk}")
            print(f"d_v          = {self.d_v}")
            print(f"s_q          = {self.s_q}")
            print(f"s_kv         = {self.s_kv}")
            print(f"h_q          = {self.h_q}")
            print(f"h_k          = {self.h_k}")
            print(f"h_v          = {self.h_v}")
            print(f"shape_q      = {self.shape_q}, gaps_q={self.gaps_q}")
            print(f"shape_k      = {self.shape_k}, gaps_k={self.gaps_k}")
            print(f"shape_v      = {self.shape_v}, gaps_v={self.gaps_v}")
            print(f"shape_o      = {self.shape_o}, gaps_o={self.gaps_o}")
            print(f"stride_q     = {self.stride_q}, elems_q={self.elems_q:,}")
            print(f"stride_k     = {self.stride_k}, elems_k={self.elems_k:,}")
            print(f"stride_v     = {self.stride_v}, elems_v={self.elems_v:,}")
            print(f"stride_o     = {self.stride_o}, elems_o={self.elems_o:,}")
            print(f"is_infer     = {self.is_infer}")
            print(f"is_causal    = {self.is_causal}")
            print(f"is_alibi     = {self.is_alibi}")
            print(f"is_paged     = {self.is_paged} (block_size={self.block_size})")
            print(f"is_bias      = {self.is_bias}")
            print(f"is_padding   = {self.is_padding}")
            print(f"is_ragged    = {self.is_ragged}")
            print(f"is_causal_br = {self.is_causal_br}")
            print(f"is_sliding_w = {self.is_sliding_w}")
            print(f"is_dropout   = {self.is_dropout}")
            print(f"is_determin  = {self.is_determin}")
            print(f"data_type    = {self.data_type}")
            if reg_run:
                cmd_opts = f"--geom_seed {self.geom_seed} --data_seed {self.data_seed}"
                print(f"repro_cmd    = {request.node.path.name}::{request.node.name} {cmd_opts}")
        elif request.config.option.dryrun == 1:
            print(f"\npytest -vv -s -rA --tb=short {request.module.__file__}::{request.node.name} --geom_seed {self.geom_seed} --data_seed {self.data_seed}")
        elif request.config.option.dryrun == 2:
            print(f"\npytest -vv -s -rA --tb=short {request.module.__file__}::test_repro --repro \"{self.config_str()}\"")
        else:
            assert False, "wrong --dryrun command line option"

    def fixed_layout(self, test_no, is_infer, data_type, head_group, in_layout, is_causal, request):
        assert data_type in data_type_options, "wrong data type"
        assert head_group in head_group_options, "wrong head group"
        assert in_layout in fixed_layout_options, "wrong layout"

        # Get the initial seed from the 'test_no' sequence. Add to it the test name hash to generate unique RNG seed.
        self.geom_seed = test_no[2] + tname_hash(request.node.name)
        self.data_seed = test_no[2]

        # Overwrite RNG seed from the command line.
        self.geom_seed = int(request.config.option.geom_seed) if request.config.option.geom_seed != None else self.geom_seed
        self.data_seed = int(request.config.option.data_seed) if request.config.option.data_seed != None else self.data_seed

        self.rng_geom.seed(self.geom_seed)
        self.rng_data.manual_seed(self.data_seed)

        self.head_group   = head_group
        self.data_type    = data_type
        self.in_layout    = in_layout

        self.is_infer     = is_infer
        self.is_causal    = is_causal
        self.is_alibi     = self.rng_geom.choice([True, False]) if self.is_causal else False  # ALiBi mask requires is_causal
        self.is_paged     = self.rng_geom.choice([True, False]) if self.is_infer else False
        self.is_bias      = self.rng_geom.choice([True, False])
        self.is_padding   = False
        self.is_ragged    = False
        self.is_causal_br = self.rng_geom.choice([True, False])
        self.is_sliding_w = self.rng_geom.choice([True, False])
        self.is_dropout   = self.rng_geom.choice([True, False])
        self.is_determin  = self.rng_geom.choice([True, False]) if not self.is_infer else True   # TODO: what is this

        # Bottom right causal mask is only supported with is_bias=False, is_alibi=False, is_dropout=False.
        if self.is_causal_br and (self.is_bias != False or self.is_alibi != False or self.is_dropout != False):
            self.is_causal_br = False

        # Sliding window attention is only supported with is_causal=True, is_dropout=False, is_bias=False.
        if self.is_sliding_w and (self.is_causal != True or self.is_dropout != False or self.is_bias != False):
            self.is_sliding_w = False

        # The is_causal_br=True and is_causal=True settings cannot be both enabled.
        if self.is_causal and self.is_causal_br:
            self.is_causal_br = False

        # Ragged mode (packed variable sequence length) is only tested with thd_thd_thd and t3hd.
        if self.is_ragged and not (self.in_layout == "bshd_bshd_bshd" or self.layout == "bs3hd"):
            self.is_ragged = False

        # Paged caches can only be used in combination with padding mask (variable sequence length).
        if self.is_paged and not self.is_padding:
            self.is_paged = False

        # Paged caches cannot be used with ragged offsets (packed variable sequence lengths).
        if self.is_paged and self.is_ragged:
            self.is_paged = False

        # Paged attention is only tested bshd_bshd_bshd.
        if self.is_paged and not self.in_layout == "bshd_bshd_bshd":
            self.is_paged = False

        # Overwrite all boolean varaibles from the command line including 'is_infer' and 'is_causal'.
        self.is_infer     = bool_cli_option(self.is_infer, request, "--mha_is_infer")
        self.is_causal    = bool_cli_option(self.is_causal, request, "--mha_is_causal")
        self.is_alibi     = bool_cli_option(self.is_alibi, request, "--mha_is_alibi")
        self.is_paged     = bool_cli_option(self.is_paged, request, "--mha_is_paged")
        self.is_bias      = bool_cli_option(self.is_bias, request, "--mha_is_bias")
        self.is_padding   = bool_cli_option(self.is_padding, request, "--mha_is_padding")
        self.is_ragged    = bool_cli_option(self.is_ragged, request, "--mha_is_ragged")
        self.is_causal_br = bool_cli_option(self.is_causal_br, request, "--mha_is_causal_br")
        self.is_sliding_w = bool_cli_option(self.is_sliding_w, request, "--mha_is_sliding_w")
        self.is_dropout   = bool_cli_option(self.is_dropout, request, "--mha_is_dropout")
        self.is_determin  = bool_cli_option(self.is_determin, request, "--mha_is_determin")

        # Ragged tensor is only tested with packed variable length tensors.
        assert self.is_ragged != True or self.is_padding != False, "is_ragged=True and is_padding=False not allowed"

        self.batches = self.rng_geom.randint(self.min_batches, self.max_batches)

        self.s_q = self.rng_geom.choice(get_powers_of_two(self.min_s_q, self.max_s_q))
        if self.in_layout == "bshd_bshd_bshd":
            self.s_kv = self.rng_geom.choice(get_powers_of_two(self.min_s_kv, self.max_s_kv))
        else:
            self.s_kv = self.s_q

        # Sliding window attention is not supported with s_q > s_kv.
        if self.is_sliding_w and (self.s_q > self.s_kv):
            self.is_sliding_w = False

        # When is_causal_br=True, s_q and s_kv have to be multiple of 64 and s_q <= s_kv.
        if self.is_causal_br and (self.s_q > self.s_kv or self.s_q % 64 != 0 or self.s_kv % 64 != 0):
            range_lo = max(self.min_s_q, self.min_s_kv)
            range_lo = (range_lo + 63) // 64 * 64  # include first multiple of 64
            range_hi = min(self.max_s_q, self.max_s_kv)
            if range_lo <= range_hi and self.in_layout == "bshd_bshd_bshd":
                self.s_kv = self.rng_geom.choice(get_multiples_of(64, range_lo, range_hi))
                self.s_q  = self.rng_geom.choice(get_multiples_of(64, range_lo, self.s_kv))
            else:
                self.is_causal_br = False

        # Overwrite batches, s_q, s_kv from command line arguments.
        self.batches = int(request.config.option.mha_batches) if request.config.option.mha_batches != None else self.batches
        self.s_q = int(request.config.option.mha_s_q) if request.config.option.mha_s_q != None else self.s_q
        self.s_kv = int(request.config.option.mha_s_kv) if request.config.option.mha_s_kv != None else self.s_kv
    
        self.d_qk = self.rng_geom.choice(get_multiples_of(8, self.min_d_qk, self.max_d_qk))
        if (self.in_layout == "bshd_bshd_bshd" and not self.is_ragged):
            self.d_v = self.rng_geom.choice(get_multiples_of(8, self.min_d_v, self.max_d_v))
        else:
            self.d_v = self.d_qk

        # Overwrite d_qk, d_v from command line arguments.
        self.d_qk = int(request.config.option.mha_d_qk) if request.config.option.mha_d_qk != None else self.d_qk
        self.d_v = int(request.config.option.mha_d_v) if request.config.option.mha_d_v != None else self.d_v

        self.h_q = self.rng_geom.randint(self.min_h_qkv, self.max_h_qkv)
        if self.head_group == "MHA":
            self.h_k = self.h_q
            self.h_v = self.h_q
        elif self.head_group == "GQA":
            h_kv_sizes = get_all_divisers(self.h_q)
            self.h_k = self.rng_geom.choice(h_kv_sizes)
            self.h_v = self.rng_geom.choice(h_kv_sizes) if self.in_layout == "bshd_bshd_bshd" else self.h_k
        elif self.head_group == "MQA":
            self.h_k = 1
            self.h_v = 1
        else:
            assert False, "wrong attention flavor"

        # Overwrite h_q, h_k, h_v from command line arguments.
        self.h_q = int(request.config.option.mha_h_q) if request.config.option.mha_h_q != None else self.h_q
        self.h_k = int(request.config.option.mha_h_k) if request.config.option.mha_h_k != None else self.h_k
        self.h_v = int(request.config.option.mha_h_v) if request.config.option.mha_h_v != None else self.h_v

        # Block size for paged attention in fprop (must be power of 2 and minimum 1).
        if self.is_infer and self.is_paged:
            self.block_size = self.rng_geom.choice(get_powers_of_two(self.min_blk_sz, self.max_blk_sz))
        else:
            self.block_size = 0

        # Overwrite block_size from command line.
        self.block_size = int(request.config.option.mha_block_size) if request.config.option.mha_block_size != None else self.block_size

        # Generator for layout combinations
        #
        # | in_layout       | GQA             |
        # |-----------------|-----------------|
        # | bshd_bshd_bshd  | bshd_bshd_bshd  |
        # | bs3hd           | bshd_bs2hd      |
        # | sbh3d           | sbhd_sbh2d      |

        batches = self.batches
        d_qk    = self.d_qk
        d_v     = self.d_v
        s_q     = self.s_q
        s_kv    = self.s_kv
        h_q     = self.h_q
        h_k     = self.h_k
        h_v     = self.h_v

        self.shape_q = (batches, h_q, s_q, d_qk)
        self.shape_k = (batches, h_k, s_kv, d_qk)
        self.shape_v = (batches, h_v, s_kv, d_v)
        self.shape_o = (batches, h_q, s_q, d_v)

        if self.in_layout == "bshd_bshd_bshd":
            self.stride_q = (s_q * h_q * d_qk, d_qk, h_q * d_qk, 1)
            self.stride_k = (s_kv * h_k * d_qk, d_qk, h_k * d_qk, 1)
            self.stride_v = (s_kv * h_v * d_v, d_v, h_v * d_v, 1)
            self.stride_o = (s_q * h_q * d_v, d_v, h_q * d_v, 1)
            self.offset_q = 0
            self.offset_k = self.offset_q + batches * s_q * h_q * d_qk
            self.offset_v = self.offset_k + batches * s_kv * h_k * d_qk
            self.out_layout = 'bshd'
        elif self.in_layout == "bs3hd":
            if self.head_group == "MHA":
                # bs3hd
                assert (h_q == h_k == h_v) and (s_q == s_kv) and (d_qk == d_v)
                self.stride_q = (s_q * 3 * h_q * d_qk, d_qk, 3 * h_q * d_qk, 1)
                self.stride_k = (s_q * 3 * h_q * d_qk, d_qk, 3 * h_q * d_qk, 1)
                self.stride_v = (s_q * 3 * h_q * d_qk, d_qk, 3 * h_q * d_qk, 1)
                self.stride_o = (s_q * h_q * d_qk, d_qk, h_q * d_qk, 1)
                self.offset_q = 0
                self.offset_k = self.offset_q + h_q * d_qk
                self.offset_v = self.offset_k + h_q * d_qk
            else:
                # bshd_bs2hd
                assert (h_k == h_v) and (s_q == s_kv) and (d_qk == d_v)
                self.stride_q = (s_q * h_q * d_qk, d_qk, h_q * d_qk, 1)
                self.stride_k = (s_q * 2 * h_k * d_qk, d_qk, 2 * h_k * d_qk, 1)
                self.stride_v = (s_q * 2 * h_k * d_qk, d_qk, 2 * h_k * d_qk, 1)
                self.stride_o = (s_q * h_q * d_qk, d_qk, h_q * d_qk, 1)
                self.offset_q = 0
                self.offset_k = self.offset_q + s_q * batches * h_q * d_qk
                self.offset_v = self.offset_k + h_k * d_qk
            self.out_layout = 'bshd'
        elif self.in_layout == "sbh3d":
            if self.head_group == "MHA":
                # sbh3d
                assert (h_q == h_k == h_v) and (s_q == s_kv) and (d_qk == d_v)
                self.stride_q = (h_q * 3 * d_qk, 3 * d_qk, batches * h_q * 3 * d_qk, 1)
                self.stride_k = (h_q * 3 * d_qk, 3 * d_qk, batches * h_q * 3 * d_qk, 1)
                self.stride_v = (h_q * 3 * d_qk, 3 * d_qk, batches * h_q * 3 * d_qk, 1)
                self.stride_o = (h_q * d_qk, d_qk, batches * h_q * d_qk, 1)
                self.offset_q = 0
                self.offset_k = self.offset_q + d_qk
                self.offset_v = self.offset_k + d_qk
            else:
                # sbhd_sbh2d
                assert (h_k == h_v) and (s_q == s_kv) and (d_qk == d_v)
                self.stride_q = (h_q * d_qk, d_qk, batches * h_q * d_qk, 1)
                self.stride_k = (h_k * 2 * d_qk, 2 * d_qk, batches * h_k * 2 * d_qk, 1)
                self.stride_v = (h_k * 2 * d_qk, 2 * d_qk, batches * h_k * 2 * d_qk, 1)
                self.stride_o = (h_q * d_qk, d_qk, batches * h_q * d_qk, 1)
                self.offset_q = 0
                self.offset_k = self.offset_q + s_q * batches * h_q * d_qk
                self.offset_v = self.offset_k + d_qk
            self.out_layout = 'sbhd'
        else:
            assert False, "layout must be 'bshd_bshd_bshd', 'bs3hd', or 'sbh3d'"

        # Compute dense tensor sizes in elements.
        self.gaps_q = self.gaps_k = self.gaps_v = self.gaps_o = (0, 0, 0, 0)
        self.elems_q = math.prod(self.shape_q)
        self.elems_k = math.prod(self.shape_k)
        self.elems_v = math.prod(self.shape_v)
        self.elems_o = math.prod(self.shape_o)

        self.showConfig(test_no, request)

    def random_layout(self, test_no, is_infer, data_type, head_group, layout_type, is_causal, request):
        assert data_type in data_type_options, "wrong data type"
        assert head_group in head_group_options, "wrong head group"
        assert layout_type in random_layout_options, "wrong layout type"

        # Get the initial seed from the 'test_no' sequence. Add to it the test name hash to generate unique RNG seed.
        self.geom_seed = test_no[2] + tname_hash(request.node.name)
        self.data_seed = test_no[2]

        # Overwrite RNG seeds from the command line.
        self.geom_seed = int(request.config.option.geom_seed) if request.config.option.geom_seed != None else self.geom_seed
        self.data_seed = int(request.config.option.data_seed) if request.config.option.data_seed != None else self.data_seed

        self.rng_geom.seed(self.geom_seed)
        self.rng_data.manual_seed(self.data_seed)

        self.head_group     = head_group
        self.data_type      = data_type

        self.is_infer     = is_infer
        self.is_causal    = is_causal
        self.is_alibi     = self.rng_geom.choice([True, False]) if self.is_causal else False  # ALiBi mask requires is_causal
        self.is_paged     = self.rng_geom.choice([True, False]) if self.is_infer else False
        self.is_bias      = self.rng_geom.choice([True, False])
        self.is_padding   = False
        self.is_ragged    = False
        self.is_causal_br = self.rng_geom.choice([True, False])
        self.is_sliding_w = self.rng_geom.choice([True, False])
        self.is_dropout   = self.rng_geom.choice([True, False])
        self.is_determin  = self.rng_geom.choice([True, False]) if not self.is_infer else True  # TODO: what is this

        # Bottom right causal mask is only supported with is_bias=False, is_alibi=False, is_dropout=False.
        if self.is_causal_br and (self.is_bias != False or self.is_alibi != False or self.is_dropout != False):
            self.is_causal_br = False

        # Sliding window attention is only supported with is_causal=True, is_dropout=False, is_bias=False.
        if self.is_sliding_w and (self.is_causal != True or self.is_dropout != False or self.is_bias != False):
            self.is_sliding_w = False

        # The is_causal_br=True and is_causal=True settings cannot be both enabled.
        if self.is_causal and self.is_causal_br:
            self.is_causal_br = False

        # TODO: Ragged mode (packed variable sequence length) is only tested with some layouts.
        # TODO: need to figure out which layouts out of 216 choices are supported.
        # if self.is_ragged and not (self.in_layout == "bshd_bshd_bshd"):
        #     self.is_ragged = False

        # Paged caches can only be used in combination with padding mask (variable sequence length).
        if self.is_paged and not self.is_padding:
            self.is_paged = False

        # Paged caches cannot be used with ragged offsets (packed variable sequence lengths).
        if self.is_paged and self.is_ragged:
            self.is_paged = False

        # TODO: Paged attention is only tested with some layouts.
        # TODO: need to figure out which layouts out of 216 choices are supported.
        # if self.is_paged and not self.in_layout == "bshd_bshd_bshd":
        #     self.is_paged = False

        # Overwrite all boolean varaibles from the command line including 'is_infer' and 'is_causal'.
        self.is_infer     = bool_cli_option(self.is_infer, request, "--mha_is_infer")
        self.is_causal    = bool_cli_option(self.is_causal, request, "--mha_is_causal")
        self.is_alibi     = bool_cli_option(self.is_alibi, request, "--mha_is_alibi")
        self.is_paged     = bool_cli_option(self.is_paged, request, "--mha_is_paged")
        self.is_bias      = bool_cli_option(self.is_bias, request, "--mha_is_bias")
        self.is_padding   = bool_cli_option(self.is_padding, request, "--mha_is_padding")
        self.is_ragged    = bool_cli_option(self.is_ragged, request, "--mha_is_ragged")
        self.is_causal_br = bool_cli_option(self.is_causal_br, request, "--mha_is_causal_br")
        self.is_sliding_w = bool_cli_option(self.is_sliding_w, request, "--mha_is_sliding_w")
        self.is_dropout   = bool_cli_option(self.is_dropout, request, "--mha_is_dropout")
        self.is_determin  = bool_cli_option(self.is_determin, request, "--mha_is_determin")

        # Ragged tensor is only tested with packed variable length tensors.
        assert self.is_ragged != True or self.is_padding != False, "is_ragged=True and is_padding=False not allowed"

        if layout_type == "edge_random":
            self.batches = self.max_batches
            self.s_q = self.max_s_q
            self.s_kv = self.max_s_kv
        elif layout_type == "inner_random":
            self.batches = self.rng_geom.randint(1, self.max_batches)
            self.s_q = self.rng_geom.randint(1, self.max_s_q)
            draw = self.rng_geom.random()
            if (draw < 0.707 and self.s_q <= self.max_s_kv):
                self.s_kv = self.s_q
            else:
                self.s_kv = self.rng_geom.randint(1, self.max_s_kv)
        else:
            assert False, "wrong layout type"

        # Sliding window attention is not supported with s_q > s_kv.
        if self.is_sliding_w and (self.s_q > self.s_kv):
            self.is_sliding_w = False

        # When is_causal_br=True, s_q and s_kv have to be multiple of 64 and s_q <= s_kv.
        if self.is_causal_br and (self.s_q > self.s_kv or self.s_q % 64 != 0 or self.s_kv % 64 != 0):
            range_lo = max(self.min_s_q, self.min_s_kv)
            range_lo = (range_lo + 63) // 64 * 64  # include first multiple of 64
            range_hi = min(self.max_s_q, self.max_s_kv)
            if range_lo <= range_hi and layout_type == "inner_random":
                self.s_kv = self.rng_geom.choice(get_multiples_of(64, range_lo, range_hi))
                self.s_q  = self.rng_geom.choice(get_multiples_of(64, range_lo, self.s_kv))
            else:
                self.is_causal_br = False

        # Overwrite batches, s_q, s_kv from command line arguments.
        self.batches = int(request.config.option.mha_batches) if request.config.option.mha_batches != None else self.batches
        self.s_q = int(request.config.option.mha_s_q) if request.config.option.mha_s_q != None else self.s_q
        self.s_kv = int(request.config.option.mha_s_kv) if request.config.option.mha_s_kv != None else self.s_kv

        # Make sure all Q,K,V vectors in a tensor are aliagned to 16 bytes.
        # For dense tensors, vectors should be divisible into 16B chunks.
        tmp = torch.tensor([1.0], dtype=data_type)
        elem_align = int(16 / tmp.element_size())

        assert self.max_d_qk >= elem_align, "Value max_d_qk too small"
        assert self.max_d_v >= elem_align, "Value max_d_v too small"

        if layout_type == "edge_random":
            self.d_qk = self.max_d_qk
            self.d_v = self.max_d_v
        elif layout_type == "inner_random":
            self.d_qk = self.rng_geom.randint(elem_align, self.max_d_qk)
            self.d_qk = round_down(self.d_qk, elem_align)
            draw = self.rng_geom.random()
            if (draw < 0.5 and self.d_qk <= self.max_d_v):
                self.d_v = self.d_qk
            else:
                self.d_v = self.rng_geom.randint(elem_align, self.max_d_v)
                self.d_v = round_down(self.d_v, elem_align)
        else:
            assert False, "wrong layout type"

        # Overwrite d_qk, d_v from command line arguments.
        self.d_qk = int(request.config.option.mha_d_qk) if request.config.option.mha_d_qk != None else self.d_qk
        self.d_v = int(request.config.option.mha_d_v) if request.config.option.mha_d_v != None else self.d_v

        if (layout_type == "edge_random"):
            self.h_q = self.max_h_qkv
            self.h_k = self.h_q
            self.h_v = self.h_q
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
                assert False, "wrong attention flavor"
        else:
            assert False, "wrong layout type"

        # Overwrite h_q, h_k, h_v from command line arguments.
        self.h_q = int(request.config.option.mha_h_q) if request.config.option.mha_h_q != None else self.h_q
        self.h_k = int(request.config.option.mha_h_k) if request.config.option.mha_h_k != None else self.h_k
        self.h_v = int(request.config.option.mha_h_v) if request.config.option.mha_h_v != None else self.h_v

        # Block size for paged attention in fprop (must be power of 2 and minimum 1).
        if self.is_infer and self.is_paged:
            self.block_size = self.rng_geom.choice(get_powers_of_two(self.min_blk_sz, self.max_blk_sz))
        else:
            self.block_size = 0

        # Overwrite block_size from command line.
        self.block_size = int(request.config.option.mha_block_size) if request.config.option.mha_block_size != None else self.block_size

        # Using the 'bhsd' order for shape.
        self.shape_q = (self.batches, self.h_q, self.s_q, self.d_qk)
        self.shape_k = (self.batches, self.h_k, self.s_kv, self.d_qk)
        self.shape_v = (self.batches, self.h_v, self.s_kv, self.d_v)
        self.shape_o = (self.batches, self.h_q, self.s_q, self.d_v)

        # Q strides, permute first three dimensions in the original layout 'bhsd'.
        base_indices = [0, 1, 2]
        self.rng_geom.shuffle(base_indices)
        base_indices.append(3)
        gaps = [0, 0, 0, 0]
        draw = self.rng_geom.random()
        if (draw < 0.5):
            gaps = [self.rng_geom.randint(0, 8) for _ in range(3)]
            gaps.append(elem_align * self.rng_geom.randint(0, 2))
        (self.stride_q, self.gaps_q, self.elems_q) = get_layout_strides(self.shape_q, base_indices, gaps)
        self.in_layout = get_layout_name("bhsd", base_indices) + '_'

        # For K strides, decide with some probability if a new layout should be used.
        indices = base_indices
        draw = self.rng_geom.random()
        if (draw < 0.707):
            indices = [0, 1, 2]
            self.rng_geom.shuffle(indices)
            indices.append(3)
        gaps = [0, 0, 0, 0]
        draw = self.rng_geom.random()
        if (draw < 0.5):
            gaps = [self.rng_geom.randint(0, 8) for _ in range(3)]
            gaps.append(elem_align * self.rng_geom.randint(0, 2))
        (self.stride_k, self.gaps_k, self.elems_k) = get_layout_strides(self.shape_k, indices, gaps)
        self.in_layout += get_layout_name("bhsd", indices) + '_'

        # For V strides, decide with some probability if a new layout should be used.
        indices = base_indices
        draw = self.rng_geom.random()
        if (draw < 0.707):
            indices = [0, 1, 2]
            self.rng_geom.shuffle(indices)
            indices.append(3)
        gaps = [0, 0, 0, 0]
        draw = self.rng_geom.random()
        if (draw < 0.5):
            gaps = [self.rng_geom.randint(0, 8) for _ in range(3)]
            gaps.append(elem_align * self.rng_geom.randint(0, 2))
        (self.stride_v, self.gaps_v, self.elems_v) = get_layout_strides(self.shape_v, indices, gaps)
        self.in_layout += get_layout_name("bhsd", indices)

        # Q, K, V buffers are not interleaved.
        self.offset_q = 0
        self.offset_k = self.offset_q + self.elems_q
        self.offset_v = self.offset_k + self.elems_k

        # For O strides, decide with some probability if a new layout should be used
        indices = base_indices
        draw = self.rng_geom.random()
        if (draw < 0.5):
            indices = [0, 1, 2]
            self.rng_geom.shuffle(indices)
            indices.append(3)
        gaps = [0, 0, 0, 0]
        draw = self.rng_geom.random()
        if (draw < 0.5):
            gaps = [self.rng_geom.randint(0, 8) for _ in range(3)]
            gaps.append(elem_align * self.rng_geom.randint(0, 2))
        (self.stride_o, self.gaps_o, self.elems_o) = get_layout_strides(self.shape_o, indices, gaps)
        self.out_layout = get_layout_name("bhsd", indices)

        self.showConfig(test_no, request)

def compute_ref(
    q,
    k,
    v,
    attn_scale=1.0,
    bias=None,
    is_alibi=False,
    padding=None,
    is_causal=False,
    is_causal_br=False,
    sliding_window_length=None,
    dropout_prob=0.0,
    dropout_mask=None,
    compute_stats=False,
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

    if is_causal_br:
        causal_mask_bottom_right_zero = torch.ones(
            1, 1, s_q, 1, dtype=torch.bool, device=device
        )
        causal_mask_bottom_right_zero[:, :, : s_q - s_kv, :] = False

    if sliding_window_length is not None:
        swa_mask_zero = torch.ones(1, 1, s_q, 1, dtype=torch.bool, device=device)
        swa_mask_zero[:, :, s_kv + sliding_window_length - 1 :, :] = False
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
    if is_causal:
        causal_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
        causal_mask.triu_(diagonal=1)
        s = s.masked_fill(causal_mask, float("-inf"))
    if is_causal_br:
        causal_mask_bottom_right = None
        if padding:
            causal_mask_bottom_right = torch.ones(
                b, 1, s_q, s_kv, dtype=torch.bool, device=device
            )
            seq_len_q, seq_len_kv = padding
            for i in range(b):
                causal_mask_bottom_right[i, :, :, :].triu_(
                    diagonal=seq_len_kv[i] - seq_len_q[i] + 1
                )
        else:
            causal_mask_bottom_right = torch.ones(
                s_q, s_kv, dtype=torch.bool, device=device
            )
            causal_mask_bottom_right.triu_(diagonal=s_kv - s_q + 1)
        s = s.masked_fill(causal_mask_bottom_right, float("-inf"))
    if sliding_window_length is not None:
        assert is_causal == True
        swa_mask = torch.ones(s_q, s_kv, dtype=torch.bool, device=device)
        swa_mask.tril_(diagonal=-1 * sliding_window_length)
        swa_mask &= swa_mask_zero.view(s_q, 1)
        s = s.masked_fill(swa_mask, float("-inf"))

    p = torch.softmax(s, dim=-1)

    if sliding_window_length is not None:
        p = p * swa_mask_zero
    if padding is not None:
        p = p.masked_fill(p_mask, 0.0)

    # apply dropout mask over softmax outputs
    if dropout_prob != 0.0:
        assert (
            dropout_mask != None
        ), "PyTorch reference must have dropout_mask for dropout"
        p = (p * dropout_mask) / (1 - dropout_prob)

    o = torch.einsum("bhqk,bhkd->bhqd", p, v)

    # softmax stats is used for backwards computation
    if compute_stats:
        # amax (NOT absolute max) is used here to evenly distribute gradient
        row_max = torch.amax(s, -1, True)
        row_exp = torch.exp(s - row_max)
        row_sum = torch.sum(row_exp, -1, True)
        stats = row_max + torch.log(row_sum)
        return o, stats

    return o


def generate_ragged_offset(layout, head_group, shape_q, shape_k, shape_v, shape_o, seq_len_q, seq_len_kv):
    b, h_q, s_q, d_qk = shape_q
    b, h_k, s_kv, d_qk = shape_k
    b, h_v, s_kv, d_v = shape_v
    b, h_q, s_q, d_v = shape_o

    assert shape_q == (b, h_q, s_q, d_qk)
    assert shape_k == (b, h_k, s_kv, d_qk)
    assert shape_v == (b, h_v, s_kv, d_v)
    assert shape_o == (b, h_q, s_q, d_v)

    # Compute the exclusive prefix sum for ragged sequence dimension
    # tensor has shape (B, 1, 1, 1)
    # output has shape (B+1, 1, 1, 1)
    # ex) tensor = [[[[2, 4, 1, 6]]]]
    #     output = [[[[0, 2, 6, 7, 13]]]]
    def compute_exclusive_prefix_sum(tensor):
        assert tensor.size(1) == tensor.size(2) == tensor.size(3) == 1
        return torch.cat(
            (
                torch.zeros(1, 1, 1, 1, dtype=tensor.dtype, device=tensor.device),
                torch.cumsum(tensor, dim=0),
            )
        )

    if layout == "bshd_bshd_bshd":
        # thd_thd_thd
        q_ragged_offset = compute_exclusive_prefix_sum(seq_len_q) * h_q * d_qk
        k_ragged_offset = compute_exclusive_prefix_sum(seq_len_kv) * h_k * d_qk
        v_ragged_offset = compute_exclusive_prefix_sum(seq_len_kv) * h_v * d_v
        o_ragged_offset = compute_exclusive_prefix_sum(seq_len_q) * h_q * d_v
    elif layout == "bs3hd":
        if head_group == "MHA":
            # t3hd
            assert torch.equal(seq_len_q, seq_len_kv)
            assert (h_q == h_k == h_v) and (d_qk == d_v)
            seq_len, h, d = seq_len_q, h_q, d_qk
            q_ragged_offset = compute_exclusive_prefix_sum(seq_len) * 3 * h * d
            k_ragged_offset = compute_exclusive_prefix_sum(seq_len) * 3 * h * d
            v_ragged_offset = compute_exclusive_prefix_sum(seq_len) * 3 * h * d
            o_ragged_offset = compute_exclusive_prefix_sum(seq_len) * h * d
        else:
            # thd_t2hd
            assert (h_k == h_v) and (d_qk == d_v)
            seq_len, h_kv, d = seq_len_q, h_k, d_qk
            q_ragged_offset = compute_exclusive_prefix_sum(seq_len_q) * h_q * d
            k_ragged_offset = compute_exclusive_prefix_sum(seq_len_kv) * 2 * h_kv * d
            v_ragged_offset = compute_exclusive_prefix_sum(seq_len_kv) * 2 * h_kv * d
            o_ragged_offset = compute_exclusive_prefix_sum(seq_len_q) * h_q * d
    else:
        assert False, "wrong layout"

    q_ragged_offset = q_ragged_offset.to(dtype=seq_len_q.dtype)
    k_ragged_offset = k_ragged_offset.to(dtype=seq_len_kv.dtype)
    v_ragged_offset = v_ragged_offset.to(dtype=seq_len_kv.dtype)
    o_ragged_offset = o_ragged_offset.to(dtype=seq_len_q.dtype)

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


def generate_actual_seq_lens(b, s_q, s_kv, layout, head_group, is_padding, force_sq_less_or_equal_than_skv):
    seq_len_q_gpu = None
    seq_len_kv_gpu = None

    if is_padding:
        seq_len_q_gpu = torch.randint(
            1, s_q + 1, (b, 1, 1, 1), dtype=torch.int32, device="cuda"
        )

        if not (layout == "bs3hd" and head_group == "MHA"):
            seq_len_kv_gpu = torch.randint(
                1, s_kv + 1, (b, 1, 1, 1), dtype=torch.int32, device="cuda"
            )
            # Avoid seq_len_q > seq_len_kv (known limitation):
            if force_sq_less_or_equal_than_skv:
                seq_len_q_gpu = torch.max(
                    torch.tensor(1), seq_len_q_gpu % seq_len_kv_gpu
                )
        else:
            seq_len_kv_gpu = seq_len_q_gpu

    return (seq_len_q_gpu, seq_len_kv_gpu)
    

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
        return

    head_group   = cfg.head_group
    data_type    = cfg.data_type
    rng_data     = cfg.rng_data

    is_causal    = cfg.is_causal
    is_alibi     = cfg.is_alibi
    is_infer     = cfg.is_infer
    is_paged     = cfg.is_paged
    is_bias      = cfg.is_bias
    is_padding   = cfg.is_padding
    is_ragged    = cfg.is_ragged
    is_causal_br = cfg.is_causal_br
    is_sliding_w = cfg.is_sliding_w
    is_dropout   = cfg.is_dropout
    is_determin  = cfg.is_determin

    batches      = cfg.batches
    d_qk         = cfg.d_qk
    d_v          = cfg.d_v
    s_q          = cfg.s_q
    s_kv         = cfg.s_kv
    h_q          = cfg.h_q
    h_k          = cfg.h_k
    h_v          = cfg.h_v
    block_size   = cfg.block_size
    layout       = cfg.in_layout

    shape_q      = cfg.shape_q
    stride_q     = cfg.stride_q
    offset_q     = cfg.offset_q
    elems_q      = cfg.elems_q

    shape_k      = cfg.shape_k
    stride_k     = cfg.stride_k
    offset_k     = cfg.offset_k
    elems_k      = cfg.elems_k

    shape_v      = cfg.shape_v
    stride_v     = cfg.stride_v
    offset_v     = cfg.offset_v
    elems_v      = cfg.elems_v

    shape_o      = cfg.shape_o
    stride_o     = cfg.stride_o
    elems_o      = cfg.elems_o

    cudnn_version = LooseVersion(cudnn.backend_version_string())

    if cudnn_version < "8.9.3":
        pytest.fail("SDPA function requires cudnn 8.9.3 or higher")

    if not is_infer:
        assert is_paged == False and block_size == 0, "wrong input for backward pass"

        if is_bias and cudnn_version < "8.9.6":
            print("ERROR: dBias is only supported 8.9.6 onwards.")
            pytest.skip("insufficient cuDNN version")

        if is_bias and cudnn_version < "9" and torch.cuda.get_device_capability()[0] < 9:
            print("ERROR: dBias is only supported on hopper before v9.")
            pytest.skip("insufficient cuDNN version")

    if head_group != "MHA" and cudnn_version < "8.9.7":
        print("ERROR: GQA and MQA is only supported 8.9.7 onwards.")
        pytest.skip("insufficient cuDNN version")

    if is_alibi and cudnn_version < "8.9.4":
        print("ERROR: ALiBi mask is only supported 8.9.4 onwards.")
        pytest.skip("insufficient cuDNN version")

    if is_padding and cudnn_version < "8.9.3":
        print("ERROR: Padding mask is only supported 8.9.3 onwards.")
        pytest.skip("insufficient cuDNN version")

    if is_dropout and cudnn_version < "8.9.6":
        print("ERROR: Dropout reference is only supported on 8.9.6 onwards.")
        pytest.skip("insufficient cuDNN version")

    if is_ragged and cudnn_version < "9":
        print("ERROR: Ragged tensor is only supported 9.0.0 onwards.")
        pytest.skip("insufficient cuDNN version")

    if is_ragged and layout == "bs3hd" and cudnn_version < "9.1.0":
        print("ERROR: t3hd is only supported on 9.1.0 onwards.")
        pytest.skip("insufficient cuDNN version")

    if is_paged and cudnn_version < "9.5":
        print("ERROR: Paged attention only tested with cuDNNv9.5 or greater.")
        pytest.skip("insufficient cuDNN version")

    if d_qk != d_v and cudnn_version < "8.9.6":
        print("ERROR: d_qk != d_v is only supported on 8.9.6 onwards.")
        pytest.skip("insufficient cuDNN version")

    if d_qk != d_v and is_ragged and cudnn_version < "9.1":
        print("ERROR: d_qk != d_v is not supported with ragged offset.")
        pytest.skip("insufficient cuDNN version")

    if is_ragged and torch.cuda.get_device_capability()[0] < 9:
        print("ERROR: Ragged tensor is only supported hopper.")
        pytest.skip("insufficient GPU version")

    qkv_num_elems = elems_q + elems_k + elems_v

    if offset_q + offset_k + offset_v == 0:
        q_gpu = alloc_tensor(shape_q, data_type, elems=elems_q, strides=stride_q, rng=rng_data, mean=-0.5, std=1.0)
        k_gpu = alloc_tensor(shape_k, data_type, elems=elems_k, strides=stride_k, rng=rng_data, mean=-0.5, std=1.0)
        v_gpu = alloc_tensor(shape_v, data_type, elems=elems_v, strides=stride_v, rng=rng_data, mean=-0.5, std=1.0)
        bias_gpu = (alloc_tensor((1, h_q, s_q, s_kv), data_type, rng=rng_data, mean=0.0, std=1.0) if is_bias else None)
    else:
        qkv_gpu = torch.randn(qkv_num_elems, dtype=data_type, generator=rng_data, device="cuda") - 0.5
        q_gpu = torch.as_strided(qkv_gpu, shape_q, stride_q, storage_offset=offset_q)
        k_gpu = torch.as_strided(qkv_gpu, shape_k, stride_k, storage_offset=offset_k)
        v_gpu = torch.as_strided(qkv_gpu, shape_v, stride_v, storage_offset=offset_v)
        bias_gpu = (torch.randn(1, h_q, s_q, s_kv, dtype=data_type, generator=rng_data, device="cuda") if is_bias else None)

    if not is_infer:
        if offset_q + offset_k + offset_v == 0:
            (dQ_gpu, dQ_sep, dQ_raw) = alloc_tensor(shape_q, data_type, elems=elems_q, strides=stride_q)
            (dK_gpu, dK_sep, dK_raw) = alloc_tensor(shape_k, data_type, elems=elems_k, strides=stride_k)
            (dV_gpu, dV_sep, dV_raw) = alloc_tensor(shape_v, data_type, elems=elems_v, strides=stride_v)
            (dBias_gpu, dBias_sep, dBias_raw) = (alloc_tensor((1, h_q, s_q, s_kv), data_type) if is_bias else (None, None, None))
            dO_gpu = alloc_tensor(shape_o, data_type, elems=elems_o, strides=stride_o, rng=rng_data, mean=0.0, std=0.1)
        else:
            dQKV_gpu = torch.empty(qkv_num_elems, dtype=data_type, device="cuda")
            dQ_gpu = torch.as_strided(dQKV_gpu, shape_q, stride_q, storage_offset=offset_q)
            dK_gpu = torch.as_strided(dQKV_gpu, shape_k, stride_k, storage_offset=offset_k)
            dV_gpu = torch.as_strided(dQKV_gpu, shape_v, stride_v, storage_offset=offset_v)
            dBias_gpu = (torch.empty((1, h_q, s_q, s_kv), dtype=data_type, device="cuda") if is_bias else None)
            dO_gpu = 0.1 * torch.randn(elems_o, dtype=data_type, generator=rng_data, device="cuda").as_strided(shape_o, stride_o)
            dQ_sep = dK_sep = dV_sep = dBias_sep = None
            dQ_raw = dK_raw = dV_raw = dBias_raw = None

    seq_len_q_gpu, seq_len_kv_gpu = generate_actual_seq_lens(batches, s_q, s_kv, layout, head_group, is_padding, is_sliding_w or is_causal_br)

    if is_dropout:
        seed_gpu = torch.full((1, 1, 1, 1), 123456, dtype=torch.int64, device="cuda")
        offset_gpu = torch.full((1, 1, 1, 1), 789, dtype=torch.int64, device="cuda")

    rng_dump_gpu = (
        torch.zeros((batches, h_q, s_q, s_kv), dtype=torch.float32, device="cuda")
        if is_dropout
        else None
    )

    if is_ragged:
        (
            q_ragged_offset_gpu,
            k_ragged_offset_gpu,
            v_ragged_offset_gpu,
            o_ragged_offset_gpu,
        ) = generate_ragged_offset(
            layout,
            head_group,
            shape_q,
            shape_k,
            shape_v,
            shape_o,
            seq_len_q_gpu,
            seq_len_kv_gpu,
        )

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

    sliding_window_length = None
    if is_sliding_w:
        sliding_window_length = s_kv // 4

    attn_scale = 0.125
    
    o, stats = graph.sdpa(
        name="sdpa",
        q=q,
        k=k,
        v=v,
        is_inference=is_infer,
        attn_scale=attn_scale,
        bias=bias,
        use_alibi_mask=is_alibi,
        use_padding_mask=is_padding,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        use_causal_mask=is_causal,
        use_causal_mask_bottom_right=is_causal_br,
        sliding_window_length=sliding_window_length,
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
        print(f"ERROR: not supported forward graph. {e}")
        pytest.xfail("not supported forward graph")
    except Exception as e:
        print(f"ERROR: unexpected '{e.__class__.__name__}' exception during forward graph validate. {e}")
        pytest.fail("unexpected exception during forward graph validate", pytrace=False)

    try:
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans()
    except cudnn.cudnnGraphNotSupportedError as e:
        print(f"ERROR: not supported forward graph after validate. {e}")
        pytest.xfail("not supported forward graph after validate")
    except Exception as e:
        print(f"ERROR: Unexpected '{e.__class__.__name__}' exception after forward validate. {e}")
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

    # Execute forward cuDNN graph
    (workspace, ws_sep, _) = alloc_tensor(graph.get_workspace_size(), torch.uint8)
    graph.execute(variant_pack, workspace, handle=cudnn_handle)
    torch.cuda.synchronize()

    if not torch.all(ws_sep==-1).item():
        print("ERROR: forward workspace overwritten outside its boundaries")
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

        # Backward cuDNN graph
        graph = cudnn.pygraph(
            io_data_type=convert_to_cudnn_type(data_type),
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
            handle=cudnn_handle,
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
            use_causal_mask=is_causal,
            use_causal_mask_bottom_right=is_causal_br,
            sliding_window_length=sliding_window_length,
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
            print(f"ERROR: not supported backward graph. {e}")
            pytest.xfail("not supported backward graph")
        except Exception as e:
            print(f"ERROR: unexpected '{e.__class__.__name__}' exception during backward graph validate. {e}")
            pytest.fail("unexpected exception during backward graph validate", pytrace=False)

        try:
            graph.build_operation_graph()
            graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
            graph.check_support()
            graph.build_plans()
        except cudnn.cudnnGraphNotSupportedError as e:
            print(f"ERROR: not supported backward graph after validate. {e}")
            pytest.xfail("not supported backward graph after validate")
        except Exception as e:
            print(f"ERROR: unexpected '{e.__class__.__name__}' exception after backward validate. {e}")
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

        # Execute backward cuDNN graph
        (workspace, ws_sep, _) = alloc_tensor(graph.get_workspace_size(), torch.uint8)
        graph.execute(variant_pack, workspace, handle=cudnn_handle)
        torch.cuda.synchronize()

        if not torch.all(ws_sep==-1).item():
            print("ERROR: backward workspace overwritten outside its boundaries")
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
        is_causal=is_causal,
        is_causal_br=is_causal_br,
        sliding_window_length=sliding_window_length,
        dropout_prob=dropout_prob,
        dropout_mask=rng_dump_ref,
        compute_stats=(is_infer == False),
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

    diffs = request.config.option.diffs

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
        print("ERROR: disallowed mismatches")
        pytest.fail("disallowed mismatches", pytrace=False)


@pytest.fixture(scope="function")
def config0():
    cfg = testConfig()
    cfg.setBatches(min_batches=2, max_batches=2)
    cfg.setSequences(min_s_q=8, max_s_q=2048, min_s_kv=8, max_s_kv=2048)
    cfg.setVectors(min_d_v=64, max_d_v=128, min_d_qk=32, max_d_qk=128)
    cfg.setHeads(min_h_qkv=1, max_h_qkv=6)
    cfg.setBlockSize(min_blk_sz=32, max_blk_sz=128)
    return cfg

# =====================================
# L0 fprop tests (legacy randomization)
# =====================================

@pytest.mark.skipif("not config.getoption('--unlock')", reason="used with '--unlock' only")
@pytest.mark.parametrize("test_no", tlist(num_tests=4, rng_seed=741), ids=lambda p: f"test{p[0]}")
@pytest.mark.parametrize("data_type", data_type_options, ids=lambda p: str(p))
@pytest.mark.parametrize("is_causal", [False, True], ids=lambda p: "causal" if p else "noncausal")
@pytest.mark.parametrize("in_layout", fixed_layout_options)
@pytest.mark.parametrize("head_group", head_group_options)
@pytest.mark.parametrize("is_infer", [True], ids=lambda p: "FWD" if p else "BWD")
@pytest.mark.L0
def test_sdpa_fixed_fwd(config0, test_no, data_type, is_infer, head_group, in_layout, is_causal, request, cudnn_handle):
    config0.fixed_layout(test_no, is_infer, data_type, head_group, in_layout, is_causal, request)
    exec_sdpa(config0, request, cudnn_handle)

# =====================================
# L0 bprop tests (legacy randomization)
# =====================================

@pytest.mark.skipif("not config.getoption('--unlock')", reason="used with '--unlock' only")
@pytest.mark.parametrize("test_no", tlist(num_tests=4, rng_seed=555), ids=lambda p: f"test{p[0]}")
@pytest.mark.parametrize("data_type", data_type_options, ids=lambda p: str(p))
@pytest.mark.parametrize("is_causal", [False, True], ids=lambda p: "causal" if p else "noncausal")
@pytest.mark.parametrize("in_layout", fixed_layout_options)
@pytest.mark.parametrize("head_group", head_group_options)
@pytest.mark.parametrize("is_infer", [False], ids=lambda p: "FWD" if p else "BWD")
@pytest.mark.L0
def test_sdpa_fixed_bwd(config0, test_no, data_type, is_infer, head_group, in_layout, is_causal, request, cudnn_handle):
    config0.fixed_layout(test_no, is_infer, data_type, head_group, in_layout, is_causal, request)
    exec_sdpa(config0, request, cudnn_handle)

@pytest.fixture(scope="function")
def config1():
    cfg = testConfig()
    cfg.setBatches(max_batches=8)
    cfg.setSequences(max_s_q=64, max_s_kv=64)
    cfg.setVectors(max_d_v=32, max_d_qk=32)
    cfg.setHeads(max_h_qkv=8)
    cfg.setBlockSize(max_blk_sz=256)
    return cfg

# ==================================
# L0 fprop tests (new randomization)
# ==================================

@pytest.mark.skipif("not config.getoption('--unlock')", reason="used with '--unlock' only")
@pytest.mark.parametrize("test_no", tlist(num_tests=64, rng_seed=888), ids=lambda p: f"test{p[0]}")
@pytest.mark.parametrize("data_type", data_type_options, ids=lambda p: str(p))
@pytest.mark.parametrize("is_causal", [False, True], ids=lambda p: "causal" if p else "noncausal")
@pytest.mark.parametrize("layout", random_layout_options)
@pytest.mark.parametrize("head_group", head_group_options)
@pytest.mark.parametrize("is_infer", [True], ids=lambda p: "FWD" if p else "BWD")
@pytest.mark.L0
def test_sdpa_random_fwd(config1, test_no, data_type, is_infer, head_group, layout, is_causal, request, cudnn_handle):
    config1.random_layout(test_no, is_infer, data_type, head_group, layout, is_causal, request)
    exec_sdpa(config1, request, cudnn_handle)

# ==================================
# L0 bprop tests (new randomization)
# ==================================

@pytest.mark.skipif("not config.getoption('--unlock')", reason="used with '--unlock' only")
@pytest.mark.parametrize("test_no", tlist(num_tests=64, rng_seed=123), ids=lambda p: f"test{p[0]}")
@pytest.mark.parametrize("data_type", data_type_options, ids=lambda p: str(p))
@pytest.mark.parametrize("is_causal", [False, True], ids=lambda p: "causal" if p else "noncausal")
@pytest.mark.parametrize("layout", random_layout_options)
@pytest.mark.parametrize("head_group", head_group_options)
@pytest.mark.parametrize("is_infer", [False], ids=lambda p: "FWD" if p else "BWD")
@pytest.mark.L0
def test_sdpa_random(config1, test_no, data_type, is_infer, head_group, layout, is_causal, request, cudnn_handle):
    config1.random_layout(test_no, is_infer, data_type, head_group, layout, is_causal, request)
    exec_sdpa(config1, request, cudnn_handle)

# ===================
# Single repro test
# ===================

@pytest.mark.skipif("not config.getoption('--repro')", reason="used with '--repro' only")
def test_repro(request, cudnn_handle):
    repro_str = request.config.getoption("--repro")
    cfg = testConfig()
    cfg.load_config(repro_str)
    cfg.showConfig((1,1), request, False)
    exec_sdpa(cfg, request, cudnn_handle)

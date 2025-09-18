from .NSA_select_attn_fwd_hmma import HopperSelectAttentionFwd
from typing import Tuple, Type
import math

from cuda.bindings import driver as cuda
import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cudnn.datatypes import _convert_to_cutlass_data_type


class SelectionAttention:
    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_v: torch.Tensor,
        sample_o: torch.Tensor,
        sample_l: torch.Tensor,
        sample_m: torch.Tensor,
        sample_block_indices: torch.Tensor,
        sample_block_counts: torch.Tensor,
        sample_seq_offsets: torch.Tensor | None = None,
        acc_dtype: torch.dtype = torch.float32,
        max_s: int | None = 1024,
        block_size: int = 64,
        scale_softmax: float | None = None,
    ):
        # Store sample tensors only; defer validation to check_support
        self.sample_q = sample_q
        self.sample_k = sample_k
        self.sample_v = sample_v
        self.sample_o = sample_o
        self.sample_l = sample_l
        self.sample_m = sample_m
        self.sample_block_indices = sample_block_indices
        self.sample_block_counts = sample_block_counts
        self.sample_seq_offsets = sample_seq_offsets

        # Types and kernel configuration
        self.acc_dtype = acc_dtype
        self.block_size = block_size
        self.max_s = max_s

        # Derived attributes (populated in check_support)
        self.input_layout = None
        self.dtype = None
        self.h_q = None
        self.h_kv = None
        self.gqa_group_size = None
        self.head_dim = None
        self.value_dim = None

        self.scale_softmax = scale_softmax

        # Compiled kernel cache
        self._compiled_selection_attention = None

    def check_support(self) -> bool:
        # Shape normalization and validation
        if self.sample_q.ndim == 4:
            # B, H_q, S, D  format
            self.input_layout = "B,H,S,D"

            raise NotImplementedError("B, H_q, S, D format not implemented")
        elif self.sample_q.ndim == 3:
            # T, H_q, D  format
            self.input_layout = "T,H,D"

            t, h_q, d = self.sample_q.shape
            assert (
                self.sample_k.ndim == 3
                and self.sample_k.shape[0] == t
                and self.sample_k.shape[2] == d
            ), "K must be (T, H_kv, D)"
            h_kv = self.sample_k.shape[1]

            assert (
                self.sample_v.ndim == 3
                and self.sample_v.shape[0] == t
                and self.sample_v.shape[1] == h_kv
            ), "V must be (T, H_kv, D_v)"
            d_v = self.sample_v.shape[2]

            assert (
                self.sample_o.ndim == 3
                and self.sample_o.shape[0] == t
                and self.sample_o.shape[1] == h_q
                and self.sample_o.shape[2] == d_v
            ), "O must be (T, H_q, D_v)"
            assert self.sample_l.ndim == 2 and self.sample_l.shape == (
                t,
                h_q,
            ), "L must be (T, H_q)"
            assert self.sample_m.ndim == 2 and self.sample_m.shape == (
                t,
                h_q,
            ), "M must be (T, H_q)"

            assert self.sample_seq_offsets is not None and isinstance(
                self.sample_seq_offsets, torch.Tensor
            ), "seq_offsets (torch.Tensor) is required when using (T, H, D) format"
            self.batch_size = len(self.sample_seq_offsets) - 1
            assert self.batch_size > 0, "batch_size must be greater than 0"
            assert self.sample_seq_offsets.dtype in (
                torch.int32,
                torch.int64,
            ), "seq_offsets must be int32 or int64"
        else:
            raise AssertionError("sample_q must be rank-3 (T,H,D) or rank-4 (B,H,S,D)")

        # Shared derived attributes
        assert h_q % h_kv == 0, "H_q must be a multiple of H_kv (GQA/MQA constraint)"
        self.h_q = h_q
        self.h_kv = h_kv
        self.gqa_group_size = h_q // h_kv
        self.head_dim = d
        self.value_dim = d_v

        # Validate dtypes and config
        self.dtype = self.sample_q.dtype
        assert (
            self.dtype
            == self.sample_k.dtype
            == self.sample_v.dtype
            == self.sample_o.dtype
        ), "All input/output tensors must have the same dtype"
        assert self.dtype in {
            torch.float16,
            torch.bfloat16,
        }, "dtype must be Float16 or BFloat16"
        assert self.acc_dtype in {torch.float32}, "acc_dtype must be Float32"
        assert self.block_size in {16, 32, 64}, "block_size must be 16, 32, or 64"

        # Compute default scale_softmax if needed
        if self.scale_softmax is None:
            self.scale_softmax = 1.0 / math.sqrt(self.head_dim)

        if not torch.cuda.is_available():
            raise AssertionError("CUDA is not available")

        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor

        if compute_capability < 90:
            raise AssertionError(
                f"Requires SM90+ compute capability, but found SM{compute_capability} on device {device}"
            )

        return True

    def _reshape_tensors(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        o: torch.Tensor,
        l: torch.Tensor,
        m: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Reshape tensors from input format to kernel expected format:
        - Q: (gqa_group_size, d, T, h_kv)
        - K: (T, d, h_kv)
        - V: (T, d_v, h_kv)
        - O: (gqa_group_size, d_v, T, h_kv)
        - L: (gqa_group_size, T, h_kv)
        - M: (gqa_group_size, T, h_kv)
        """
        if self.input_layout == "B,H,S,D":
            raise NotImplementedError("B,H,S,D format not implemented")
        elif self.input_layout == "T,H,D":
            T, h_q, d = q.shape
            _, h_kv, _ = k.shape
            _, _, d_v = v.shape

            # Reshape Q: (T, H_q, D) -> (gqa_group_size, D, T, H_kv)
            q_reshaped = q.view(T, h_kv, self.gqa_group_size, d).permute(2, 3, 0, 1)
            # Reshape K: (T, H_kv, D) -> (T, D, H_kv)
            k_reshaped = k.permute(0, 2, 1)
            # Reshape V: (T, H_kv, D_v) -> (T, D_v, H_kv)
            v_reshaped = v.permute(0, 2, 1)
            # Reshape O: (T, H_q, D_v) -> (gqa_group_size, D_v, T, H_kv)
            o_reshaped = o.view(T, h_kv, self.gqa_group_size, d_v).permute(2, 3, 0, 1)
            # Reshape L: (T, H_q) -> (gqa_group_size, T, H_kv)
            l_reshaped = l.view(T, h_kv, self.gqa_group_size).permute(2, 0, 1)
            # Reshape M: (T, H_q) -> (gqa_group_size, T, H_kv)
            m_reshaped = m.view(T, h_kv, self.gqa_group_size).permute(2, 0, 1)
        else:
            raise ValueError(f"Invalid input layout: {self.input_layout}")

        # Temporary: assert that no memory is copied during reshape
        # Long term, we'd instead want to handle copying output tensors back to their original tensors
        def shares_memory(original, reshaped):
            return original.data_ptr() == reshaped.data_ptr()

        assert shares_memory(
            q, q_reshaped
        ), "Q tensor memory changed during reshape - expected view operation"
        assert shares_memory(
            k, k_reshaped
        ), "K tensor memory changed during reshape - expected view operation"
        assert shares_memory(
            v, v_reshaped
        ), "V tensor memory changed during reshape - expected view operation"
        assert shares_memory(
            o, o_reshaped
        ), "O tensor memory changed during reshape - expected view operation"
        assert shares_memory(
            l, l_reshaped
        ), "L tensor memory changed during reshape - expected view operation"
        assert shares_memory(
            m, m_reshaped
        ), "M tensor memory changed during reshape - expected view operation"

        return q_reshaped, k_reshaped, v_reshaped, o_reshaped, l_reshaped, m_reshaped

    def compile(self, current_stream: cuda.CUstream = None) -> None:
        if current_stream is None:
            current_stream = cutlass.cuda.default_stream()
        if (
            self.h_q is None
            or self.h_kv is None
            or self.gqa_group_size is None
            or self.head_dim is None
            or self.value_dim is None
        ):
            assert self.check_support()

        selection_attention = HopperSelectAttentionFwd(
            head_dim=self.head_dim,
            value_dim=self.value_dim,
            GQA_group_size=self.gqa_group_size,
            block_size=self.block_size,
            dtype=_convert_to_cutlass_data_type(self.dtype),
            acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
        )

        q_reshaped, k_reshaped, v_reshaped, o_reshaped, l_reshaped, m_reshaped = (
            self._reshape_tensors(
                self.sample_q,
                self.sample_k,
                self.sample_v,
                self.sample_o,
                self.sample_l,
                self.sample_m,
            )
        )

        mQ = from_dlpack(q_reshaped, assumed_align=128)
        mK = from_dlpack(k_reshaped, assumed_align=128)
        mV = from_dlpack(v_reshaped, assumed_align=128)
        mO = from_dlpack(o_reshaped, assumed_align=128)
        mL = from_dlpack(l_reshaped)
        mM = from_dlpack(m_reshaped)
        m_block_indices = from_dlpack(self.sample_block_indices)
        m_block_counts = from_dlpack(self.sample_block_counts)
        m_seq_offsets = from_dlpack(self.sample_seq_offsets)

        compiled_selection_attention = cute.compile(
            selection_attention,
            mQ,
            mK,
            mV,
            mO,
            mL,
            mM,
            m_block_indices,
            m_block_counts,
            self.max_s,
            m_seq_offsets,
            self.scale_softmax,
            current_stream,
        )
        self._compiled_selection_attention = compiled_selection_attention

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        l_tensor: torch.Tensor,
        m_tensor: torch.Tensor,
        block_indices_tensor: torch.Tensor,
        block_counts_tensor: torch.Tensor,
        seq_offsets_tensor: torch.Tensor,
        scale_softmax: float | None = None,
        current_stream: cuda.CUstream = None,
        skip_compile: bool = False,
    ):
        if current_stream is None:
            current_stream = cutlass.cuda.default_stream()
        if not skip_compile:
            assert (
                self._compiled_selection_attention is not None
            ), "SelectionAttention kernel not compiled"

        q_reshaped, k_reshaped, v_reshaped, o_reshaped, l_reshaped, m_reshaped = (
            self._reshape_tensors(
                q_tensor, k_tensor, v_tensor, o_tensor, l_tensor, m_tensor
            )
        )

        mQ = from_dlpack(q_reshaped, assumed_align=128)
        mK = from_dlpack(k_reshaped, assumed_align=128)
        mV = from_dlpack(v_reshaped, assumed_align=128)
        mO = from_dlpack(o_reshaped, assumed_align=128)
        mL = from_dlpack(l_reshaped, assumed_align=128)
        mM = from_dlpack(m_reshaped, assumed_align=128)
        m_block_indices = from_dlpack(block_indices_tensor, assumed_align=128)
        m_block_counts = from_dlpack(block_counts_tensor, assumed_align=128)
        m_seq_offsets = from_dlpack(seq_offsets_tensor, assumed_align=128)

        scale_softmax = self.scale_softmax if scale_softmax is None else scale_softmax

        if not skip_compile:
            self._compiled_selection_attention(
                mQ,
                mK,
                mV,
                mO,
                mL,
                mM,
                m_block_indices,
                m_block_counts,
                self.max_s,
                m_seq_offsets,
                scale_softmax,
                current_stream,
            )
        else:
            selection_attention = HopperSelectAttentionFwd(
                head_dim=self.head_dim,
                value_dim=self.value_dim,
                GQA_group_size=self.gqa_group_size,
                block_size=self.block_size,
                dtype=_convert_to_cutlass_data_type(self.dtype),
                acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
            )
            selection_attention(
                mQ,
                mK,
                mV,
                mO,
                mL,
                mM,
                m_block_indices,
                m_block_counts,
                self.max_s,
                m_seq_offsets,
                scale_softmax,
                current_stream,
            )

    def __call__(self, *args, **kwargs):
        self.execute(*args, skip_compile=True, **kwargs)


# Cache for SelectionAttention objects
_cache_of_SelectionAttentionObjects = {}


def SelectionAttentionWrapper(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    block_indices_tensor: torch.Tensor,
    block_counts_tensor: torch.Tensor,
    seq_offsets_tensor: torch.Tensor,
    block_size: int = 64,
    scale_softmax: float | None = None,
    acc_dtype: torch.dtype = torch.float32,
    max_s: int | None = None,
    stream: cuda.CUstream = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Selection Attention Wrapper that returns output tensors directly.

    Returns:
        tuple: (o_tensor, l_tensor, m_tensor) - Output, logsumexp, and max tensors
    """
    if stream is None:
        stream = cutlass.cuda.default_stream()
    dtype = q_tensor.dtype
    max_s = (
        max(seq_offsets_tensor[1:] - seq_offsets_tensor[:-1]).item()
        if max_s is None
        else max_s
    )

    t, h_q, d = q_tensor.shape
    _, h_kv, d_v = v_tensor.shape

    o_tensor = torch.zeros((t, h_q, d_v), dtype=dtype).cuda()
    l_tensor = torch.zeros((t, h_q), dtype=torch.float32).cuda()
    m_tensor = torch.zeros((t, h_q), dtype=torch.float32).cuda()

    # TODO: cache
    # if (q_tensor.shape, q_tensor.dtype, q_tensor.stride(),
    #     k_tensor.shape, k_tensor.dtype, k_tensor.stride(),
    #     v_tensor.shape, v_tensor.dtype, v_tensor.stride(),
    #     o_tensor.shape, o_tensor.dtype, o_tensor.stride(),
    #     l_tensor.shape, l_tensor.dtype, l_tensor.stride(),
    #     m_tensor.shape, m_tensor.dtype, m_tensor.stride(),
    #     block_indices_tensor.shape, block_indices_tensor.dtype, block_indices_tensor.stride(),
    #     block_counts_tensor.shape, block_counts_tensor.dtype, block_counts_tensor.stride(),
    #     seq_offsets_tensor.shape, seq_offsets_tensor.dtype, seq_offsets_tensor.stride(),
    #     max_s, block_size, scale_softmax, acc_dtype) in _cache_of_SelectionAttentionObjects:
    #     selection_attention_object = _cache_of_SelectionAttentionObjects[(q_tensor.shape, q_tensor.dtype, q_tensor.stride(),
    #                                                                        k_tensor.shape, k_tensor.dtype, k_tensor.stride(),
    #                                                                        v_tensor.shape, v_tensor.dtype, v_tensor.stride(),
    #                                                                        o_tensor.shape, o_tensor.dtype, o_tensor.stride(),
    #                                                                        l_tensor.shape, l_tensor.dtype, l_tensor.stride(),
    #                                                                        m_tensor.shape, m_tensor.dtype, m_tensor.stride(),
    #                                                                        block_indices_tensor.shape, block_indices_tensor.dtype, block_indices_tensor.stride(),
    #                                                                        block_counts_tensor.shape, block_counts_tensor.dtype, block_counts_tensor.stride(),
    #                                                                        seq_offsets_tensor.shape, seq_offsets_tensor.dtype, seq_offsets_tensor.stride(),
    #                                                                        max_s, block_size, scale_softmax, acc_dtype)]
    if False:
        pass
    else:
        selection_attention_object = SelectionAttention(
            sample_q=q_tensor,
            sample_k=k_tensor,
            sample_v=v_tensor,
            sample_o=o_tensor,
            sample_l=l_tensor,
            sample_m=m_tensor,
            sample_block_indices=block_indices_tensor,
            sample_block_counts=block_counts_tensor,
            sample_seq_offsets=seq_offsets_tensor,
            # dtype=dtype,
            acc_dtype=acc_dtype,
            max_s=max_s,
            block_size=block_size,
            scale_softmax=scale_softmax,
        )
        selection_attention_object.check_support()
        selection_attention_object.compile()
        selection_attention_object.execute(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            o_tensor=o_tensor,
            l_tensor=l_tensor,
            m_tensor=m_tensor,
            block_indices_tensor=block_indices_tensor,
            block_counts_tensor=block_counts_tensor,
            seq_offsets_tensor=seq_offsets_tensor,
            scale_softmax=scale_softmax,
            current_stream=stream,
            # skip_compile=True,
        )
        # _cache_of_SelectionAttentionObjects[(q_tensor.shape, q_tensor.dtype, q_tensor.stride(),
        #                                      k_tensor.shape, k_tensor.dtype, k_tensor.stride(),
        #                                      v_tensor.shape, v_tensor.dtype, v_tensor.stride(),
        #                                      o_tensor.shape, o_tensor.dtype, o_tensor.stride(),
        #                                      l_tensor.shape, l_tensor.dtype, l_tensor.stride(),
        #                                      m_tensor.shape, m_tensor.dtype, m_tensor.stride(),
        #                                      block_indices_tensor.shape, block_indices_tensor.dtype, block_indices_tensor.stride(),
        #                                      block_counts_tensor.shape, block_counts_tensor.dtype, block_counts_tensor.stride(),
        #                                      seq_offsets_tensor.shape, seq_offsets_tensor.dtype, seq_offsets_tensor.stride(),
        #                                      max_s, block_size, scale_softmax, acc_dtype)] = selection_attention_object

    return o_tensor, l_tensor, m_tensor

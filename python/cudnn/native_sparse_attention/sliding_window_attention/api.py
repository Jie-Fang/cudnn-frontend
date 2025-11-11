from __future__ import annotations
import cudnn
import torch

from cuda.bindings import driver as cuda
from cudnn.datatypes import _torch_to_cudnn_data_type
from cudnn.api_base import APIBase
from typing import Optional


class SlidingWindowAttention(APIBase):
    def __init__(
        self,
        sample_q: torch.Tensor,
        sample_k: torch.Tensor,
        sample_v: torch.Tensor,
        sample_o: torch.Tensor,
        sample_stats: torch.Tensor,
        left_bound: int = 0,
        right_bound: int = 0,
        sample_seq_len_q: torch.Tensor = None,
        sample_seq_len_kv: torch.Tensor = None,
        sample_q_ragged_offset: torch.Tensor = None,
        sample_k_ragged_offset: torch.Tensor = None,
        sample_v_ragged_offset: torch.Tensor = None,
        sample_o_ragged_offset: torch.Tensor = None,
        sample_stats_ragged_offset: torch.Tensor = None,
        max_seq_len_q: int = None,
        max_seq_len_kv: int = None,
        is_infer: bool = False,
        attn_scale: float = 1.0,
        intermediate_data_type: torch.dtype = torch.float32,
        compute_data_type: torch.dtype = torch.float32,
        cudnn_handle: cudnn.handle = None,
    ):
        super().__init__()
        self._logger.debug("Entering __init__")

        self.sample_q = sample_q
        self.sample_k = sample_k
        self.sample_v = sample_v
        self.sample_o = sample_o
        self.sample_stats = self._pad_tensor_to_ndim(
            sample_stats, self.sample_o.ndim, "sample_stats"
        )
        self.left_bound = left_bound
        self.right_bound = right_bound

        self.sample_seq_len_q = self._pad_tensor_to_ndim(
            sample_seq_len_q, 4, "sample_seq_len_q"
        )
        self.sample_seq_len_kv = self._pad_tensor_to_ndim(
            sample_seq_len_kv, 4, "sample_seq_len_kv"
        )
        self.max_seq_len_q = max_seq_len_q
        self.max_seq_len_kv = max_seq_len_kv
        self.sample_q_ragged_offset = self._pad_tensor_to_ndim(
            sample_q_ragged_offset, 4, "sample_q_ragged_offset"
        )
        self.sample_k_ragged_offset = self._pad_tensor_to_ndim(
            sample_k_ragged_offset, 4, "sample_k_ragged_offset"
        )
        self.sample_v_ragged_offset = self._pad_tensor_to_ndim(
            sample_v_ragged_offset, 4, "sample_v_ragged_offset"
        )
        self.sample_o_ragged_offset = self._pad_tensor_to_ndim(
            sample_o_ragged_offset, 4, "sample_o_ragged_offset"
        )
        self.sample_stats_ragged_offset = self._pad_tensor_to_ndim(
            sample_stats_ragged_offset, 4, "sample_stats_ragged_offset"
        )

        self.is_infer = is_infer
        self.attn_scale = attn_scale
        self.intermediate_data_type = intermediate_data_type
        self.compute_data_type = compute_data_type

        self.dtype = None
        self.sm_version = None
        self.input_layout = None

        if cudnn_handle is None:
            self._logger.critical(
                "cudnn_handle not provided, creating new handle. This is not recommended as this is significant overhead and will occur for each SlidingWindowAttention object created."
            )
        self._cudnn_handle = (
            cudnn_handle if cudnn_handle is not None else cudnn.create_handle()
        )
        self._cudnn_swa_graph = None
        self._cudnn_compiled = False
        self._logger.debug(
            f"__init__ completed with args: sample_q {tuple(sample_q.shape)}, sample_k {tuple(sample_k.shape)}, sample_v {tuple(sample_v.shape)}, sample_o {tuple(sample_o.shape)}, sample_stats {None if sample_stats is None else tuple(sample_stats.shape)}, left_bound {left_bound}, right_bound {right_bound}, sample_seq_len_q {None if sample_seq_len_q is None else tuple(sample_seq_len_q.shape)}, sample_seq_len_kv {None if sample_seq_len_kv is None else tuple(sample_seq_len_kv.shape)}, max_seq_len_q {max_seq_len_q}, max_seq_len_kv {max_seq_len_kv}, is_infer {is_infer}, attn_scale {attn_scale}, intermediate_data_type {intermediate_data_type}, compute_data_type {compute_data_type}"
        )

    # _pad_tensor_to_ndim is provided by APIBase

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        self.dtype = self.sample_q.dtype
        self.sm_version = (
            torch.cuda.get_device_capability()[0] * 10
            + torch.cuda.get_device_capability()[1]
        )
        if self.sample_q.ndim == 4:
            self._logger.debug("Inferred bshd layout")
            self.input_layout = "bshd"
        elif self.sample_q.ndim == 3:
            self._logger.debug("Inferred thd layout")
            self.input_layout = "thd"
        else:
            raise ValueError(f"Invalid input layout: {self.sample_q.ndim}")
        if not self.is_infer:
            if self.sample_stats is None:
                raise ValueError("sample_stats output must be provided for training")

        swa_graph = cudnn.pygraph(
            io_data_type=_torch_to_cudnn_data_type(self.dtype),
            intermediate_data_type=_torch_to_cudnn_data_type(
                self.intermediate_data_type
            ),
            compute_data_type=_torch_to_cudnn_data_type(self.compute_data_type),
            handle=self._cudnn_handle,
            sm_version=self.sm_version,
        )

        (
            self.q_cudnn,
            self.k_cudnn,
            self.v_cudnn,
            self.seq_len_q_cudnn,
            self.seq_len_kv_cudnn,
            self.q_ragged_offset_cudnn,
            self.k_ragged_offset_cudnn,
            self.v_ragged_offset_cudnn,
            self.o_ragged_offset_cudnn,
            self.stats_ragged_offset_cudnn,
        ) = (None, None, None, None, None, None, None, None, None, None)
        if self.input_layout == "bshd":
            b, h_q, s_q, d_qk = self.sample_q.shape
            b, h_kv, s_kv, d_qk = self.sample_k.shape
            b, h_kv, s_kv, d_v = self.sample_v.shape
            b, h_q, s_q, d_v = self.sample_o.shape

            if self.sample_q.shape != (b, h_q, s_q, d_qk):
                raise ValueError(
                    f"Input shape mismatch: expected Q tensor shape {b, h_q, s_q, d_qk}, got {self.sample_q.shape}"
                )
            if self.sample_k.shape != (b, h_kv, s_kv, d_qk):
                raise ValueError(
                    f"Input shape mismatch: expected K tensor shape {b, h_kv, s_kv, d_qk}, got {self.sample_k.shape}"
                )
            if self.sample_v.shape != (b, h_kv, s_kv, d_v):
                raise ValueError(
                    f"Input shape mismatch: expected V tensor shape {b, h_kv, s_kv, d_v}, got {self.sample_v.shape}"
                )
            if self.sample_o.shape != (b, h_q, s_q, d_v):
                raise ValueError(
                    f"Output shape mismatch: expected O tensor shape {b, h_q, s_q, d_v}, got {self.sample_o.shape}"
                )
            if not self.is_infer:
                self.sample_stats = self._pad_tensor_to_ndim(
                    self.sample_stats, 4, "sample_stats"
                )
                if self.sample_stats.shape != (b, h_q, s_q, 1):
                    raise ValueError(
                        f"Output shape mismatch: expected Stats tensor shape {b, h_q, s_q, 1}, got {self.sample_stats.shape}"
                    )
            if self.sample_seq_len_q is not None or self.sample_seq_len_kv is not None:
                raise ValueError(
                    f"sample_seq_len_q and sample_seq_len_kv should be None for bshd layout, got {self.sample_seq_len_q} and {self.sample_seq_len_kv}"
                )
            if self.max_seq_len_q is not None or self.max_seq_len_kv is not None:
                raise ValueError(
                    f"max_seq_len_q and max_seq_len_kv should be None for bshd layout, got {self.max_seq_len_q} and {self.max_seq_len_kv}"
                )
            if (
                self.sample_q_ragged_offset is not None
                or self.sample_k_ragged_offset is not None
                or self.sample_v_ragged_offset is not None
                or self.sample_o_ragged_offset is not None
                or self.sample_stats_ragged_offset is not None
            ):
                raise ValueError(
                    f"sample_q_ragged_offset, sample_k_ragged_offset, sample_v_ragged_offset, sample_o_ragged_offset, and sample_stats_ragged_offset should be None for bshd layout, got {self.sample_q_ragged_offset}, {self.sample_k_ragged_offset}, {self.sample_v_ragged_offset}, {self.sample_o_ragged_offset}, and {self.sample_stats_ragged_offset}"
                )

            self.q_cudnn = swa_graph.tensor_like(self.sample_q)
            self.k_cudnn = swa_graph.tensor_like(self.sample_k)
            self.v_cudnn = swa_graph.tensor_like(self.sample_v)
        elif self.input_layout == "thd":
            t, h_q, d_qk = self.sample_q.shape
            t, h_kv, d_qk = self.sample_k.shape
            t, h_kv, d_v = self.sample_v.shape
            t, h_q, d_v = self.sample_o.shape

            if self.sample_q.shape != (t, h_q, d_qk):
                raise ValueError(
                    f"Input shape mismatch: expected Q tensor shape {t, h_q, d_qk}, got {self.sample_q.shape}"
                )
            if self.sample_k.shape != (t, h_kv, d_qk):
                raise ValueError(
                    f"Input shape mismatch: expected K tensor shape {t, h_kv, d_qk}, got {self.sample_k.shape}"
                )
            if self.sample_v.shape != (t, h_kv, d_v):
                raise ValueError(
                    f"Input shape mismatch: expected V tensor shape {t, h_kv, d_v}, got {self.sample_v.shape}"
                )
            if self.sample_o.shape != (t, h_q, d_v):
                raise ValueError(
                    f"Output shape mismatch: expected O tensor shape {t, h_q, d_v}, got {self.sample_o.shape}"
                )
            if not self.is_infer:
                self.sample_stats = self._pad_tensor_to_ndim(
                    self.sample_stats, 3, "sample_stats"
                )
                if self.sample_stats.shape != (t, h_q, 1):
                    raise ValueError(
                        f"Output shape mismatch: expected Stats tensor shape {t, h_q, 1}, got {self.sample_stats.shape}"
                    )

            if self.sample_seq_len_q is None or self.sample_seq_len_kv is None:
                raise ValueError(
                    f"sample_seq_len_q and sample_seq_len_kv must be provided for thd layout, got {self.sample_seq_len_q} and {self.sample_seq_len_kv}"
                )
            if self.max_seq_len_q is None or self.max_seq_len_kv is None:
                raise ValueError(
                    f"max_seq_len_q and max_seq_len_kv must be provided for thd layout, got {self.max_seq_len_q} and {self.max_seq_len_kv}"
                )
            if (
                self.sample_q_ragged_offset is None
                or self.sample_k_ragged_offset is None
                or self.sample_v_ragged_offset is None
                or self.sample_o_ragged_offset is None
                or self.sample_stats_ragged_offset is None
            ):
                raise ValueError(
                    f"sample_q_ragged_offset, sample_k_ragged_offset, sample_v_ragged_offset, sample_o_ragged_offset, and sample_stats_ragged_offset must be provided for thd layout, got {self.sample_q_ragged_offset}, {self.sample_k_ragged_offset}, {self.sample_v_ragged_offset}, {self.sample_o_ragged_offset}, and {self.sample_stats_ragged_offset}"
                )

            b = len(self.sample_seq_len_q)
            self.q_cudnn = swa_graph.tensor(
                dim=(b, h_q, self.max_seq_len_q, d_qk),
                stride=(self.max_seq_len_q * h_q * d_qk, d_qk, h_q * d_qk, 1),
                data_type=_torch_to_cudnn_data_type(self.dtype),
            )
            self.k_cudnn = swa_graph.tensor(
                dim=(b, h_kv, self.max_seq_len_kv, d_qk),
                stride=(self.max_seq_len_kv * h_kv * d_qk, d_qk, h_kv * d_qk, 1),
                data_type=_torch_to_cudnn_data_type(self.dtype),
            )
            self.v_cudnn = swa_graph.tensor(
                dim=(b, h_kv, self.max_seq_len_kv, d_v),
                stride=(self.max_seq_len_kv * h_kv * d_v, d_v, h_kv * d_v, 1),
                data_type=_torch_to_cudnn_data_type(self.dtype),
            )
            self.seq_len_q_cudnn = swa_graph.tensor_like(self.sample_seq_len_q)
            self.seq_len_kv_cudnn = swa_graph.tensor_like(self.sample_seq_len_kv)
            self.q_ragged_offset_cudnn = swa_graph.tensor_like(
                self.sample_q_ragged_offset
            )
            self.k_ragged_offset_cudnn = swa_graph.tensor_like(
                self.sample_k_ragged_offset
            )
            self.v_ragged_offset_cudnn = swa_graph.tensor_like(
                self.sample_v_ragged_offset
            )
            self.o_ragged_offset_cudnn = swa_graph.tensor_like(
                self.sample_o_ragged_offset
            )
            if not self.is_infer:
                self.stats_ragged_offset_cudnn = swa_graph.tensor_like(
                    self.sample_stats_ragged_offset
                )

            self.q_cudnn.set_ragged_offset(self.q_ragged_offset_cudnn)
            self.k_cudnn.set_ragged_offset(self.k_ragged_offset_cudnn)
            self.v_cudnn.set_ragged_offset(self.v_ragged_offset_cudnn)

        self.o_cudnn, self.stats_cudnn = swa_graph.sdpa(
            name="sdpa",
            q=self.q_cudnn,
            k=self.k_cudnn,
            v=self.v_cudnn,
            generate_stats=not self.is_infer,
            attn_scale=self.attn_scale,
            bias=None,
            use_alibi_mask=False,
            use_padding_mask=(self.input_layout == "thd"),
            seq_len_q=self.seq_len_q_cudnn,
            seq_len_kv=self.seq_len_kv_cudnn,
            diagonal_band_left_bound=self.left_bound,
            diagonal_band_right_bound=self.right_bound,
            diagonal_alignment=cudnn.diagonal_alignment.TOP_LEFT,
            dropout=None,
            rng_dump=None,
            paged_attention_k_table=None,
            paged_attention_v_table=None,
            paged_attention_max_seq_len_kv=None,
        )
        self.o_cudnn.set_output(True)

        if self.input_layout == "bshd":
            self.o_cudnn.set_dim(self.sample_o.shape).set_stride(self.sample_o.stride())
            if not self.is_infer:
                self.stats_cudnn.set_output(True).set_data_type(cudnn.data_type.FLOAT)
                self.stats_cudnn.set_dim(self.sample_stats.shape).set_stride(
                    self.sample_stats.stride()
                )
        elif self.input_layout == "thd":
            self.o_cudnn.set_dim((b, h_q, self.max_seq_len_q, d_v))
            self.o_cudnn.set_stride((self.max_seq_len_q * h_q * d_v, d_v, h_q * d_v, 1))
            self.o_cudnn.set_ragged_offset(self.o_ragged_offset_cudnn)

            if not self.is_infer:
                self.stats_cudnn.set_output(True).set_data_type(cudnn.data_type.FLOAT)
                self.stats_cudnn.set_dim((b, h_q, self.max_seq_len_q, 1))
                self.stats_cudnn.set_stride((self.max_seq_len_q * h_q, 1, h_q, 1))
                self.stats_cudnn.set_ragged_offset(self.stats_ragged_offset_cudnn)

        try:
            swa_graph.validate()
        except cudnn.cudnnGraphNotSupportedError as e:
            self._logger.error(
                f"Graph not supported (cudnnGraphNotSupportedError): {e}"
            )
            return False
        except Exception as e:
            self._logger.error(f"Graph not supported: {e}")
            return False

        self._cudnn_swa_graph = swa_graph
        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self, current_stream: Optional[cuda.CUstream] = None) -> None:
        if current_stream is not None:
            self._logger.warning(
                "Overwriting cudnn_handle stream with provided cuda stream. Do not pass in current_stream if this is not intended."
            )
            cudnn.set_stream(self._cudnn_handle, current_stream)
        self._ensure_support_checked()

        self._cudnn_swa_graph.build_operation_graph()
        self._cudnn_swa_graph.create_execution_plans(
            [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]
        )
        self._cudnn_swa_graph.check_support()
        self._cudnn_swa_graph.build_plans()

        self._cudnn_compiled = True
        self._logger.debug("SlidingWindowAttention kernel compiled successfully")

    def execute(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
        stats_tensor: torch.Tensor | None,
        seq_len_q_tensor: torch.Tensor = None,
        seq_len_kv_tensor: torch.Tensor = None,
        q_ragged_offset_tensor: torch.Tensor = None,
        k_ragged_offset_tensor: torch.Tensor = None,
        v_ragged_offset_tensor: torch.Tensor = None,
        o_ragged_offset_tensor: torch.Tensor = None,
        stats_ragged_offset_tensor: torch.Tensor = None,
        current_stream: Optional[cuda.CUstream] = None,
        cudnn_handle: cudnn.handle = None,
        skip_compile: bool = False,
    ) -> None:
        self._logger.debug("Entering execute")
        cudnn_handle = self._cudnn_handle if cudnn_handle is None else cudnn_handle
        if current_stream is not None:
            self._logger.info(
                "Overwriting cudnn_handle stream with provided cuda stream. Do not pass in current_stream if this is not intended."
            )
            cudnn.set_stream(cudnn_handle, current_stream)

        if skip_compile:
            raise NotImplementedError(
                "cudnn sliding window attention kernel does not support skip_compile"
            )
        if self._cudnn_swa_graph is None or not self._cudnn_compiled:
            raise ValueError("SlidingWindowAttention kernel not compiled")
        self._logger.debug("Executing with compiled kernel")

        self._logger.debug("Reshaping tensors to kernel expected format")
        stats_tensor = self._pad_tensor_to_ndim(
            stats_tensor, self.sample_o.ndim, "stats_tensor"
        )
        seq_len_q_tensor = self._pad_tensor_to_ndim(
            seq_len_q_tensor, 4, "seq_len_q_tensor"
        )
        seq_len_kv_tensor = self._pad_tensor_to_ndim(
            seq_len_kv_tensor, 4, "seq_len_kv_tensor"
        )
        q_ragged_offset_tensor = self._pad_tensor_to_ndim(
            q_ragged_offset_tensor, 4, "q_ragged_offset_tensor"
        )
        k_ragged_offset_tensor = self._pad_tensor_to_ndim(
            k_ragged_offset_tensor, 4, "k_ragged_offset_tensor"
        )
        v_ragged_offset_tensor = self._pad_tensor_to_ndim(
            v_ragged_offset_tensor, 4, "v_ragged_offset_tensor"
        )
        o_ragged_offset_tensor = self._pad_tensor_to_ndim(
            o_ragged_offset_tensor, 4, "o_ragged_offset_tensor"
        )
        stats_ragged_offset_tensor = self._pad_tensor_to_ndim(
            stats_ragged_offset_tensor, 4, "stats_ragged_offset_tensor"
        )

        if self.input_layout == "thd":
            if seq_len_q_tensor is None or seq_len_kv_tensor is None:
                raise ValueError(
                    f"seq_len_q_tensor and seq_len_kv_tensor must be provided for thd layout, got {seq_len_q_tensor} and {seq_len_kv_tensor}"
                )
            if (
                q_ragged_offset_tensor is None
                or k_ragged_offset_tensor is None
                or v_ragged_offset_tensor is None
                or o_ragged_offset_tensor is None
            ):
                raise ValueError(
                    f"q_ragged_offset_tensor, k_ragged_offset_tensor, v_ragged_offset_tensor, and o_ragged_offset_tensor must be provided for thd layout, got {q_ragged_offset_tensor}, {k_ragged_offset_tensor}, {v_ragged_offset_tensor}, and {o_ragged_offset_tensor}"
                )
            if not self.is_infer:
                if stats_ragged_offset_tensor is None:
                    raise ValueError(
                        f"stats_ragged_offset_tensor must be provided for thd layout, got {stats_ragged_offset_tensor}"
                    )

        variant_pack = {
            self.q_cudnn: q_tensor,
            self.k_cudnn: k_tensor,
            self.v_cudnn: v_tensor,
            self.o_cudnn: o_tensor,
            self.stats_cudnn: stats_tensor,
            self.seq_len_q_cudnn: seq_len_q_tensor,
            self.seq_len_kv_cudnn: seq_len_kv_tensor,
            self.q_ragged_offset_cudnn: q_ragged_offset_tensor,
            self.k_ragged_offset_cudnn: k_ragged_offset_tensor,
            self.v_ragged_offset_cudnn: v_ragged_offset_tensor,
            self.o_ragged_offset_cudnn: o_ragged_offset_tensor,
            self.stats_ragged_offset_cudnn: stats_ragged_offset_tensor,
        }

        workspace = torch.empty(
            self._cudnn_swa_graph.get_workspace_size(),
            device="cuda",
            dtype=torch.uint8,
        )
        self._cudnn_swa_graph.execute(variant_pack, workspace, handle=cudnn_handle)
        torch.cuda.synchronize()
        self._logger.debug("Executed successfully")

    def __call__(self, *args, **kwargs) -> None:
        self.execute(*args, skip_compile=True, **kwargs)


import logging

_logger = logging.getLogger(__name__)
_cache_of_SlidingWindowAttentionObjects = {}


def sliding_window_attention_wrapper(
    q_tensor: torch.Tensor,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
    seq_len_q_tensor: torch.Tensor,
    seq_len_kv_tensor: torch.Tensor,
    q_ragged_offset_tensor: torch.Tensor,
    k_ragged_offset_tensor: torch.Tensor,
    v_ragged_offset_tensor: torch.Tensor,
    o_ragged_offset_tensor: torch.Tensor,
    stats_ragged_offset_tensor: torch.Tensor,
    left_bound: int = 0,
    right_bound: int = 0,
    is_infer: bool = False,
    attn_scale: float = 1.0,
    intermediate_data_type: torch.dtype = torch.float32,
    compute_data_type: torch.dtype = torch.float32,
    cudnn_handle: cudnn.handle = None,
    stream: Optional[cuda.CUstream] = None,
):
    o_tensor, stats_tensor = None, None
    if q_tensor.ndim == 3:  # thd
        _logger.debug(
            "sliding_window_attention_wrapper: Creating empty output tensors o and stats for thd layout"
        )
        o_tensor = torch.empty(
            q_tensor.shape[0],
            q_tensor.shape[1],
            v_tensor.shape[2],
            device="cuda",
            dtype=q_tensor.dtype,
        )
        stats_tensor = torch.empty(
            q_tensor.shape[0], q_tensor.shape[1], 1, device="cuda", dtype=torch.float32
        )
    else:  # bshd
        _logger.debug(
            "sliding_window_attention_wrapper: Creating empty output tensors o and stats for bshd layout"
        )
        o_tensor = torch.empty(
            q_tensor.shape[0],
            q_tensor.shape[1],
            q_tensor.shape[2],
            v_tensor.shape[3],
            device="cuda",
            dtype=q_tensor.dtype,
        )
        stats_tensor = torch.empty(
            q_tensor.shape[0],
            q_tensor.shape[1],
            q_tensor.shape[2],
            1,
            device="cuda",
            dtype=torch.float32,
        )

    cache_key = (
        q_tensor.shape,
        k_tensor.shape,
        v_tensor.shape,
        seq_len_q_tensor.shape if seq_len_q_tensor is not None else None,
        seq_len_kv_tensor.shape if seq_len_kv_tensor is not None else None,
        q_ragged_offset_tensor.shape if q_ragged_offset_tensor is not None else None,
        k_ragged_offset_tensor.shape if k_ragged_offset_tensor is not None else None,
        v_ragged_offset_tensor.shape if v_ragged_offset_tensor is not None else None,
        o_ragged_offset_tensor.shape if o_ragged_offset_tensor is not None else None,
        (
            stats_ragged_offset_tensor.shape
            if stats_ragged_offset_tensor is not None
            else None
        ),
        q_tensor.stride(),
        k_tensor.stride(),
        v_tensor.stride(),
        seq_len_q_tensor.stride() if seq_len_q_tensor is not None else None,
        seq_len_kv_tensor.stride() if seq_len_kv_tensor is not None else None,
        q_ragged_offset_tensor.stride() if q_ragged_offset_tensor is not None else None,
        k_ragged_offset_tensor.stride() if k_ragged_offset_tensor is not None else None,
        v_ragged_offset_tensor.stride() if v_ragged_offset_tensor is not None else None,
        o_ragged_offset_tensor.stride() if o_ragged_offset_tensor is not None else None,
        (
            stats_ragged_offset_tensor.stride()
            if stats_ragged_offset_tensor is not None
            else None
        ),
        q_tensor.dtype,
        k_tensor.dtype,
        v_tensor.dtype,
        left_bound,
        right_bound,
        is_infer,
        attn_scale,
        intermediate_data_type,
        compute_data_type,
    )
    sliding_window_attention_object = None
    if cache_key in _cache_of_SlidingWindowAttentionObjects:
        _logger.debug(
            "sliding_window_attention_wrapper: Using previously cached SlidingWindowAttention object"
        )
        sliding_window_attention_object = _cache_of_SlidingWindowAttentionObjects[
            cache_key
        ]

        assert sliding_window_attention_object.check_support()
        sliding_window_attention_object.compile(current_stream=stream)
        sliding_window_attention_object.execute(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            o_tensor=o_tensor,
            stats_tensor=stats_tensor,
            seq_len_q_tensor=seq_len_q_tensor,
            seq_len_kv_tensor=seq_len_kv_tensor,
            q_ragged_offset_tensor=q_ragged_offset_tensor,
            k_ragged_offset_tensor=k_ragged_offset_tensor,
            v_ragged_offset_tensor=v_ragged_offset_tensor,
            o_ragged_offset_tensor=o_ragged_offset_tensor,
            stats_ragged_offset_tensor=stats_ragged_offset_tensor,
            current_stream=stream,
        )
    else:
        _logger.debug(
            "sliding_window_attention_wrapper: No previously cached SlidingWindowAttention object found, creating new SlidingWindowAttention object"
        )
        sliding_window_attention_object = SlidingWindowAttention(
            sample_q=q_tensor,
            sample_k=k_tensor,
            sample_v=v_tensor,
            sample_o=o_tensor,
            sample_stats=stats_tensor,
            sample_seq_len_q=seq_len_q_tensor,
            sample_seq_len_kv=seq_len_kv_tensor,
            sample_q_ragged_offset=q_ragged_offset_tensor,
            sample_k_ragged_offset=k_ragged_offset_tensor,
            sample_v_ragged_offset=v_ragged_offset_tensor,
            sample_o_ragged_offset=o_ragged_offset_tensor,
            sample_stats_ragged_offset=stats_ragged_offset_tensor,
            max_seq_len_q=(
                max(seq_len_q_tensor).item() if seq_len_q_tensor is not None else None
            ),
            max_seq_len_kv=(
                max(seq_len_kv_tensor).item() if seq_len_kv_tensor is not None else None
            ),
            left_bound=left_bound,
            right_bound=right_bound,
            is_infer=is_infer,
            attn_scale=attn_scale,
            intermediate_data_type=intermediate_data_type,
            compute_data_type=compute_data_type,
            cudnn_handle=cudnn_handle,
        )

        assert sliding_window_attention_object.check_support()
        sliding_window_attention_object.compile(current_stream=stream)
        sliding_window_attention_object.execute(
            q_tensor=q_tensor,
            k_tensor=k_tensor,
            v_tensor=v_tensor,
            o_tensor=o_tensor,
            stats_tensor=stats_tensor,
            seq_len_q_tensor=seq_len_q_tensor,
            seq_len_kv_tensor=seq_len_kv_tensor,
            q_ragged_offset_tensor=q_ragged_offset_tensor,
            k_ragged_offset_tensor=k_ragged_offset_tensor,
            v_ragged_offset_tensor=v_ragged_offset_tensor,
            o_ragged_offset_tensor=o_ragged_offset_tensor,
            stats_ragged_offset_tensor=stats_ragged_offset_tensor,
            current_stream=stream,
        )
        _cache_of_SlidingWindowAttentionObjects[cache_key] = (
            sliding_window_attention_object
        )

    return o_tensor, stats_tensor

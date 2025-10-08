from .dense_gemm_persistent_swiglu import PersistentDenseGemmKernel

from cuda.bindings import driver as cuda
import torch
from typing import Tuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cutlass.cute.math as math

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import APIBase


class GemmSwiglu(APIBase):
    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_b: torch.Tensor,
        sample_c: torch.Tensor,
        sample_glu: torch.Tensor,
        alpha: float = 1.0,
        acc_dtype: torch.dtype = torch.float32,
        use_2cta_instrs: bool = False,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
    ):
        super().__init__()
        self._kernel = PersistentDenseGemmKernel

        self._logger.warning("GemmSwiglu is an experimental API")
        self._logger.critical("GemmSwiglu is currently failing ref check")
        self._logger.debug("Entering __init__")

        self.sample_a = sample_a
        self.sample_b = sample_b
        self.sample_c = sample_c
        self.sample_glu = sample_glu
        self.alpha = alpha
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn

        self._logger.debug(
            f"__init__ completed with args: sample_a {sample_a.shape}, sample_b {sample_b.shape}, sample_c {sample_c.shape}, sample_glu {sample_glu.shape}, alpha {alpha}, acc_dtype {acc_dtype}, use_2cta_instrs {use_2cta_instrs}, mma_tiler_mn {mma_tiler_mn}, cluster_shape_mn {cluster_shape_mn}"
        )

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")
        m, k, l = self.sample_a.shape
        n, k, l = self.sample_b.shape
        m, n, l = self.sample_c.shape
        m, n_2, l = self.sample_glu.shape
        ab_dtype = self.sample_a.dtype
        c_dtype = self.sample_c.dtype

        assert self.sample_a.shape == (m, k, l), "Input/Output shape mismatch"
        assert self.sample_b.shape == (n, k, l), "Input/Output shape mismatch"
        assert self.sample_c.shape == (m, n, l), "Input/Output shape mismatch"
        assert self.sample_glu.shape == (m, n // 2, l), "Input/Output shape mismatch"
        assert (
            self.sample_a.dtype == self.sample_b.dtype
        ), "A and B tensor dtypes must match"

        if self.sample_a.stride() == (1, m, m * k):
            a_major = "m"
        elif self.sample_a.stride() == (k, 1, m * k):
            a_major = "k"
        else:
            raise ValueError(
                f"Unsupported A tensor stride pattern: {self.sample_a.stride()}. Expected (1, m, m * k) or (k, 1, m * k)"
            )

        if self.sample_b.stride() == (1, n, n * k):
            b_major = "n"
        elif self.sample_b.stride() == (k, 1, n * k):
            b_major = "k"
        else:
            raise ValueError(
                f"Unsupported B tensor stride pattern: {self.sample_b.stride()}. Expected (1, n, n * k) or (k, 1, n * k)"
            )

        if self.sample_c.stride() == (1, m, m * n) and self.sample_glu.stride() == (
            1,
            m,
            m * n_2,
        ):
            c_major = "m"
        elif self.sample_c.stride() == (n, 1, m * n) and self.sample_glu.stride() == (
            n_2,
            1,
            m * n_2,
        ):
            c_major = "n"
        else:
            raise ValueError(
                f"Unsupported C/glu tensor stride pattern: {self.sample_c.stride()}. Expected (1, m, m * n) or (n, 1, m * n) for C and (1, m, m * n_2) or (n_2, 1, m * n_2) for GLU"
            )

        assert self._kernel.is_valid_dtypes(
            _convert_to_cutlass_data_type(ab_dtype),
            _convert_to_cutlass_data_type(self.acc_dtype),
            _convert_to_cutlass_data_type(c_dtype),
        ), "Unsupported data types"

        assert self._kernel.is_valid_mma_tiler_and_cluster_shape(
            self.use_2cta_instrs, self.mma_tiler_mn, self.cluster_shape_mn
        ), "Invalid MMA tile shape or cluster shape"

        assert self._kernel.is_valid_tensor_alignment(
            m,
            n,
            k,
            l,
            _convert_to_cutlass_data_type(ab_dtype),
            _convert_to_cutlass_data_type(c_dtype),
            a_major,
            b_major,
            c_major,
        ), "Invalid tensor alignment for problem shape"

        assert self._kernel.is_valid_epilog_store_option(
            self.use_2cta_instrs, True, m, n, self.mma_tiler_mn
        ), "Invalid epilogue store option"

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self, current_stream: Optional[cuda.CUstream] = None) -> None:
        self._logger.debug("Entering compile")
        current_stream = self._get_default_stream(current_stream)
        self._ensure_support_checked()

        gemm_swiglu = self._kernel(
            _convert_to_cutlass_data_type(self.acc_dtype),
            self.use_2cta_instrs,
            self.mma_tiler_mn,
            self.cluster_shape_mn,
            True,
        )
        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        )

        self._logger.debug("Compiling gemm_swiglu")
        self._compiled_kernel = cute.compile(
            gemm_swiglu,
            from_dlpack(self.sample_a),
            from_dlpack(self.sample_b),
            from_dlpack(self.sample_c),
            from_dlpack(self.sample_glu),
            self.alpha,
            max_active_clusters,
            current_stream,
            lambda x: x / (1 + math.exp(-1 * x, True)),
        )
        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        a_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        glu_tensor: torch.Tensor,
        alpha: float = 1.0,
        current_stream: Optional[cuda.CUstream] = None,
        skip_compile: bool = False,
    ) -> None:
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        if not skip_compile:
            assert self._compiled_kernel is not None, "GemmSwiglu not compiled"
            self._logger.debug("Executing with compiled kernel")
            self._compiled_kernel(
                from_dlpack(a_tensor),
                from_dlpack(b_tensor),
                from_dlpack(c_tensor),
                from_dlpack(glu_tensor),
                alpha,
                current_stream,
            )
            self._logger.debug("Executed with compiled kernel successfully")
        else:
            self._logger.debug("Executing without compiled kernel (JIT)")
            gemm_swiglu = self._kernel(
                _convert_to_cutlass_data_type(self.acc_dtype),
                self.use_2cta_instrs,
                self.mma_tiler_mn,
                self.cluster_shape_mn,
                True,
            )
            gemm_swiglu(
                from_dlpack(a_tensor),
                from_dlpack(b_tensor),
                from_dlpack(c_tensor),
                from_dlpack(glu_tensor),
                alpha,
                current_stream,
            )
            self._logger.debug("Executed successfully")


import logging

_logger = logging.getLogger(__name__)
_cache_of_GemmSwigluObjects = {}


def gemm_swiglu_wrapper(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    alpha: float = 1.0,
    c_major: str = "n",
    c_dtype: torch.dtype = torch.float32,
    glu_dtype: torch.dtype = torch.float16,
    acc_dtype: torch.dtype = torch.float32,
    use_2cta_instrs: bool = False,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    stream: Optional[cuda.CUstream] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    _logger.debug("gemm_swiglu_wrapper: Creating empty output tensors c and glu")
    m, k, l = a_tensor.shape
    n, k, l = b_tensor.shape
    c_tensor, glu_tensor = None, None
    if c_major == "m":

        c_tensor = torch.empty_strided(
            (m, n, l), (1, m, m * n), dtype=c_dtype, device="cuda"
        )
        glu_tensor = torch.empty_strided(
            (m, n // 2, l), (1, m, m * n // 2), dtype=glu_dtype, device="cuda"
        )
    elif c_major == "n":
        c_tensor = torch.empty_strided(
            (m, n, l), (n, 1, m * n), dtype=c_dtype, device="cuda"
        )
        glu_tensor = torch.empty_strided(
            (m, n // 2, l), (n // 2, 1, m * n // 2), dtype=glu_dtype, device="cuda"
        )
    else:
        raise ValueError(f"c_major must be either 'm' or 'n', got {c_major}")

    cache_key = (
        a_tensor.shape,
        b_tensor.shape,
        a_tensor.dtype,
        b_tensor.dtype,
        a_tensor.stride(),
        b_tensor.stride(),
        alpha,
        c_major,
        c_dtype,
        glu_dtype,
        acc_dtype,
        use_2cta_instrs,
        mma_tiler_mn,
        cluster_shape_mn,
    )
    if cache_key in _cache_of_GemmSwigluObjects:
        _logger.debug("gemm_swiglu_wrapper: Using previously cached GemmSwiglu object")
        gemm_swiglu = _cache_of_GemmSwigluObjects[cache_key]
        gemm_swiglu.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            c_tensor=c_tensor,
            glu_tensor=glu_tensor,
            alpha=alpha,
            current_stream=stream,
        )
    else:
        _logger.debug(
            "gemm_swiglu_wrapper: No previously cached GemmSwiglu object found, creating new GemmSwiglu object"
        )
        gemm_swiglu = GemmSwiglu(
            sample_a=a_tensor,
            sample_b=b_tensor,
            sample_c=c_tensor,
            sample_glu=glu_tensor,
            alpha=alpha,
            acc_dtype=acc_dtype,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
        )
        assert gemm_swiglu.check_support(), "Unsupported testcase"
        gemm_swiglu.compile(current_stream=stream)
        gemm_swiglu.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            c_tensor=c_tensor,
            glu_tensor=glu_tensor,
            alpha=alpha,
            current_stream=stream,
        )
        _cache_of_GemmSwigluObjects[cache_key] = gemm_swiglu

    return c_tensor, glu_tensor

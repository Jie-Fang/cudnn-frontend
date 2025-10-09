from .dense_blockscaled_gemm_persistent_amax import (
    Sm100BlockScaledPersistentDenseGemmKernel,
)

from cuda.bindings import driver as cuda
import torch
from typing import Tuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import APIBase


class GemmAmax(APIBase):
    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_b: torch.Tensor,
        sample_sfa: torch.Tensor,
        sample_sfb: torch.Tensor,
        sample_c: torch.Tensor,
        sample_amax: torch.Tensor,
        acc_dtype: torch.dtype = torch.float32,
        mma_tiler_mn: Tuple[int, int] = (128, 128),
        cluster_shape_mn: Tuple[int, int] = (1, 1),
        sf_vec_size: int = 32,
    ):
        super().__init__()
        self._kernel = Sm100BlockScaledPersistentDenseGemmKernel

        self._logger.warning("GemmAmax is an experimental API")
        self._logger.debug("Entering __init__")

        self.sample_a = sample_a
        self.sample_b = sample_b
        self.sample_sfa = sample_sfa
        self.sample_sfb = sample_sfb
        self.sample_c = sample_c
        self.sample_amax = sample_amax
        if self.sample_amax.dim() < 3:
            self._logger.info(
                f"Reshaping sample_amax to (1, 1, 1) from {self.sample_amax.shape}"
            )
            for _ in range(3 - self.sample_amax.dim()):
                self.sample_amax = self.sample_amax.unsqueeze(-1)
        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = mma_tiler_mn
        self.cluster_shape_mn = cluster_shape_mn
        self.sf_vec_size = sf_vec_size

        # used to reshape sfa/sfb tensors to atom layout
        self.atom_m = (32, 4)
        self.atom_k = 4

        self._logger.debug(
            f"__init__ completed with args: sample_a {sample_a.shape}, sample_b {sample_b.shape}, sample_sfa {sample_sfa.shape}, sample_sfb {sample_sfb.shape}, sample_c {sample_c.shape}, sample_amax {sample_amax.shape}, acc_dtype {acc_dtype}, mma_tiler_mn {mma_tiler_mn}, cluster_shape_mn {cluster_shape_mn}, sf_vec_size {sf_vec_size}"
        )

    def check_support(self) -> bool:
        self._logger.debug("Entering check_support")

        ab_dtype = self.sample_a.dtype
        sf_dtype = self.sample_sfa.dtype
        c_dtype = self.sample_c.dtype
        if ab_dtype == torch.float4_e2m1fn_x2 or c_dtype == torch.float4_e2m1fn_x2:
            self._logger.warning(
                "Running GemmAmax with float4_e2m1fn_x2 is not numerically correct due to limited support and may not give correct results."
            )
        m, k, l = self.sample_a.shape
        n, k, l = self.sample_b.shape
        m, n, l = self.sample_c.shape
        _, _, m_div_atom_m0_m1, _, sf_k_div_atom_k, l = self.sample_sfa.shape
        _, _, n_div_atom_m0_m1, _, sf_k_div_atom_k, l = self.sample_sfb.shape
        _, _, _ = self.sample_amax.shape

        self._logger.debug("Checking dtypes and sf_vec_size")
        assert (
            self.sample_a.dtype == self.sample_b.dtype
        ), "A and B tensor dtypes must match"
        assert ab_dtype in {
            torch.float4_e2m1fn_x2,
            torch.float8_e5m2,
            torch.float8_e4m3fn,
        }, "Unsupported ab_dtype"
        assert self.sf_vec_size in {16, 32}, "Unsupported sf_vec_size"
        assert (
            sf_dtype != torch.float8_e8m0fnu
        ), "Please pass in sf tensors as torch.int8 instead of torch.float8_e8m0fnu"
        assert sf_dtype in {
            torch.float8_e8m0fnu,
            torch.float8_e4m3fn,
            torch.int8,
        }, "Unsupported sf_dtype"
        assert not (
            sf_dtype == torch.float8_e4m3fn and self.sf_vec_size == 32
        ), "Unsupported sf_dtype and sf_vec_size combination"
        assert not (
            ab_dtype in {torch.float8_e5m2, torch.float8_e4m3fn}
            and self.sf_vec_size == 16
        ), "Unsupported ab_dtype and sf_vec_size combination"
        assert c_dtype in {
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.float8_e5m2,
            torch.float8_e4m3fn,
            torch.float4_e2m1fn_x2,
        }, "Unsupported c_dtype"
        assert (
            self.acc_dtype == torch.float32
        ), "Unsupported acc_dtype: accumulator dtype must be float32"

        self._logger.debug("Checking tensor layout")
        assert self.sample_a.shape == (m, k, l), "Input/Output shape mismatch"
        assert self.sample_b.shape == (n, k, l), "Input/Output shape mismatch"
        assert self.sample_c.shape == (m, n, l), "Input/Output shape mismatch"
        assert self.sample_sfa.shape == (
            self.atom_m[0],
            self.atom_m[1],
            m_div_atom_m0_m1,
            self.atom_k,
            sf_k_div_atom_k,
            l,
        ), "Input/Output shape mismatch"
        assert self.sample_sfb.shape == (
            self.atom_m[0],
            self.atom_m[1],
            n_div_atom_m0_m1,
            self.atom_k,
            sf_k_div_atom_k,
            l,
        ), "Input/Output shape mismatch"
        assert self.sample_amax.shape == (1, 1, 1), "Input/Output shape mismatch"
        assert m_div_atom_m0_m1 == (m + self.atom_m[0] * self.atom_m[1] - 1) // (
            self.atom_m[0] * self.atom_m[1]
        ), "Input/Output shape mismatch"
        assert n_div_atom_m0_m1 == (n + self.atom_m[0] * self.atom_m[1] - 1) // (
            self.atom_m[0] * self.atom_m[1]
        ), "Input/Output shape mismatch"
        if self.sample_a.stride() == (1, m, m * k):
            self.a_major = "m"
        elif self.sample_a.stride() == (k, 1, m * k):
            self.a_major = "k"
        else:
            raise ValueError(
                f"Unsupported A tensor stride pattern: {self.sample_a.stride()}. Expected (1, m, m * k) or (k, 1, m * k)"
            )
        if self.sample_b.stride() == (1, n, n * k):
            self.b_major = "n"
        elif self.sample_b.stride() == (k, 1, n * k):
            self.b_major = "k"
        else:
            raise ValueError(
                f"Unsupported B tensor stride pattern: {self.sample_b.stride()}. Expected (1, n, n * k) or (k, 1, n * k)"
            )
        if self.sample_c.stride() == (1, m, m * n):
            self.c_major = "m"
        elif self.sample_c.stride() == (n, 1, m * n):
            self.c_major = "n"
        else:
            raise ValueError(
                f"Unsupported C tensor stride pattern: {self.sample_c.stride()}. Expected (1, m, m * n) or (n, 1, m * n) for C"
            )
        assert not (
            ab_dtype is torch.float4_e2m1fn_x2
            and not (self.a_major == "k" and self.b_major == "k")
        ), "Unsupported ab_dtype and layout combination"
        assert not (
            c_dtype is torch.float4_e2m1fn_x2 and self.c_major == "m"
        ), "Unsupported c_dtype and layout combination"

        self._logger.debug("Checking mma tiler and cluster shape")
        assert self.mma_tiler_mn[0] in [128, 256], "Unsupported mma tile shape"
        assert self.mma_tiler_mn[1] in [128, 256], "Unsupported mma tile shape"
        assert (
            self.cluster_shape_mn[0] % (2 if self.mma_tiler_mn[0] == 256 else 1) == 0
        ), "Illegal cluster shape"

        def is_power_of_2(x):
            return x > 0 and (x & (x - 1)) == 0

        assert (
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1] <= 16
            and self.cluster_shape_mn[0] > 0
            and self.cluster_shape_mn[1] > 0
        ), "Invalid cluster shape"
        # Special cluster shape check for scale factor multicasts.
        # Due to limited size of scale factors, we can't multicast among more than 4 CTAs.
        assert (
            self.cluster_shape_mn[0] <= 4
            and self.cluster_shape_mn[1] <= 4
            and is_power_of_2(self.cluster_shape_mn[0])
            and is_power_of_2(self.cluster_shape_mn[1])
        ), "Invalid cluster shape"

        self._logger.debug("Checking tensor alignment")

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = (
                16 * 8 // (_convert_to_cutlass_data_type(dtype).width)
            )
            return num_major_elements % num_contiguous_elements == 0

        assert (
            check_contigous_16B_alignment(ab_dtype, self.a_major == "m", (m, k, l))
            and check_contigous_16B_alignment(ab_dtype, self.b_major == "n", (n, k, l))
            and check_contigous_16B_alignment(c_dtype, self.c_major == "m", (m, n, l))
        ), "Unsupported tensor alignment"

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self, current_stream: Optional[cuda.CUstream] = None) -> None:
        self._logger.debug("Entering compile")
        current_stream = self._get_default_stream(current_stream)
        self._ensure_support_checked()

        gemm_amax = self._kernel(
            self.sf_vec_size,
            self.mma_tiler_mn,
            self.cluster_shape_mn,
        )
        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        )

        sample_a_cute = from_dlpack(
            self.sample_a, assumed_align=16
        ).mark_compact_shape_dynamic(
            mode=1 if self.a_major == "k" else 0,
            stride_order=(2, 0, 1) if self.a_major == "k" else (2, 1, 0),
            divisibility=32 if self.sample_a.dtype == torch.float4_e2m1fn_x2 else 16,
        )
        sample_b_cute = from_dlpack(
            self.sample_b, assumed_align=16
        ).mark_compact_shape_dynamic(
            mode=1 if self.b_major == "k" else 0,
            stride_order=(2, 0, 1) if self.b_major == "k" else (2, 1, 0),
            divisibility=32 if self.sample_b.dtype == torch.float4_e2m1fn_x2 else 16,
        )
        sample_c_cute = from_dlpack(
            self.sample_c, assumed_align=16
        ).mark_compact_shape_dynamic(
            mode=1 if self.c_major == "n" else 0,
            stride_order=(2, 0, 1) if self.c_major == "n" else (2, 1, 0),
            divisibility=32 if self.sample_c.dtype == torch.float4_e2m1fn_x2 else 16,
        )

        self._logger.debug("Compiling gemm_amax")
        self._compiled_kernel = cute.compile(
            gemm_amax,
            sample_a_cute,
            sample_b_cute,
            from_dlpack(self.sample_sfa, assumed_align=16),
            from_dlpack(self.sample_sfb, assumed_align=16),
            sample_c_cute,
            from_dlpack(self.sample_amax, assumed_align=16),
            max_active_clusters,
            current_stream,
        )
        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        a_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        sfb_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        amax_tensor: torch.Tensor,
        current_stream: Optional[cuda.CUstream] = None,
        skip_compile: bool = False,
    ) -> None:
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        if amax_tensor.dim() < 3:
            self._logger.info(
                f"Reshaping amax_tensor to (1, 1, 1) from {amax_tensor.shape}"
            )
            for _ in range(3 - amax_tensor.dim()):
                amax_tensor = amax_tensor.unsqueeze(-1)

        a_tensor_cute = from_dlpack(
            a_tensor, assumed_align=16
        ).mark_compact_shape_dynamic(
            mode=1 if self.a_major == "k" else 0,
            stride_order=(2, 0, 1) if self.a_major == "k" else (2, 1, 0),
            divisibility=32 if self.sample_a.dtype == torch.float4_e2m1fn_x2 else 16,
        )
        b_tensor_cute = from_dlpack(
            b_tensor, assumed_align=16
        ).mark_compact_shape_dynamic(
            mode=1 if self.b_major == "k" else 0,
            stride_order=(2, 0, 1) if self.b_major == "k" else (2, 1, 0),
            divisibility=32 if self.sample_b.dtype == torch.float4_e2m1fn_x2 else 16,
        )
        c_tensor_cute = from_dlpack(
            c_tensor, assumed_align=16
        ).mark_compact_shape_dynamic(
            mode=1 if self.c_major == "n" else 0,
            stride_order=(2, 0, 1) if self.c_major == "n" else (2, 1, 0),
            divisibility=32 if self.sample_c.dtype == torch.float4_e2m1fn_x2 else 16,
        )
        if not skip_compile:
            assert self._compiled_kernel is not None, "GemmAmax not compiled"
            self._logger.debug("Executing with compiled kernel")
            self._compiled_kernel(
                a_tensor_cute,
                b_tensor_cute,
                from_dlpack(sfa_tensor, assumed_align=16),
                from_dlpack(sfb_tensor, assumed_align=16),
                c_tensor_cute,
                from_dlpack(amax_tensor, assumed_align=16),
                current_stream,
            )
            self._logger.debug("Executed with compiled kernel successfully")
        else:
            self._logger.debug("Executing without compiled kernel (JIT)")
            gemm_amax = self._kernel(
                self.sf_vec_size,
                self.mma_tiler_mn,
                self.cluster_shape_mn,
            )
            gemm_amax(
                a_tensor_cute,
                b_tensor_cute,
                from_dlpack(sfa_tensor, assumed_align=16),
                from_dlpack(sfb_tensor, assumed_align=16),
                c_tensor_cute,
                from_dlpack(amax_tensor, assumed_align=16),
                current_stream,
            )
            self._logger.debug("Executed successfully")


import logging

_logger = logging.getLogger(__name__)
_cache_of_GemmAmaxObjects = {}


def gemm_amax_wrapper(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    sfa_tensor: torch.Tensor,
    sfb_tensor: torch.Tensor,
    c_major: str = "n",
    c_dtype: torch.dtype = torch.float32,
    acc_dtype: torch.dtype = torch.float32,
    mma_tiler_mn: Tuple[int, int] = (128, 128),
    cluster_shape_mn: Tuple[int, int] = (1, 1),
    sf_vec_size: int = 32,
    stream: Optional[cuda.CUstream] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    _logger.debug("gemm_amax_wrapper: Creating empty output tensors c and amax")

    m, _, l = a_tensor.shape
    n, _, l = b_tensor.shape
    c_tensor = None
    if c_major == "m":
        c_tensor = torch.empty_strided(
            (m, n, l), (1, m, m * n), dtype=c_dtype, device="cuda"
        )
    elif c_major == "n":
        c_tensor = torch.empty_strided(
            (m, n, l), (n, 1, m * n), dtype=c_dtype, device="cuda"
        )
    else:
        raise ValueError(f"c_major must be either 'm' or 'n', got {c_major}")
    amax_tensor = torch.full(
        (1, 1, 1), -float("inf"), device="cuda", dtype=torch.float32
    )

    cache_key = (
        a_tensor.shape,
        b_tensor.shape,
        sfa_tensor.shape,
        sfb_tensor.shape,
        a_tensor.dtype,
        b_tensor.dtype,
        sfa_tensor.dtype,
        sfb_tensor.dtype,
        a_tensor.stride(),
        b_tensor.stride(),
        sfa_tensor.stride(),
        sfb_tensor.stride(),
        c_major,
        c_dtype,
        acc_dtype,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
    )
    if cache_key in _cache_of_GemmAmaxObjects:
        _logger.debug("gemm_amax_wrapper: Using previously cached GemmAmax object")
        gemm_amax = _cache_of_GemmAmaxObjects[cache_key]
        gemm_amax.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            c_tensor=c_tensor,
            amax_tensor=amax_tensor,
            current_stream=stream,
        )
    else:
        _logger.debug(
            "gemm_amax_wrapper: No previously cached GemmAmax object found, creating new GemmAmax object"
        )
        gemm_amax = GemmAmax(
            sample_a=a_tensor,
            sample_b=b_tensor,
            sample_sfa=sfa_tensor,
            sample_sfb=sfb_tensor,
            sample_c=c_tensor,
            sample_amax=amax_tensor,
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
        )
        assert gemm_amax.check_support()
        gemm_amax.compile(current_stream=stream)
        gemm_amax.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            c_tensor=c_tensor,
            amax_tensor=amax_tensor,
            current_stream=stream,
        )
        _cache_of_GemmAmaxObjects[cache_key] = gemm_amax

    return c_tensor, amax_tensor

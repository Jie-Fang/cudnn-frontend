# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
API for Grouped GEMM dSwiGLU Backward Kernel (SM100+)

This module provides the API class for contiguous grouped block-scaled GEMM
backward pass with dSwiGLU activation gradient for MoE (Mixture of Experts) workloads.
"""

from .grouped_gemm_dswiglu_quant import (
    BlockScaledContiguousGroupedGemmKernel,
    BlockScaledContiguousGroupedGemmKernelNoDlpack,
)
from cuda.bindings import driver as cuda
import torch
from typing import Tuple, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from packaging import version

from cudnn.datatypes import _convert_to_cutlass_data_type
from cudnn.api_base import APIBase, TupleDict, ceil_div, is_power_of_2


class GroupedGemmDswigluSm100(APIBase):
    """API class for Grouped GEMM dSwiGLU backward operation on SM100+ GPUs.

    This kernel performs contiguous grouped block-scaled GEMM backward pass
    with dSwiGLU activation gradient, designed for MoE (Mixture of Experts) workloads.

    Key features:
    - Supports variable M per group (aligned to m_aligned)
    - Contiguous memory layout for A and D tensors
    - Block-scaled quantization support (MXF8, MXF4, NVF4)
    - Uses padded_offsets interface for expert mapping

    Example:
        >>> api = GroupedGemmDswigluSm100(
        ...     sample_a=a_tensor,
        ...     ...
        ... )
        >>> api.check_support()
        >>> api.compile()
        >>> api.execute(..., stream)
    """

    def __init__(
        self,
        sample_a: torch.Tensor,
        sample_b: torch.Tensor,
        sample_c: torch.Tensor,
        sample_d_row: torch.Tensor,
        sample_d_col: torch.Tensor,
        sample_sfa: torch.Tensor,
        sample_sfb: torch.Tensor,
        sample_padded_offsets: torch.Tensor,
        sample_alpha: torch.Tensor,
        sample_beta: torch.Tensor,
        sample_prob: torch.Tensor,
        sample_dprob: torch.Tensor,
        # Optional quantization output arguments
        sample_sfd_row: Optional[torch.Tensor] = None,
        sample_sfd_col: Optional[torch.Tensor] = None,
        sample_amax: Optional[torch.Tensor] = None,
        sample_norm_const: Optional[torch.Tensor] = None,
        # Configuration
        acc_dtype: torch.dtype = torch.float32,
        mma_tiler_mn: Tuple[int, int] = (256, 256),
        cluster_shape_mn: Optional[Tuple[int, int]] = None,
        sf_vec_size: int = 16,
        vector_f32: bool = False,
        m_aligned: int = 256,
        discrete_col_sfd: bool = False,
        epilogue_op: Optional[str] = None,
    ):
        """Initialize the GroupedGemmDswigluSm100 API.

        :param sample_a: Sample A tensor (valid_m, k, 1)
        :param sample_b: Sample B tensor (n, k, l) where l = num_groups
        :param sample_c: Sample C tensor for intermediate storage (valid_m, 2n, 1)
        :param sample_d_row: Sample D row output tensor (valid_m, 2n, 1) after dSwiGLU
        :param sample_d_col: Sample D column output tensor (valid_m, 2n, 1) after dSwiGLU
        :param sample_sfa: Sample scale factor A tensor
        :param sample_sfb: Sample scale factor B tensor
        :param sample_padded_offsets: End offset for each expert after padding, shape (expert_cnt,)
        :param sample_alpha: Per-group alpha scaling factors
        :param sample_beta: Per-group beta scaling factors
        :param sample_prob: Per-row probability tensor (valid_m, 1, 1)
        :param sample_dprob: Gradient of probability tensor (valid_m, 1, 1). Must be zero-initialized.
        :param sample_sfd_row: Optional row scale factor for D
        :param sample_sfd_col: Optional column scale factor for D
        :param sample_amax: Optional amax tensor for quantization, shape (l, 2, 1)
        :param sample_norm_const: Optional normalization constant
        :param acc_dtype: Accumulator data type
        :param mma_tiler_mn: MMA tiler shape (M, N)
        :param cluster_shape_mn: Cluster shape (M, N)
        :param sf_vec_size: Scale factor vector size
        :param vector_f32: Use vectorized f32 operations
        :param m_aligned: Alignment for group M dimension
        :param discrete_col_sfd: Boolean, True to generate discrete col-major scale factor tensor
        :param epilogue_op: Optional epilogue operation. Valid values: None, "none", "identity", "relu", "srelu"
        """
        super().__init__()

        self._logger.warning("GroupedGemmDswigluSm100 is an experimental API")
        self._logger.debug("Entering __init__")

        # Store sample tensors
        self.sample_a = sample_a
        self.sample_b = sample_b
        self.sample_c = sample_c
        self.sample_d_row = sample_d_row
        self.sample_d_col = sample_d_col
        self.sample_sfa = sample_sfa
        self.sample_sfb = sample_sfb
        self.sample_padded_offsets = sample_padded_offsets
        self.sample_alpha = sample_alpha
        self.sample_beta = sample_beta
        self.sample_prob = sample_prob
        self.sample_dprob = sample_dprob

        # Optional quantization outputs
        self.sample_sfd_row = sample_sfd_row
        self.sample_sfd_col = sample_sfd_col
        self.sample_amax = sample_amax
        self.sample_norm_const = self._unpad_tensor_to_ndim(sample_norm_const, 1, "norm_const")

        # Configuration
        self.acc_dtype = acc_dtype
        self.mma_tiler_mn = mma_tiler_mn
        self.use_2cta_instrs = mma_tiler_mn[0] == 256
        if cluster_shape_mn is None:
            self.cluster_shape_mn = (2, 1) if self.use_2cta_instrs else (1, 1)
        else:
            self.cluster_shape_mn = cluster_shape_mn
        self.sf_vec_size = sf_vec_size
        self.vector_f32 = vector_f32
        self.m_aligned = m_aligned
        self.discrete_col_sfd = discrete_col_sfd

        # expert_cnt derived from padded_offsets shape
        self.expert_cnt = sample_padded_offsets.shape[0]

        # Epilogue operation
        if epilogue_op in [None, "none", "identity"]:
            self.epilogue_op = lambda x: x
        elif epilogue_op == "relu":
            self.epilogue_op = lambda x: cute.where(x > 0, x, cute.full_like(x, 0))
        elif epilogue_op == "srelu":
            self.epilogue_op = lambda x: cute.where(x > 0, x, cute.full_like(x, 0)) ** 2
        else:
            raise ValueError(f"Invalid epilogue operation: {epilogue_op}. Valid values: None, 'none', 'identity', 'relu', 'srelu'")

        self._interpret_uint8_as_fp4x2 = True
        # Determine kernel variant based on sample tensor dtypes
        # NoDlpack kernels are required for:
        # - FP4 dtypes (any of ab_dtype, c_dtype, d_dtype)
        # - FP8 dtypes on PyTorch < 2.10.0
        ab_dtype = self.sample_a.dtype
        c_dtype = self.sample_c.dtype
        d_dtype = self.sample_d_row.dtype
        torch_version = version.parse(torch.__version__)
        is_ab_fp4 = self._is_fp4x2(ab_dtype)
        is_c_fp4 = self._is_fp4x2(c_dtype)
        is_d_fp4 = self._is_fp4x2(d_dtype)
        is_ab_fp8 = self._is_fp8(ab_dtype)
        is_c_fp8 = self._is_fp8(c_dtype)
        is_d_fp8 = self._is_fp8(d_dtype)
        _fp8_dlpack_supported = version.parse(torch_version.base_version) >= version.parse("2.10.0")
        use_no_dlpack_kernel = is_ab_fp4 or is_c_fp4 or is_d_fp4 or ((is_ab_fp8 or is_c_fp8 or is_d_fp8) and not _fp8_dlpack_supported)

        if use_no_dlpack_kernel:
            self._logger.debug("Using NoDlpack kernel due to FP4 dtype or FP8 dtype on incompatible torch version")
            self._kernel = BlockScaledContiguousGroupedGemmKernelNoDlpack
        else:
            self._kernel = BlockScaledContiguousGroupedGemmKernel
        self._logger.debug(f"__init__ completed")

    def check_support(self) -> bool:
        """Check if the kernel configuration is supported.

        :return: True if supported, raises exception otherwise
        """
        self._logger.debug("Entering check_support")

        all_none = all(x is None for x in [self.sample_sfd_row, self.sample_sfd_col, self.sample_norm_const])
        none_none = all(x is not None for x in [self.sample_sfd_row, self.sample_sfd_col, self.sample_norm_const])
        if not (all_none or none_none):
            raise ValueError("sample_sfd_row, sample_sfd_col, and norm_const must be all None or all not None")
        self.generate_sfd = none_none
        if self.discrete_col_sfd and not self.generate_sfd:
            self._logger.warning("discrete_col_sfd is True but generate_sfd is False, discrete_col_sfd will be ignored")
            self.discrete_col_sfd = False

        self._logger.debug("Checking tensor shapes and strides")
        tensor_m, k, _one = self._tensor_shape(self.sample_a, name="sample_a")
        n, _, l = self._tensor_shape(self.sample_b, name="sample_b")
        _, _, _one = self._tensor_shape(self.sample_c, name="sample_c")
        self._check_tensor_shape(self.sample_a, (tensor_m, k, 1), "A")
        self._check_tensor_shape(self.sample_b, (n, k, l), "B")
        self._check_tensor_shape(self.sample_c, (tensor_m, n * 2, 1), "C")
        self._check_tensor_shape(self.sample_d_row, (tensor_m, n * 2, 1), "D_row")
        self._check_tensor_shape(self.sample_d_col, (tensor_m, n * 2, 1), "D_col")

        rest_k = ceil_div(ceil_div(k, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sample_sfa, (32, 4, ceil_div(tensor_m, 128), 4, rest_k, 1), "SFA")
        self._check_tensor_shape(self.sample_sfb, (32, 4, ceil_div(n, 128), 4, rest_k, l), "SFB")
        # SFD uses full n dimension since D has n columns (interleaved ab and dswiglu)
        rest_n2_full = ceil_div(ceil_div(n * 2, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sample_sfd_row, (32, 4, ceil_div(tensor_m, 128), 4, rest_n2_full, 1), "SFD_row")
        rest_m = ceil_div(ceil_div(tensor_m, self.sf_vec_size), 4)
        self._check_tensor_shape(self.sample_sfd_col, (32, 4, ceil_div(n * 2, 128), 4, rest_m, 1), "SFD_col")

        self._check_tensor_shape(self.sample_padded_offsets, (l,), "padded_offsets")
        self._check_tensor_shape(self.sample_alpha, (l,), "alpha")
        self._check_tensor_shape(self.sample_beta, (l,), "beta")
        self._check_tensor_shape(self.sample_prob, (tensor_m, 1, 1), "prob")
        self._check_tensor_shape(self.sample_dprob, (tensor_m, 1, 1), "dprob")
        self._check_tensor_shape(self.sample_amax, (l, 2, 1), "amax")
        self._check_tensor_shape(self.sample_norm_const, (1,), "norm_const")

        _, self.a_stride_order = self._check_tensor_stride(self.sample_a, stride=[(k, 1, tensor_m * k)], extra_error_msg="A must have k-major layout")
        if self._is_fp8(self.sample_a.dtype):
            _, self.b_stride_order = self._check_tensor_stride(
                self.sample_b, stride=[(k, 1, n * k), (1, n, n * k)], extra_error_msg="For fp8 ab_dtype, B must have k- or n-major layout"
            )
        else:
            _, self.b_stride_order = self._check_tensor_stride(
                self.sample_b, stride=[(k, 1, n * k)], extra_error_msg="For fp4 ab_dtype, B must have k-major layout"
            )
        _, self.c_stride_order = self._check_tensor_stride(self.sample_c, stride=[(n * 2, 1, tensor_m * n * 2)], extra_error_msg="C must have n-major layout")
        # D has same shape as C (n columns for interleaved ab and dswiglu)
        _, self.d_stride_order = self._check_tensor_stride(
            self.sample_d_row, stride=[(n * 2, 1, tensor_m * n * 2)], extra_error_msg="D_row must have n-major layout"
        )
        _, self.d_col_stride_order = self._check_tensor_stride(
            self.sample_d_col, stride=[(n * 2, 1, tensor_m * n * 2)], extra_error_msg="D_col must have n-major layout"
        )
        self.cd_stride_order = self.c_stride_order

        self._logger.debug("Checking data types")
        self.ab_dtype = self._check_dtype(
            self.sample_a,
            dtype=[
                torch.float4_e2m1fn_x2,
                torch.uint8,
                torch.float8_e5m2,
                torch.float8_e4m3fn,
            ],
            name="A/B",
        )
        self._check_dtype(self.sample_b, dtype=self.ab_dtype, name="B", extra_error_msg="B must have the same dtype as A")

        self.sf_dtype = self._check_dtype(
            self.sample_sfa,
            dtype=[torch.float8_e8m0fnu, torch.float8_e4m3fn],
            name="SFA/SFB/SFD_row/SFD_col",
        )
        self._check_dtype(self.sample_sfb, dtype=self.sf_dtype, name="SFB", extra_error_msg="SFB must have the same dtype as SFA")
        self._check_dtype(self.sample_sfd_row, dtype=self.sf_dtype, name="SFD_row", extra_error_msg="SFD_row must have the same dtype as SFA")
        self._check_dtype(self.sample_sfd_col, dtype=self.sf_dtype, name="SFD_col", extra_error_msg="SFD_col must have the same dtype as SFA")

        if self.sf_vec_size not in [16, 32]:
            raise ValueError(f"sf_vec_size must be 16 or 32, got {self.sf_vec_size}")
        if self.sf_dtype in [torch.float8_e4m3fn] and self.sf_vec_size == 32:
            raise ValueError(f"sf_dtype {self.sf_dtype} and sf_vec_size {self.sf_vec_size} combination is not supported")
        if self._is_fp8(self.ab_dtype) and self.sf_vec_size == 16:
            raise ValueError(f"ab_dtype {self.ab_dtype} and sf_vec_size {self.sf_vec_size} combination is not supported")

        self._check_dtype(self.acc_dtype, dtype=torch.float32, name="Accumulator", extra_error_msg="Accumulator must be float32")
        self._check_dtype(self.sample_prob, dtype=torch.float32, name="Prob", extra_error_msg="Prob must be float32")
        self._check_dtype(self.sample_dprob, dtype=torch.float32, name="Dprob", extra_error_msg="Dprob must be float32")
        self.c_dtype = self._check_dtype(
            self.sample_c,
            dtype=[torch.float32, torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2],
            name="C",
        )
        if self._is_fp8(self.c_dtype) and self.vector_f32:
            raise ValueError(
                f"Invalid configuration: fp8 ab_dtype, c_dtype, and vector_f32 is not supported. Please use vector_f32=False or c_dtype=bfloat16 instead"
            )

        if self._is_fp4x2(self.ab_dtype):
            self.d_dtype = self._check_dtype(
                self.sample_d_row,
                dtype=[torch.float16, torch.bfloat16, torch.float32],
                name="D_row",
                extra_error_msg="D_row must be fp16, bf16, or float32 when ab_dtype is fp4",
            )
        elif self._is_fp8(self.ab_dtype):
            self.d_dtype = self._check_dtype(
                self.sample_d_row,
                dtype=[
                    torch.float8_e4m3fn,
                    torch.float8_e5m2,
                ],
                name="D_row",
                extra_error_msg="D_row must be fp8 dtype when ab_dtype is fp8",
            )
        else:
            raise NotImplementedError(f"Invalid ab_dtype: {self.ab_dtype}, expected fp4 or fp8")
        self._check_dtype(self.sample_d_col, dtype=self.d_dtype, name="D_col", extra_error_msg="D_col must have the same dtype as D_row")

        self._logger.debug("Checking MMA tile shape and cluster shape")
        if not self.use_2cta_instrs and self.mma_tiler_mn[0] not in [64, 128]:
            raise ValueError(f"MMA tiler M must be 64 or 128 when use_2cta_instrs=False, got {self.mma_tiler_mn[0]}")
        if self.use_2cta_instrs and self.mma_tiler_mn[0] not in [128, 256]:
            raise ValueError(f"MMA tiler M must be 128 or 256 when use_2cta_instrs=True, got {self.mma_tiler_mn[0]}")
        if self.mma_tiler_mn[1] not in [128, 256]:
            raise ValueError(f"MMA tiler N must be 128 or 256, got {self.mma_tiler_mn[1]}")
        if self.cluster_shape_mn[0] % (2 if self.use_2cta_instrs else 1) != 0:
            raise ValueError(f"cluster_shape_mn[0] must be divisible by 2 when use_2cta_instrs=True, got {self.cluster_shape_mn[0]}")
        if not (
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1] <= 16
            and self.cluster_shape_mn[0] > 0
            and self.cluster_shape_mn[1] > 0
            and self.cluster_shape_mn[0] <= 4
            and self.cluster_shape_mn[1] <= 4
            and is_power_of_2(self.cluster_shape_mn[0])
            and is_power_of_2(self.cluster_shape_mn[1])
        ):
            raise ValueError(
                f"Invalid cluster shape: expected values to be powers of 2 and cluster_shape_mn[0] * cluster_shape_mn[1] <= 16, got {self.cluster_shape_mn[0]},{self.cluster_shape_mn[1]}"
            )
        cluster_tiler_m = (self.cluster_shape_mn[0] // (2 if self.use_2cta_instrs else 1)) * self.mma_tiler_mn[0]
        if cluster_tiler_m not in [128, 256]:
            raise ValueError(f"Invalid cluster tiler shape: expected cluster_tiler_m in {{128, 256}}, got {cluster_tiler_m}")
        if self.m_aligned % self.mma_tiler_mn[0] != 0:
            raise ValueError(f"Invalid m_aligned: expected m_aligned to be divisible by mma_tiler_mn[0], got {self.m_aligned} % {self.mma_tiler_mn[0]} != 0")
        if self.m_aligned != BlockScaledContiguousGroupedGemmKernel.FIX_PAD_SIZE:
            raise ValueError(
                f"Invalid m_aligned: expected m_aligned to be equal to FIX_PAD_SIZE, got {self.m_aligned} != {BlockScaledContiguousGroupedGemmKernel.FIX_PAD_SIZE}"
            )

        self._logger.debug("Checking tensor alignment")

        def check_contigous_16B_alignment(dtype, stride_order, tensor_shape):
            is_mode0_major = stride_order == (0, 1, 2)
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // (_convert_to_cutlass_data_type(dtype, interpret_uint8_as_fp4x2=self._interpret_uint8_as_fp4x2).width)
            return num_major_elements % num_contiguous_elements == 0

        if not (
            check_contigous_16B_alignment(self.ab_dtype, self.a_stride_order, (tensor_m, k, l))
            and check_contigous_16B_alignment(self.ab_dtype, self.b_stride_order, (n, k, l))
            and check_contigous_16B_alignment(self.d_dtype, self.cd_stride_order, (tensor_m, n, l))
        ):
            raise ValueError("Invalid tensor alignment: tensors must be 16B aligned")

        # Check expert_cnt constraint
        if self.expert_cnt > 1024:
            raise ValueError(f"expert_cnt must be <= 1024, got {self.expert_cnt}")

        # Check environment
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        compute_capability = major * 10 + minor
        if compute_capability < 100:
            raise RuntimeError(f"GroupedGemmDswiglu requires SM100+ compute capability, " f"but found SM{compute_capability} on device {device}")

        self._is_supported = True
        self._logger.debug("check_support completed successfully")
        return True

    def compile(self, current_stream: Optional[cuda.CUstream] = None) -> None:
        """Compile the kernel.

        :param current_stream: CUDA stream to use
        """
        self._logger.debug("Entering compile")
        current_stream = self._get_default_stream(current_stream)
        self._ensure_support_checked()

        gemm_dswiglu = self._kernel(
            sf_vec_size=self.sf_vec_size,
            acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
            use_2cta_instrs=self.use_2cta_instrs,
            mma_tiler_mn=self.mma_tiler_mn,
            cluster_shape_mn=self.cluster_shape_mn,
            vectorized_f32=self.vector_f32,
            discrete_col_sfd=self.discrete_col_sfd,
            expert_cnt=self.expert_cnt,
            use_mono_increase_expert_idx=True,
        )

        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])

        if self._kernel is BlockScaledContiguousGroupedGemmKernel:
            self._logger.debug("Compiling grouped_gemm_dswiglu kernel (dlpack)")
            self._compiled_kernel = cute.compile(
                gemm_dswiglu,
                a=from_dlpack(self.sample_a, assumed_align=16),
                b=from_dlpack(self.sample_b, assumed_align=16),
                c=from_dlpack(self.sample_c, assumed_align=16),
                d=from_dlpack(self.sample_d_row, assumed_align=16),
                d_col=from_dlpack(self.sample_d_col, assumed_align=16) if self.sample_d_col is not None else None,
                sfa=from_dlpack(self.sample_sfa, assumed_align=16),
                sfb=from_dlpack(self.sample_sfb, assumed_align=16),
                sfd_row_tensor=from_dlpack(self.sample_sfd_row, assumed_align=16) if self.sample_sfd_row is not None else None,
                sfd_col_tensor=from_dlpack(self.sample_sfd_col, assumed_align=16) if self.sample_sfd_col is not None else None,
                amax_tensor=from_dlpack(self.sample_amax, assumed_align=16) if self.sample_amax is not None else None,
                norm_const_tensor=from_dlpack(self.sample_norm_const) if self.sample_norm_const is not None else None,
                padded_offsets=from_dlpack(self.sample_padded_offsets, assumed_align=16),
                alpha=from_dlpack(self.sample_alpha, assumed_align=16),
                beta=from_dlpack(self.sample_beta, assumed_align=16),
                prob=from_dlpack(self.sample_prob, assumed_align=16),
                dprob=from_dlpack(self.sample_dprob, assumed_align=16),
                max_active_clusters=max_active_clusters,
                epilogue_op=self.epilogue_op,
                stream=current_stream,
            )
        elif self._kernel is BlockScaledContiguousGroupedGemmKernelNoDlpack:
            self._logger.debug("Compiling grouped_gemm_dswiglu kernel (no_dlpack)")
            # Create cute pointers/tensors manually to avoid DLPack requirements
            a_ptr, a_shape, a_order = self._make_cute_tensor_descriptor(self.sample_a, name="A")
            b_ptr, b_shape, b_order = self._make_cute_tensor_descriptor(self.sample_b, name="B")
            c_ptr, c_shape, c_order = self._make_cute_tensor_descriptor(self.sample_c, name="C")
            d_row_ptr, d_row_shape, d_row_order = self._make_cute_tensor_descriptor(self.sample_d_row, name="D_row")
            d_col_ptr, d_col_shape, d_col_order = self._make_cute_tensor_descriptor(self.sample_d_col, name="D_col")
            sfa_ptr, sfa_shape, sfa_order = self._make_cute_tensor_descriptor(self.sample_sfa, name="SFA")
            sfb_ptr, sfb_shape, sfb_order = self._make_cute_tensor_descriptor(self.sample_sfb, name="SFB")
            sfd_row_ptr, sfd_row_shape, sfd_row_order = self._make_cute_tensor_descriptor(self.sample_sfd_row, name="SFD_row")
            sfd_col_ptr, sfd_col_shape, sfd_col_order = self._make_cute_tensor_descriptor(self.sample_sfd_col, name="SFD_col")
            amax_ptr, amax_shape, amax_order = self._make_cute_tensor_descriptor(self.sample_amax, name="amax")
            norm_const_ptr, norm_const_shape, norm_const_order = self._make_cute_tensor_descriptor(self.sample_norm_const, name="norm_const")
            padded_offsets_ptr, padded_offsets_shape, padded_offsets_order = self._make_cute_tensor_descriptor(
                self.sample_padded_offsets, name="padded_offsets"
            )
            alpha_ptr, alpha_shape, alpha_order = self._make_cute_tensor_descriptor(self.sample_alpha, name="alpha")
            beta_ptr, beta_shape, beta_order = self._make_cute_tensor_descriptor(self.sample_beta, name="beta")
            prob_ptr, prob_shape, prob_order = self._make_cute_tensor_descriptor(self.sample_prob, name="prob")
            dprob_ptr, dprob_shape, dprob_order = self._make_cute_tensor_descriptor(self.sample_dprob, name="dprob")

            self._compiled_kernel = cute.compile(
                gemm_dswiglu,
                a_ptr=a_ptr,
                a_shape=a_shape,
                a_order=a_order,
                b_ptr=b_ptr,
                b_shape=b_shape,
                b_order=b_order,
                c_ptr=c_ptr,
                c_shape=c_shape,
                c_order=c_order,
                d_ptr=d_row_ptr,
                d_shape=d_row_shape,
                d_order=d_row_order,
                d_col_ptr=d_col_ptr,
                d_col_shape=d_col_shape,
                d_col_order=d_col_order,
                sfa_ptr=sfa_ptr,
                sfa_shape=sfa_shape,
                sfa_order=sfa_order,
                sfb_ptr=sfb_ptr,
                sfb_shape=sfb_shape,
                sfb_order=sfb_order,
                sfd_row_ptr=sfd_row_ptr,
                sfd_row_shape=sfd_row_shape,
                sfd_row_order=sfd_row_order,
                sfd_col_ptr=sfd_col_ptr,
                sfd_col_shape=sfd_col_shape,
                sfd_col_order=sfd_col_order,
                amax_ptr=amax_ptr,
                amax_shape=amax_shape,
                amax_order=amax_order,
                norm_const_ptr=norm_const_ptr,
                norm_const_shape=norm_const_shape,
                norm_const_order=norm_const_order,
                padded_offsets_ptr=padded_offsets_ptr,
                padded_offsets_shape=padded_offsets_shape,
                padded_offsets_order=padded_offsets_order,
                alpha_ptr=alpha_ptr,
                alpha_shape=alpha_shape,
                alpha_order=alpha_order,
                beta_ptr=beta_ptr,
                beta_shape=beta_shape,
                beta_order=beta_order,
                prob_ptr=prob_ptr,
                prob_shape=prob_shape,
                prob_order=prob_order,
                dprob_ptr=dprob_ptr,
                dprob_shape=dprob_shape,
                dprob_order=dprob_order,
                max_active_clusters=max_active_clusters,
                epilogue_op=self.epilogue_op,
                stream=current_stream,
            )
        else:
            raise NotImplementedError(f"Unreachable: invalid kernel type {self._kernel}")

        self._logger.debug("Kernel compiled successfully")

    def execute(
        self,
        a_tensor: torch.Tensor,
        b_tensor: torch.Tensor,
        c_tensor: torch.Tensor,
        d_row_tensor: torch.Tensor,
        d_col_tensor: torch.Tensor,
        sfa_tensor: torch.Tensor,
        sfb_tensor: torch.Tensor,
        padded_offsets: torch.Tensor,
        alpha_tensor: torch.Tensor,
        beta_tensor: torch.Tensor,
        prob_tensor: torch.Tensor,
        dprob_tensor: torch.Tensor,
        sfd_row_tensor: Optional[torch.Tensor] = None,
        sfd_col_tensor: Optional[torch.Tensor] = None,
        amax_tensor: Optional[torch.Tensor] = None,
        norm_const_tensor: Optional[torch.Tensor] = None,
        current_stream: Optional[cuda.CUstream] = None,
        skip_compile: bool = False,
    ) -> None:
        """Execute the compiled kernel.

        :param a_tensor: Input A tensor
        :param b_tensor: Input B tensor (weights)
        :param c_tensor: Intermediate C tensor (from forward pass)
        :param d_row_tensor: Output D row tensor
        :param d_col_tensor: Output D column tensor
        :param sfa_tensor: Scale factor A
        :param sfb_tensor: Scale factor B
        :param padded_offsets: End offset per expert after padding
        :param alpha_tensor: Per-group alpha scaling factors
        :param beta_tensor: Per-group beta scaling factors
        :param prob_tensor: Per-row probability tensor
        :param dprob_tensor: Gradient of probability tensor. Must be zero-initialized.
        :param sfd_row_tensor: Optional row scale factor D
        :param sfd_col_tensor: Optional column scale factor D
        :param amax_tensor: Optional amax tensor
        :param norm_const_tensor: Optional normalization constant
        :param current_stream: CUDA stream
        :param skip_compile: If True, use JIT execution without prior compilation
        """
        self._logger.debug("Entering execute")
        current_stream = self._get_default_stream(current_stream)

        norm_const_tensor = self._unpad_tensor_to_ndim(norm_const_tensor, 1, "norm_const")

        if not skip_compile:
            if self._compiled_kernel is None:
                raise RuntimeError("Kernel not compiled; call compile() first or use skip_compile=True")

            if self._kernel is BlockScaledContiguousGroupedGemmKernel:
                self._logger.debug("Executing grouped_gemm_dswiglu kernel (dlpack)")
                self._compiled_kernel(
                    a=from_dlpack(a_tensor, assumed_align=16),
                    b=from_dlpack(b_tensor, assumed_align=16),
                    c=from_dlpack(c_tensor, assumed_align=16),
                    d=from_dlpack(d_row_tensor, assumed_align=16),
                    d_col=from_dlpack(d_col_tensor, assumed_align=16) if d_col_tensor is not None else None,
                    sfa=from_dlpack(sfa_tensor, assumed_align=16),
                    sfb=from_dlpack(sfb_tensor, assumed_align=16),
                    sfd_row_tensor=from_dlpack(sfd_row_tensor, assumed_align=16) if sfd_row_tensor is not None else None,
                    sfd_col_tensor=from_dlpack(sfd_col_tensor, assumed_align=16) if sfd_col_tensor is not None else None,
                    amax_tensor=from_dlpack(amax_tensor, assumed_align=16) if amax_tensor is not None else None,
                    norm_const_tensor=from_dlpack(norm_const_tensor, assumed_align=16) if norm_const_tensor is not None else None,
                    padded_offsets=from_dlpack(padded_offsets, assumed_align=16),
                    alpha=from_dlpack(alpha_tensor, assumed_align=16),
                    beta=from_dlpack(beta_tensor, assumed_align=16),
                    prob=from_dlpack(prob_tensor, assumed_align=16),
                    dprob=from_dlpack(dprob_tensor, assumed_align=16),
                    stream=current_stream,
                )
            elif self._kernel is BlockScaledContiguousGroupedGemmKernelNoDlpack:
                self._logger.debug("Executing grouped_gemm_dswiglu kernel (no_dlpack)")
                # Create cute pointers manually to avoid DLPack requirements
                a_ptr = self._make_cute_pointer(a_tensor, assumed_align=16)
                b_ptr = self._make_cute_pointer(b_tensor, assumed_align=16)
                c_ptr = self._make_cute_pointer(c_tensor, assumed_align=16)
                d_row_ptr = self._make_cute_pointer(d_row_tensor, assumed_align=16)
                d_col_ptr = self._make_cute_pointer(d_col_tensor, assumed_align=16)
                sfa_ptr = self._make_cute_pointer(sfa_tensor, assumed_align=16)
                sfb_ptr = self._make_cute_pointer(sfb_tensor, assumed_align=16)
                sfd_row_ptr = self._make_cute_pointer(sfd_row_tensor, assumed_align=16)
                sfd_col_ptr = self._make_cute_pointer(sfd_col_tensor, assumed_align=16)
                amax_ptr = self._make_cute_pointer(amax_tensor, assumed_align=16)
                norm_const_ptr = self._make_cute_pointer(norm_const_tensor, assumed_align=16)
                padded_offsets_ptr = self._make_cute_pointer(padded_offsets, assumed_align=16)
                alpha_ptr = self._make_cute_pointer(alpha_tensor, assumed_align=16)
                beta_ptr = self._make_cute_pointer(beta_tensor, assumed_align=16)
                prob_ptr = self._make_cute_pointer(prob_tensor, assumed_align=16)
                dprob_ptr = self._make_cute_pointer(dprob_tensor, assumed_align=16)

                self._compiled_kernel(
                    a_ptr=a_ptr,
                    b_ptr=b_ptr,
                    c_ptr=c_ptr,
                    d_ptr=d_row_ptr,
                    d_col_ptr=d_col_ptr,
                    sfa_ptr=sfa_ptr,
                    sfb_ptr=sfb_ptr,
                    sfd_row_ptr=sfd_row_ptr,
                    sfd_col_ptr=sfd_col_ptr,
                    amax_ptr=amax_ptr,
                    norm_const_ptr=norm_const_ptr,
                    padded_offsets_ptr=padded_offsets_ptr,
                    alpha_ptr=alpha_ptr,
                    beta_ptr=beta_ptr,
                    prob_ptr=prob_ptr,
                    dprob_ptr=dprob_ptr,
                    stream=current_stream,
                )
            else:
                raise NotImplementedError(f"Unreachable: invalid kernel type {self._kernel}")
        else:
            self._logger.debug("Executing without compiled kernel (JIT)")
            generate_sfd = sfd_row_tensor is not None and sfd_col_tensor is not None and norm_const_tensor is not None
            discrete_col_sfd = self.discrete_col_sfd and generate_sfd

            gemm_dswiglu = self._kernel(
                sf_vec_size=self.sf_vec_size,
                acc_dtype=_convert_to_cutlass_data_type(self.acc_dtype),
                use_2cta_instrs=self.use_2cta_instrs,
                mma_tiler_mn=self.mma_tiler_mn,
                cluster_shape_mn=self.cluster_shape_mn,
                vectorized_f32=self.vector_f32,
                discrete_col_sfd=discrete_col_sfd,
                expert_cnt=self.expert_cnt,
                use_mono_increase_expert_idx=True,
            )

            hardware_info = cutlass.utils.HardwareInfo()
            max_active_clusters = hardware_info.get_max_active_clusters(self.cluster_shape_mn[0] * self.cluster_shape_mn[1])

            if self._kernel is BlockScaledContiguousGroupedGemmKernel:
                self._logger.debug("JIT executing grouped_gemm_dswiglu kernel (dlpack)")
                gemm_dswiglu(
                    a=from_dlpack(a_tensor, assumed_align=16),
                    b=from_dlpack(b_tensor, assumed_align=16),
                    c=from_dlpack(c_tensor, assumed_align=16),
                    d=from_dlpack(d_row_tensor, assumed_align=16),
                    d_col=from_dlpack(d_col_tensor, assumed_align=16) if d_col_tensor is not None else None,
                    sfa=from_dlpack(sfa_tensor, assumed_align=16),
                    sfb=from_dlpack(sfb_tensor, assumed_align=16),
                    sfd_row_tensor=from_dlpack(sfd_row_tensor, assumed_align=16) if sfd_row_tensor is not None else None,
                    sfd_col_tensor=from_dlpack(sfd_col_tensor, assumed_align=16) if sfd_col_tensor is not None else None,
                    amax_tensor=from_dlpack(amax_tensor, assumed_align=16) if amax_tensor is not None else None,
                    norm_const_tensor=from_dlpack(norm_const_tensor) if norm_const_tensor is not None else None,
                    padded_offsets=from_dlpack(padded_offsets, assumed_align=16),
                    alpha=from_dlpack(alpha_tensor, assumed_align=16),
                    beta=from_dlpack(beta_tensor, assumed_align=16),
                    prob=from_dlpack(prob_tensor, assumed_align=16),
                    dprob=from_dlpack(dprob_tensor, assumed_align=16),
                    max_active_clusters=max_active_clusters,
                    epilogue_op=self.epilogue_op,
                    stream=current_stream,
                )
            elif self._kernel is BlockScaledContiguousGroupedGemmKernelNoDlpack:
                self._logger.debug("JIT executing grouped_gemm_dswiglu kernel (no_dlpack)")
                # Create cute tensor descriptors manually to avoid DLPack requirements
                a_ptr, a_shape, a_order = self._make_cute_tensor_descriptor(a_tensor, name="A")
                b_ptr, b_shape, b_order = self._make_cute_tensor_descriptor(b_tensor, name="B")
                c_ptr, c_shape, c_order = self._make_cute_tensor_descriptor(c_tensor, name="C")
                d_row_ptr, d_row_shape, d_row_order = self._make_cute_tensor_descriptor(d_row_tensor, name="D_row")
                d_col_ptr, d_col_shape, d_col_order = self._make_cute_tensor_descriptor(d_col_tensor, name="D_col")
                sfa_ptr, sfa_shape, sfa_order = self._make_cute_tensor_descriptor(sfa_tensor, name="SFA")
                sfb_ptr, sfb_shape, sfb_order = self._make_cute_tensor_descriptor(sfb_tensor, name="SFB")
                sfd_row_ptr, sfd_row_shape, sfd_row_order = self._make_cute_tensor_descriptor(sfd_row_tensor, name="SFD_row")
                sfd_col_ptr, sfd_col_shape, sfd_col_order = self._make_cute_tensor_descriptor(sfd_col_tensor, name="SFD_col")
                amax_ptr, amax_shape, amax_order = self._make_cute_tensor_descriptor(amax_tensor, name="amax")
                norm_const_ptr, norm_const_shape, norm_const_order = self._make_cute_tensor_descriptor(norm_const_tensor, name="norm_const")
                padded_offsets_ptr, padded_offsets_shape, padded_offsets_order = self._make_cute_tensor_descriptor(padded_offsets, name="padded_offsets")
                alpha_ptr, alpha_shape, alpha_order = self._make_cute_tensor_descriptor(alpha_tensor, name="alpha")
                beta_ptr, beta_shape, beta_order = self._make_cute_tensor_descriptor(beta_tensor, name="beta")
                prob_ptr, prob_shape, prob_order = self._make_cute_tensor_descriptor(prob_tensor, name="prob")
                dprob_ptr, dprob_shape, dprob_order = self._make_cute_tensor_descriptor(dprob_tensor, name="dprob")

                gemm_dswiglu(
                    a_ptr=a_ptr,
                    a_shape=a_shape,
                    a_order=a_order,
                    b_ptr=b_ptr,
                    b_shape=b_shape,
                    b_order=b_order,
                    c_ptr=c_ptr,
                    c_shape=c_shape,
                    c_order=c_order,
                    d_ptr=d_row_ptr,
                    d_shape=d_row_shape,
                    d_order=d_row_order,
                    d_col_ptr=d_col_ptr,
                    d_col_shape=d_col_shape,
                    d_col_order=d_col_order,
                    sfa_ptr=sfa_ptr,
                    sfa_shape=sfa_shape,
                    sfa_order=sfa_order,
                    sfb_ptr=sfb_ptr,
                    sfb_shape=sfb_shape,
                    sfb_order=sfb_order,
                    sfd_row_ptr=sfd_row_ptr,
                    sfd_row_shape=sfd_row_shape,
                    sfd_row_order=sfd_row_order,
                    sfd_col_ptr=sfd_col_ptr,
                    sfd_col_shape=sfd_col_shape,
                    sfd_col_order=sfd_col_order,
                    amax_ptr=amax_ptr,
                    amax_shape=amax_shape,
                    amax_order=amax_order,
                    norm_const_ptr=norm_const_ptr,
                    norm_const_shape=norm_const_shape,
                    norm_const_order=norm_const_order,
                    padded_offsets_ptr=padded_offsets_ptr,
                    padded_offsets_shape=padded_offsets_shape,
                    padded_offsets_order=padded_offsets_order,
                    alpha_ptr=alpha_ptr,
                    alpha_shape=alpha_shape,
                    alpha_order=alpha_order,
                    beta_ptr=beta_ptr,
                    beta_shape=beta_shape,
                    beta_order=beta_order,
                    prob_ptr=prob_ptr,
                    prob_shape=prob_shape,
                    prob_order=prob_order,
                    dprob_ptr=dprob_ptr,
                    dprob_shape=dprob_shape,
                    dprob_order=dprob_order,
                    max_active_clusters=max_active_clusters,
                    epilogue_op=self.epilogue_op,
                    stream=current_stream,
                )
            else:
                raise NotImplementedError(f"Unreachable: invalid kernel type {self._kernel}")

        self._logger.debug("Execute completed")


import logging

_logger = logging.getLogger(__name__)
_cache_of_GroupedGemmDswigluSm100Objects = {}


def grouped_gemm_dswiglu_wrapper_sm100(
    a_tensor: torch.Tensor,
    b_tensor: torch.Tensor,
    c_tensor: torch.Tensor,  # Intermediate from forward pass (required)
    sfa_tensor: torch.Tensor,
    sfb_tensor: torch.Tensor,
    padded_offsets: torch.Tensor,
    alpha_tensor: torch.Tensor,
    beta_tensor: torch.Tensor,
    prob_tensor: torch.Tensor,
    norm_const_tensor: Optional[torch.Tensor] = None,
    acc_dtype: torch.dtype = torch.float32,
    d_dtype: torch.dtype = torch.bfloat16,
    cd_major: str = "n",
    mma_tiler_mn: Tuple[int, int] = (256, 256),
    cluster_shape_mn: Optional[Tuple[int, int]] = None,
    sf_vec_size: int = 16,
    vector_f32: bool = False,
    m_aligned: int = 256,
    discrete_col_sfd: bool = False,
    epilogue_op: Optional[str] = None,
    current_stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Convenience wrapper for grouped GEMM dSwiGLU backward operation.

    This function creates the API, compiles, and executes in one call.
    Compiled kernels are cached for reuse when called with the same configuration.

    Args:
        a_tensor: Input A tensor (valid_m, k, 1)
        b_tensor: Weight B tensor (n, k, l)
        c_tensor: Intermediate C tensor from forward pass (valid_m, 2n, 1)
        sfa_tensor: Scale factor A
        sfb_tensor: Scale factor B
        padded_offsets: End offset per expert after padding (l,)
        alpha_tensor: Per-group alpha scaling
        beta_tensor: Per-group beta scaling
        prob_tensor: Per-row probability tensor
        norm_const_tensor: Optional normalization constant
        acc_dtype: Accumulator data type
        d_dtype: Output D tensor data type
        cd_major: CD major dimension (note: only "n"-major layout is supported)
        mma_tiler_mn: MMA tiler shape
        cluster_shape_mn: Cluster shape
        sf_vec_size: Scale factor vector size
        vector_f32: Use vectorized f32
        m_aligned: M alignment
        discrete_col_sfd: Boolean, True to generate discrete col-major scale factor tensor
        epilogue_op: Optional epilogue operation. Valid values: None, "none", "identity", "relu", "srelu"
        current_stream: CUDA stream

    Returns:
        TupleDict: A dictionary-like object containing output tensors that can also be unpacked as a tuple.
            Dictionary keys (also the unpacking order):
            - **d_row_tensor** (torch.Tensor): Final output tensor after dSwiGLU
            - **d_col_tensor** (torch.Tensor): Column-wise output tensor
            - **dprob_tensor** (torch.Tensor): Gradient tensor for prob (shape `(valid_m, 1, 1)`)
            - **amax_tensor** (torch.Tensor or None): Absolute maximum values (shape l, 2, 1)
            - **sfd_row_tensor** (torch.Tensor or None): Row-wise scale factors for D
            - **sfd_col_tensor** (torch.Tensor or None): Column-wise scale factors for D
    """
    valid_m = a_tensor.shape[0]
    n, _, l = b_tensor.shape

    _logger.debug("grouped_gemm_dswiglu_wrapper_sm100: Creating output tensors d_row_tensor, d_col_tensor, dprob_tensor")

    if cd_major == "n":
        d_row_tensor = torch.empty_strided((valid_m, n * 2, 1), (n * 2, 1, valid_m * n * 2), dtype=d_dtype, device=a_tensor.device)
        d_col_tensor = torch.empty_strided((valid_m, n * 2, 1), (n * 2, 1, valid_m * n * 2), dtype=d_dtype, device=a_tensor.device)
        dprob_tensor = torch.zeros((valid_m, 1, 1), dtype=torch.float32, device=a_tensor.device)
    else:
        raise ValueError(f"cd_major must be 'n', got {cd_major}")

    sfd_row_tensor = None
    sfd_col_tensor = None
    amax_tensor = None

    if a_tensor.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and sfa_tensor.dtype in [torch.float8_e8m0fnu, torch.float8_e4m3fn]:
        _logger.debug("grouped_gemm_dswiglu_wrapper_sm100: Detected fp8 a_dtype and sfa_dtype, constructing sfd_row_tensor and sfd_col_tensor")

        sf_dtype = sfa_tensor.dtype
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        sf_k_row = ceil_div(n * 2, sf_vec_size)
        mma_shape_row = (
            1,
            ceil_div(valid_m, 128),
            ceil_div(sf_k_row, 4),
            32,
            4,
            4,
        )
        sfd_row_tensor = torch.empty(mma_shape_row, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)

        sf_k_col = ceil_div(valid_m, sf_vec_size)
        mma_shape_col = (
            1,
            ceil_div(n * 2, 128),
            ceil_div(sf_k_col, 4),
            32,
            4,
            4,
        )
        sfd_col_tensor = torch.empty(mma_shape_col, dtype=sf_dtype, device=a_tensor.device).permute(mma_permute_order)

    if d_dtype in [torch.bfloat16, torch.float16]:
        _logger.debug("grouped_gemm_dswiglu_wrapper_sm100: Detected bf16/float16 d_dtype, constructing amax_tensor")
        amax_tensor = torch.full((l, 2, 1), float("-inf"), dtype=torch.float32, device=a_tensor.device)

    cache_key = (
        a_tensor.shape,
        b_tensor.shape,
        c_tensor.shape,
        a_tensor.dtype,
        b_tensor.dtype,
        c_tensor.dtype,
        a_tensor.stride(),
        b_tensor.stride(),
        c_tensor.stride(),
        sfa_tensor.shape,
        sfb_tensor.shape,
        sfa_tensor.stride(),
        sfb_tensor.stride(),
        sfa_tensor.dtype,
        sfb_tensor.dtype,
        padded_offsets.shape,
        padded_offsets.stride(),
        padded_offsets.dtype,
        norm_const_tensor.shape if norm_const_tensor is not None else None,
        norm_const_tensor.stride() if norm_const_tensor is not None else None,
        norm_const_tensor.dtype if norm_const_tensor is not None else None,
        acc_dtype,
        d_dtype,
        cd_major,
        mma_tiler_mn,
        cluster_shape_mn,
        sf_vec_size,
        vector_f32,
        m_aligned,
        discrete_col_sfd,
        epilogue_op,
    )

    if cache_key in _cache_of_GroupedGemmDswigluSm100Objects:
        _logger.debug("group_gemm_dswiglu_wrapper_sm100: Using previously cached GroupedGemmDswigluSm100 object")
        grouped_gemm_dswiglu = _cache_of_GroupedGemmDswigluSm100Objects[cache_key]
        grouped_gemm_dswiglu.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            c_tensor=c_tensor,
            d_row_tensor=d_row_tensor,
            d_col_tensor=d_col_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            beta_tensor=beta_tensor,
            prob_tensor=prob_tensor,
            dprob_tensor=dprob_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            current_stream=current_stream,
        )
    else:
        _logger.debug(
            "group_gemm_dswiglu_wrapper_sm100: No previously cached GroupedGemmDswigluSm100 object found, creating new GroupedGemmDswigluSm100 object"
        )
        grouped_gemm_dswiglu = GroupedGemmDswigluSm100(
            sample_a=a_tensor,
            sample_b=b_tensor,
            sample_c=c_tensor,
            sample_d_row=d_row_tensor,
            sample_d_col=d_col_tensor,
            sample_sfa=sfa_tensor,
            sample_sfb=sfb_tensor,
            sample_padded_offsets=padded_offsets,
            sample_alpha=alpha_tensor,
            sample_beta=beta_tensor,
            sample_prob=prob_tensor,
            sample_dprob=dprob_tensor,
            sample_amax=amax_tensor,
            sample_sfd_row=sfd_row_tensor,
            sample_sfd_col=sfd_col_tensor,
            sample_norm_const=norm_const_tensor,
            acc_dtype=acc_dtype,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            sf_vec_size=sf_vec_size,
            vector_f32=vector_f32,
            m_aligned=m_aligned,
            discrete_col_sfd=discrete_col_sfd,
            epilogue_op=epilogue_op,
        )

        assert grouped_gemm_dswiglu.check_support(), "Unsupported configuration"
        grouped_gemm_dswiglu.compile(current_stream=current_stream)
        grouped_gemm_dswiglu.execute(
            a_tensor=a_tensor,
            b_tensor=b_tensor,
            c_tensor=c_tensor,
            d_row_tensor=d_row_tensor,
            d_col_tensor=d_col_tensor,
            sfa_tensor=sfa_tensor,
            sfb_tensor=sfb_tensor,
            padded_offsets=padded_offsets,
            alpha_tensor=alpha_tensor,
            beta_tensor=beta_tensor,
            prob_tensor=prob_tensor,
            dprob_tensor=dprob_tensor,
            sfd_row_tensor=sfd_row_tensor,
            sfd_col_tensor=sfd_col_tensor,
            amax_tensor=amax_tensor,
            norm_const_tensor=norm_const_tensor,
            current_stream=current_stream,
        )
        _cache_of_GroupedGemmDswigluSm100Objects[cache_key] = grouped_gemm_dswiglu

    return TupleDict(
        d_row_tensor=d_row_tensor,
        d_col_tensor=d_col_tensor,
        dprob_tensor=dprob_tensor,
        amax_tensor=amax_tensor,
        sfd_row_tensor=sfd_row_tensor,
        sfd_col_tensor=sfd_col_tensor,
    )

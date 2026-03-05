# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

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

from typing import Type, Tuple, Optional
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.typing import Int32, Int64, Float32
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils

try:
    from . import fmha_utils as _fmha
except ImportError:
    import fmha_utils as _fmha

for _sym in (
    "compute_grid",
    "create_fmha_static_tile_scheduler",
    "make_thread_cooperative_group",
    "FmhaStaticTileScheduler",
    "FmhaStaticTileSchedulerParams",
    "FusedMask",
    "MaskEnum",
    "SM100_TMEM_CAPACITY_COLUMNS",
):
    globals()[_sym] = getattr(_fmha, _sym)


# ---- Tile / layout constants used in __init__ ----
_TILE_Q = 128
_TILE_KV = 128
_TILE_D = 256
_WARP_SIZE = 32


class BlackwellFusedAttentionDQKernel:
    def __init__(
        self,
        element_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        mma_tiler: Tuple[int, int, int],
        varlen: bool,
        is_causal: bool,
        mask_type: MaskEnum,
        window_size_left: int | None,
        window_size_right: int | None,
        is_persistent: bool,
    ):
        self.element_dtype = element_dtype
        self.acc_dtype = acc_dtype
        self.mma_tiler = mma_tiler
        self.varlen = varlen
        self.is_causal = is_causal
        self.mask_type = mask_type
        self.is_persistent = is_persistent

        # Window sizes: negative => None; causal => left=None, right=0
        win_l = None if window_size_left is None or window_size_left < 0 else cutlass.Int32(window_size_left)
        win_r = None if window_size_right is None or window_size_right < 0 else cutlass.Int32(window_size_right)
        self.window_size_left = None if is_causal else win_l
        self.window_size_right = cutlass.Int32(0) if is_causal else win_r

        assert mma_tiler[0] == _TILE_Q and mma_tiler[1] == _TILE_KV, "Only 128x128 tile impl is supported"
        assert mma_tiler[2] == _TILE_D, "Only 256 is supported for 128x128 tile impl"

        # CTA and MMA tiling
        self.cta_tiler = (mma_tiler[0], mma_tiler[1], mma_tiler[2])
        self.qk_mma_tiler = (2 * mma_tiler[0], mma_tiler[1], self.cta_tiler[2])
        self.dov_mma_tiler = self.qk_mma_tiler
        self.dsk_mma_tiler = (2 * mma_tiler[0], self.cta_tiler[2], mma_tiler[1])
        self.dsk_block_tiler = (self.dsk_mma_tiler[0] // 2, self.dsk_mma_tiler[1], self.dsk_mma_tiler[2])

        self.cluster_shape_mn = (2, 1)
        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.tmem_warp_shape_mn = (4, 1)

        # Warp assignment
        self.compute_warp_ids = (0, 1, 2, 3)
        self.epilogue_warp_ids = (4, 5, 6, 7)
        self.mma_warp_id = 8
        self.load_warp_id = 9
        self.empty_warp_id = (10, 11)
        self.num_compute_warps = len(self.compute_warp_ids)
        self.threads_per_warp = _WARP_SIZE
        self.threads_per_cta = self.threads_per_warp * (len(self.compute_warp_ids) + len(self.epilogue_warp_ids) + 1 + 1 + len(self.empty_warp_id))

        self.cta_sync_bar_id = 0
        self.tmem_alloc_sync_bar_id = 1
        self.compute_sync_bar_id = 2
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=self.threads_per_warp * (1 + len(self.compute_warp_ids) + len(self.epilogue_warp_ids)),
        )

        # TMEM settings
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS
        self.tmem_s_offset = 0
        self.tmem_dp_offset = 128
        self.tmem_dq_offset = 256

        # Register reconfiguration
        self.num_regs_compute = 256
        self.num_regs_epilogue = 160
        self.num_regs_other = 32

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.q_stage = 1
        self.k_stage = 1
        self.do_stage = 1
        self.v_stage = 1
        self.kt_stage = 1
        self.qk_acc_stage = 1
        self.dov_acc_stage = 1
        self.dsk_acc_stage = 1
        self.epi_stage = 1
        self.load_compute_LSE_stage = 1
        self.load_compute_sum_OdO_stage = 1

    @cute.jit
    def __call__(
        self,
        problem_size: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        q_tensor: cute.Tensor,
        k_tensor: cute.Tensor,
        v_tensor: cute.Tensor,
        o_tensor: cute.Tensor,
        dq_tensor: cute.Tensor,
        do_tensor: cute.Tensor,
        lse_tensor: cute.Tensor,
        sum_odo_tensor: cute.Tensor,
        cum_seqlen_q: Optional[cute.Tensor],
        cum_seqlen_k: Optional[cute.Tensor],
        scale_softmax: cutlass.Float32,
        workspace: cute.Tensor,
        stream: cuda.CUstream,
    ):
        seq_q, seq_k, head_dim, ((num_heads_q, num_heads_kv), batch) = problem_size
        lse_len = lse_tensor.shape[0]
        seq_q_i64 = Int64(seq_q)
        seq_k_i64 = Int64(seq_k)
        lse_len_i64 = Int64(lse_len)
        head_dim_i64 = cute.assume(Int64(head_dim), divby=128)
        num_heads_q_i64 = Int64(num_heads_q)
        num_heads_kv_i64 = Int64(num_heads_kv)
        batch_i64 = Int64(batch)

        use_varlen_q = cum_seqlen_q is not None
        use_varlen_k = cum_seqlen_k is not None
        seq_q_total = q_tensor.shape[1] if use_varlen_q else seq_q_i64
        seq_k_total = k_tensor.shape[1] if use_varlen_k else seq_k_i64
        batch_stride_qo = num_heads_q_i64 * num_heads_kv_i64 * seq_q_i64 * head_dim_i64 if not use_varlen_q else 0
        batch_stride_kv = num_heads_kv_i64 * seq_k_i64 * head_dim_i64 if not use_varlen_k else 0
        batch_lse = batch_i64 if not use_varlen_q else 1
        batch_stride_lse = num_heads_q_i64 * num_heads_kv_i64 * lse_len_i64 if not use_varlen_q else 0

        q_layout = cute.make_layout(
            (seq_q_total, head_dim, ((num_heads_q, num_heads_kv), batch)),
            stride=(head_dim_i64 * num_heads_q_i64 * num_heads_kv_i64, 1, ((head_dim_i64, head_dim_i64 * num_heads_q_i64), batch_stride_qo)),
        )
        q = cute.make_tensor(q_tensor.iterator, q_layout)
        do_layout = cute.make_layout(
            (seq_q_total, head_dim, ((num_heads_q, num_heads_kv), batch)),
            stride=(head_dim_i64 * num_heads_q_i64 * num_heads_kv_i64, 1, ((head_dim_i64, head_dim_i64 * num_heads_q_i64), batch_stride_qo)),
        )
        do = cute.make_tensor(do_tensor.iterator, do_layout)
        k_layout = cute.make_layout(
            (seq_k_total, head_dim, ((num_heads_q, num_heads_kv), batch)),
            stride=(head_dim_i64 * num_heads_kv_i64, 1, ((0, head_dim_i64), batch_stride_kv)),
        )
        k = cute.make_tensor(k_tensor.iterator, k_layout)
        kt_layout = cute.make_layout(
            (head_dim, seq_k_total, ((num_heads_q, num_heads_kv), batch)),
            stride=(1, head_dim_i64 * num_heads_kv_i64, ((0, head_dim_i64), batch_stride_kv)),
        )
        kt = cute.make_tensor(k_tensor.iterator, kt_layout)
        v_layout = cute.make_layout(
            (seq_k_total, head_dim, ((num_heads_q, num_heads_kv), batch)),
            stride=(head_dim_i64 * num_heads_kv_i64, 1, ((0, head_dim_i64), batch_stride_kv)),
        )
        v = cute.make_tensor(v_tensor.iterator, v_layout)
        lse = cute.make_tensor(lse_tensor.iterator, lse_tensor.layout)
        sum_odo_layout = cute.make_layout(
            (lse_len_i64, ((num_heads_q, num_heads_kv), batch_lse)),
            stride=(1, ((lse_len_i64, num_heads_q_i64 * lse_len_i64), batch_stride_lse)),
        )
        sum_odo = cute.make_tensor(sum_odo_tensor.iterator, sum_odo_layout)
        dq_layout = cute.make_layout(
            (seq_q_total, head_dim, ((num_heads_q, num_heads_kv), batch)),
            stride=(head_dim_i64 * num_heads_q_i64 * num_heads_kv_i64, 1, ((head_dim_i64, head_dim_i64 * num_heads_q_i64), batch_stride_qo)),
        )
        dq = cute.make_tensor(dq_tensor.iterator, dq_layout)

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.do_dtype = do.element_type
        self.dq_dtype = dq.element_type
        self.tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.q_dtype.width

        self.tile_sched_params, grid = compute_grid(
            (seq_q, dq.shape[1], dq.shape[2]) if use_varlen_q else dq.shape,
            self.cta_tiler,
            self.is_persistent,
        )

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.do_major_mode = utils.LayoutEnum.from_tensor(do).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.dq_layout = utils.LayoutEnum.from_tensor(dq)

        required_major = tcgen05.OperandMajorMode.K
        for name, mode in (
            ("q", self.q_major_mode),
            ("k", self.k_major_mode),
            ("v", self.v_major_mode),
            ("do", self.do_major_mode),
        ):
            if cutlass.const_expr(mode != required_major):
                raise RuntimeError(f"Unsupported layout for {name}")

        ref_dtype = self.q_dtype
        for name, dt in (("k", self.k_dtype), ("v", self.v_dtype), ("do", self.do_dtype)):
            if cutlass.const_expr(ref_dtype != dt):
                raise TypeError(f"Type mismatch: {name} has {dt}, expected {ref_dtype}")

        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.TWO
        # the intermediate tensor p is from tmem & k-major
        ds_source = tcgen05.OperandSource.TMEM
        ds_major_mode = tcgen05.OperandMajorMode.K
        k_trans_major_mode = tcgen05.OperandMajorMode.MN
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.acc_dtype,
            cta_group,
            self.qk_mma_tiler[:2],
        )
        dov_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.do_dtype,
            self.do_major_mode,
            self.v_major_mode,
            self.acc_dtype,
            cta_group,
            self.dov_mma_tiler[:2],
        )
        dsk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            ds_major_mode,
            k_trans_major_mode,
            self.acc_dtype,
            cta_group,
            self.dsk_mma_tiler[:2],
            ds_source,
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        self.epi_tile = self.dsk_block_tiler[:2]

        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.q_dtype,
            self.q_stage,
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.qk_mma_tiler,
            self.k_dtype,
            self.k_stage,
        )
        do_smem_layout_staged = sm100_utils.make_smem_layout_a(
            dov_tiled_mma,
            self.dov_mma_tiler,
            self.do_dtype,
            self.do_stage,
        )
        v_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dov_tiled_mma,
            self.dov_mma_tiler,
            self.v_dtype,
            self.v_stage,
        )
        ds_tmem_layout_staged = sm100_utils.make_smem_layout_a(
            dsk_tiled_mma,
            self.dsk_mma_tiler,
            self.q_dtype,
            self.qk_acc_stage,
        )
        ds_tmem_layout = cute.select(ds_tmem_layout_staged, mode=[0, 1, 2])
        kt_smem_layout_staged = sm100_utils.make_smem_layout_b(
            dsk_tiled_mma,
            self.dsk_mma_tiler,
            self.k_dtype,
            self.dsk_acc_stage,
        )

        # TMA load for Q
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)

        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q,
            q_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for K
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            k,
            k_smem_layout,
            self.qk_mma_tiler,
            qk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for dO
        do_smem_layout = cute.select(do_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_do, tma_tensor_do = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            do,
            do_smem_layout,
            self.dov_mma_tiler,
            dov_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for V
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            v,
            v_smem_layout,
            self.dov_mma_tiler,
            dov_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        # TMA load for KT
        kt_smem_layout = cute.select(kt_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_kt, tma_tensor_kt = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            kt,
            kt_smem_layout,
            self.dsk_mma_tiler,
            dsk_tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        lse_smem_layout = cute.make_layout((self.cta_tiler[0], self.load_compute_LSE_stage))
        sum_odo_smem_layout = cute.make_layout((self.cta_tiler[0], self.load_compute_sum_OdO_stage))

        q_copy_size = cute.size_in_bytes(self.q_dtype, q_smem_layout)
        k_copy_size = cute.size_in_bytes(self.k_dtype, k_smem_layout)
        do_copy_size = cute.size_in_bytes(self.do_dtype, do_smem_layout)
        v_copy_size = cute.size_in_bytes(self.v_dtype, v_smem_layout)
        kt_copy_size = cute.size_in_bytes(self.k_dtype, kt_smem_layout)
        self.tma_copy_q_bytes = q_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_k_bytes = k_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_do_bytes = do_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_v_bytes = v_copy_size * cute.size(qk_tiled_mma.thr_id.shape)
        self.tma_copy_kt_bytes = kt_copy_size * cute.size(qk_tiled_mma.thr_id.shape)

        @cute.struct
        class SharedStorage:
            # TMA G2S load barriers: LOAD warp (producer) -> MMA warp (consumer)
            load_q_mbar_ptr: cute.struct.MemRange[Int64, self.q_stage * 2]  # load_q_{producer,consumer}
            load_do_mbar_ptr: cute.struct.MemRange[Int64, self.do_stage * 2]  # load_do_{producer,consumer}
            load_k_mbar_ptr: cute.struct.MemRange[Int64, self.k_stage * 2]  # load_k_{producer,consumer}
            load_kt_mbar_ptr: cute.struct.MemRange[Int64, self.kt_stage * 2]  # load_kt_{producer,consumer}
            load_v_mbar_ptr: cute.struct.MemRange[Int64, self.v_stage * 2]  # load_v_{producer,consumer}
            mma_s_mbar_ptr: cute.struct.MemRange[Int64, self.qk_acc_stage * 2]
            mma_dp_mbar_ptr: cute.struct.MemRange[Int64, self.dov_acc_stage * 2]
            mma_dq_mbar_ptr: cute.struct.MemRange[Int64, self.epi_stage * 2]
            ds_mma_mbar_ptr: cute.struct.MemRange[Int64, self.dsk_acc_stage * 2]
            lse_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_compute_LSE_stage * 2]
            sum_odo_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_compute_sum_OdO_stage * 2]
            tmem_dealloc_mbar_ptr: Int64
            tmem_holding_buf: Int32

        self.shared_storage = SharedStorage

        grid = cute.round_up(grid, self.cluster_shape_mnk)
        # Launch the kernel synchronously
        self.kernel(
            problem_size,
            qk_tiled_mma,
            dov_tiled_mma,
            dsk_tiled_mma,
            tma_atom_q,
            tma_tensor_q,
            tma_atom_k,
            tma_tensor_k,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_do,
            tma_tensor_do,
            tma_atom_kt,
            tma_tensor_kt,
            lse,
            sum_odo,
            dq,
            cum_seqlen_q,
            cum_seqlen_k,
            scale_softmax,
            self.window_size_left,
            self.window_size_right,
            self.cluster_layout_vmnk,
            q_smem_layout_staged,
            k_smem_layout_staged,
            v_smem_layout_staged,
            do_smem_layout_staged,
            kt_smem_layout_staged,
            ds_tmem_layout,
            lse_smem_layout,
            sum_odo_smem_layout,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        problem_size: tuple[Int32, Int32, Int32, tuple[tuple[Int32, Int32], Int32]],
        qk_tiled_mma: cute.TiledMma,
        dov_tiled_mma: cute.TiledMma,
        dsk_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        gQ: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        gK: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        gV: cute.Tensor,
        tma_atom_do: cute.CopyAtom,
        gdO: cute.Tensor,
        tma_atom_kt: cute.CopyAtom,
        gKT: cute.Tensor,
        gLSE: cute.Tensor,
        gSumOdO: cute.Tensor,
        gdQ: cute.Tensor,
        cum_seqlen_q: Optional[cute.Tensor],
        cum_seqlen_k: Optional[cute.Tensor],
        scale_softmax: Float32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        cluster_layout_vmnk: cute.Layout,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        do_smem_layout_staged: cute.ComposedLayout,
        kt_smem_layout_staged: cute.ComposedLayout,
        ds_tmem_layout: cute.ComposedLayout,
        lse_smem_layout: cute.Layout,
        sum_odo_smem_layout: cute.Layout,
        tile_sched_params: FmhaStaticTileSchedulerParams,
    ):
        widx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if widx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_do)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_kt)

        bidx, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        # Alloc Shared Storage
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.q_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_k_producer, load_k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_k_bytes,
            barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_v_producer, load_v_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.v_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_v_bytes,
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_do_producer, load_do_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.do_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_do_bytes,
            barrier_storage=storage.load_do_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        load_kt_producer, load_kt_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.kt_stage,
            producer_group=make_thread_cooperative_group(len([self.load_warp_id])),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            tx_count=self.tma_copy_kt_bytes,
            barrier_storage=storage.load_kt_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_s_producer, mma_s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.qk_acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.compute_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_s_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_dp_producer, mma_dp_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.dov_acc_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.compute_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_dp_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        ds_mma_producer, ds_mma_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.dsk_acc_stage,
            producer_group=make_thread_cooperative_group(
                len(self.compute_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            consumer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            barrier_storage=storage.ds_mma_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        mma_dq_producer, mma_dq_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.epi_stage,
            producer_group=make_thread_cooperative_group(len([self.mma_warp_id])),
            consumer_group=make_thread_cooperative_group(
                len(self.epilogue_warp_ids) * self.threads_per_warp * self.cluster_shape_mnk[0],
            ),
            barrier_storage=storage.mma_dq_mbar_ptr.data_ptr(),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        load_lse_producer, load_lse_consumer = pipeline.PipelineCpAsync.create(
            num_stages=self.load_compute_LSE_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * self.num_compute_warps),
            barrier_storage=storage.lse_mbar_ptr.data_ptr(),
        ).make_participants()
        load_sum_odo_producer, load_sum_odo_consumer = pipeline.PipelineCpAsync.create(
            num_stages=self.load_compute_sum_OdO_stage,
            producer_group=make_thread_cooperative_group(self.threads_per_warp),
            consumer_group=make_thread_cooperative_group(self.threads_per_warp * self.num_compute_warps),
            barrier_storage=storage.sum_odo_mbar_ptr.data_ptr(),
        ).make_participants()

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_ids[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )
        # Cluster arrive after barrier init
        pipeline.pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sQ = smem.allocate_tensor(
            element_type=self.q_dtype,
            layout=q_smem_layout_staged.outer,
            swizzle=q_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sK = smem.allocate_tensor(
            element_type=self.k_dtype,
            layout=k_smem_layout_staged.outer,
            swizzle=k_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sV = smem.allocate_tensor(
            element_type=self.v_dtype,
            layout=v_smem_layout_staged.outer,
            swizzle=v_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sdO = smem.allocate_tensor(
            element_type=self.do_dtype,
            layout=do_smem_layout_staged.outer,
            swizzle=do_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sKT = smem.allocate_tensor(
            element_type=self.k_dtype,
            layout=kt_smem_layout_staged.outer,
            swizzle=kt_smem_layout_staged.inner,
            byte_alignment=128,
        )
        sLSE = smem.allocate_tensor(
            element_type=self.acc_dtype,
            layout=lse_smem_layout,
            byte_alignment=128,
        )
        sSum_OdO = smem.allocate_tensor(
            element_type=self.acc_dtype,
            layout=sum_odo_smem_layout,
            byte_alignment=128,
        )

        mma_tile_coord_v = bidx % cute.size(qk_tiled_mma.thr_id.shape)
        qk_thr_mma = qk_tiled_mma.get_slice(mma_tile_coord_v)  # default 1sm
        dov_thr_mma = dov_tiled_mma.get_slice(mma_tile_coord_v)  # default 1sm
        dsk_thr_mma = dsk_tiled_mma.get_slice(mma_tile_coord_v)  # default 1sm
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tdPrdO = dov_thr_mma.make_fragment_A(sdO)
        tdPrV = dov_thr_mma.make_fragment_B(sV)
        tdQrKT = dsk_thr_mma.make_fragment_B(sKT)
        qk_acc_shape = qk_thr_mma.partition_shape_C((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        tStS = qk_thr_mma.make_fragment_C(cute.append(qk_acc_shape, self.qk_acc_stage))
        dov_acc_shape = dov_thr_mma.partition_shape_C((self.dov_mma_tiler[0], self.dov_mma_tiler[1]))
        tdPtdP = dov_thr_mma.make_fragment_C(cute.append(dov_acc_shape, self.dov_acc_stage))
        dsk_acc_shape = dsk_thr_mma.partition_shape_C((self.dsk_mma_tiler[0], self.dsk_mma_tiler[1]))
        tdQtdQ = dsk_thr_mma.make_fragment_C(dsk_acc_shape)
        tdQtdQ_layout = cute.append(
            tdQtdQ.layout,
            cute.make_layout(
                1,
                stride=self.dsk_mma_tiler[1] // self.tmem_warp_shape_mn[1],
            ),
        )
        tStS = cute.make_tensor(tStS.iterator + self.tmem_s_offset, tStS.layout)
        tdPtdP = cute.make_tensor(tdPtdP.iterator + self.tmem_dp_offset, tdPtdP.layout)
        tdQtdQ_staged = cute.make_tensor(tdQtdQ.iterator + self.tmem_dq_offset, tdQtdQ_layout)

        # ---- Empty warps ----
        if widx == self.empty_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

        tile_sched = create_fmha_static_tile_scheduler(tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim())
        tile_info = tile_sched.initial_work_tile_info()

        # Cluster wait
        pipeline.pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        # ---- Load warp: TMA + LSE/sum_OdO ----
        if widx == self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

            while tile_info.is_valid_tile:
                block_coord = tile_info.tile_idx
                mma_coord = (
                    block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    block_coord[1],
                    block_coord[2],
                )
                skip = False
                batch_coord = block_coord[2][1]
                seqlen_q = gQ.shape[0]
                seqlen_k = gK.shape[0]
                cuseqlen_q = Int32(0)
                cuseqlen_k = Int32(0)
                block_offset = (
                    Int32(0),
                    Int32(0),
                    Int32(0),
                    ((Int32(0), Int32(0)), Int32(0)),
                )
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                    block_offset = (
                        cuseqlen_q,
                        cuseqlen_k,
                        Int32(0),
                        ((Int32(0), Int32(0)), Int32(0)),
                    )
                    skip = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_coord[0],
                        seqlen_q,
                    )
                if not skip:
                    gQ_ = cute.domain_offset(cute.select(block_offset, mode=[0, 2, 3]), gQ)
                    gK_ = cute.domain_offset(cute.select(block_offset, mode=[1, 2, 3]), gK)
                    gdO_ = cute.domain_offset(cute.select(block_offset, mode=[0, 2, 3]), gdO)
                    gV_ = cute.domain_offset(cute.select(block_offset, mode=[1, 2, 3]), gV)
                    gKT_ = cute.domain_offset(cute.select(block_offset, mode=[2, 1, 3]), gKT)
                    LSE = cute.domain_offset(cute.select(block_offset, mode=[0, 3]), gLSE)
                    sum_OdO = cute.domain_offset(cute.select(block_offset, mode=[0, 3]), gSumOdO)

                    # Local tile partition global tensors
                    q_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
                    # (bM, bK, loopM, loopK, loopL)
                    gQ_qdl = cute.flat_divide(gQ_, cute.select(self.qk_mma_tiler, mode=[0, 2]))
                    tSgQ_qdl = qk_thr_mma.partition_A(gQ_qdl)
                    tQsQ, tQgQ_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_q,
                        cluster_coord[2],
                        q_cta_layout,
                        cute.group_modes(sQ, 0, 3),
                        cute.group_modes(tSgQ_qdl, 0, 3),
                    )
                    k_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
                    gK_kdl = cute.flat_divide(gK_, cute.select(self.qk_mma_tiler, mode=[1, 2]))
                    tSgK_kdl = qk_thr_mma.partition_B(gK_kdl)
                    tKsK, tKgK_kdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_k,
                        cluster_coord[1],
                        k_cta_layout,
                        cute.group_modes(sK, 0, 3),
                        cute.group_modes(tSgK_kdl, 0, 3),
                    )
                    do_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
                    # (bM, bK, loopM, loopK, loopL)
                    gdO_qdl = cute.flat_divide(gdO_, cute.select(self.dov_mma_tiler, mode=[0, 2]))
                    tdPgdO_qdl = dov_thr_mma.partition_A(gdO_qdl)
                    tdOsdO, tdOgdO_qdl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_do,
                        cluster_coord[2],
                        do_cta_layout,
                        cute.group_modes(sdO, 0, 3),
                        cute.group_modes(tdPgdO_qdl, 0, 3),
                    )
                    v_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
                    gV_dkl = cute.flat_divide(gV_, cute.select(self.dov_mma_tiler, mode=[1, 2]))
                    tSgV_dkl = dov_thr_mma.partition_B(gV_dkl)
                    tVsV, tVgV_dkl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_v,
                        cluster_coord[1],
                        v_cta_layout,
                        cute.group_modes(sV, 0, 3),
                        cute.group_modes(tSgV_dkl, 0, 3),
                    )
                    # kt layout
                    kt_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
                    gK_dkl = cute.flat_divide(gKT_, cute.select(self.dsk_mma_tiler, mode=[1, 2]))
                    tdQgK_dkl = dsk_thr_mma.partition_B(gK_dkl)
                    tKTsKT, tKgK_dkl = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_kt,
                        cluster_coord[1],
                        kt_cta_layout,
                        cute.group_modes(sKT, 0, 3),
                        cute.group_modes(tdQgK_dkl, 0, 3),
                    )
                    # ((atom_v, rest_v), RestK)
                    tQgQ = tQgQ_qdl[None, mma_coord[0], None, mma_coord[2]]
                    # ((atom_v, rest_v), RestK)
                    tdOgdO = tdOgdO_qdl[None, mma_coord[0], None, mma_coord[2]]
                    # ((atom_v, rest_v), RestN, RestK)
                    tKgK = tKgK_kdl[None, None, None, mma_coord[2]]
                    # ((atom_v, rest_v), RestN, RestK)
                    tVgV = tVgV_dkl[None, None, None, mma_coord[2]]
                    # ((atom_v, rest_v), RestN, RestK)
                    tKTgKT = tKgK_dkl[None, None, None, mma_coord[2]]

                    seqlen_kv_loop_start = FusedMask.get_trip_start(
                        self.mask_type,
                        mma_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                    )
                    seqlen_kv_loop_steps = FusedMask.get_trip_count(
                        self.mask_type,
                        mma_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                        window_size_right,
                    )
                    # LSE
                    lse_handle = load_lse_producer.acquire_and_advance()
                    # 32 threads loading 128 values of 32b each
                    # so 4*32b = 128b
                    thread_idx = tidx % self.threads_per_warp
                    async_copy_num_elts = sLSE.shape[0] // self.threads_per_warp
                    atom_async_copy = cute.make_copy_atom(
                        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
                        self.acc_dtype,
                        num_bits_per_copy=self.acc_dtype.width,
                    )
                    sLSE_for_copy = cute.flat_divide(sLSE, (1,))
                    LSE_for_copy = cute.flat_divide(LSE, (1,))
                    for i in cutlass.range_constexpr(async_copy_num_elts):
                        LSE_idx = self.cta_tiler[0] * block_coord[0] + thread_idx * async_copy_num_elts
                        if cute.elem_less(LSE_idx + i, problem_size[0]):
                            cute.copy(
                                atom_async_copy,
                                LSE_for_copy[None, LSE_idx + i, block_coord[2]],
                                sLSE_for_copy[
                                    None,
                                    thread_idx * async_copy_num_elts + i,
                                    lse_handle.index,
                                ],
                            )
                        else:
                            sLSE_for_copy[
                                None,
                                thread_idx * async_copy_num_elts + i,
                                lse_handle.index,
                            ].fill(0.0)
                    lse_handle.commit()

                    sum_odo_handle = load_sum_odo_producer.acquire_and_advance()
                    sSum_OdO_for_copy = cute.flat_divide(sSum_OdO, (1,))
                    sum_OdO_for_copy = cute.flat_divide(sum_OdO, (1,))
                    for i in cutlass.range_constexpr(async_copy_num_elts):
                        sum_OdO_idx = self.cta_tiler[0] * block_coord[0] + thread_idx * async_copy_num_elts
                        if cute.elem_less(sum_OdO_idx + i, problem_size[0]):
                            cute.copy(
                                atom_async_copy,
                                sum_OdO_for_copy[None, sum_OdO_idx + i, block_coord[2]],
                                sSum_OdO_for_copy[
                                    None,
                                    thread_idx * async_copy_num_elts + i,
                                    sum_odo_handle.index,
                                ],
                            )
                        else:
                            sSum_OdO_for_copy[
                                None,
                                thread_idx * async_copy_num_elts + i,
                                sum_odo_handle.index,
                            ].fill(0.0)
                    sum_odo_handle.commit()

                    # Q
                    q_handle = load_q_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_q,
                        tQgQ[None, 0],
                        tQsQ[None, q_handle.index],
                        tma_bar_ptr=q_handle.barrier,
                    )
                    # dO
                    do_handle = load_do_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_do,
                        tdOgdO[None, 0],
                        tdOsdO[None, do_handle.index],
                        tma_bar_ptr=do_handle.barrier,
                    )

                    kv_coord = seqlen_kv_loop_start
                    for i in cutlass.range(0, seqlen_kv_loop_steps, 1, unroll=1):
                        # Ki
                        k_handle = load_k_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_k,
                            tKgK[None, kv_coord, 0],
                            tKsK[None, k_handle.index],
                            tma_bar_ptr=k_handle.barrier,
                        )
                        # Vi
                        v_handle = load_v_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_v,
                            tVgV[None, kv_coord, 0],
                            tVsV[None, v_handle.index],
                            tma_bar_ptr=v_handle.barrier,
                        )
                        # KTi
                        kt_handle = load_kt_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_kt,
                            tKTgKT[None, 0, kv_coord],
                            tKTsKT[None, kt_handle.index],
                            tma_bar_ptr=kt_handle.barrier,
                        )
                        kv_coord += 1

                tile_sched.advance_to_next_work()
                tile_info = tile_sched.get_current_work()
            load_k_producer.tail()
            load_v_producer.tail()
            load_kt_producer.tail()
            load_q_producer.tail()
            load_do_producer.tail()
            load_lse_producer.tail()
            load_sum_odo_producer.tail()

        # ---- MMA warp: QK, dOV, dSK ----
        if widx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            tmem.wait_for_alloc()

            while tile_info.is_valid_tile:
                block_coord = tile_info.tile_idx
                mma_coord = (
                    block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    block_coord[1],
                    block_coord[2],
                )
                skip = False
                seqlen_q = gQ.shape[0]
                seqlen_k = gK.shape[0]
                batch_coord = block_coord[2][1]
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    skip = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_coord[0],
                        seqlen_q,
                    )

                if not skip:
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k

                    seqlen_kv_loop_start = FusedMask.get_trip_start(
                        self.mask_type,
                        mma_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                    )
                    seqlen_kv_loop_steps = FusedMask.get_trip_count(
                        self.mask_type,
                        mma_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                        window_size_right,
                    )

                    cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
                    is_leader_cta = cta_rank_in_cluster % 2 == 0
                    load_q_releaser = load_q_consumer.clone()
                    load_do_releaser = load_do_consumer.clone()
                    dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    K_PHASE_UNROLL = 8

                    if is_leader_cta:
                        dq_handle = mma_dq_producer.acquire_and_advance()
                        if seqlen_kv_loop_steps > 1:
                            # QK0
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            load_q_consumer.wait_and_advance()
                            tSrQ_slice = tSrQ[None, None, None, 0]

                            k_handle = load_k_consumer.wait_and_advance()
                            tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                            num_kphases = cute.size(tSrQ_slice, mode=[2])
                            if cutlass.const_expr(num_kphases % K_PHASE_UNROLL == 0):
                                num_outer_iter = num_kphases // K_PHASE_UNROLL
                                for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                    for kphase_idx in cutlass.range(K_PHASE_UNROLL, unroll_full=True):
                                        kphase_coord = (None, None, outer_iter * K_PHASE_UNROLL + kphase_idx)
                                        cute.gemm(
                                            qk_tiled_mma,
                                            tStS_slice,
                                            tSrQ_slice[kphase_coord],
                                            tSrK_trans_slice[kphase_coord],
                                            tStS_slice,
                                        )
                                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            else:
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        qk_tiled_mma,
                                        tStS_slice,
                                        tSrQ_slice[kphase_coord],
                                        tSrK_trans_slice[kphase_coord],
                                        tStS_slice,
                                    )
                                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            k_handle.release()
                            s_handle.commit()

                            # dOV0
                            dp_handle = mma_dp_producer.acquire_and_advance()
                            tdPtdP_slice = tdPtdP[None, None, None, dp_handle.index]
                            dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            load_do_consumer.wait_and_advance()
                            tdPrdO_slice = tdPrdO[None, None, None, 0]
                            v_handle = load_v_consumer.wait_and_advance()
                            tdPrV_trans_slice = tdPrV[None, None, None, v_handle.index]
                            num_kphases = cute.size(tdPrdO_slice, mode=[2])
                            if cutlass.const_expr(num_kphases % K_PHASE_UNROLL == 0):
                                num_outer_iter = num_kphases // K_PHASE_UNROLL
                                for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                    for kphase_idx in cutlass.range(K_PHASE_UNROLL, unroll_full=True):
                                        kphase_coord = (None, None, outer_iter * K_PHASE_UNROLL + kphase_idx)
                                        cute.gemm(
                                            dov_tiled_mma,
                                            tdPtdP_slice,
                                            tdPrdO_slice[kphase_coord],
                                            tdPrV_trans_slice[kphase_coord],
                                            tdPtdP_slice,
                                        )
                                        dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            else:
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        dov_tiled_mma,
                                        tdPtdP_slice,
                                        tdPrdO_slice[kphase_coord],
                                        tdPrV_trans_slice[kphase_coord],
                                        tdPtdP_slice,
                                    )
                                    dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            v_handle.release()
                            dp_handle.commit()

                            for i in cutlass.range(1, seqlen_kv_loop_steps - 1, 1, unroll=1):
                                # QKi
                                s_handle = mma_s_producer.acquire_and_advance()

                                tStS_slice = tStS[None, None, None, s_handle.index]
                                qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                                tSrQ_slice = tSrQ[None, None, None, 0]
                                k_handle = load_k_consumer.wait_and_advance()
                                tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                                num_kphases = cute.size(tSrQ_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % K_PHASE_UNROLL == 0):
                                    num_outer_iter = num_kphases // K_PHASE_UNROLL
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(K_PHASE_UNROLL, unroll_full=True):
                                            kphase_coord = (None, None, outer_iter * K_PHASE_UNROLL + kphase_idx)
                                            cute.gemm(
                                                qk_tiled_mma,
                                                tStS_slice,
                                                tSrQ_slice[kphase_coord],
                                                tSrK_trans_slice[kphase_coord],
                                                tStS_slice,
                                            )
                                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            qk_tiled_mma,
                                            tStS_slice,
                                            tSrQ_slice[kphase_coord],
                                            tSrK_trans_slice[kphase_coord],
                                            tStS_slice,
                                        )
                                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                k_handle.release()
                                s_handle.commit()

                                # dSKTi
                                ds_handle = ds_mma_consumer.wait_and_advance()
                                dsk_whether_acc = dsk_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                                kt_handle = load_kt_consumer.wait_and_advance()
                                dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, dsk_whether_acc)
                                tdQtdQ_slice = tdQtdQ_staged[None, None, None, 0]
                                tdStdS_slice = tdPtdP[None, None, None, ds_handle.index]
                                tdS = cute.make_tensor(tdStdS_slice.iterator, ds_tmem_layout.outer)
                                tdQrdS = dsk_thr_mma.make_fragment_A(tdS)
                                tdQrdS_slice = cute.make_tensor(
                                    cute.recast_ptr(tdStdS_slice.iterator, dtype=self.q_dtype),
                                    tdQrdS.layout,
                                )

                                tdQrKT_slice = tdQrKT[None, None, None, kt_handle.index]
                                num_kphases = cute.size(tdQrKT_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % K_PHASE_UNROLL == 0):
                                    num_outer_iter = num_kphases // K_PHASE_UNROLL
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(K_PHASE_UNROLL, unroll_full=True):
                                            kphase_coord = (None, None, outer_iter * K_PHASE_UNROLL + kphase_idx)
                                            cute.gemm(
                                                dsk_tiled_mma,
                                                tdQtdQ_slice,
                                                tdQrdS_slice[kphase_coord],
                                                tdQrKT_slice[kphase_coord],
                                                tdQtdQ_slice,
                                            )
                                            dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            dsk_tiled_mma,
                                            tdQtdQ_slice,
                                            tdQrdS_slice[kphase_coord],
                                            tdQrKT_slice[kphase_coord],
                                            tdQtdQ_slice,
                                        )
                                        dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                kt_handle.release()
                                ds_handle.release()

                                # dOVi
                                dp_handle = mma_dp_producer.acquire_and_advance()
                                tdPtdP_slice = tdPtdP[None, None, None, dp_handle.index]
                                dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                                tdPrdO_slice = tdPrdO[None, None, None, 0]
                                v_handle = load_v_consumer.wait_and_advance()
                                tdPrV_trans_slice = tdPrV[None, None, None, v_handle.index]
                                num_kphases = cute.size(tdPrdO_slice, mode=[2])
                                if cutlass.const_expr(num_kphases % K_PHASE_UNROLL == 0):
                                    num_outer_iter = num_kphases // K_PHASE_UNROLL
                                    for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                        for kphase_idx in cutlass.range(K_PHASE_UNROLL, unroll_full=True):
                                            kphase_coord = (None, None, outer_iter * K_PHASE_UNROLL + kphase_idx)
                                            cute.gemm(
                                                dov_tiled_mma,
                                                tdPtdP_slice,
                                                tdPrdO_slice[kphase_coord],
                                                tdPrV_trans_slice[kphase_coord],
                                                tdPtdP_slice,
                                            )
                                            dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                else:
                                    for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                        kphase_coord = (None, None, kphase_idx)
                                        cute.gemm(
                                            dov_tiled_mma,
                                            tdPtdP_slice,
                                            tdPrdO_slice[kphase_coord],
                                            tdPrV_trans_slice[kphase_coord],
                                            tdPtdP_slice,
                                        )
                                        dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                                v_handle.release()
                                dp_handle.commit()

                            # QKend
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            tSrQ_slice = tSrQ[None, None, None, 0]
                            k_handle = load_k_consumer.wait_and_advance()

                            tSrK_trans_slice = tSrK[None, None, None, k_handle.index]

                            num_kphases = cute.size(tSrQ_slice, mode=[2])
                            if cutlass.const_expr(num_kphases % K_PHASE_UNROLL == 0):
                                num_outer_iter = num_kphases // K_PHASE_UNROLL
                                for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                    for kphase_idx in cutlass.range(K_PHASE_UNROLL, unroll_full=True):
                                        kphase_coord = (None, None, outer_iter * K_PHASE_UNROLL + kphase_idx)
                                        cute.gemm(
                                            qk_tiled_mma,
                                            tStS_slice,
                                            tSrQ_slice[kphase_coord],
                                            tSrK_trans_slice[kphase_coord],
                                            tStS_slice,
                                        )
                                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            else:
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        qk_tiled_mma,
                                        tStS_slice,
                                        tSrQ_slice[kphase_coord],
                                        tSrK_trans_slice[kphase_coord],
                                        tStS_slice,
                                    )
                                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            k_handle.release()
                            load_q_releaser.release()
                            load_q_releaser.advance()
                            s_handle.commit()

                            # dSKTend - 1
                            ds_handle = ds_mma_consumer.wait_and_advance()
                            dsk_whether_acc = dsk_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                            kt_handle = load_kt_consumer.wait_and_advance()
                            dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, dsk_whether_acc)
                            tdQtdQ_slice = tdQtdQ_staged[None, None, None, 0]
                            tdStdS_slice = tdPtdP[None, None, None, ds_handle.index]
                            tdS = cute.make_tensor(tdStdS_slice.iterator, ds_tmem_layout.outer)
                            tdQrdS = dsk_thr_mma.make_fragment_A(tdS)
                            tdQrdS_slice = cute.make_tensor(
                                cute.recast_ptr(tdStdS_slice.iterator, dtype=self.q_dtype),
                                tdQrdS.layout,
                            )

                            tdQrKT_slice = tdQrKT[None, None, None, kt_handle.index]
                            num_kphases = cute.size(tdQrKT_slice, mode=[2])
                            if cutlass.const_expr(num_kphases % K_PHASE_UNROLL == 0):
                                num_outer_iter = num_kphases // K_PHASE_UNROLL
                                for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                    for kphase_idx in cutlass.range(K_PHASE_UNROLL, unroll_full=True):
                                        kphase_coord = (None, None, outer_iter * K_PHASE_UNROLL + kphase_idx)
                                        cute.gemm(
                                            dsk_tiled_mma,
                                            tdQtdQ_slice,
                                            tdQrdS_slice[kphase_coord],
                                            tdQrKT_slice[kphase_coord],
                                            tdQtdQ_slice,
                                        )
                                        dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            else:
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        dsk_tiled_mma,
                                        tdQtdQ_slice,
                                        tdQrdS_slice[kphase_coord],
                                        tdQrKT_slice[kphase_coord],
                                        tdQtdQ_slice,
                                    )
                                    dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            kt_handle.release()
                            ds_handle.release()

                            # dOVend
                            dp_handle = mma_dp_producer.acquire_and_advance()
                            tdPtdP_slice = tdPtdP[None, None, None, dp_handle.index]
                            dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            tdPrdO_slice = tdPrdO[None, None, None, 0]
                            v_handle = load_v_consumer.wait_and_advance()
                            tdPrV_trans_slice = tdPrV[None, None, None, v_handle.index]
                            num_kphases = cute.size(tdPrdO_slice, mode=[2])
                            if cutlass.const_expr(num_kphases % K_PHASE_UNROLL == 0):
                                num_outer_iter = num_kphases // K_PHASE_UNROLL
                                for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                    for kphase_idx in cutlass.range(K_PHASE_UNROLL, unroll_full=True):
                                        kphase_coord = (None, None, outer_iter * K_PHASE_UNROLL + kphase_idx)
                                        cute.gemm(
                                            dov_tiled_mma,
                                            tdPtdP_slice,
                                            tdPrdO_slice[kphase_coord],
                                            tdPrV_trans_slice[kphase_coord],
                                            tdPtdP_slice,
                                        )
                                        dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            else:
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        dov_tiled_mma,
                                        tdPtdP_slice,
                                        tdPrdO_slice[kphase_coord],
                                        tdPrV_trans_slice[kphase_coord],
                                        tdPtdP_slice,
                                    )
                                    dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            v_handle.release()
                            load_do_releaser.release()
                            load_do_releaser.advance()
                            dp_handle.commit()
                            # dSKTend
                            ds_handle = ds_mma_consumer.wait_and_advance()
                            dsk_whether_acc = dsk_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                            kt_handle = load_kt_consumer.wait_and_advance()
                            dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, dsk_whether_acc)
                            tdQtdQ_slice = tdQtdQ_staged[None, None, None, 0]
                            tdStdS_slice = tdPtdP[None, None, None, ds_handle.index]
                            tdS = cute.make_tensor(tdStdS_slice.iterator, ds_tmem_layout.outer)
                            tdQrdS = dsk_thr_mma.make_fragment_A(tdS)
                            tdQrdS_slice = cute.make_tensor(
                                cute.recast_ptr(tdStdS_slice.iterator, dtype=self.q_dtype),
                                tdQrdS.layout,
                            )

                            tdQrKT_slice = tdQrKT[None, None, None, kt_handle.index]
                            num_kphases = cute.size(tdQrKT_slice, mode=[2])
                            if cutlass.const_expr(num_kphases % K_PHASE_UNROLL == 0):
                                num_outer_iter = num_kphases // K_PHASE_UNROLL
                                for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                    for kphase_idx in cutlass.range(K_PHASE_UNROLL, unroll_full=True):
                                        kphase_coord = (None, None, outer_iter * K_PHASE_UNROLL + kphase_idx)
                                        cute.gemm(
                                            dsk_tiled_mma,
                                            tdQtdQ_slice,
                                            tdQrdS_slice[kphase_coord],
                                            tdQrKT_slice[kphase_coord],
                                            tdQtdQ_slice,
                                        )
                                        dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            else:
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        dsk_tiled_mma,
                                        tdQtdQ_slice,
                                        tdQrdS_slice[kphase_coord],
                                        tdQrKT_slice[kphase_coord],
                                        tdQtdQ_slice,
                                    )
                                    dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            kt_handle.release()
                            ds_handle.release()
                        else:
                            # QK0
                            s_handle = mma_s_producer.acquire_and_advance()
                            tStS_slice = tStS[None, None, None, s_handle.index]
                            qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                            load_q_consumer.wait_and_advance()
                            tSrQ_slice = tSrQ[None, None, None, 0]
                            k_handle = load_k_consumer.wait_and_advance()
                            tSrK_trans_slice = tSrK[None, None, None, k_handle.index]
                            num_kphases = cute.size(tSrQ_slice, mode=[2])
                            if cutlass.const_expr(num_kphases % K_PHASE_UNROLL == 0):
                                num_outer_iter = num_kphases // K_PHASE_UNROLL
                                for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                    for kphase_idx in cutlass.range(K_PHASE_UNROLL, unroll_full=True):
                                        kphase_coord = (None, None, outer_iter * K_PHASE_UNROLL + kphase_idx)
                                        cute.gemm(
                                            qk_tiled_mma,
                                            tStS_slice,
                                            tSrQ_slice[kphase_coord],
                                            tSrK_trans_slice[kphase_coord],
                                            tStS_slice,
                                        )
                                        qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            else:
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        qk_tiled_mma,
                                        tStS_slice,
                                        tSrQ_slice[kphase_coord],
                                        tSrK_trans_slice[kphase_coord],
                                        tStS_slice,
                                    )
                                    qk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            k_handle.release()
                            load_q_releaser.release()
                            load_q_releaser.advance()
                            s_handle.commit()

                            # dOV0
                            dp_handle = mma_dp_producer.acquire_and_advance()
                            tdPtdP_slice = tdPtdP[None, None, None, dp_handle.index]
                            dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                            load_do_consumer.wait_and_advance()
                            tdPrdO_slice = tdPrdO[None, None, None, 0]
                            v_handle = load_v_consumer.wait_and_advance()
                            tdPrV_trans_slice = tdPrV[None, None, None, v_handle.index]
                            num_kphases = cute.size(tdPrdO_slice, mode=[2])
                            if cutlass.const_expr(num_kphases % K_PHASE_UNROLL == 0):
                                num_outer_iter = num_kphases // K_PHASE_UNROLL
                                for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                    for kphase_idx in cutlass.range(K_PHASE_UNROLL, unroll_full=True):
                                        kphase_coord = (None, None, outer_iter * K_PHASE_UNROLL + kphase_idx)
                                        cute.gemm(
                                            dov_tiled_mma,
                                            tdPtdP_slice,
                                            tdPrdO_slice[kphase_coord],
                                            tdPrV_trans_slice[kphase_coord],
                                            tdPtdP_slice,
                                        )
                                        dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            else:
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        dov_tiled_mma,
                                        tdPtdP_slice,
                                        tdPrdO_slice[kphase_coord],
                                        tdPrV_trans_slice[kphase_coord],
                                        tdPtdP_slice,
                                    )
                                    dov_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            v_handle.release()
                            load_do_releaser.release()
                            load_do_releaser.advance()
                            dp_handle.commit()

                            # dSKT0
                            ds_handle = ds_mma_consumer.wait_and_advance()
                            dsk_whether_acc = dsk_tiled_mma.get(tcgen05.Field.ACCUMULATE)
                            kt_handle = load_kt_consumer.wait_and_advance()
                            dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, dsk_whether_acc)
                            tdQtdQ_slice = tdQtdQ_staged[None, None, None, 0]
                            tdStdS_slice = tdPtdP[None, None, None, ds_handle.index]
                            tdS = cute.make_tensor(tdStdS_slice.iterator, ds_tmem_layout.outer)
                            tdQrdS = dsk_thr_mma.make_fragment_A(tdS)
                            tdQrdS_slice = cute.make_tensor(
                                cute.recast_ptr(tdStdS_slice.iterator, dtype=self.q_dtype),
                                tdQrdS.layout,
                            )

                            tdQrKT_slice = tdQrKT[None, None, None, kt_handle.index]
                            num_kphases = cute.size(tdQrKT_slice, mode=[2])
                            if cutlass.const_expr(num_kphases % K_PHASE_UNROLL == 0):
                                num_outer_iter = num_kphases // K_PHASE_UNROLL
                                for outer_iter in cutlass.range(num_outer_iter, unroll=1):
                                    for kphase_idx in cutlass.range(K_PHASE_UNROLL, unroll_full=True):
                                        kphase_coord = (None, None, outer_iter * K_PHASE_UNROLL + kphase_idx)
                                        cute.gemm(
                                            dsk_tiled_mma,
                                            tdQtdQ_slice,
                                            tdQrdS_slice[kphase_coord],
                                            tdQrKT_slice[kphase_coord],
                                            tdQtdQ_slice,
                                        )
                                        dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            else:
                                for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
                                    kphase_coord = (None, None, kphase_idx)
                                    cute.gemm(
                                        dsk_tiled_mma,
                                        tdQtdQ_slice,
                                        tdQrdS_slice[kphase_coord],
                                        tdQrKT_slice[kphase_coord],
                                        tdQtdQ_slice,
                                    )
                                    dsk_tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            kt_handle.release()
                            ds_handle.release()
                        dq_handle.commit()
                tile_sched.advance_to_next_work()
                tile_info = tile_sched.get_current_work()
            mma_s_producer.tail()
            mma_dp_producer.tail()
            mma_dq_producer.tail()

        # ---- Compute warps: softmax + dP from LSE/sum_OdO ----
        if widx >= self.compute_warp_ids[0] and widx <= self.compute_warp_ids[-1]:
            # increase register after decreasing
            cute.arch.setmaxregister_increase(self.num_regs_compute)
            tmem.wait_for_alloc()
            while tile_info.is_valid_tile:
                block_coord = tile_info.tile_idx
                mma_coord = (
                    block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    block_coord[1],
                    block_coord[2],
                )
                batch_coord = block_coord[2][1]
                skip = False
                seqlen_q = gQ.shape[0]
                seqlen_k = gK.shape[0]
                cuseqlen_q = Int32(0)
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    skip = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_coord[0],
                        seqlen_q,
                    )
                if not skip:
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k

                    start_count = FusedMask.get_trip_start(
                        self.mask_type,
                        mma_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                    )
                    trip_count = FusedMask.get_trip_count(
                        self.mask_type,
                        mma_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                        window_size_right,
                    )
                    unmask_count = FusedMask.get_unmasked_trip_count(
                        self.mask_type,
                        mma_coord,
                        self.qk_mma_tiler,
                        seqlen_q,
                        seqlen_k,
                        window_size_left,
                        window_size_right,
                    )

                    cS_base = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
                    cS = cute.domain_offset((mma_coord[0] * self.qk_mma_tiler[0], 0), cS_base)
                    tScS = qk_thr_mma.partition_C(cS)

                    cdP_base = cute.make_identity_tensor((self.dov_mma_tiler[0], self.dov_mma_tiler[1]))
                    cdP = cute.domain_offset((mma_coord[0] * self.dov_mma_tiler[0], 0), cdP_base)
                    tdPcdP = dov_thr_mma.partition_C(cdP)

                    lse_handle = load_lse_consumer.wait_and_advance()
                    sum_odo_handle = load_sum_odo_consumer.wait_and_advance()
                    for step in cutlass.range(start_count, trip_count, 1, unroll=1):
                        cS_iter = cute.domain_offset((0, step * self.qk_mma_tiler[1]), cS)
                        tScS_iter = qk_thr_mma.partition_C(cS_iter)

                        cdP_iter = cute.domain_offset((0, step * self.dov_mma_tiler[1]), cdP)

                        tdPcdP_iter = dov_thr_mma.partition_C(cdP_iter)

                        # Si, dPi -> dSi
                        mma_s_consumer, mma_dp_consumer, ds_mma_producer = self.compute_step(
                            (step >= unmask_count, window_size_left, window_size_right),
                            (
                                seqlen_q,
                                seqlen_k,
                                scale_softmax,
                                batch_coord,
                            ),
                            (tStS, tScS_iter, tdPtdP, tdPcdP_iter, sLSE, sSum_OdO),
                            (mma_s_consumer, mma_dp_consumer, ds_mma_producer, lse_handle, sum_odo_handle),
                            step,
                        )
                    lse_handle.release()
                    sum_odo_handle.release()
                tile_sched.advance_to_next_work()
                tile_info = tile_sched.get_current_work()
            ds_mma_producer.tail()

        # ---- Epilogue: dQ writeback ----
        if widx >= self.epilogue_warp_ids[0] and widx <= self.epilogue_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_epilogue)
            tmem.allocate(self.tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            while tile_info.is_valid_tile:
                block_coord = tile_info.tile_idx
                mma_coord = (
                    block_coord[0] // cute.size(qk_tiled_mma.thr_id.shape),
                    block_coord[1],
                    block_coord[2],
                )
                batch_coord = block_coord[2][1]
                seqlen_q = gQ.shape[0]
                seqlen_k = gK.shape[0]
                skip = False
                cuseqlen_q = Int32(0)
                if cutlass.const_expr(cum_seqlen_q is not None):
                    cuseqlen_q = cum_seqlen_q[batch_coord]
                    seqlen_q = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                    skip = not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.qk_mma_tiler[0],
                        mma_coord[0],
                        seqlen_q,
                    )

                if not skip:
                    if cutlass.const_expr(cum_seqlen_k is not None):
                        cuseqlen_k = cum_seqlen_k[batch_coord]
                        seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k

                    gdQ_eff = gdQ
                    if cutlass.const_expr(cum_seqlen_q is not None):
                        block_offset_dQ = (
                            cuseqlen_q,
                            Int32(0),
                            Int32(0),
                            ((Int32(0), Int32(0)), Int32(0)),
                        )
                        gdQ_eff = cute.domain_offset(cute.select(block_offset_dQ, mode=[0, 2, 3]), gdQ)

                    # (bM, bN, loopM, loopN, loopL)
                    gdQ_qdl = cute.flat_divide(gdQ_eff, cute.select(self.dsk_block_tiler, mode=[0, 1]))
                    cdQ_qdl = cute.flat_divide(
                        cute.make_identity_tensor(gdQ_eff.shape),
                        cute.select(self.dsk_block_tiler, mode=[0, 1]),
                    )

                    gdQ_staged = gdQ_qdl[None, None, block_coord[0], None, block_coord[2]]
                    cdQ_staged = cdQ_qdl[None, None, block_coord[0], None, block_coord[2]]

                    # dQ TMEM to GMEM
                    mma_dq_consumer = self.dQ_epilogue(
                        (seqlen_q, cuseqlen_q, gQ.shape[0], batch_coord),
                        (mma_dq_consumer, gdQ_staged, cdQ_staged, tdQtdQ_staged),
                        self.epi_tile,
                    )
                tile_sched.advance_to_next_work()
                tile_info = tile_sched.get_current_work()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        if widx > self.load_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

        return

    @cute.jit
    def compute_step(
        self,
        mask_args: Tuple,
        value_args: Tuple,
        tensor_args: Tuple,
        pipeline_args: Tuple,
        step: Int32,
    ) -> Tuple[Float32, Float32, pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        need_apply_mask, window_size_left, window_size_right = mask_args
        seqlen_q, seqlen_k, scale_softmax, batch_coord = value_args
        tStS, tScS, tdPtdP, tdPcdP, sLSE, sSum_OdO = tensor_args
        mma_s_consumer, mma_dp_consumer, ds_mma_producer, lse_handle, sum_odo_handle = pipeline_args

        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.compute_warp_ids))
        s_handle = mma_s_consumer.wait_and_advance()
        tStS_slice = tStS[(None, None), 0, 0, s_handle.index]
        tScS_slice = tScS[(None, None), 0, 0]
        tmem_load_atom = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(16)), self.acc_dtype)
        tmem_tiled_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_slice)
        thr_load = tmem_tiled_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_load.partition_S(tStS_slice)
        tTMEM_LOADcS = thr_load.partition_D(tScS_slice)
        tTMEM_LOADrS = cute.make_rmem_tensor(tTMEM_LOADcS.shape, self.acc_dtype)
        cute.copy(tmem_tiled_load, tTMEM_LOADtS, tTMEM_LOADrS)
        cute.arch.fence_view_async_tmem_load()
        s_handle.release()
        if need_apply_mask:
            FusedMask.apply_mask(
                self.mask_type,
                tTMEM_LOADrS,
                tTMEM_LOADcS,
                seqlen_q,
                seqlen_k,
                window_size_left,
                window_size_right,
            )

        log2_e = cutlass.Float32(math.log2(math.e))
        softmax_scale_log2_e = scale_softmax * log2_e
        tTMEM_STORErP = cute.make_rmem_tensor(tTMEM_LOADrS.shape, self.q_dtype)
        for k in cutlass.range(0, cute.size(tTMEM_LOADrS), 2, unroll_full=True):
            lse = (
                sLSE[
                    cute.get(tTMEM_LOADcS[k], mode=[0]) - bidx * self.cta_tiler[0],
                    lse_handle.index,
                ],
                sLSE[
                    cute.get(tTMEM_LOADcS[k + 1], mode=[0]) - bidx * self.cta_tiler[0],
                    lse_handle.index,
                ],
            )
            tTMEM_LOADrS[k], tTMEM_LOADrS[k + 1] = cute.arch.fma_packed_f32x2(
                (tTMEM_LOADrS[k], tTMEM_LOADrS[k + 1]),
                (softmax_scale_log2_e, softmax_scale_log2_e),
                lse,
            )
            tTMEM_LOADrS[k] = cute.math.exp2(tTMEM_LOADrS[k], fastmath=True)
            tTMEM_LOADrS[k + 1] = cute.math.exp2(tTMEM_LOADrS[k + 1], fastmath=True)

        dp_handle = mma_dp_consumer.wait_and_advance()
        tdPtdP_slice = tdPtdP[(None, None), 0, 0, dp_handle.index]
        tdPcdP_slice = tdPcdP[(None, None), 0, 0]
        thr_load = tmem_tiled_load.get_slice(thread_idx)
        tTMEM_LOADtdP = thr_load.partition_S(tdPtdP_slice)
        tTMEM_LOADcdP = thr_load.partition_D(tdPcdP_slice)
        tTMEM_LOADrdP = cute.make_rmem_tensor(tTMEM_LOADcdP.shape, self.acc_dtype)
        cute.copy(tmem_tiled_load, tTMEM_LOADtdP, tTMEM_LOADrdP)
        cute.arch.fence_view_async_tmem_load()
        dp_handle.release()
        tTMEM_STORErdP = cute.make_rmem_tensor(tTMEM_LOADrdP.shape, self.q_dtype)

        for k in cutlass.range(0, cute.size(tTMEM_LOADrdP), 2, unroll_full=True):
            tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1] = cute.arch.add_packed_f32x2(
                (tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1]),
                (
                    sSum_OdO[
                        cute.get(tTMEM_LOADcdP[k], mode=[0]) - bidx * self.cta_tiler[0],
                        sum_odo_handle.index,
                    ],
                    sSum_OdO[
                        cute.get(tTMEM_LOADcdP[k + 1], mode=[0]) - bidx * self.cta_tiler[0],
                        sum_odo_handle.index,
                    ],
                ),
            )
            tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1] = cute.arch.mul_packed_f32x2(
                (tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1]), (tTMEM_LOADrS[k], tTMEM_LOADrS[k + 1])
            )
            tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1] = cute.arch.mul_packed_f32x2((tTMEM_LOADrdP[k], tTMEM_LOADrdP[k + 1]), (scale_softmax, scale_softmax))
        dp_vec = tTMEM_LOADrdP.load()
        tTMEM_STORErdP.store(dp_vec.to(self.q_dtype))

        ds_handle = ds_mma_producer.acquire_and_advance()
        tmem_store_atom = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(32)), self.acc_dtype)
        tdPtdP_dS_layout = cute.composition(
            tdPtdP_slice.layout,
            cute.make_layout((tdPtdP_slice.shape[0], self.tilePlikeFP32)),
        )
        tdPtdP_dS = cute.make_tensor(tdPtdP_slice.iterator, tdPtdP_dS_layout)
        tdPcdP_dS_layout = cute.composition(
            tdPcdP_slice.layout,
            cute.make_layout((tdPcdP_slice.shape[0], self.tilePlikeFP32)),
        )
        tdPcdP_dS = cute.make_tensor(tdPcdP_slice.iterator, tdPcdP_dS_layout)
        tmem_tiled_store = tcgen05.make_tmem_copy(tmem_store_atom, tdPtdP_dS)

        thr_store = tmem_tiled_store.get_slice(thread_idx)
        tTMEM_STOREtdS = thr_store.partition_D(tdPtdP_dS)
        tTMEM_STOREcdP = thr_store.partition_S(tdPcdP_dS)
        tTMEM_STORErdS_ = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErdP.iterator, dtype=self.acc_dtype),
            tTMEM_STOREcdP.shape,
        )
        cute.copy(tmem_tiled_store, tTMEM_STORErdS_, tTMEM_STOREtdS)
        cute.arch.fence_view_async_tmem_store()
        ds_handle.commit()
        return mma_s_consumer, mma_dp_consumer, ds_mma_producer

    @cute.jit
    def dQ_epilogue(
        self,
        value_args: Tuple,
        dq_args: Tuple,
        epi_tile: cute.Tile,
    ) -> Tuple[pipeline.PipelineConsumer, pipeline.PipelineProducer]:
        seqlen_q, cuseqlen_q, total_q, batch_coord = value_args
        mma_dq_consumer, gdQ_staged, cdQ_staged, tdQtdQ_staged = dq_args
        dq_handle = mma_dq_consumer.wait_and_advance()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cute.arch.fence_view_async_shared()

        gdQ = gdQ_staged[None, None, 0]
        cdQ = cdQ_staged[None, None, 0]
        tdQtdQ = tdQtdQ_staged[(None, None), 0, 0, 0]
        tdQtdQ_epi = cute.zipped_divide(tdQtdQ, epi_tile)
        cdQ_epi = cute.zipped_divide(cdQ, epi_tile)
        gdQ_epi = cute.zipped_divide(gdQ, epi_tile)
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.epilogue_warp_ids))
        tmem_copy_atom = cute.make_copy_atom(tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype)
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_copy_atom, tdQtdQ_epi)
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        tTMEM_LOADtdQ = thr_tmem_load.partition_S(tdQtdQ_epi)
        tTMEM_LOADgdQ = thr_tmem_load.partition_D(gdQ_epi)
        tTMEM_LOADcdQ = thr_tmem_load.partition_D(cdQ_epi)

        for i in cutlass.range(cute.size(tTMEM_LOADtdQ, mode=[1]), unroll_full=True):
            tTMEM_LOADtdQ_i = tTMEM_LOADtdQ[None, i, 0]
            tTMEM_LOADgdQ_i = tTMEM_LOADgdQ[None, i, 0]
            tTMEM_LOADcdQ_i = tTMEM_LOADcdQ[None, i, 0]
            tTMrdQ = cute.make_rmem_tensor(tTMEM_LOADcdQ[None, 0, i].shape, self.acc_dtype)
            cute.copy(tiled_tmem_load, tTMEM_LOADtdQ_i, tTMrdQ)
            tSMrdQ = cute.make_rmem_tensor(tTMrdQ.shape, self.q_dtype)
            dq_vec = tTMrdQ.load()
            tSMrdQ.store(dq_vec.to(self.q_dtype))
            if cute.elem_less(tTMEM_LOADcdQ_i[0][0], seqlen_q):
                cute.autovec_copy(tSMrdQ, tTMEM_LOADgdQ_i)
        dq_handle.release()
        return mma_dq_consumer

/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include "nvrtc_shim.h"
#include "attention_utils.h"
#include "../generated/sdpa/sm90/prefill/full_seqlens/d128_fprop_kernel.h"
#include "../generated/sdpa/sm90/prefill/full_seqlens/d64_fprop_kernel.h"
#include "../graph_helpers.h"
#include "../../cudnn_frontend_shim.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Suppress MSVC C4127 "conditional expression is constant" triggered by
// RETURN_CUDNN_FRONTEND_ERROR_IF macro (which has `if (retval == error_code_t::OK)`)
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4127)
#endif

#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace cudnn_frontend {
namespace experimental {

// ============================================================
// Kernel spec lookup (replaces KernelFactory)
// ============================================================

inline const KernelSpec*
lookup_kernel_spec(int d, int sm_version) {
    if (sm_version != 90) return nullptr;

    static const KernelSpec spec_d128 = {
        generated::d128_fprop_source,
        generated::d128_fprop_source_len,
        generated::d128_fprop_flags,
        generated::d128_fprop_flags_len,
        "cudnn_generated_oss_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x128_4x1x1_cga1x1x1_kernel0_0",
        64,
        128,
        128,
        232448  // smem_bytes for d=128 (227 * 1024)
    };

    static const KernelSpec spec_d64 = {
        generated::d64_fprop_source,
        generated::d64_fprop_source_len,
        generated::d64_fprop_flags,
        generated::d64_fprop_flags_len,
        "cudnn_generated_oss_sdpa_sm90_flash_fprop_wgmma_f16_knob_7_64x128x64_4x1x1_cga1x1x1_kernel0_0",
        64,
        128,
        64,
        122880  // smem_bytes for d=64 (120 * 1024)
    };

    switch (d) {
        case 128:
            return &spec_d128;
        case 64:
            return &spec_d64;
        default:
            return nullptr;
    }
}
// ============================================================
// Sm90SdpaPrefillEngine
// ============================================================

class Sm90SdpaPrefillEngine {
   public:
    Sm90SdpaPrefillEngine() = default;

    ~Sm90SdpaPrefillEngine() {
        if (module_) {
            detail::cu_library_unload(module_);
        }
    }

    // Non-copyable
    Sm90SdpaPrefillEngine(const Sm90SdpaPrefillEngine&) = delete;
    Sm90SdpaPrefillEngine&
    operator=(const Sm90SdpaPrefillEngine&) = delete;

    // Movable
    Sm90SdpaPrefillEngine(Sm90SdpaPrefillEngine&& other) noexcept
        : spec_(other.spec_),
          d_(other.d_),
          sm_version_(other.sm_version_),
          sm_count_(other.sm_count_),
          support_checked_(other.support_checked_),
          module_(other.module_),
          kernelPtr_(other.kernelPtr_),
          cubin_(std::move(other.cubin_)),
          cubinSize_(other.cubinSize_),
          tile_m_(other.tile_m_),
          tile_n_(other.tile_n_),
          tile_k_(other.tile_k_),
          smemBytes_(other.smemBytes_),
          kernelName_(std::move(other.kernelName_)),
          built_(other.built_) {
        other.module_    = nullptr;
        other.kernelPtr_ = nullptr;
        other.built_     = false;
    }

    Sm90SdpaPrefillEngine&
    operator=(Sm90SdpaPrefillEngine&& other) noexcept {
        if (this != &other) {
            if (module_) {
                detail::cu_library_unload(module_);
            }
            spec_            = other.spec_;
            d_               = other.d_;
            sm_version_      = other.sm_version_;
            sm_count_        = other.sm_count_;
            support_checked_ = other.support_checked_;
            module_          = other.module_;
            kernelPtr_       = other.kernelPtr_;
            cubin_           = std::move(other.cubin_);
            cubinSize_       = other.cubinSize_;
            tile_m_          = other.tile_m_;
            tile_n_          = other.tile_n_;
            tile_k_          = other.tile_k_;
            smemBytes_       = other.smemBytes_;
            kernelName_      = std::move(other.kernelName_);
            built_           = other.built_;
            other.module_    = nullptr;
            other.kernelPtr_ = nullptr;
            other.built_     = false;
        }
        return *this;
    }

    // ---- Phase 1: Support check (spec lookup, cache SM count) ----
    error_t
    check_support(AttentionShape_t shape, int sm_version) {
        spec_ = lookup_kernel_spec(shape.d_qk, sm_version);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            spec_ == nullptr,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "Unsupported config: d=" + std::to_string(shape.d_qk) + " sm_version=" + std::to_string(sm_version));

        d_          = shape.d_qk;
        sm_version_ = sm_version;
        tile_m_     = spec_->tile_m;
        tile_n_     = spec_->tile_n;
        tile_k_     = spec_->tile_k;
        smemBytes_  = spec_->smem_bytes;
        kernelName_ = spec_->kernel_name;

        // Cache SM count for later use in execute()
        cudaDeviceProp devProp;
        int device_ordinal   = 0;
        cudaError_t cuda_err = detail::cuda_get_device(&device_ordinal);
        RETURN_CUDNN_FRONTEND_ERROR_IF(cuda_err != cudaSuccess, error_code_t::CUDA_API_FAILED, "cudaGetDevice failed");
        cuda_err = detail::cuda_get_device_properties(&devProp, device_ordinal);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            cuda_err != cudaSuccess, error_code_t::CUDA_API_FAILED, "cudaGetDeviceProperties failed");
        sm_count_ = devProp.multiProcessorCount;

        support_checked_ = true;
        return {error_code_t::OK, ""};
    }

    // Workspace size needed by execute() (tile_id_counter: 4 bytes, 16-byte aligned)
    static int64_t
    get_workspace_size() {
        return 16;
    }

    // ---- Phase 2: NVRTC compilation + module loading ----
    error_t
    build() {
        RETURN_CUDNN_FRONTEND_ERROR_IF(!support_checked_ || spec_ == nullptr,
                                       error_code_t::INVALID_VALUE,
                                       "build() called before check_support()");

        // Ensure NVRTC library is available
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !detail::nvrtc_is_loaded(), error_code_t::CUDA_API_FAILED, "NVRTC library could not be loaded");

        // Parse flags from embedded string
        std::vector<std::string> flags = parse_flags_string(spec_->flags_raw, spec_->flags_len);
        std::vector<const char*> flag_ptrs;
        flag_ptrs.reserve(flags.size());
        for (auto& f : flags) {
            flag_ptrs.push_back(f.c_str());
        }

        // Create NVRTC program from embedded source
        nvrtcProgram prog;
        nvrtcResult nvrtc_err;

        nvrtc_err = detail::nvrtc_create_program(&prog, spec_->source, "kernel.cu", 0, nullptr, nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            nvrtc_err != NVRTC_SUCCESS, error_code_t::CUDA_API_FAILED, "nvrtcCreateProgram failed");

        // Compile
        nvrtcResult compResult = detail::nvrtc_compile_program(prog, (int)flag_ptrs.size(), flag_ptrs.data());

        if (cudnn_frontend::isLoggingEnabled()) {
            // Write the source to a file
            std::ofstream ofs("kernel.cu");
            ofs << spec_->source;

            // Try to dump PTX for debugging (cicc may have succeeded even if ptxas failed)
            size_t ptxSize = 0;
            if (detail::nvrtc_get_ptx_size(prog, &ptxSize) == NVRTC_SUCCESS && ptxSize > 1) {
                std::vector<char> ptx(ptxSize);
                if (detail::nvrtc_get_ptx(prog, ptx.data()) == NVRTC_SUCCESS) {
                    std::ofstream ptx_ofs("kernel.ptx");
                    ptx_ofs.write(ptx.data(), static_cast<std::streamsize>(ptxSize - 1));
                }
            }
        }

        if (compResult != NVRTC_SUCCESS) {
            // Try to retrieve the compilation log for diagnostics
            size_t logSize = 0;
            detail::nvrtc_get_program_log_size(prog, &logSize);
            std::string log_msg = "NVRTC compilation failed";
            if (logSize > 1) {
                std::vector<char> log(logSize);
                detail::nvrtc_get_program_log(prog, log.data());
                log_msg += ": ";
                log_msg += log.data();
            }

            CUDNN_FE_LOG_LABEL_ENDL("ERROR: NVRTC compilation failed: " << log_msg);
            detail::nvrtc_destroy_program(&prog);
            return {error_code_t::CUDA_API_FAILED, log_msg};
        }

        // Extract CUBIN
        nvrtc_err = detail::nvrtc_get_cubin_size(prog, &cubinSize_);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            nvrtc_err != NVRTC_SUCCESS, error_code_t::CUDA_API_FAILED, "nvrtcGetCUBINSize failed");

        cubin_ = std::make_unique<char[]>(cubinSize_);

        nvrtc_err = detail::nvrtc_get_cubin(prog, cubin_.get());
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            nvrtc_err != NVRTC_SUCCESS, error_code_t::CUDA_API_FAILED, "nvrtcGetCUBIN failed");

        nvrtc_err = detail::nvrtc_destroy_program(&prog);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            nvrtc_err != NVRTC_SUCCESS, error_code_t::CUDA_API_FAILED, "nvrtcDestroyProgram failed");

        // Load module + extract kernel function
        CUresult cu_err;

        cu_err = detail::cu_library_load_data(&module_, cubin_.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
        RETURN_CUDNN_FRONTEND_ERROR_IF(cu_err != CUDA_SUCCESS,
                                       error_code_t::CUDA_API_FAILED,
                                       "cuLibraryLoadData failed (cubin_size=" + std::to_string(cubinSize_) +
                                           ", CUresult=" + detail::cu_result_to_string(cu_err) + ")");

        cu_err = detail::cu_library_get_kernel(&kernelPtr_, module_, kernelName_.c_str());
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            cu_err != CUDA_SUCCESS,
            error_code_t::CUDA_API_FAILED,
            "cuLibraryGetKernel failed: " + detail::cu_result_to_string(cu_err) + " (kernel: " + kernelName_ + ")");

        built_ = true;
        return {error_code_t::OK, ""};
    }

    // ---- Phase 3: Set up TMA descriptors, pack params, launch ----
    // Strides are in ELEMENTS (not bytes), matching PyTorch's .stride() convention.
    // All stride vectors have 4 elements in (b, h, s, d) order.
    // For max/sum_exp the 4th element (d-stride) is typically 1 and is unused internally.
    error_t
    execute(int batch,
            int heads_q,
            int heads_kv,
            int seq_q,
            int seq_kv,
            int d,
            void* d_Q,
            std::vector<int64_t> const& q_strides,
            void* d_K,
            std::vector<int64_t> const& k_strides,
            void* d_V,
            std::vector<int64_t> const& v_strides,
            void* d_O,
            std::vector<int64_t> const& o_strides,
            void* d_max,
            std::vector<int64_t> const& max_strides,
            void* d_sum_exp,
            std::vector<int64_t> const& se_strides,
            void* workspace,
            CUdevice device,
            cudaStream_t stream,
            std::optional<float> user_attn_scale = std::nullopt) {
        RETURN_CUDNN_FRONTEND_ERROR_IF(!built_, error_code_t::INVALID_VALUE, "execute() called before build()");
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            q_strides.size() < 4 || k_strides.size() < 4 || v_strides.size() < 4 || o_strides.size() < 4,
            error_code_t::INVALID_VALUE,
            "Q/K/V/O strides must have at least 4 elements");
        RETURN_CUDNN_FRONTEND_ERROR_IF(max_strides.size() < 3 || se_strides.size() < 3,
                                       error_code_t::INVALID_VALUE,
                                       "max/sum_exp strides must have at least 3 elements");

        int B      = batch;
        int H_q    = heads_q;
        int H_kv   = heads_kv;
        int S_q    = seq_q;
        int S_kv   = seq_kv;
        int D      = d;
        int TILE_M = tile_m_;
        int TILE_N = tile_n_;

        // ---- Set up AttentionDescriptor ----
        AttentionDescriptor_t attnDesc = {};
        attnDesc.b                     = B;
        attnDesc.q_h                   = H_q;
        attnDesc.k_h                   = H_kv;
        attnDesc.v_h                   = H_kv;
        attnDesc.s_q                   = S_q;
        attnDesc.s_kv                  = S_kv;
        attnDesc.d_qk                  = D;
        attnDesc.d_v                   = D;
        attnDesc.q_heads_per_k         = (H_q >= H_kv && H_kv > 0) ? (uint16_t)(H_q / H_kv) : 1;
        attnDesc.q_heads_per_v         = (H_q >= H_kv && H_kv > 0) ? (uint16_t)(H_q / H_kv) : 1;
        attnDesc.min_q_heads_per_kv    = attnDesc.q_heads_per_k;

        // ---- Tile scheduling ----
        int effective_tile_m       = TILE_M * 2;
        int tiles_r                = div_up(S_q, effective_tile_m);
        int tiles_hr               = H_q * tiles_r;
        int num_tiles              = B * tiles_hr;
        FastDivisor_t tiles_hr_div = make_fast_divisor(tiles_hr);
        FastDivisor_t tiles_r_div  = make_fast_divisor(tiles_r);

        // ---- TMA descriptors (4D) ----
        // Caller strides are in elements; TMA needs bytes.
        uint32_t elem_bytes   = sizeof(half);
        uint32_t elems_per_ld = std::min(64, D);

        // Q TMA (tma_tensor_3): dims = (D, S_q, H_q, B)
        cudaTmaDesc tma_q = {};
        CHECK_CUDNN_FRONTEND_ERROR(create_tma_desc_4d(&tma_q,
                                                      d_Q,
                                                      CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                                                      D,
                                                      S_q,
                                                      H_q,
                                                      B,
                                                      (uint64_t)q_strides[2] * elem_bytes,
                                                      (uint64_t)q_strides[1] * elem_bytes,
                                                      (uint64_t)q_strides[0] * elem_bytes,
                                                      elems_per_ld,
                                                      TILE_M * 2,
                                                      CU_TENSOR_MAP_SWIZZLE_128B));

        // K TMA (tma_tensor_2)
        cudaTmaDesc tma_k = {};
        CHECK_CUDNN_FRONTEND_ERROR(create_tma_desc_4d(&tma_k,
                                                      d_K,
                                                      CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                                                      D,
                                                      S_kv,
                                                      H_kv,
                                                      B,
                                                      (uint64_t)k_strides[2] * elem_bytes,
                                                      (uint64_t)k_strides[1] * elem_bytes,
                                                      (uint64_t)k_strides[0] * elem_bytes,
                                                      elems_per_ld,
                                                      TILE_N,
                                                      CU_TENSOR_MAP_SWIZZLE_128B));

        // V TMA (tma_tensor_1)
        cudaTmaDesc tma_v = {};
        CHECK_CUDNN_FRONTEND_ERROR(create_tma_desc_4d(&tma_v,
                                                      d_V,
                                                      CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                                                      D,
                                                      S_kv,
                                                      H_kv,
                                                      B,
                                                      (uint64_t)v_strides[2] * elem_bytes,
                                                      (uint64_t)v_strides[1] * elem_bytes,
                                                      (uint64_t)v_strides[0] * elem_bytes,
                                                      elems_per_ld,
                                                      TILE_N,
                                                      CU_TENSOR_MAP_SWIZZLE_128B));

        // O TMA (tma_tensor_5)
        cudaTmaDesc tma_o = {};
        CHECK_CUDNN_FRONTEND_ERROR(create_tma_desc_4d(&tma_o,
                                                      d_O,
                                                      CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                                                      D,
                                                      S_q,
                                                      H_q,
                                                      B,
                                                      (uint64_t)o_strides[2] * elem_bytes,
                                                      (uint64_t)o_strides[1] * elem_bytes,
                                                      (uint64_t)o_strides[0] * elem_bytes,
                                                      elems_per_ld,
                                                      16,
                                                      CU_TENSOR_MAP_SWIZZLE_128B));

        // ---- Scaling ----
        float attn_scale = user_attn_scale.value_or(1.0f / std::sqrt((float)D));
        float tensor_13  = -INFINITY;

        // ---- Tensor descriptors for max and sum_exp (f32, [B, H_q, S_q]) ----
        // Strides provided by caller in elements.
        tensor_descriptor desc_max = {};
        desc_max.num_dims          = 3;
        desc_max.dims[0]           = B;
        desc_max.dims[1]           = H_q;
        desc_max.dims[2]           = S_q;
        desc_max.strides[0]        = max_strides[0];
        desc_max.strides[1]        = max_strides[1];
        desc_max.strides[2]        = max_strides[2];

        tensor_descriptor desc_sum_exp = {};
        desc_sum_exp.num_dims          = 3;
        desc_sum_exp.dims[0]           = B;
        desc_sum_exp.dims[1]           = H_q;
        desc_sum_exp.dims[2]           = S_q;
        desc_sum_exp.strides[0]        = se_strides[0];
        desc_sum_exp.strides[1]        = se_strides[1];
        desc_sum_exp.strides[2]        = se_strides[2];

        // ---- Grid/Block ----
        int CTAs = std::min(sm_count_, num_tiles);
        dim3 grid(CTAs, 1, 1);
        dim3 block(384, 1, 1);

        // ---- Memset tile_id_counter (uses caller-provided workspace) ----
        RETURN_CUDNN_FRONTEND_ERROR_IF(!workspace,
                                       error_code_t::INVALID_VALUE,
                                       "workspace must not be null (need at least 16 bytes for tile_id_counter)");
        int32_t* d_tile_id_counter = reinterpret_cast<int32_t*>(workspace);
        {
            cudaError_t memset_err = detail::cuda_mem_set_async(d_tile_id_counter, 0, sizeof(int32_t), stream);
            RETURN_CUDNN_FRONTEND_ERROR_IF(
                memset_err != cudaSuccess, error_code_t::CUDA_API_FAILED, "cudaMemsetAsync for tile_id_counter failed");
        }

        // ---- Build kernel params array ----
        void* kernelParams[] = {
            (void*)&attnDesc,
            (void*)&num_tiles,
            (void*)&tiles_hr_div,
            (void*)&tiles_r_div,
            (void*)&d_tile_id_counter,
            (void*)&tma_q,
            (void*)&tma_k,
            (void*)&attn_scale,
            (void*)&tensor_13,
            (void*)&d_max,
            (void*)&desc_max,
            (void*)&d_sum_exp,
            (void*)&desc_sum_exp,
            (void*)&tma_v,
            (void*)&tma_o,
        };

        // ---- Set shared memory attribute and launch ----
        CUresult cu_err;

        cu_err = detail::cu_kernel_set_attribute(
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smemBytes_, kernelPtr_, device);
        RETURN_CUDNN_FRONTEND_ERROR_IF(cu_err != CUDA_SUCCESS,
                                       error_code_t::CUDA_API_FAILED,
                                       "cuKernelSetAttribute failed (smem=" + std::to_string(smemBytes_) +
                                           " bytes, CUresult=" + detail::cu_result_to_string(cu_err) + ")");

        cu_err               = detail::cu_launch_kernel((CUfunction)kernelPtr_,
                                          grid.x,
                                          grid.y,
                                          grid.z,
                                          block.x,
                                          block.y,
                                          block.z,
                                          smemBytes_,
                                          stream,
                                          kernelParams,
                                          nullptr);
        cudaError_t last_err = detail::cuda_get_last_error();
        RETURN_CUDNN_FRONTEND_ERROR_IF(cu_err != CUDA_SUCCESS,
                                       error_code_t::CUDA_API_FAILED,
                                       "cuLaunchKernel failed (grid=" + std::to_string(grid.x) +
                                           ", block=384, smem=" + std::to_string(smemBytes_) +
                                           ", CUresult=" + detail::cu_result_to_string(cu_err) + ")");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            last_err != cudaSuccess,
            error_code_t::CUDA_API_FAILED,
            "cudaGetLastError after kernel launch: " + detail::cuda_error_to_string(last_err));

        return {error_code_t::OK, ""};
    }

   private:
    // State from check_support()
    const KernelSpec* spec_ = nullptr;
    int d_                  = 0;
    int sm_version_         = 0;
    int sm_count_           = 0;
    bool support_checked_   = false;

    // State from build()
    CUlibrary module_   = nullptr;
    CUkernel kernelPtr_ = nullptr;
    std::unique_ptr<char[]> cubin_;
    size_t cubinSize_ = 0;
    int tile_m_ = 0, tile_n_ = 0, tile_k_ = 0;
    int smemBytes_ = 0;
    std::string kernelName_;
    bool built_ = false;
};

}  // namespace experimental
}  // namespace cudnn_frontend

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

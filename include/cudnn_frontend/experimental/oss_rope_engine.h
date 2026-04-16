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

#include "oss_engine_interface.h"
#include "../generated/rope/rope_fwd_kernel.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <memory>

namespace cudnn_frontend {
namespace experimental {

class RoPEEngine : public IOssRoPEEngine {
   public:
    // ---- Phase 1: Support check ----
    error_t
    check_support(RoPEShape_t shape, int sm_version) override {
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            sm_version < 80, error_code_t::GRAPH_NOT_SUPPORTED, "RoPE engine requires SM80+");

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            shape.head_dim <= 0 || (shape.head_dim % 2 != 0),
            error_code_t::GRAPH_NOT_SUPPORTED,
            "RoPE requires even head_dim, got " + std::to_string(shape.head_dim));

        RETURN_CUDNN_FRONTEND_ERROR_IF(
            shape.head_dim / 2 > 256,
            error_code_t::GRAPH_NOT_SUPPORTED,
            "RoPE supports head_dim/2 <= 256, got " + std::to_string(shape.head_dim / 2));

        sm_version_ = sm_version;
        is_bf16_    = shape.is_bf16;

        support_checked_ = true;
        return {error_code_t::OK, ""};
    }

    // ---- Phase 2: NVRTC compilation ----
    error_t
    build() override {
        RETURN_CUDNN_FRONTEND_ERROR_IF(!support_checked_, error_code_t::INVALID_VALUE, "build() before check_support()");

        // Include headers first, then dtype define, then kernel body.
        // The kernel source already has #include <cuda_bf16.h> etc. at the top,
        // so we insert the DTYPE define after it using a separate preamble.
        std::string preamble = "#include <cuda_bf16.h>\n#include <cuda_fp16.h>\n";
        std::string dtype_def =
            is_bf16_ ? "using DTYPE = nv_bfloat16;\n" : "using DTYPE = __half;\n";

        std::string full_source = preamble + dtype_def + std::string(generated::rope_fwd_kernel_source);

        // Parse compile flags
        std::vector<std::string> flags =
            parse_flags_string(generated::rope_fwd_kernel_flags, generated::rope_fwd_kernel_flags_len);

        // Override GPU architecture
        {
            std::string target_arch;
            if (sm_version_ >= 100)
                target_arch = "sm_100a";
            else if (sm_version_ == 90)
                target_arch = "sm_90a";
            else if (sm_version_ > 90 && sm_version_ < 100)
                target_arch = "sm_90";
            else if (sm_version_ >= 89)
                target_arch = "sm_89";
            else if (sm_version_ >= 86)
                target_arch = "sm_86";
            else
                target_arch = "sm_80";

            for (auto& f : flags) {
                if (f.find("--gpu-architecture=") == 0) {
                    f = "--gpu-architecture=" + target_arch;
                    break;
                }
            }
        }

        // Add CUDA include paths for NVRTC to resolve cuda_bf16.h / cuda_fp16.h
        std::string cuda_include = "/usr/local/cuda/include";
        if (auto env0 = std::getenv("CUDA_HOME")) {
            cuda_include = std::string(env0) + "/include";
        } else if (auto env1 = std::getenv("CUDA_PATH")) {
            cuda_include = std::string(env1) + "/include";
        }
        flags.push_back("--include-path=" + cuda_include);

        std::vector<const char*> flag_ptrs;
        for (auto& f : flags) flag_ptrs.push_back(f.c_str());

        // NVRTC compile
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            !detail::nvrtc_is_loaded(), error_code_t::CUDA_API_FAILED, "NVRTC library could not be loaded");

        nvrtcProgram prog;
        nvrtcResult nvrtc_err;

        nvrtc_err = detail::nvrtc_create_program(&prog, full_source.c_str(), "rope_fwd.cu", 0, nullptr, nullptr);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            nvrtc_err != NVRTC_SUCCESS, error_code_t::CUDA_API_FAILED, "nvrtcCreateProgram failed");

        nvrtcResult compResult = detail::nvrtc_compile_program(prog, (int)flag_ptrs.size(), flag_ptrs.data());
        if (compResult != NVRTC_SUCCESS) {
            size_t logSize = 0;
            detail::nvrtc_get_program_log_size(prog, &logSize);
            std::string log_msg = "NVRTC compilation failed for RoPE kernel";
            if (logSize > 1) {
                std::vector<char> log(logSize);
                detail::nvrtc_get_program_log(prog, log.data());
                log_msg += ": ";
                log_msg += log.data();
            }
            detail::nvrtc_destroy_program(&prog);
            return {error_code_t::CUDA_API_FAILED, log_msg};
        }

        // Extract CUBIN
        nvrtc_err = detail::nvrtc_get_cubin_size(prog, &cubinSize_);
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            nvrtc_err != NVRTC_SUCCESS, error_code_t::CUDA_API_FAILED, "nvrtcGetCUBINSize failed");

        cubin_    = std::make_unique<char[]>(cubinSize_);
        nvrtc_err = detail::nvrtc_get_cubin(prog, cubin_.get());
        RETURN_CUDNN_FRONTEND_ERROR_IF(
            nvrtc_err != NVRTC_SUCCESS, error_code_t::CUDA_API_FAILED, "nvrtcGetCUBIN failed");

        detail::nvrtc_destroy_program(&prog);

        // Load module + kernel
        cudaError_t cuda_err;
        cuda_err = detail::cuda_library_load_data(&module_, cubin_.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
        RETURN_CUDNN_FRONTEND_ERROR_IF(cuda_err != cudaSuccess,
                                       error_code_t::CUDA_API_FAILED,
                                       "cudaLibraryLoadData failed: " + detail::cuda_error_to_string(cuda_err));

        cuda_err = detail::cuda_library_get_kernel(&kernelPtr_, module_, "rope_fwd_kernel");
        RETURN_CUDNN_FRONTEND_ERROR_IF(cuda_err != cudaSuccess,
                                       error_code_t::CUDA_API_FAILED,
                                       "cudaLibraryGetKernel failed: " + detail::cuda_error_to_string(cuda_err));

        built_ = true;
        return {error_code_t::OK, ""};
    }

    // ---- Phase 3: Execute ----
    error_t
    execute(void* input,
            void* cos,
            void* sin,
            void* output,
            int batch,
            int seq_len,
            int num_heads,
            int head_dim,
            std::vector<int64_t> const& in_strides,
            std::vector<int64_t> const& out_strides,
            int device,
            cudaStream_t stream) override {
        RETURN_CUDNN_FRONTEND_ERROR_IF(!built_, error_code_t::INVALID_VALUE, "execute() before build()");

        // Pack kernel parameters - must match RoPEParams struct in the kernel
        struct RoPEParams {
            const void* input;
            const void* cos;
            const void* sin;
            void* output;
            int batch;
            int seq_len;
            int num_heads;
            int head_dim;
            int in_stride_b;
            int in_stride_s;
            int in_stride_h;
            int out_stride_b;
            int out_stride_s;
            int out_stride_h;
        };

        RoPEParams params{};
        params.input      = input;
        params.cos        = cos;
        params.sin        = sin;
        params.output     = output;
        params.batch      = batch;
        params.seq_len    = seq_len;
        params.num_heads  = num_heads;
        params.head_dim   = head_dim;
        params.in_stride_b  = static_cast<int>(in_strides[0]);
        params.in_stride_s  = static_cast<int>(in_strides[1]);
        params.in_stride_h  = static_cast<int>(in_strides[2]);
        params.out_stride_b = static_cast<int>(out_strides[0]);
        params.out_stride_s = static_cast<int>(out_strides[1]);
        params.out_stride_h = static_cast<int>(out_strides[2]);

        // Grid: one block per (seq_pos, batch)
        dim3 grid(seq_len, batch);
        // Block: 32 threads in x (iterate over D/2), up to 8 warps in y (iterate over heads)
        int warps_y = std::min(num_heads, 8);
        dim3 block(32, warps_y);

        // Dynamic shared memory: cos + sin cached in float32
        int d2        = head_dim / 2;
        int smem_size = 2 * d2 * static_cast<int>(sizeof(float));

        // Set shared memory attribute
        cudaError_t cuda_err;
        cuda_err = detail::cuda_kernel_set_attribute_for_device(
            kernelPtr_, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size, device);
        RETURN_CUDNN_FRONTEND_ERROR_IF(cuda_err != cudaSuccess,
                                       error_code_t::CUDA_API_FAILED,
                                       "cudaKernelSetAttributeForDevice failed");

        void* kernelParams[] = {reinterpret_cast<void*>(&params)};

        cuda_err = detail::cuda_launch_kernel(
            (const void*)kernelPtr_, grid, block, kernelParams, smem_size, stream);
        RETURN_CUDNN_FRONTEND_ERROR_IF(cuda_err != cudaSuccess,
                                       error_code_t::CUDA_API_FAILED,
                                       "RoPE kernel launch failed: " + detail::cuda_error_to_string(cuda_err));

        return {error_code_t::OK, ""};
    }

    int64_t
    get_workspace_size() const override {
        return 0;  // No workspace needed
    }

   private:
    bool support_checked_ = false;
    bool built_           = false;
    int sm_version_       = 0;
    bool is_bf16_         = true;

    // NVRTC compilation artifacts
    std::unique_ptr<char[]> cubin_;
    size_t cubinSize_     = 0;
    cudaLibrary_t module_ = nullptr;
    cudaKernel_t kernelPtr_ = nullptr;
};

}  // namespace experimental
}  // namespace cudnn_frontend

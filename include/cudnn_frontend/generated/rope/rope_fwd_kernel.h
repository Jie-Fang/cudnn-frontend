// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

// RoPE (Rotary Position Embedding) forward kernel
// Non-interleaved (halved) variant: y1 = x1*cos - x2*sin, y2 = x2*cos + x1*sin
// where x1 = input[..., :D/2], x2 = input[..., D/2:]

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Woverlength-strings"
#endif

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4068)
#endif

namespace cudnn_frontend::experimental::generated {

inline constexpr const char rope_fwd_kernel_source[] =
    R"KERNEL(

// Note: cuda_bf16.h and cuda_fp16.h are included via the preamble in oss_rope_engine.h
// DTYPE is defined there as either nv_bfloat16 or __half

struct RoPEParams {
    const void* input;   // [B, S, H, D] - BSHD layout
    const void* cos;     // [S, D/2]
    const void* sin;     // [S, D/2]
    void* output;        // [B, S, H, D] - BSHD layout
    int batch;
    int seq_len;
    int num_heads;
    int head_dim;
    // Input strides (in elements)
    int in_stride_b;     // stride for batch dimension
    int in_stride_s;     // stride for sequence dimension
    int in_stride_h;     // stride for head dimension
    // Output strides (in elements)
    int out_stride_b;
    int out_stride_s;
    int out_stride_h;
};

extern "C" __global__ void rope_fwd_kernel(RoPEParams params) {
    const int s_id = blockIdx.x;
    const int b_id = blockIdx.y;

    if (s_id >= params.seq_len || b_id >= params.batch) return;

    const int d2 = params.head_dim / 2;

    // Load cos/sin for this sequence position into shared memory (float32)
    extern __shared__ float smem[];
    float* sh_cos = smem;
    float* sh_sin = smem + d2;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;
    const DTYPE* cos_ptr = reinterpret_cast<const DTYPE*>(params.cos);
    const DTYPE* sin_ptr = reinterpret_cast<const DTYPE*>(params.sin);

    for (int i = tid; i < d2; i += block_size) {
        sh_cos[i] = static_cast<float>(cos_ptr[s_id * d2 + i]);
        sh_sin[i] = static_cast<float>(sin_ptr[s_id * d2 + i]);
    }
    __syncthreads();

    const DTYPE* in_ptr  = reinterpret_cast<const DTYPE*>(params.input);
    DTYPE* out_ptr       = reinterpret_cast<DTYPE*>(params.output);

    const int in_base  = b_id * params.in_stride_b  + s_id * params.in_stride_s;
    const int out_base = b_id * params.out_stride_b + s_id * params.out_stride_s;

    for (int h = threadIdx.y; h < params.num_heads; h += blockDim.y) {
        const int in_head_off  = in_base  + h * params.in_stride_h;
        const int out_head_off = out_base + h * params.out_stride_h;

        for (int d = threadIdx.x; d < d2; d += blockDim.x) {
            // x1 = input[..., d], x2 = input[..., d + d2]
            float x1 = static_cast<float>(in_ptr[in_head_off + d]);
            float x2 = static_cast<float>(in_ptr[in_head_off + d + d2]);
            float c  = sh_cos[d];
            float s  = sh_sin[d];

            // y1 = x1 * cos - x2 * sin
            // y2 = x2 * cos + x1 * sin
            out_ptr[out_head_off + d]      = static_cast<DTYPE>(x1 * c - x2 * s);
            out_ptr[out_head_off + d + d2] = static_cast<DTYPE>(x2 * c + x1 * s);
        }
    }
}

)KERNEL";
inline constexpr size_t rope_fwd_kernel_source_len = sizeof(rope_fwd_kernel_source) - 1;

inline constexpr const char rope_fwd_kernel_flags[] = R"FLAGS(--gpu-architecture=sm_80
--std=c++17
-w
--define-macro=__CUDACC_RTC__
-default-device
--use_fast_math
)FLAGS";
inline constexpr size_t rope_fwd_kernel_flags_len = sizeof(rope_fwd_kernel_flags) - 1;

}  // namespace cudnn_frontend::experimental::generated

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

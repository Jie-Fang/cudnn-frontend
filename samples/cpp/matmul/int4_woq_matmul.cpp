/*
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <catch2/catch_test_macros.hpp>

#include <random>

#include "../utils/helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("Int4 WoQ Matmul", "[matmul][graph][int4][woq]") {
#if (CUDNN_VERSION < 91100)
    SKIP("WoQ requires 9.11.0 and above");
#endif
    if (cudnnGetCudartVersion() < 12000) {
        SKIP("Test requires cuda toolkit 12.0 or above");
    }
    if (!(is_ada_arch() || get_compute_capability() == 120)) {
        SKIP("WoQ only supported on Ada or Blackwell");
    }

    namespace fe             = cudnn_frontend;
    constexpr int kBlockSize = 128;

    // matmul problem size
    int64_t const b = 4;
    int64_t const m = 16;
    int64_t const n = 16;
    int64_t const k = 512;

    // Initialize input tensors
    Surface<half> A_gpu(b * m * k, false);
    // int4 tensor, but use int8 for memory allocation
    Surface<int8_t> Bq_gpu(k * n / 2, false);
    // scale tensor, 128 block size
    Surface<half> S_gpu(k / kBlockSize, false);

    // Make cudnn graph
    fe::graph::Graph graph{};

    // Create the two non-virtual input tensors A and B.
    // There are read from global memory.
    auto A_attributes = fe::graph::Tensor_attributes()
                            .set_name("A")
                            .set_dim({b, m, k})
                            .set_stride({m * k, k, 1})
                            .set_data_type(fe::DataType_t::HALF);
    auto A             = graph.tensor(A_attributes);
    auto Bq_attributes = fe::graph::Tensor_attributes()
                             .set_name("Bq")
                             .set_dim({1, k, n})
                             .set_stride({k * n, 1, k})
                             .set_data_type(fe::DataType_t::INT4);
    auto Bq           = graph.tensor(Bq_attributes);
    auto S_attributes = fe::graph::Tensor_attributes()
                            .set_name("S")
                            .set_dim({1, k / kBlockSize, n})
                            .set_stride({k * n / kBlockSize, 1, k / kBlockSize})
                            .set_data_type(fe::DataType_t::HALF);
    auto S = graph.tensor(S_attributes);

    // Dequantize Bq to half.
    auto B = graph.block_scale_dequantize(Bq,
                                          S,
                                          fe::graph::Block_scale_dequantize_attributes()
                                              .set_block_size(kBlockSize)
                                              .set_compute_data_type(fe::DataType_t::FLOAT));
    B->set_data_type(fe::DataType_t::HALF);

    auto matmul_attributes =
        fe::graph::Matmul_attributes().set_name("GEMM").set_compute_data_type(fe::DataType_t::FLOAT);
    auto C = graph.matmul(A, B, matmul_attributes);
    C->set_output(true).set_data_type(fe::DataType_t::HALF);

    // Create a unique_ptr for the cuDNN handle
    auto handle_ptr = create_cudnn_handle();
    auto handle     = *handle_ptr;

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());

    REQUIRE(graph.check_support(handle).is_good());

    REQUIRE(graph.build_plans(handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE).is_good());

    // Run cudnn graph
    Surface<half> C_gpu(b * m * n, false);
    int64_t workspace_size;
    REQUIRE(graph.get_workspace_size(workspace_size).is_good());
    Surface<int8_t> workspace(workspace_size, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {A, A_gpu.devPtr}, {Bq, Bq_gpu.devPtr}, {S, S_gpu.devPtr}, {C, C_gpu.devPtr}};

    std::cout << graph.print() << std::endl;
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());
}

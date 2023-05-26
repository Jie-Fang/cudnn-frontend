/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
#include "../helpers.h"

#include <cudnn_frontend.h>

TEST_CASE("Matmul SBR Graph", "[matmul][graph]") {

    cudnn_frontend::graph::Graph graph("matmul_sbr");
    graph.set_io_data_type(cudnn_frontend::DataType_t::HALF)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
    
    auto matmul = cudnn_frontend::graph::Matmul("matmul")
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Matmul::PORTS::A, "image"}
                        , {cudnn_frontend::graph::Matmul::PORTS::B, "filter"}
                        , {cudnn_frontend::graph::Matmul::PORTS::C, "response"}
                    });
    graph.insert_node(matmul);

    auto pw_scale = cudnn_frontend::graph::Pointwise("pw_scale")
                    .set_mode(cudnn_frontend::PointwiseMode_t::MUL)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, matmul.get_tensor_at_port(cudnn_frontend::graph::Matmul::PORTS::C)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "scale"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "scale_output"}
                    });
    graph.insert_node(pw_scale);

    auto pw_bias = cudnn_frontend::graph::Pointwise("pw_bias")
                    .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, pw_scale.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "bias"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "bias_output"}
                    });
    graph.insert_node(pw_bias);

    auto pw_relu = cudnn_frontend::graph::Pointwise("pw_relu")
                    .set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, pw_bias.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "output"}
                    });
    graph.insert_node(pw_relu);
    
    graph.insert_tensor(cudnn_frontend::graph::Tensor("image").set_dim({4, 16, 64}));
    graph.insert_tensor(cudnn_frontend::graph::Tensor("filter").set_dim({4, 64, 32}));
    graph.insert_tensor(cudnn_frontend::graph::Tensor("response").set_is_virtual(true));
    graph.insert_tensor(cudnn_frontend::graph::Tensor("scale").set_dim({4, 16, 32}));
    graph.insert_tensor(cudnn_frontend::graph::Tensor("scale_output").set_is_virtual(true));
    graph.insert_tensor(cudnn_frontend::graph::Tensor("bias").set_dim({4, 16, 32}));
    graph.insert_tensor(cudnn_frontend::graph::Tensor("bias_output").set_is_virtual(true));
    graph.insert_tensor(cudnn_frontend::graph::Tensor("output"));

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    #if (CUDNN_VERSION >= 8500)
        REQUIRE(cudnn_frontend::error_t::OK == graph.build(handle));
    #else
        SKIP("Cudnn 8.4.1 and below did not support matmul epilogue fusion with Column Major layout");
    #endif

    auto plans = graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_A)
                    .build_plans(handle);

    REQUIRE(cudnn_frontend::error_t::OK == graph.set_executor(plans));

    Surface<half> x_tensor(4*16*64, false);
    Surface<half> w_tensor(4*64*32, false);
    Surface<half> s_tensor(4*16*32, false);
    Surface<half> b_tensor(4*16*32, false);
    Surface<half> y_tensor(4*16*32, false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"image", x_tensor.devPtr}
        , {"filter", w_tensor.devPtr}
        , {"scale", s_tensor.devPtr}
        , {"bias", b_tensor.devPtr}
        , {"output", y_tensor.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == graph.execute(handle, variant_pack));
    cudnnDestroy(handle);
}
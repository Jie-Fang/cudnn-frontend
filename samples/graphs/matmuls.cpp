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

#include "matmuls.h"

void test_matmul_relu_graph() {
    cudnn_frontend::graph::Graph graph("matmul_sbr");
    
    auto matmul = cudnn_frontend::graph::Matmul("matmul")
                    .set_compute_type(cudnn_frontend::DataType_t::FLOAT)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Matmul::PORTS::X, "image"}
                        , {cudnn_frontend::graph::Matmul::PORTS::W, "filter"}
                        , {cudnn_frontend::graph::Matmul::PORTS::Y, "response"}
                    });
    graph.insert_node(matmul);

    auto image = cudnn_frontend::graph::Tensor("image").set_dim({4, 16, 64}).set_data_type(cudnn_frontend::DataType_t::HALF);
    graph.insert_tensor(image);

    auto filter = cudnn_frontend::graph::Tensor("filter").set_dim({4, 64, 32}).set_data_type(cudnn_frontend::DataType_t::HALF);
    graph.insert_tensor(filter);
    
    auto response = cudnn_frontend::graph::Tensor("response").set_is_virtual(true).set_data_type(cudnn_frontend::DataType_t::FLOAT);
    graph.insert_tensor(response);

    auto pw_relu = cudnn_frontend::graph::Pointwise("pw_relu")
                    .set_compute_type(cudnn_frontend::DataType_t::FLOAT)
                    .set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::X, matmul.get_tensor_at_port(cudnn_frontend::graph::Matmul::PORTS::Y)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::Y, "output"}
                    });
    graph.insert_node(pw_relu);
    
    auto output = cudnn_frontend::graph::Tensor("output").set_data_type(cudnn_frontend::DataType_t::HALF);
    graph.insert_tensor(output);

    REQUIRE(cudnn_frontend::error_t::OK == graph.build());

    Surface<half> x_tensor(image.get_tensor_size(), false);
    Surface<half> w_tensor(filter.get_tensor_size(), false);
    Surface<half> y_tensor(output.get_tensor_size(), false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"image", x_tensor.devPtr}
        , {"filter", w_tensor.devPtr}
        , {"output", y_tensor.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == graph.execute(variant_pack));
}

void test_matmul_scale_bias_relu_graph() {

    cudnn_frontend::graph::Graph graph("matmul_sbr");
    
    auto matmul = cudnn_frontend::graph::Matmul("matmul")
                    .set_compute_type(cudnn_frontend::DataType_t::FLOAT)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Matmul::PORTS::X, "image"}
                        , {cudnn_frontend::graph::Matmul::PORTS::W, "filter"}
                        , {cudnn_frontend::graph::Matmul::PORTS::Y, "response"}
                    });
    graph.insert_node(matmul);

    auto image = cudnn_frontend::graph::Tensor("image").set_dim({4, 16, 64}).set_data_type(cudnn_frontend::DataType_t::HALF);
    graph.insert_tensor(image);

    auto filter = cudnn_frontend::graph::Tensor("filter").set_dim({4, 64, 32}).set_data_type(cudnn_frontend::DataType_t::HALF);
    graph.insert_tensor(filter);
    
    auto response = cudnn_frontend::graph::Tensor("response").set_is_virtual(true).set_data_type(cudnn_frontend::DataType_t::FLOAT);
    graph.insert_tensor(response);

    auto pw_scale = cudnn_frontend::graph::Pointwise("pw_scale")
                    .set_compute_type(cudnn_frontend::DataType_t::FLOAT)
                    .set_mode(cudnn_frontend::PointwiseMode_t::MUL)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::X, matmul.get_tensor_at_port(cudnn_frontend::graph::Matmul::PORTS::Y)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::B, "scale"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::Y, "scale_output"}
                    });
    graph.insert_node(pw_scale);

    auto scale = cudnn_frontend::graph::Tensor("scale").set_dim({4, 16, 32}).set_data_type(cudnn_frontend::DataType_t::HALF);
    graph.insert_tensor(scale);

    auto scale_output = cudnn_frontend::graph::Tensor("scale_output").set_is_virtual(true).set_data_type(cudnn_frontend::DataType_t::FLOAT);
    graph.insert_tensor(scale_output);

    auto pw_bias = cudnn_frontend::graph::Pointwise("pw_bias")
                    .set_compute_type(cudnn_frontend::DataType_t::FLOAT)
                    .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::X, pw_scale.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::Y)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::B, "bias"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::Y, "bias_output"}
                    });
    graph.insert_node(pw_bias);

    auto bias = cudnn_frontend::graph::Tensor("bias").set_dim({4, 16, 32}).set_data_type(cudnn_frontend::DataType_t::HALF);
    graph.insert_tensor(bias);
    
    auto bias_output = cudnn_frontend::graph::Tensor("bias_output").set_is_virtual(true).set_data_type(cudnn_frontend::DataType_t::FLOAT);
    graph.insert_tensor(bias_output);

    auto pw_relu = cudnn_frontend::graph::Pointwise("pw_relu")
                    .set_compute_type(cudnn_frontend::DataType_t::FLOAT)
                    .set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::X, pw_bias.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::Y)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::Y, "output"}
                    });
    graph.insert_node(pw_relu);
    
    auto output = cudnn_frontend::graph::Tensor("output").set_data_type(cudnn_frontend::DataType_t::HALF);
    graph.insert_tensor(output);

    #if (CUDNN_VERSION >= 8500)
        REQUIRE(cudnn_frontend::error_t::OK == graph.build());
    #else
        SKIP("Cudnn 8.4.1 and below did not support matmul epilogue fusion with Column Major layout");
    #endif
    Surface<half> x_tensor(image.get_tensor_size(), false);
    Surface<half> w_tensor(filter.get_tensor_size(), false);
    Surface<half> s_tensor(scale.get_tensor_size(), false);
    Surface<half> b_tensor(bias.get_tensor_size(), false);
    Surface<half> y_tensor(output.get_tensor_size(), false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"image", x_tensor.devPtr}
        , {"filter", w_tensor.devPtr}
        , {"scale", s_tensor.devPtr}
        , {"bias", b_tensor.devPtr}
        , {"output", y_tensor.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == graph.execute(variant_pack));
}
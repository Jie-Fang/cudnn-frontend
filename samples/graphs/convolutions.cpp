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

#include "convolutions.h"

void test_convolution_scale_bias_relu_graph() {
    cudnn_frontend::cuDNNFEContext context;
    cudnn_frontend::graph::Graph graph("conv_sbr", context);
    
    auto conv = std::make_shared<cudnn_frontend::graph::convolution>("conv");
    conv->set_padding({1, 1});
    conv->set_stride({1, 1});
    conv->set_dilation({1, 1});
    conv->set_tensor_data_type(cudnn_frontend::DataType_t::HALF);
    conv->set_compute_type(cudnn_frontend::DataType_t::FLOAT);

    conv->map_port_to_tensor({
        {cudnn_frontend::graph::convolution::PORTS::X, "image"}
        , {cudnn_frontend::graph::convolution::PORTS::W, "filter"}
        , {cudnn_frontend::graph::convolution::PORTS::Y, "response"}
    });
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_node(conv));

    auto image = std::make_shared<cudnn_frontend::graph::Tensor>("image");
    image->set_dim({4, 32, 16, 16});
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(image));

    auto filter = std::make_shared<cudnn_frontend::graph::Tensor>("filter");
    filter->set_dim({64, 32, 3, 3});
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(filter));
    
    auto response = std::make_shared<cudnn_frontend::graph::Tensor>("response");
    response->set_is_virtual(true);
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(response));

    auto pw_scale = std::make_shared<cudnn_frontend::graph::pointwise>("pw_scale");
    pw_scale->set_tensor_data_type(cudnn_frontend::DataType_t::HALF);
    pw_scale->set_compute_type(cudnn_frontend::DataType_t::FLOAT);
    pw_scale->set_mode(cudnn_frontend::PointwiseMode_t::MUL);
    pw_scale->map_port_to_tensor({
        {cudnn_frontend::graph::pointwise::PORTS::X, conv->get_port_name(cudnn_frontend::graph::convolution::PORTS::Y)}
        , {cudnn_frontend::graph::pointwise::PORTS::B, "scale"}
        , {cudnn_frontend::graph::pointwise::PORTS::Y, "scale_output"}
    });
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_node(pw_scale));

    auto scale = std::make_shared<cudnn_frontend::graph::Tensor>("scale");
    scale->set_dim({1, 64, 1, 1});
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(scale));

    auto scale_output = std::make_shared<cudnn_frontend::graph::Tensor>("scale_output");
    scale_output->set_is_virtual(true);
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(scale_output));

    auto pw_bias = std::make_shared<cudnn_frontend::graph::pointwise>("pw_bias");
    pw_bias->set_tensor_data_type(cudnn_frontend::DataType_t::HALF);
    pw_bias->set_compute_type(cudnn_frontend::DataType_t::FLOAT);
    pw_bias->set_mode(cudnn_frontend::PointwiseMode_t::ADD);
    pw_bias->map_port_to_tensor({
        {cudnn_frontend::graph::pointwise::PORTS::X, pw_scale->get_port_name(cudnn_frontend::graph::pointwise::PORTS::Y)}
        , {cudnn_frontend::graph::pointwise::PORTS::B, "bias"}
        , {cudnn_frontend::graph::pointwise::PORTS::Y, "bias_output"}
    });
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_node(pw_bias));

    auto bias = std::make_shared<cudnn_frontend::graph::Tensor>("bias");
    bias->set_dim({1, 64, 1, 1});
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(bias));
    
    auto bias_output = std::make_shared<cudnn_frontend::graph::Tensor>("bias_output");
    bias_output->set_is_virtual(true);
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(bias_output));

    auto pw_relu = std::make_shared<cudnn_frontend::graph::pointwise>("pw_relu");
    pw_relu->set_tensor_data_type(cudnn_frontend::DataType_t::HALF);
    pw_relu->set_compute_type(cudnn_frontend::DataType_t::FLOAT);
    pw_relu->set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD);
    pw_relu->map_port_to_tensor({
        {cudnn_frontend::graph::pointwise::PORTS::X, pw_bias->get_port_name(cudnn_frontend::graph::pointwise::PORTS::Y)}
        , {cudnn_frontend::graph::pointwise::PORTS::Y, "output"}
    });
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_node(pw_relu));
    
    auto output = std::make_shared<cudnn_frontend::graph::Tensor>("output");
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(output));

    REQUIRE(cudnn_frontend::error_t::OK == graph.build());

    Surface<half> x_tensor(image->get_tensor_size(), false);
    Surface<half> w_tensor(filter->get_tensor_size(), false);
    Surface<half> s_tensor(scale->get_tensor_size(), false);
    Surface<half> b_tensor(bias->get_tensor_size(), false);
    Surface<half> y_tensor(output->get_tensor_size(), false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"image", x_tensor.devPtr}
        , {"filter", w_tensor.devPtr}
        , {"scale", s_tensor.devPtr}
        , {"bias", b_tensor.devPtr}
        , {"output", y_tensor.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == graph.execute(variant_pack));
}
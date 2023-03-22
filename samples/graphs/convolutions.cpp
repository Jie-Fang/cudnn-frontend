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
    cudnn_frontend::Graph graph("conv_sbr", context);
    
    auto conv_props = std::make_shared<cudnn_frontend::convolution_properties>("conv");
    conv_props->set_padding({1, 1});
    conv_props->set_stride({1, 1});
    conv_props->set_dilation({1, 1});
    conv_props->set_tensor_data_type(CUDNN_DATA_HALF);
    conv_props->set_compute_type(CUDNN_DATA_FLOAT);

    conv_props->map_port_to_tensor({
        {cudnn_frontend::convolution_properties::PORTS::X, "image"}
        , {cudnn_frontend::convolution_properties::PORTS::W, "filter"}
        , {cudnn_frontend::convolution_properties::PORTS::Y, "response"}
    });
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_node(conv_props));

    auto image_props = std::make_shared<cudnn_frontend::tensor_properties>("image");
    image_props->set_dim({4, 32, 16, 16});
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(image_props));

    auto filter_props = std::make_shared<cudnn_frontend::tensor_properties>("filter");
    filter_props->set_dim({64, 32, 3, 3});
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(filter_props));
    
    auto response_props = std::make_shared<cudnn_frontend::tensor_properties>("response");
    response_props->set_is_virtual(true);
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(response_props));

    auto pw_scale_props = std::make_shared<cudnn_frontend::pointwise_properties>("pw_scale");
    pw_scale_props->set_tensor_data_type(CUDNN_DATA_HALF);
    pw_scale_props->set_compute_type(CUDNN_DATA_FLOAT);
    pw_scale_props->set_mode(cudnn_frontend::PointwiseMode_t::MUL);
    pw_scale_props->map_port_to_tensor({
        {cudnn_frontend::pointwise_properties::PORTS::X, conv_props->get_port_name(cudnn_frontend::convolution_properties::PORTS::Y)}
        , {cudnn_frontend::pointwise_properties::PORTS::B, "scale"}
        , {cudnn_frontend::pointwise_properties::PORTS::Y, "scale_output"}
    });
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_node(pw_scale_props));

    auto scale_props = std::make_shared<cudnn_frontend::tensor_properties>("scale");
    scale_props->set_dim({1, 64, 1, 1});
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(scale_props));

    auto scale_output = std::make_shared<cudnn_frontend::tensor_properties>("scale_output");
    scale_output->set_is_virtual(true);
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(scale_output));

    auto pw_bias_props = std::make_shared<cudnn_frontend::pointwise_properties>("pw_bias");
    pw_bias_props->set_tensor_data_type(CUDNN_DATA_HALF);
    pw_bias_props->set_compute_type(CUDNN_DATA_FLOAT);
    pw_bias_props->set_mode(cudnn_frontend::PointwiseMode_t::ADD);
    pw_bias_props->map_port_to_tensor({
        {cudnn_frontend::pointwise_properties::PORTS::X, pw_scale_props->get_port_name(cudnn_frontend::pointwise_properties::PORTS::Y)}
        , {cudnn_frontend::pointwise_properties::PORTS::B, "bias"}
        , {cudnn_frontend::pointwise_properties::PORTS::Y, "bias_output"}
    });
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_node(pw_bias_props));

    auto bias_props = std::make_shared<cudnn_frontend::tensor_properties>("bias");
    bias_props->set_dim({1, 64, 1, 1});
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(bias_props));
    
    auto bias_output_props = std::make_shared<cudnn_frontend::tensor_properties>("bias_output");
    bias_output_props->set_is_virtual(true);
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(bias_output_props));

    auto pw_relu_props = std::make_shared<cudnn_frontend::pointwise_properties>("pw_relu");
    pw_relu_props->set_tensor_data_type(CUDNN_DATA_HALF);
    pw_relu_props->set_compute_type(CUDNN_DATA_FLOAT);
    pw_relu_props->set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD);
    pw_relu_props->map_port_to_tensor({
        {cudnn_frontend::pointwise_properties::PORTS::X, pw_bias_props->get_port_name(cudnn_frontend::pointwise_properties::PORTS::Y)}
        , {cudnn_frontend::pointwise_properties::PORTS::Y, "output"}
    });
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_node(pw_relu_props));
    
    auto output_props = std::make_shared<cudnn_frontend::tensor_properties>("output");
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(output_props));

    REQUIRE(cudnn_frontend::error_t::OK == graph.build());

    Surface<half> x_tensor(image_props->get_tensor_size(), false);
    Surface<half> w_tensor(filter_props->get_tensor_size(), false);
    Surface<half> s_tensor(scale_props->get_tensor_size(), false);
    Surface<half> b_tensor(bias_props->get_tensor_size(), false);
    Surface<half> y_tensor(output_props->get_tensor_size(), false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"image", x_tensor.devPtr}
        , {"filter", w_tensor.devPtr}
        , {"scale", s_tensor.devPtr}
        , {"bias", b_tensor.devPtr}
        , {"output", y_tensor.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == graph.execute(variant_pack));
}
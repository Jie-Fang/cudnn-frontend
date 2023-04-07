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
        {cudnn_frontend::graph::pointwise::PORTS::X, conv->get_tensor_at_port(cudnn_frontend::graph::convolution::PORTS::Y)}
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
        {cudnn_frontend::graph::pointwise::PORTS::X, pw_scale->get_tensor_at_port(cudnn_frontend::graph::pointwise::PORTS::Y)}
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
        {cudnn_frontend::graph::pointwise::PORTS::X, pw_bias->get_tensor_at_port(cudnn_frontend::graph::pointwise::PORTS::Y)}
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


void test_convolution_batchnorm_infernece_graph() {
    cudnn_frontend::cuDNNFEContext context;
    cudnn_frontend::graph::Graph graph("conv_bn_inference", context);
    
    //////////////////////////////////////////////////////////////////////////////
    // response = conv(image, filter)
    //////////////////////////////////////////////////////////////////////////////
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

    //////////////////////////////////////////////////////////////////////////////
    // mean_sub_output = response - mean
    //////////////////////////////////////////////////////////////////////////////

    auto mean_sub = std::make_shared<cudnn_frontend::graph::pointwise>("mean_sub");
    mean_sub->set_tensor_data_type(cudnn_frontend::DataType_t::HALF);
    mean_sub->set_compute_type(cudnn_frontend::DataType_t::FLOAT);
    mean_sub->set_mode(cudnn_frontend::PointwiseMode_t::SUB);
    mean_sub->map_port_to_tensor({
        {cudnn_frontend::graph::pointwise::PORTS::X, conv->get_tensor_at_port(cudnn_frontend::graph::convolution::PORTS::Y)}
        , {cudnn_frontend::graph::pointwise::PORTS::B, "mean"}
        , {cudnn_frontend::graph::pointwise::PORTS::Y, "mean_sub_output"}
    });
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_node(mean_sub));

    auto mean = std::make_shared<cudnn_frontend::graph::Tensor>("mean");
    mean->set_dim({1, 64, 1, 1});
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(mean));

    auto mean_sub_output = std::make_shared<cudnn_frontend::graph::Tensor>("mean_sub_output");
    mean_sub_output->set_is_virtual(true);
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(mean_sub_output));

    //////////////////////////////////////////////////////////////////////////////
    // var_with_epsilon = var + epsilon
    //////////////////////////////////////////////////////////////////////////////
    auto epsilon_add = std::make_shared<cudnn_frontend::graph::pointwise>("epsilon_add");
    epsilon_add->set_tensor_data_type(cudnn_frontend::DataType_t::HALF);
    epsilon_add->set_compute_type(cudnn_frontend::DataType_t::FLOAT);
    epsilon_add->set_mode(cudnn_frontend::PointwiseMode_t::ADD);
    epsilon_add->map_port_to_tensor({
        {cudnn_frontend::graph::pointwise::PORTS::X, "var"}
        , {cudnn_frontend::graph::pointwise::PORTS::B, "epsilon"}
        , {cudnn_frontend::graph::pointwise::PORTS::Y, "var_with_epsilon"}
    });
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_node(epsilon_add));

    auto var = std::make_shared<cudnn_frontend::graph::Tensor>("var");
    var->set_dim({1, 64, 1, 1});
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(var));
    
    auto epsilon = std::make_shared<cudnn_frontend::graph::Tensor>("epsilon");
    epsilon->set_dim({1, 1, 1, 1});
    epsilon->set_is_pass_by_value(true);
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(epsilon));

    auto var_with_epsilon = std::make_shared<cudnn_frontend::graph::Tensor>("var_with_epsilon");
    var_with_epsilon->set_is_virtual(true);
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(var_with_epsilon));

    //////////////////////////////////////////////////////////////////////////////
    // inv_var = rsqrt(var_with_epsilon)
    //////////////////////////////////////////////////////////////////////////////
    auto rsqrt_var = std::make_shared<cudnn_frontend::graph::pointwise>("rsqrt_var");
    rsqrt_var->set_tensor_data_type(cudnn_frontend::DataType_t::HALF);
    rsqrt_var->set_compute_type(cudnn_frontend::DataType_t::FLOAT);
    rsqrt_var->set_mode(cudnn_frontend::PointwiseMode_t::RSQRT);
    rsqrt_var->map_port_to_tensor({
        {cudnn_frontend::graph::pointwise::PORTS::X, epsilon_add->get_tensor_at_port(cudnn_frontend::graph::pointwise::PORTS::Y)}
        , {cudnn_frontend::graph::pointwise::PORTS::Y, "inv_var"}
    });
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_node(rsqrt_var));

    auto inv_var = std::make_shared<cudnn_frontend::graph::Tensor>("inv_var");
    inv_var->set_is_virtual(true);
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(inv_var));

    //////////////////////////////////////////////////////////////////////////////
    // norm_input = mean_sub_output * inv_var
    //////////////////////////////////////////////////////////////////////////////
    auto inv_var_mul = std::make_shared<cudnn_frontend::graph::pointwise>("inv_var_mul");
    inv_var_mul->set_tensor_data_type(cudnn_frontend::DataType_t::HALF);
    inv_var_mul->set_compute_type(cudnn_frontend::DataType_t::FLOAT);
    inv_var_mul->set_mode(cudnn_frontend::PointwiseMode_t::MUL);
    inv_var_mul->map_port_to_tensor({
        {cudnn_frontend::graph::pointwise::PORTS::X, mean_sub->get_tensor_at_port(cudnn_frontend::graph::pointwise::PORTS::Y)}
        , {cudnn_frontend::graph::pointwise::PORTS::B, rsqrt_var->get_tensor_at_port(cudnn_frontend::graph::pointwise::PORTS::Y)}
        , {cudnn_frontend::graph::pointwise::PORTS::Y, "norm_input"}
    });
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_node(inv_var_mul));

    auto norm_input = std::make_shared<cudnn_frontend::graph::Tensor>("norm_input");
    norm_input->set_is_virtual(true);
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(norm_input));

    //////////////////////////////////////////////////////////////////////////////
    // scale_output = norm_input * scale
    //////////////////////////////////////////////////////////////////////////////
    auto scale_mul = std::make_shared<cudnn_frontend::graph::pointwise>("scale_mul");
    scale_mul->set_tensor_data_type(cudnn_frontend::DataType_t::HALF);
    scale_mul->set_compute_type(cudnn_frontend::DataType_t::FLOAT);
    scale_mul->set_mode(cudnn_frontend::PointwiseMode_t::MUL);
    scale_mul->map_port_to_tensor({
        {cudnn_frontend::graph::pointwise::PORTS::X, inv_var_mul->get_tensor_at_port(cudnn_frontend::graph::pointwise::PORTS::Y)}
        , {cudnn_frontend::graph::pointwise::PORTS::B, "scale"}
        , {cudnn_frontend::graph::pointwise::PORTS::Y, "scale_output"}
    });
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_node(scale_mul));

    auto scale = std::make_shared<cudnn_frontend::graph::Tensor>("scale");
    scale->set_dim({1, 64, 1, 1});
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(scale));

    auto scale_output = std::make_shared<cudnn_frontend::graph::Tensor>("scale_output");
    scale_output->set_is_virtual(true);
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(scale_output));

    //////////////////////////////////////////////////////////////////////////////
    // bias_output = scale_output + bias
    //////////////////////////////////////////////////////////////////////////////
    auto bias_add = std::make_shared<cudnn_frontend::graph::pointwise>("bias_add");
    bias_add->set_tensor_data_type(cudnn_frontend::DataType_t::HALF);
    bias_add->set_compute_type(cudnn_frontend::DataType_t::FLOAT);
    bias_add->set_mode(cudnn_frontend::PointwiseMode_t::ADD);
    bias_add->map_port_to_tensor({
        {cudnn_frontend::graph::pointwise::PORTS::X, scale_mul->get_tensor_at_port(cudnn_frontend::graph::pointwise::PORTS::Y)}
        , {cudnn_frontend::graph::pointwise::PORTS::B, "bias"}
        , {cudnn_frontend::graph::pointwise::PORTS::Y, "bias_output"}
    });
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_node(bias_add));

    auto bias = std::make_shared<cudnn_frontend::graph::Tensor>("bias");
    bias->set_dim({1, 64, 1, 1});
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(bias));

    auto bias_output = std::make_shared<cudnn_frontend::graph::Tensor>("bias_output");
    REQUIRE(cudnn_frontend::error_t::OK == graph.insert_tensor(bias_output));

    #if (CUDNN_VERSION >= 8500)
        REQUIRE(cudnn_frontend::error_t::OK == graph.build());
    #elif (CUDNN_VERSION >= 8300)
        SKIP("Passing tensors by value, here epsilon, is not supported prior to 8500.");
    #else
        SKIP("RSQRT pointwise mode does not exist prior to 8300.");
    #endif


    Surface<half> x_tensor(image->get_tensor_size(), false);
    Surface<half> w_tensor(filter->get_tensor_size(), false);
    Surface<float> m_tensor(mean->get_tensor_size(), false);
    Surface<float> v_tensor(var->get_tensor_size(), false);
    float e_tensor;
    Surface<float> s_tensor(scale->get_tensor_size(), false);
    Surface<float> b_tensor(bias->get_tensor_size(), false);
    Surface<half> y_tensor(bias_output->get_tensor_size(), false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"image", x_tensor.devPtr}
        , {"filter", w_tensor.devPtr}
        , {"mean", m_tensor.devPtr}
        , {"var", v_tensor.devPtr}
        , {"epsilon", &e_tensor}
        , {"scale", s_tensor.devPtr}
        , {"bias", b_tensor.devPtr}
        , {"bias_output", y_tensor.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == graph.execute(variant_pack));
}
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

    cudnn_frontend::graph::Graph graph("conv_sbr");
    graph.set_io_data_type(cudnn_frontend::DataType_t::HALF)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
    
    auto conv = cudnn_frontend::graph::Convolution("conv")
                .set_padding({1, 1})
                .set_stride({1, 1})
                .set_dilation({1, 1})
                .map_port_to_tensor({
                    {cudnn_frontend::graph::Convolution::PORTS::X, "image"}
                    , {cudnn_frontend::graph::Convolution::PORTS::W, "filter"}
                    , {cudnn_frontend::graph::Convolution::PORTS::Y, "response"}
                });
    graph.insert_node(conv);

    auto pw_scale = cudnn_frontend::graph::Pointwise("pw_scale")
                    .set_mode(cudnn_frontend::PointwiseMode_t::MUL)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::X, conv.get_tensor_at_port(cudnn_frontend::graph::Convolution::PORTS::Y)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::B, "scale"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::Y, "scale_output"}
                    });
    graph.insert_node(pw_scale);

    auto pw_bias = cudnn_frontend::graph::Pointwise("pw_bias")
                    .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::X, pw_scale.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::Y)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::B, "bias"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::Y, "bias_output"}
                    });
    graph.insert_node(pw_bias);

    auto pw_relu = cudnn_frontend::graph::Pointwise("pw_relu")
                    .set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::X, pw_bias.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::Y)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::Y, "output"}
                    });
    graph.insert_node(pw_relu);
    
    graph.insert_tensor(cudnn_frontend::graph::Tensor("image").set_dim({4, 32, 16, 16}))
         .insert_tensor(cudnn_frontend::graph::Tensor("filter").set_dim({64, 32, 3, 3}))
         .insert_tensor(cudnn_frontend::graph::Tensor("response").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("scale").set_dim({1, 64, 1, 1}))
         .insert_tensor(cudnn_frontend::graph::Tensor("scale_output").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("bias").set_dim({1, 64, 1, 1}))
         .insert_tensor(cudnn_frontend::graph::Tensor("bias_output").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("output"));

    REQUIRE(cudnn_frontend::error_t::OK == graph.build());

    Surface<half> x_tensor(4*32*16*16, false);
    Surface<half> w_tensor(64*32*3*3, false);
    Surface<half> s_tensor(64, false);
    Surface<half> b_tensor(64, false);
    Surface<half> y_tensor(4*64*3*3, false);

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
    cudnn_frontend::graph::Graph graph("conv_bn_inference");
    graph.set_io_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
    
    //////////////////////////////////////////////////////////////////////////////
    // response = conv(image, filter)
    //////////////////////////////////////////////////////////////////////////////
    auto conv = cudnn_frontend::graph::Convolution("conv")
                .set_padding({1, 1})
                .set_stride({1, 1})
                .set_dilation({1, 1})
                .map_port_to_tensor({
                    {cudnn_frontend::graph::Convolution::PORTS::X, "image"}
                    , {cudnn_frontend::graph::Convolution::PORTS::W, "filter"}
                    , {cudnn_frontend::graph::Convolution::PORTS::Y, "response"}
                });
    graph.insert_node(conv);

    //////////////////////////////////////////////////////////////////////////////
    // mean_sub_output = response - mean
    //////////////////////////////////////////////////////////////////////////////

    auto mean_sub = cudnn_frontend::graph::Pointwise("mean_sub")
                    .set_mode(cudnn_frontend::PointwiseMode_t::SUB)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::X, conv.get_tensor_at_port(cudnn_frontend::graph::Convolution::PORTS::Y)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::B, "mean"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::Y, "mean_sub_output"}
                    });
    graph.insert_node(mean_sub);

    //////////////////////////////////////////////////////////////////////////////
    // var_with_epsilon = var + epsilon
    //////////////////////////////////////////////////////////////////////////////
    auto epsilon_add = cudnn_frontend::graph::Pointwise("epsilon_add")
                        .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
                        .map_port_to_tensor({
                            {cudnn_frontend::graph::Pointwise::PORTS::X, "var"}
                            , {cudnn_frontend::graph::Pointwise::PORTS::B, "epsilon"}
                            , {cudnn_frontend::graph::Pointwise::PORTS::Y, "var_with_epsilon"}
                        });
    graph.insert_node(epsilon_add);

    //////////////////////////////////////////////////////////////////////////////
    // inv_var = rsqrt(var_with_epsilon)
    //////////////////////////////////////////////////////////////////////////////
    auto rsqrt_var = cudnn_frontend::graph::Pointwise("rsqrt_var")
                        .set_mode(cudnn_frontend::PointwiseMode_t::RSQRT)
                        .map_port_to_tensor({
                            {cudnn_frontend::graph::Pointwise::PORTS::X, epsilon_add.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::Y)}
                            , {cudnn_frontend::graph::Pointwise::PORTS::Y, "inv_var"}
                        });
    graph.insert_node(rsqrt_var);

    //////////////////////////////////////////////////////////////////////////////
    // norm_input = mean_sub_output * inv_var
    //////////////////////////////////////////////////////////////////////////////
    auto inv_var_mul = cudnn_frontend::graph::Pointwise("inv_var_mul")
                        .set_mode(cudnn_frontend::PointwiseMode_t::MUL)
                        .map_port_to_tensor({
                            {cudnn_frontend::graph::Pointwise::PORTS::X, mean_sub.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::Y)}
                            , {cudnn_frontend::graph::Pointwise::PORTS::B, rsqrt_var.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::Y)}
                            , {cudnn_frontend::graph::Pointwise::PORTS::Y, "norm_input"}
                        });
    graph.insert_node(inv_var_mul);


    //////////////////////////////////////////////////////////////////////////////
    // scale_output = norm_input * scale
    //////////////////////////////////////////////////////////////////////////////
    auto scale_mul = cudnn_frontend::graph::Pointwise("scale_mul")
                        .set_mode(cudnn_frontend::PointwiseMode_t::MUL)
                        .map_port_to_tensor({
                            {cudnn_frontend::graph::Pointwise::PORTS::X, inv_var_mul.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::Y)}
                            , {cudnn_frontend::graph::Pointwise::PORTS::B, "scale"}
                            , {cudnn_frontend::graph::Pointwise::PORTS::Y, "scale_output"}
                        });
    graph.insert_node(scale_mul);

    //////////////////////////////////////////////////////////////////////////////
    // bias_output = scale_output + bias
    //////////////////////////////////////////////////////////////////////////////
    auto bias_add = cudnn_frontend::graph::Pointwise("bias_add")
                    .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::X, scale_mul.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::Y)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::B, "bias"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::Y, "bias_output"}
                    });
    graph.insert_node(bias_add);

    graph.insert_tensor(cudnn_frontend::graph::Tensor("image").set_dim({4, 32, 16, 16}).set_data_type(cudnn_frontend::DataType_t::HALF))
         .insert_tensor(cudnn_frontend::graph::Tensor("filter").set_dim({64, 32, 3, 3}).set_data_type(cudnn_frontend::DataType_t::HALF))
         .insert_tensor(cudnn_frontend::graph::Tensor("response").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("mean").set_dim({1, 64, 1, 1}))
         .insert_tensor(cudnn_frontend::graph::Tensor("mean_sub_output").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("var").set_dim({1, 64, 1, 1}))
         .insert_tensor(cudnn_frontend::graph::Tensor("epsilon").set_dim({1, 1, 1, 1}).set_is_pass_by_value(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("var_with_epsilon").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("inv_var").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("norm_input").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("scale").set_dim({1, 64, 1, 1}))
         .insert_tensor(cudnn_frontend::graph::Tensor("scale_output").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("bias").set_dim({1, 64, 1, 1}))
         .insert_tensor(cudnn_frontend::graph::Tensor("bias_output").set_data_type(cudnn_frontend::DataType_t::HALF));

    #if (CUDNN_VERSION >= 8500)
        REQUIRE(cudnn_frontend::error_t::OK == graph.build());
    #elif (CUDNN_VERSION >= 8300)
        SKIP("Passing tensors by value, here epsilon, is not supported prior to 8500.");
    #else
        SKIP("RSQRT pointwise mode does not exist prior to 8300.");
    #endif


    Surface<half> x_tensor(4*32*16*16, false);
    Surface<half> w_tensor(64*32*3*3, false);
    Surface<float> m_tensor(64, false);
    Surface<float> v_tensor(64, false);
    float e_tensor;
    Surface<float> s_tensor(64, false);
    Surface<float> b_tensor(64, false);
    Surface<half> y_tensor(4*64*16*16, false);

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
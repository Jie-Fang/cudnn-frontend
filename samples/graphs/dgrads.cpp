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

TEST_CASE("Dgrad Graph", "[dgrad][graph]") {
    cudnn_frontend::graph::Graph graph("dgrad");
    graph.set_io_data_type(cudnn_frontend::DataType_t::HALF)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
    
    auto dgrad = cudnn_frontend::graph::Dgrad("dgrad")
                .set_padding({1, 1})
                .set_stride({1, 1})
                .set_dilation({1, 1})
                .map_port_to_tensor({
                    {cudnn_frontend::graph::Dgrad::PORTS::DY, "grad"}
                    , {cudnn_frontend::graph::Dgrad::PORTS::W, "input"}
                    , {cudnn_frontend::graph::Dgrad::PORTS::DX, "output"}
                });
    graph.insert_node(dgrad);

    graph.insert_tensor(cudnn_frontend::graph::Tensor("grad").set_dim({4, 64, 16, 16}))
         .insert_tensor(cudnn_frontend::graph::Tensor("input").set_dim({64, 32, 3, 3}))
         .insert_tensor(cudnn_frontend::graph::Tensor("output"));

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    REQUIRE(cudnn_frontend::error_t::OK == graph.build(handle));

    auto plans = graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_A)
                    .build_plans(handle);

    REQUIRE(cudnn_frontend::error_t::OK == graph.set_executor(plans));

    Surface<half> dy_tensor(4*64*16*16, false);
    Surface<half> w_tensor(64*32*3*3, false);
    Surface<half> dx_tensor(4*32*16*16, false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"grad", dy_tensor.devPtr}
        , {"input", w_tensor.devPtr}
        , {"output", dx_tensor.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == graph.execute(handle, variant_pack));
    cudnnDestroy(handle);
}

TEST_CASE("Dgrad Drelu Graph", "[dgrad][graph]") {
    cudnn_frontend::graph::Graph graph("dgrad");
    graph.set_io_data_type(cudnn_frontend::DataType_t::HALF)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
    
    auto dgrad = cudnn_frontend::graph::Dgrad("dgrad")
                .set_padding({1, 1})
                .set_stride({1, 1})
                .set_dilation({1, 1})
                .map_port_to_tensor({
                    {cudnn_frontend::graph::Dgrad::PORTS::DY, "grad"}
                    , {cudnn_frontend::graph::Dgrad::PORTS::W, "weight"}
                    , {cudnn_frontend::graph::Dgrad::PORTS::DX, "dgrad_output"}
                });
    graph.insert_node(dgrad);
    
    auto drelu = cudnn_frontend::graph::Pointwise("drelu")
                .set_mode(cudnn_frontend::PointwiseMode_t::RELU_BWD)
                .map_port_to_tensor({
                    {cudnn_frontend::graph::Pointwise::PORTS::IN_0, dgrad.get_tensor_at_port(cudnn_frontend::graph::Dgrad::PORTS::DX)}
                    , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "input"}
                    , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "output"}
                });
    graph.insert_node(drelu);

    graph.insert_tensor(cudnn_frontend::graph::Tensor("grad").set_dim({4, 64, 16, 16}))
         .insert_tensor(cudnn_frontend::graph::Tensor("weight").set_dim({64, 32, 3, 3}))
         .insert_tensor(cudnn_frontend::graph::Tensor("dgrad_output").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("input").set_dim({4, 32, 16, 16}))
         .insert_tensor(cudnn_frontend::graph::Tensor("output"));

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    REQUIRE(cudnn_frontend::error_t::OK == graph.build(handle));

    auto plans = graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_A)
                    .build_plans(handle);

    REQUIRE(cudnn_frontend::error_t::OK == graph.set_executor(plans));

    Surface<half> dy_tensor(4*64*16*16, false);
    Surface<half> w_tensor(64*32*3*3, false);
    Surface<half> x_tensor(4*32*16*16, false);
    Surface<half> dx_tensor(4*32*16*16, false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"grad", dy_tensor.devPtr}
        , {"weight", w_tensor.devPtr}
        , {"input", x_tensor.devPtr}
        , {"output", dx_tensor.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == graph.execute(handle, variant_pack));
    cudnnDestroy(handle);
}

TEST_CASE("Dgrad Drelu DBNweight Graph", "[dgrad][graph]") {
    cudnn_frontend::graph::Graph graph("dgrad");
    graph.set_io_data_type(cudnn_frontend::DataType_t::HALF)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
    
    auto dgrad = cudnn_frontend::graph::Dgrad("dgrad")
                .set_padding({1, 1})
                .set_stride({1, 1})
                .set_dilation({1, 1})
                .map_port_to_tensor({
                    {cudnn_frontend::graph::Dgrad::PORTS::DY, "DY"}
                    , {cudnn_frontend::graph::Dgrad::PORTS::W, "W"}
                    , {cudnn_frontend::graph::Dgrad::PORTS::DX, "dgrad_output"}
                });
    graph.insert_node(dgrad);
    
    auto pw_mean = cudnn_frontend::graph::Pointwise("pw_mean")
                    .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, "X"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "mean"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "mean_output"}
                    });
    graph.insert_node(pw_mean);

    auto pw_inv_var = cudnn_frontend::graph::Pointwise("pw_inv_var")
                    .set_mode(cudnn_frontend::PointwiseMode_t::MUL)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, pw_mean.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "inv_var"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "inv_var_output"}
                    });
    graph.insert_node(pw_inv_var);
    
    auto pw_scale = cudnn_frontend::graph::Pointwise("pw_scale")
                    .set_mode(cudnn_frontend::PointwiseMode_t::MUL)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, pw_inv_var.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
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

    auto drelu = cudnn_frontend::graph::Pointwise("drelu")
                .set_mode(cudnn_frontend::PointwiseMode_t::RELU_BWD)
                .map_port_to_tensor({
                    {cudnn_frontend::graph::Pointwise::PORTS::IN_0, dgrad.get_tensor_at_port(cudnn_frontend::graph::Dgrad::PORTS::DX)}
                    , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, pw_bias.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                    , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "relu_output"}
                });
    graph.insert_node(drelu);
    
    auto dbn_weight = cudnn_frontend::graph::Batchnorm_backward_weight("dbn_weight")
                .map_port_to_tensor({
                    {cudnn_frontend::graph::Batchnorm_backward_weight::PORTS::DY, drelu.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                    , {cudnn_frontend::graph::Batchnorm_backward_weight::PORTS::X, "X"}
                    , {cudnn_frontend::graph::Batchnorm_backward_weight::PORTS::MEAN, "mean"}
                    , {cudnn_frontend::graph::Batchnorm_backward_weight::PORTS::INV_VARIANCE, "inv_var"}
                    , {cudnn_frontend::graph::Batchnorm_backward_weight::PORTS::SCALE, "scale"}
                    , {cudnn_frontend::graph::Batchnorm_backward_weight::PORTS::DSCALE, "dscale"}
                    , {cudnn_frontend::graph::Batchnorm_backward_weight::PORTS::DBIAS, "dbias"}
                    , {cudnn_frontend::graph::Batchnorm_backward_weight::PORTS::EQUIVALENT_BIAS, "eq_bias"}
                    , {cudnn_frontend::graph::Batchnorm_backward_weight::PORTS::EQUIVALENT_SCALE_DY, "eq_scale_dy"}
                    , {cudnn_frontend::graph::Batchnorm_backward_weight::PORTS::EQUIVALENT_SCALE_X, "eq_scale_x"}
                });
    graph.insert_node(dbn_weight);

    graph.insert_tensor(cudnn_frontend::graph::Tensor("DY").set_dim({4, 64, 16, 16}))
         .insert_tensor(cudnn_frontend::graph::Tensor("W").set_dim({64, 32, 3, 3}))
         .insert_tensor(cudnn_frontend::graph::Tensor("dgrad_output").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("X").set_dim({4, 32, 16, 16}))
         .insert_tensor(cudnn_frontend::graph::Tensor("mean").set_dim({1, 32, 1, 1}).set_data_type(cudnn_frontend::DataType_t::FLOAT))
         .insert_tensor(cudnn_frontend::graph::Tensor("mean_output").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("inv_var").set_dim({1, 32, 1, 1}).set_data_type(cudnn_frontend::DataType_t::FLOAT))
         .insert_tensor(cudnn_frontend::graph::Tensor("inv_var_output").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("scale").set_dim({1, 32, 1, 1}).set_data_type(cudnn_frontend::DataType_t::FLOAT))
         .insert_tensor(cudnn_frontend::graph::Tensor("scale_output").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("bias").set_dim({1, 32, 1, 1}).set_data_type(cudnn_frontend::DataType_t::FLOAT))
         .insert_tensor(cudnn_frontend::graph::Tensor("bias_output").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("relu_output"))
         .insert_tensor(cudnn_frontend::graph::Tensor("dscale").set_data_type(cudnn_frontend::DataType_t::FLOAT))
         .insert_tensor(cudnn_frontend::graph::Tensor("dbias").set_data_type(cudnn_frontend::DataType_t::FLOAT))
         .insert_tensor(cudnn_frontend::graph::Tensor("eq_bias").set_data_type(cudnn_frontend::DataType_t::FLOAT))
         .insert_tensor(cudnn_frontend::graph::Tensor("eq_scale_dy").set_data_type(cudnn_frontend::DataType_t::FLOAT))
         .insert_tensor(cudnn_frontend::graph::Tensor("eq_scale_x").set_data_type(cudnn_frontend::DataType_t::FLOAT));

    #if (CUDNN_VERSION < 8900)
        SKIP("DgradDreluBNBwdWeight requires cudnn 8.9 and up");
    #endif
    if (check_device_arch_newer_than("ampere") == false) {
        SKIP("DgradDreluBNBwdWeight requires ampere and above architecture.");
    }

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    REQUIRE(cudnn_frontend::error_t::OK == graph.build(handle));

    auto plans = graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_A)
                    .build_plans(handle);

    REQUIRE(cudnn_frontend::error_t::OK == graph.set_executor(plans));

    Surface<half> dy_tensor(4*64*16*16, false);
    Surface<half> w_tensor(64*32*3*3, false);
    Surface<half> x_tensor(4*32*16*16, false);
    Surface<half> relu_output_tensor(4*32*16*16, false);
    Surface<float> mean_tensor(1*32*1*1, false);
    Surface<float> inv_var_tensor(1*32*1*1, false);
    Surface<float> scale_tensor(1*32*1*1, false);
    Surface<float> bias_tensor(1*32*1*1, false);
    Surface<float> dscale_tensor(1*32*1*1, false);
    Surface<float> dbias_tensor(1*32*1*1, false);
    Surface<float> eq_scale_dy_tensor(1*32*1*1, false);
    Surface<float> eq_scale_x_tensor(1*32*1*1, false);
    Surface<float> eq_bias_tensor(1*32*1*1, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"DY", dy_tensor.devPtr}
        , {"W", w_tensor.devPtr}
        , {"X", x_tensor.devPtr}
        , {"mean", mean_tensor.devPtr}
        , {"scale", scale_tensor.devPtr}
        , {"inv_var", inv_var_tensor.devPtr}
        , {"bias", bias_tensor.devPtr}
        , {"dbias", dbias_tensor.devPtr}
        , {"dscale", dscale_tensor.devPtr}
        , {"eq_bias", eq_bias_tensor.devPtr}
        , {"eq_scale_dy", eq_scale_dy_tensor.devPtr}
        , {"eq_scale_x", eq_scale_x_tensor.devPtr}
        , {"relu_output", relu_output_tensor.devPtr}
        , {"workspace", workspace.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == graph.execute(handle, variant_pack));
    cudnnDestroy(handle);
}
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

TEST_CASE("BN Finalize Graph", "[batchnorm][graph]") {
    cudnn_frontend::graph::Graph graph("bn_finalize");
    graph.set_io_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
    
    auto bn_finalize = cudnn_frontend::graph::Batchnorm_finalize("bn_finalize")
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Batchnorm_finalize::PORTS::SUM, "SUM"}
                        , {cudnn_frontend::graph::Batchnorm_finalize::PORTS::SQUARE_SUM, "SQUARE_SUM"}
                        , {cudnn_frontend::graph::Batchnorm_finalize::PORTS::MEAN, "MEAN"}
                        , {cudnn_frontend::graph::Batchnorm_finalize::PORTS::INV_VARIANCE, "INV_VARIANCE"}
                        , {cudnn_frontend::graph::Batchnorm_finalize::PORTS::Previous_running_mean, "Previous_running_mean"}
                        , {cudnn_frontend::graph::Batchnorm_finalize::PORTS::Previous_running_var, "Previous_running_var"}
                        , {cudnn_frontend::graph::Batchnorm_finalize::PORTS::Next_running_mean, "Next_running_mean"}
                        , {cudnn_frontend::graph::Batchnorm_finalize::PORTS::Next_running_var, "Next_running_var"}
                        , {cudnn_frontend::graph::Batchnorm_finalize::PORTS::SCALE, "SCALE"}
                        , {cudnn_frontend::graph::Batchnorm_finalize::PORTS::BIAS, "BIAS"}
                        , {cudnn_frontend::graph::Batchnorm_finalize::PORTS::EPSILON, "EPSILON"}
                        , {cudnn_frontend::graph::Batchnorm_finalize::PORTS::EXP_AVG, "EXP_AVG"}
                        , {cudnn_frontend::graph::Batchnorm_finalize::PORTS::ACCUMULATION_COUNT, "ACCUMULATION_COUNT"}
                        , {cudnn_frontend::graph::Batchnorm_finalize::PORTS::EQUIVALENT_BIAS, "EQUIVALENT_BIAS"}
                        , {cudnn_frontend::graph::Batchnorm_finalize::PORTS::EQUIVALENT_SCALE, "EQUIVALENT_SCALE"}
                    });
    graph.insert_node(bn_finalize);

    graph.insert_tensor(cudnn_frontend::graph::Tensor("SUM").set_dim({1, 32, 1, 1}))
         .insert_tensor(cudnn_frontend::graph::Tensor("SQUARE_SUM"))
         .insert_tensor(cudnn_frontend::graph::Tensor("MEAN"))
         .insert_tensor(cudnn_frontend::graph::Tensor("INV_VARIANCE"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Previous_running_mean"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Previous_running_var"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Next_running_mean"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Next_running_var"))
         .insert_tensor(cudnn_frontend::graph::Tensor("SCALE"))
         .insert_tensor(cudnn_frontend::graph::Tensor("BIAS"))
         .insert_tensor(cudnn_frontend::graph::Tensor("EPSILON").set_is_pass_by_value(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("EXP_AVG").set_is_pass_by_value(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("ACCUMULATION_COUNT").set_is_pass_by_value(true).set_data_type(cudnn_frontend::DataType_t::INT64))
         .insert_tensor(cudnn_frontend::graph::Tensor("EQUIVALENT_SCALE"))
         .insert_tensor(cudnn_frontend::graph::Tensor("EQUIVALENT_BIAS"));

    #if (CUDNN_VERSION < 8400)
        SKIP("BNFinalize requires cudnn 8.4 and up");
    #endif

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    REQUIRE(cudnn_frontend::error_t::OK == graph.build(handle));
    auto plans = graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_FALLBACK)
                    .build_plans(handle);

    REQUIRE(cudnn_frontend::error_t::OK == graph.set_executor(plans));

    Surface<float> Sum_tensor(32, false);
    Surface<float> Sq_sum_tensor(32, false);
    Surface<float> Mean_tensor(32, false);
    Surface<float> Var_tensor(32, false);
    Surface<float> Previous_running_mean_tensor(32, false);
    Surface<float> Previous_running_var_tensor(32, false);
    Surface<float> Next_running_mean_tensor(32, false);
    Surface<float> Next_running_var_tensor(32, false);
    Surface<float> Scale_tensor(32, false);
    Surface<float> Bias_tensor(32, false);
    Surface<float> eq_scale_tensor(32, false);
    Surface<float> eq_bias_tensor(32, false);
    float EPS_scalar = 0.001;
    float EXP_AVG_scalar = 0.001;
    int64_t nhw = 64;

    Surface<int8_t> workspace(graph.get_workspace_size(), false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"SUM", Sum_tensor.devPtr}
        , {"SQUARE_SUM", Sq_sum_tensor.devPtr}
        , {"MEAN", Mean_tensor.devPtr}
        , {"INV_VARIANCE", Var_tensor.devPtr}
        , {"Previous_running_mean", Previous_running_mean_tensor.devPtr}
        , {"Previous_running_var", Previous_running_var_tensor.devPtr}
        , {"Next_running_mean", Next_running_mean_tensor.devPtr}
        , {"Next_running_var", Next_running_var_tensor.devPtr}
        , {"SCALE", Scale_tensor.devPtr}
        , {"BIAS", Bias_tensor.devPtr}
        , {"EPSILON", &EPS_scalar}
        , {"EXP_AVG", &EXP_AVG_scalar}
        , {"ACCUMULATION_COUNT", &nhw}
        , {"EQUIVALENT_SCALE", eq_scale_tensor.devPtr}
        , {"EQUIVALENT_BIAS", eq_bias_tensor.devPtr}
        , {"workspace", workspace.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == graph.execute(handle, variant_pack));

    cudnnDestroy(handle);
}

TEST_CASE("SGBN Graph", "[batchnorm][graph]") {
    cudnn_frontend::graph::Graph graph("SGBN");
    graph.set_io_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
    
    auto sgbn = cudnn_frontend::graph::Batchnorm("SGBN")
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Batchnorm::PORTS::X, "X"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Mean, "Mean"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Var, "Var"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Previous_running_mean, "Previous_running_mean"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Previous_running_var, "Previous_running_var"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Next_running_mean, "Next_running_mean"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Next_running_var, "Next_running_var"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Scale, "Scale"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Bias, "Bias"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::EPS, "EPS"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::EXP_AVG, "EXP_AVG"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Y, "Y"}
                    });
    graph.insert_node(sgbn);

    graph.insert_tensor(cudnn_frontend::graph::Tensor("X").set_dim({4, 32, 16, 16}).set_data_type(cudnn_frontend::DataType_t::HALF))
         .insert_tensor(cudnn_frontend::graph::Tensor("Mean"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Var"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Previous_running_mean"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Previous_running_var"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Next_running_mean"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Next_running_var"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Scale"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Bias"))
         .insert_tensor(cudnn_frontend::graph::Tensor("EPS").set_is_pass_by_value(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("EXP_AVG").set_is_pass_by_value(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("Y").set_data_type(cudnn_frontend::DataType_t::HALF));
    
    #if (CUDNN_VERSION < 8700)
        SKIP("single GPU BN is not supported in cudnn versions prior to 8.7");
    #endif

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    REQUIRE(cudnn_frontend::error_t::OK == graph.build(handle));
    auto plans = graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_FALLBACK)
                    .build_plans(handle);

    REQUIRE(cudnn_frontend::error_t::OK == graph.set_executor(plans));

    Surface<half> X_tensor(4*32*16*16, false);
    Surface<float> Mean_tensor(16, false);
    Surface<float> Var_tensor(16, false);
    Surface<float> Previous_running_mean_tensor(16, false);
    Surface<float> Previous_running_var_tensor(16, false);
    Surface<float> Next_running_mean_tensor(16, false);
    Surface<float> Next_running_var_tensor(16, false);
    Surface<float> Scale_tensor(16, false);
    Surface<float> Bias_tensor(16, false);
    float EPS_scalar = 0.001;
    float EXP_AVG_scalar = 0.001;
    Surface<half> Y_tensor(4*32*16*16, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"X", X_tensor.devPtr}
        , {"Mean", Mean_tensor.devPtr}
        , {"Var", Var_tensor.devPtr}
        , {"Previous_running_mean", Previous_running_mean_tensor.devPtr}
        , {"Previous_running_var", Previous_running_var_tensor.devPtr}
        , {"Next_running_mean", Next_running_mean_tensor.devPtr}
        , {"Next_running_var", Next_running_var_tensor.devPtr}
        , {"Scale", Scale_tensor.devPtr}
        , {"Bias", Bias_tensor.devPtr}
        , {"EPS", &EPS_scalar}
        , {"EXP_AVG", &EXP_AVG_scalar}
        , {"Y", Y_tensor.devPtr}
        , {"workspace", workspace.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == graph.execute(handle, variant_pack));

    cudnnDestroy(handle);
}

TEST_CASE("SGBN Relu Graph", "[batchnorm][graph]") {
    cudnn_frontend::graph::Graph graph("SGBN_Relu");
    graph.set_io_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
    
    auto sgbn = cudnn_frontend::graph::Batchnorm("SGBN")
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Batchnorm::PORTS::X, "X"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Mean, "Mean"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Var, "Var"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Previous_running_mean, "Previous_running_mean"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Previous_running_var, "Previous_running_var"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Next_running_mean, "Next_running_mean"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Next_running_var, "Next_running_var"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Scale, "Scale"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Bias, "Bias"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::EPS, "EPS"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::EXP_AVG, "EXP_AVG"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Y, "Y"}
                    });
    graph.insert_node(sgbn);

    auto relu = cudnn_frontend::graph::Pointwise("relu")
                    .set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, sgbn.get_tensor_at_port(cudnn_frontend::graph::Batchnorm::PORTS::Y)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "output"}
                    });
    graph.insert_node(relu);

    graph.insert_tensor(cudnn_frontend::graph::Tensor("X").set_dim({4, 32, 16, 16}).set_data_type(cudnn_frontend::DataType_t::HALF))
         .insert_tensor(cudnn_frontend::graph::Tensor("Mean"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Var"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Previous_running_mean"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Previous_running_var"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Next_running_mean"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Next_running_var"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Scale"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Bias"))
         .insert_tensor(cudnn_frontend::graph::Tensor("EPS").set_is_pass_by_value(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("EXP_AVG").set_is_pass_by_value(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("Y").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("output").set_data_type(cudnn_frontend::DataType_t::HALF));
    
    #if (CUDNN_VERSION < 8700)
        SKIP("single GPU BN is not supported in cudnn versions prior to 8.7");
    #endif

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    REQUIRE(cudnn_frontend::error_t::OK == graph.build(handle));
    auto plans = graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_FALLBACK)
                    .build_plans(handle);

    REQUIRE(cudnn_frontend::error_t::OK == graph.set_executor(plans));

    Surface<half> X_tensor(4*32*16*16, false);
    Surface<float> Mean_tensor(16, false);
    Surface<float> Var_tensor(16, false);
    Surface<float> Previous_running_mean_tensor(16, false);
    Surface<float> Previous_running_var_tensor(16, false);
    Surface<float> Next_running_mean_tensor(16, false);
    Surface<float> Next_running_var_tensor(16, false);
    Surface<float> Scale_tensor(16, false);
    Surface<float> Bias_tensor(16, false);
    float EPS_scalar = 0.001;
    float EXP_AVG_scalar = 0.001;
    Surface<half> Y_tensor(4*32*16*16, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"X", X_tensor.devPtr}
        , {"Mean", Mean_tensor.devPtr}
        , {"Var", Var_tensor.devPtr}
        , {"Previous_running_mean", Previous_running_mean_tensor.devPtr}
        , {"Previous_running_var", Previous_running_var_tensor.devPtr}
        , {"Next_running_mean", Next_running_mean_tensor.devPtr}
        , {"Next_running_var", Next_running_var_tensor.devPtr}
        , {"Scale", Scale_tensor.devPtr}
        , {"Bias", Bias_tensor.devPtr}
        , {"EPS", &EPS_scalar}
        , {"EXP_AVG", &EXP_AVG_scalar}
        , {"output", Y_tensor.devPtr}
        , {"workspace", workspace.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == graph.execute(handle, variant_pack));

    cudnnDestroy(handle);
}

TEST_CASE("SGBN Add Relu Graph", "[batchnorm][graph]") {
    cudnn_frontend::graph::Graph graph("SGBN_Add_Relu");
    graph.set_io_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
    
    auto sgbn = cudnn_frontend::graph::Batchnorm("SGBN")
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Batchnorm::PORTS::X, "X"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Mean, "Mean"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Var, "Var"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Previous_running_mean, "Previous_running_mean"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Previous_running_var, "Previous_running_var"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Next_running_mean, "Next_running_mean"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Next_running_var, "Next_running_var"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Scale, "Scale"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Bias, "Bias"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::EPS, "EPS"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::EXP_AVG, "EXP_AVG"}
                        , {cudnn_frontend::graph::Batchnorm::PORTS::Y, "Y"}
                    });
    graph.insert_node(sgbn);

    auto add = cudnn_frontend::graph::Pointwise("add")
                    .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, sgbn.get_tensor_at_port(cudnn_frontend::graph::Batchnorm::PORTS::Y)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "A"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "add_output"}
                    });
    graph.insert_node(add);

    auto relu = cudnn_frontend::graph::Pointwise("relu")
                    .set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, add.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "output"}
                    });
    graph.insert_node(relu);
    
    graph.insert_tensor(cudnn_frontend::graph::Tensor("X").set_dim({4, 32, 16, 16}).set_data_type(cudnn_frontend::DataType_t::HALF))
         .insert_tensor(cudnn_frontend::graph::Tensor("Mean"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Var"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Previous_running_mean"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Previous_running_var"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Next_running_mean"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Next_running_var"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Scale"))
         .insert_tensor(cudnn_frontend::graph::Tensor("Bias"))
         .insert_tensor(cudnn_frontend::graph::Tensor("EPS").set_is_pass_by_value(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("EXP_AVG").set_is_pass_by_value(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("Y").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("A").set_dim({4, 32, 16, 16}).set_data_type(cudnn_frontend::DataType_t::HALF))
         .insert_tensor(cudnn_frontend::graph::Tensor("add_output").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("output").set_data_type(cudnn_frontend::DataType_t::HALF));

    #if (CUDNN_VERSION < 8700)
        SKIP("single GPU BN is not supported in cudnn versions prior to 8.7");
    #endif

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    
    REQUIRE(cudnn_frontend::error_t::OK == graph.build(handle));
    auto plans = graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_FALLBACK)
                    .build_plans(handle);

    REQUIRE(cudnn_frontend::error_t::OK == graph.set_executor(plans));

    Surface<half> X_tensor(4*32*16*16, false);
    Surface<float> Mean_tensor(16, false);
    Surface<float> Var_tensor(16, false);
    Surface<float> Previous_running_mean_tensor(16, false);
    Surface<float> Previous_running_var_tensor(16, false);
    Surface<float> Next_running_mean_tensor(16, false);
    Surface<float> Next_running_var_tensor(16, false);
    Surface<float> Scale_tensor(16, false);
    Surface<float> Bias_tensor(16, false);
    float EPS_scalar = 0.001;
    float EXP_AVG_scalar = 0.001;
    Surface<half> A_tensor(4*32*16*16, false);
    Surface<half> Y_tensor(4*32*16*16, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"X", X_tensor.devPtr}
        , {"Mean", Mean_tensor.devPtr}
        , {"Var", Var_tensor.devPtr}
        , {"Previous_running_mean", Previous_running_mean_tensor.devPtr}
        , {"Previous_running_var", Previous_running_var_tensor.devPtr}
        , {"Next_running_mean", Next_running_mean_tensor.devPtr}
        , {"Next_running_var", Next_running_var_tensor.devPtr}
        , {"Scale", Scale_tensor.devPtr}
        , {"Bias", Bias_tensor.devPtr}
        , {"EPS", &EPS_scalar}
        , {"EXP_AVG", &EXP_AVG_scalar}
        , {"A", A_tensor.devPtr}
        , {"output", Y_tensor.devPtr}
        , {"workspace", workspace.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == graph.execute(handle, variant_pack));

    cudnnDestroy(handle);
}

TEST_CASE("Scale Bias Relu Conv Genstats Graph", "[conv][genstats][graph]") {
    cudnn_frontend::graph::Graph graph("genstats");
    graph.set_io_data_type(cudnn_frontend::DataType_t::HALF)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::HALF)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

    auto pw_scale = cudnn_frontend::graph::Pointwise("pw_scale")
                    .set_mode(cudnn_frontend::PointwiseMode_t::MUL)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, "image"}
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
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "relu_output"}
                    });
    graph.insert_node(pw_relu);

    auto conv = cudnn_frontend::graph::Convolution("Convolution")
                .set_padding({1, 1})
                .set_stride({1, 1})
                .set_dilation({1, 1})
                .map_port_to_tensor({
                    {cudnn_frontend::graph::Convolution::PORTS::X, pw_relu.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                    , {cudnn_frontend::graph::Convolution::PORTS::W, "weight"}
                    , {cudnn_frontend::graph::Convolution::PORTS::Y, "output"}
                });
    graph.insert_node(conv);

    auto genstats = cudnn_frontend::graph::Genstats("Genstats")
                .map_port_to_tensor({
                    {cudnn_frontend::graph::Genstats::PORTS::X, conv.get_tensor_at_port(cudnn_frontend::graph::Convolution::PORTS::Y)}
                    , {cudnn_frontend::graph::Genstats::PORTS::SUM, "sum"}
                    , {cudnn_frontend::graph::Genstats::PORTS::SQ_SUM, "sq_sum"}
                });
    graph.insert_node(genstats);

    graph.insert_tensor(cudnn_frontend::graph::Tensor("image").set_dim({4, 32, 16, 16}))
         .insert_tensor(cudnn_frontend::graph::Tensor("scale").set_dim({1, 32, 1, 1}))
         .insert_tensor(cudnn_frontend::graph::Tensor("scale_output").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("bias").set_dim({1, 32, 1, 1}))
         .insert_tensor(cudnn_frontend::graph::Tensor("bias_output").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("relu_output").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("weight").set_dim({64, 32, 3, 3}))
         .insert_tensor(cudnn_frontend::graph::Tensor("output"))
         .insert_tensor(cudnn_frontend::graph::Tensor("sum").set_data_type(cudnn_frontend::DataType_t::FLOAT))
         .insert_tensor(cudnn_frontend::graph::Tensor("sq_sum").set_data_type(cudnn_frontend::DataType_t::FLOAT));

    #if (CUDNN_VERSION < 8800)
        SKIP("ConvBNFprop requires cudnn 8.8 and up");
    #endif

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    REQUIRE(cudnn_frontend::error_t::OK == graph.build(handle));

    auto plans = graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_A)
                    .build_plans(handle);

    REQUIRE(cudnn_frontend::error_t::OK == graph.set_executor(plans));

    Surface<half> x_tensor(4*32*16*16, false);
    Surface<half> s_tensor(32, false);
    Surface<half> b_tensor(32, false);
    Surface<half> w_tensor(64*32*3*3, false);
    Surface<half> y_tensor(4*64*16*16, false);
    Surface<float> sum_tensor(64, false);
    Surface<float> sq_sum_tensor(64, false);
    
    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    
    std::unordered_map<std::string, void*> variant_pack = {
        {"image", x_tensor.devPtr}
        , {"scale", s_tensor.devPtr}
        , {"bias", b_tensor.devPtr}
        , {"weight", w_tensor.devPtr}
        , {"output", y_tensor.devPtr}
        , {"sum", sum_tensor.devPtr}
        , {"sq_sum", sq_sum_tensor.devPtr}
        , {"workspace", workspace.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == graph.execute(handle, variant_pack));
    cudnnDestroy(handle);
}

TEST_CASE("Scale Bias Add Relu Conv Genstats Graph", "[conv][genstats][graph]") {
    cudnn_frontend::graph::Graph graph("genstats");
    graph.set_io_data_type(cudnn_frontend::DataType_t::HALF)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::HALF)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

    auto pw_scale = cudnn_frontend::graph::Pointwise("pw_scale")
                    .set_mode(cudnn_frontend::PointwiseMode_t::MUL)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, "image"}
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
    
    auto pw_add = cudnn_frontend::graph::Pointwise("pw_add")
                    .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, pw_bias.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "add"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "add_output"}
                    });
    graph.insert_node(pw_add);

    auto pw_relu = cudnn_frontend::graph::Pointwise("pw_relu")
                    .set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, pw_add.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "relu_output"}
                    });
    graph.insert_node(pw_relu);

    auto conv = cudnn_frontend::graph::Convolution("Convolution")
                .set_padding({0, 0})
                .set_stride({1, 1})
                .set_dilation({1, 1})
                .map_port_to_tensor({
                    {cudnn_frontend::graph::Convolution::PORTS::X, pw_relu.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                    , {cudnn_frontend::graph::Convolution::PORTS::W, "weight"}
                    , {cudnn_frontend::graph::Convolution::PORTS::Y, "output"}
                });
    graph.insert_node(conv);

    auto genstats = cudnn_frontend::graph::Genstats("Genstats")
                .map_port_to_tensor({
                    {cudnn_frontend::graph::Genstats::PORTS::X, conv.get_tensor_at_port(cudnn_frontend::graph::Convolution::PORTS::Y)}
                    , {cudnn_frontend::graph::Genstats::PORTS::SUM, "sum"}
                    , {cudnn_frontend::graph::Genstats::PORTS::SQ_SUM, "sq_sum"}
                });
    graph.insert_node(genstats);

    graph.insert_tensor(cudnn_frontend::graph::Tensor("image").set_dim({4, 32, 16, 16}))
         .insert_tensor(cudnn_frontend::graph::Tensor("scale").set_dim({1, 32, 1, 1}))
         .insert_tensor(cudnn_frontend::graph::Tensor("scale_output").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("bias").set_dim({1, 32, 1, 1}))
         .insert_tensor(cudnn_frontend::graph::Tensor("bias_output").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("add").set_dim({4, 32, 16, 16}))
         .insert_tensor(cudnn_frontend::graph::Tensor("add_output").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("relu_output"))
         .insert_tensor(cudnn_frontend::graph::Tensor("weight").set_dim({64, 32, 1, 1}))
         .insert_tensor(cudnn_frontend::graph::Tensor("output"))
         .insert_tensor(cudnn_frontend::graph::Tensor("sum").set_data_type(cudnn_frontend::DataType_t::FLOAT))
         .insert_tensor(cudnn_frontend::graph::Tensor("sq_sum").set_data_type(cudnn_frontend::DataType_t::FLOAT));

    #if (CUDNN_VERSION < 8900)
        SKIP("DBARCS requires cudnn 8.9 and up");
    #endif
    if (check_device_arch_newer_than("hopper") == false) {
        SKIP("DBARCS requires hopper and above architecture.");
    }
    
    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    REQUIRE(cudnn_frontend::error_t::OK == graph.build(handle));

    auto plans = graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_A)
                    .build_plans(handle);

    REQUIRE(cudnn_frontend::error_t::OK == graph.set_executor(plans));

    Surface<half> x_tensor(4*32*16*16, false);
    Surface<half> s_tensor(32, false);
    Surface<half> b_tensor(32, false);
    Surface<half> a_tensor(4*32*16*16, false);
    Surface<half> relu_y_tensor(4*32*16*16, false);
    Surface<half> w_tensor(64*32*1*1, false);
    Surface<half> y_tensor(4*64*16*16, false);
    Surface<float> sum_tensor(64, false);
    Surface<float> sq_sum_tensor(64, false);
    
    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    
    std::unordered_map<std::string, void*> variant_pack = {
        {"image", x_tensor.devPtr}
        , {"scale", s_tensor.devPtr}
        , {"bias", b_tensor.devPtr}
        , {"add", a_tensor.devPtr}
        , {"relu_output", relu_y_tensor.devPtr}
        , {"weight", w_tensor.devPtr}
        , {"output", y_tensor.devPtr}
        , {"sum", sum_tensor.devPtr}
        , {"sq_sum", sq_sum_tensor.devPtr}
        , {"workspace", workspace.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == graph.execute(handle, variant_pack));
    cudnnDestroy(handle);
}


TEST_CASE("Dual Scale Bias Add Relu Conv Genstats Graph", "[conv][genstats][graph]") {
    cudnn_frontend::graph::Graph graph("genstats");
    graph.set_io_data_type(cudnn_frontend::DataType_t::HALF)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::HALF)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

    auto pw_scale1 = cudnn_frontend::graph::Pointwise("pw_scale1")
                    .set_mode(cudnn_frontend::PointwiseMode_t::MUL)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, "image"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "scale1"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "scale_output1"}
                    });
    graph.insert_node(pw_scale1);
    
    auto pw_scale2 = cudnn_frontend::graph::Pointwise("pw_scale2")
                    .set_mode(cudnn_frontend::PointwiseMode_t::MUL)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, "dual_X"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "scale2"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "scale_output2"}
                    });
    graph.insert_node(pw_scale2);

    auto pw_bias1 = cudnn_frontend::graph::Pointwise("pw_bias1")
                    .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, pw_scale1.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "bias1"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "bias_output1"}
                    });
    graph.insert_node(pw_bias1);
    
    auto pw_bias2 = cudnn_frontend::graph::Pointwise("pw_bias2")
                    .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, pw_scale2.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "bias2"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "bias_output2"}
                    });
    graph.insert_node(pw_bias2);
    
    auto pw_add = cudnn_frontend::graph::Pointwise("pw_add")
                    .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, pw_bias1.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, pw_bias2.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "add_output"}
                    });
    graph.insert_node(pw_add);

    auto pw_relu = cudnn_frontend::graph::Pointwise("pw_relu")
                    .set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, pw_add.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "relu_output"}
                    });
    graph.insert_node(pw_relu);

    auto conv = cudnn_frontend::graph::Convolution("Convolution")
                .set_padding({0, 0})
                .set_stride({1, 1})
                .set_dilation({1, 1})
                .map_port_to_tensor({
                    {cudnn_frontend::graph::Convolution::PORTS::X, pw_relu.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                    , {cudnn_frontend::graph::Convolution::PORTS::W, "weight"}
                    , {cudnn_frontend::graph::Convolution::PORTS::Y, "output"}
                });
    graph.insert_node(conv);

    auto genstats = cudnn_frontend::graph::Genstats("Genstats")
                .map_port_to_tensor({
                    {cudnn_frontend::graph::Genstats::PORTS::X, conv.get_tensor_at_port(cudnn_frontend::graph::Convolution::PORTS::Y)}
                    , {cudnn_frontend::graph::Genstats::PORTS::SUM, "sum"}
                    , {cudnn_frontend::graph::Genstats::PORTS::SQ_SUM, "sq_sum"}
                });
    graph.insert_node(genstats);

    graph.insert_tensor(cudnn_frontend::graph::Tensor("image").set_dim({4, 32, 16, 16}))
         .insert_tensor(cudnn_frontend::graph::Tensor("scale1").set_dim({1, 32, 1, 1}))
         .insert_tensor(cudnn_frontend::graph::Tensor("scale_output1").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("scale2").set_dim({1, 32, 1, 1}))
         .insert_tensor(cudnn_frontend::graph::Tensor("scale_output2").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("bias1").set_dim({1, 32, 1, 1}))
         .insert_tensor(cudnn_frontend::graph::Tensor("bias_output1").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("bias2").set_dim({1, 32, 1, 1}))
         .insert_tensor(cudnn_frontend::graph::Tensor("bias_output2").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("dual_X").set_dim({4, 32, 16, 16}))
         .insert_tensor(cudnn_frontend::graph::Tensor("add_output").set_is_virtual(true))
         .insert_tensor(cudnn_frontend::graph::Tensor("relu_output"))
         .insert_tensor(cudnn_frontend::graph::Tensor("weight").set_dim({64, 32, 1, 1}))
         .insert_tensor(cudnn_frontend::graph::Tensor("output"))
         .insert_tensor(cudnn_frontend::graph::Tensor("sum").set_data_type(cudnn_frontend::DataType_t::FLOAT))
         .insert_tensor(cudnn_frontend::graph::Tensor("sq_sum").set_data_type(cudnn_frontend::DataType_t::FLOAT));

    
    #if (CUDNN_VERSION < 8900)
        SKIP("DBARCS requires cudnn 8.9 and up");
    #endif
    if (check_device_arch_newer_than("hopper") == false) {
        SKIP("DBARCS requires hopper and above architecture.");
    }

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    REQUIRE(cudnn_frontend::error_t::OK == graph.build(handle));

    auto plans = graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_A)
                    .build_plans(handle);

    REQUIRE(cudnn_frontend::error_t::OK == graph.set_executor(plans));

    Surface<half> x_tensor(4*32*16*16, false);
    Surface<half> s1_tensor(32, false);
    Surface<half> b1_tensor(32, false);
    Surface<half> s2_tensor(32, false);
    Surface<half> b2_tensor(32, false);
    Surface<half> dual_x_tensor(4*32*16*16, false);
    Surface<half> relu_y_tensor(4*32*16*16, false);
    Surface<half> w_tensor(64*32*1*1, false);
    Surface<half> y_tensor(4*64*16*16, false);
    Surface<float> sum_tensor(64, false);
    Surface<float> sq_sum_tensor(64, false);
    
    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    
    std::unordered_map<std::string, void*> variant_pack = {
        {"image", x_tensor.devPtr}
        , {"scale1", s1_tensor.devPtr}
        , {"bias1", b1_tensor.devPtr}
        , {"scale2", s2_tensor.devPtr}
        , {"bias2", b2_tensor.devPtr}
        , {"dual_X", dual_x_tensor.devPtr}
        , {"relu_output", relu_y_tensor.devPtr}
        , {"weight", w_tensor.devPtr}
        , {"output", y_tensor.devPtr}
        , {"sum", sum_tensor.devPtr}
        , {"sq_sum", sq_sum_tensor.devPtr}
        , {"workspace", workspace.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == graph.execute(handle, variant_pack));
    cudnnDestroy(handle);
}

TEST_CASE("Conv Genstats Graph", "[conv][genstats][graph]") {
    cudnn_frontend::graph::Graph graph("genstats");
    graph.set_io_data_type(cudnn_frontend::DataType_t::HALF)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

    auto conv = cudnn_frontend::graph::Convolution("Convolution")
                .set_padding({1, 1})
                .set_stride({1, 1})
                .set_dilation({1, 1})
                .map_port_to_tensor({
                    {cudnn_frontend::graph::Convolution::PORTS::X, "image"}
                    , {cudnn_frontend::graph::Convolution::PORTS::W, "weight"}
                    , {cudnn_frontend::graph::Convolution::PORTS::Y, "output"}
                });
    graph.insert_node(conv);

    auto genstats = cudnn_frontend::graph::Genstats("Genstats")
                .map_port_to_tensor({
                    {cudnn_frontend::graph::Genstats::PORTS::X, conv.get_tensor_at_port(cudnn_frontend::graph::Convolution::PORTS::Y)}
                    , {cudnn_frontend::graph::Genstats::PORTS::SUM, "sum"}
                    , {cudnn_frontend::graph::Genstats::PORTS::SQ_SUM, "sq_sum"}
                });
    graph.insert_node(genstats);

    graph.insert_tensor(cudnn_frontend::graph::Tensor("image").set_dim({4, 32, 16, 16}))
         .insert_tensor(cudnn_frontend::graph::Tensor("weight").set_dim({64, 32, 3, 3}))
         .insert_tensor(cudnn_frontend::graph::Tensor("output"))
         .insert_tensor(cudnn_frontend::graph::Tensor("sum").set_data_type(cudnn_frontend::DataType_t::FLOAT))
         .insert_tensor(cudnn_frontend::graph::Tensor("sq_sum").set_data_type(cudnn_frontend::DataType_t::FLOAT));

    #if (CUDNN_VERSION < 8800)
        SKIP("ConvBNFprop requires cudnn 8.8 and up");
    #endif

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    REQUIRE(cudnn_frontend::error_t::OK == graph.build(handle));

    auto plans = graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_A)
                    .build_plans(handle);

    REQUIRE(cudnn_frontend::error_t::OK == graph.set_executor(plans));

    Surface<half> x_tensor(4*32*16*16, false);
    Surface<half> w_tensor(64*32*3*3, false);
    Surface<half> y_tensor(4*64*16*16, false);
    Surface<float> sum_tensor(64, false);
    Surface<float> sq_sum_tensor(64, false);
    
    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    
    std::unordered_map<std::string, void*> variant_pack = {
        {"image", x_tensor.devPtr}
        , {"weight", w_tensor.devPtr}
        , {"output", y_tensor.devPtr}
        , {"sum", sum_tensor.devPtr}
        , {"sq_sum", sq_sum_tensor.devPtr}
        , {"workspace", workspace.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == graph.execute(handle, variant_pack));
    cudnnDestroy(handle);
}

TEST_CASE("Convolution BN Inference Graph", "[conv][graph]") {
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
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, conv.get_tensor_at_port(cudnn_frontend::graph::Convolution::PORTS::Y)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "mean"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "mean_sub_output"}
                    });
    graph.insert_node(mean_sub);

    //////////////////////////////////////////////////////////////////////////////
    // var_with_epsilon = var + epsilon
    //////////////////////////////////////////////////////////////////////////////
    auto epsilon_add = cudnn_frontend::graph::Pointwise("epsilon_add")
                        .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
                        .map_port_to_tensor({
                            {cudnn_frontend::graph::Pointwise::PORTS::IN_0, "var"}
                            , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "epsilon"}
                            , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "var_with_epsilon"}
                        });
    graph.insert_node(epsilon_add);

    //////////////////////////////////////////////////////////////////////////////
    // inv_var = rsqrt(var_with_epsilon)
    //////////////////////////////////////////////////////////////////////////////
    auto rsqrt_var = cudnn_frontend::graph::Pointwise("rsqrt_var")
                        .set_mode(cudnn_frontend::PointwiseMode_t::RSQRT)
                        .map_port_to_tensor({
                            {cudnn_frontend::graph::Pointwise::PORTS::IN_0, epsilon_add.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                            , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "inv_var"}
                        });
    graph.insert_node(rsqrt_var);

    //////////////////////////////////////////////////////////////////////////////
    // norm_input = mean_sub_output * inv_var
    //////////////////////////////////////////////////////////////////////////////
    auto inv_var_mul = cudnn_frontend::graph::Pointwise("inv_var_mul")
                        .set_mode(cudnn_frontend::PointwiseMode_t::MUL)
                        .map_port_to_tensor({
                            {cudnn_frontend::graph::Pointwise::PORTS::IN_0, mean_sub.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                            , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, rsqrt_var.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                            , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "norm_input"}
                        });
    graph.insert_node(inv_var_mul);


    //////////////////////////////////////////////////////////////////////////////
    // scale_output = norm_input * scale
    //////////////////////////////////////////////////////////////////////////////
    auto scale_mul = cudnn_frontend::graph::Pointwise("scale_mul")
                        .set_mode(cudnn_frontend::PointwiseMode_t::MUL)
                        .map_port_to_tensor({
                            {cudnn_frontend::graph::Pointwise::PORTS::IN_0, inv_var_mul.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                            , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "scale"}
                            , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "scale_output"}
                        });
    graph.insert_node(scale_mul);

    //////////////////////////////////////////////////////////////////////////////
    // bias_output = scale_output + bias
    //////////////////////////////////////////////////////////////////////////////
    auto bias_add = cudnn_frontend::graph::Pointwise("bias_add")
                    .set_mode(cudnn_frontend::PointwiseMode_t::ADD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, scale_mul.get_tensor_at_port(cudnn_frontend::graph::Pointwise::PORTS::OUT_0)}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "bias"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "bias_output"}
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

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    #if (CUDNN_VERSION >= 8500)
        REQUIRE(cudnn_frontend::error_t::OK == graph.build(handle));
    #elif (CUDNN_VERSION >= 8300)
        SKIP("Passing tensors by value, here epsilon, is not supported prior to 8500.");
    #else
        SKIP("RSQRT pointwise mode does not exist prior to 8300.");
    #endif

    auto plans = graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_A)
                    .build_plans(handle);

    REQUIRE(cudnn_frontend::error_t::OK == graph.set_executor(plans));

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
    REQUIRE(cudnn_frontend::error_t::OK == graph.execute(handle, variant_pack));

    cudnnDestroy(handle);
}
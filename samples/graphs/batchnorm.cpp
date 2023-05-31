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
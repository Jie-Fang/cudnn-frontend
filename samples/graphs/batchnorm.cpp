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
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph("bn_finalize");
    graph.set_io_data_type(fe::DataType_t::FLOAT)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    fe::graph::BN_finalize::Inputs inputs;
    auto sum = graph.tensor(fe::graph::Tensor("sum").set_dim({1,32,1,1}));
    sum->generateStrides(CUDNN_TENSOR_NHWC);
    auto sq_sum = graph.tensor(fe::graph::Tensor("sq_sum"));
    auto mean = graph.tensor(fe::graph::Tensor("mean"));
    auto inv_variance = graph.tensor(fe::graph::Tensor("inv_variance"));
    auto prev_running_mean = graph.tensor(fe::graph::Tensor("prev_running_mean"));
    auto prev_running_var = graph.tensor(fe::graph::Tensor("prev_running_var"));
    auto scale = graph.tensor(fe::graph::Tensor("scale"));
    auto bias = graph.tensor(fe::graph::Tensor("bias"));
    auto epsilon = graph.tensor(fe::graph::Tensor("epsilon").set_is_pass_by_value(true));
    auto exp_avg = graph.tensor(fe::graph::Tensor("exp_avg").set_is_pass_by_value(true));
    auto accum_count = graph.tensor(fe::graph::Tensor("accum_count").set_is_pass_by_value(true).set_data_type(fe::DataType_t::INT64));

    inputs.SUM = sum;
    inputs.SQ_SUM = sq_sum;
    inputs.SCALE = scale;
    inputs.BIAS = bias;
    inputs.MEAN = mean;
    inputs.INV_VARIANCE = inv_variance;
    inputs.PREV_RUNNING_MEAN = prev_running_mean;
    inputs.PREV_RUNNING_VAR = prev_running_var;
    inputs.EPSILON = epsilon;
    inputs.EXP_AVG = exp_avg;
    inputs.ACCUM_COUNT = accum_count;

    auto bn_finalize_options = fe::graph::BN_finalize("bn_finalize");
    auto [eq_scale, eq_bias, next_running_mean, next_running_var] = graph.bn_finalize(inputs, bn_finalize_options);

    #if (CUDNN_VERSION < 8400)
        SKIP("BNFinalize requires cudnn 8.4 and up");
    #endif

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    REQUIRE(fe::error_t::OK == graph.build(handle));
    auto plans = graph.get_execution_plan_list(fe::HeurMode_t::HEUR_MODE_FALLBACK)
                    .build_plans(handle);

    REQUIRE(fe::error_t::OK == graph.set_executor(plans));

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

    auto workspace = graph.tensor(fe::graph::Tensor("workspace"));
    Surface<int8_t> workspace_tensor(graph.get_workspace_size(), false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor>, void*> variant_pack = {
        {sum, Sum_tensor.devPtr}
        , {sq_sum, Sq_sum_tensor.devPtr}
        , {mean, Mean_tensor.devPtr}
        , {inv_variance, Var_tensor.devPtr}
        , {prev_running_mean, Previous_running_mean_tensor.devPtr}
        , {prev_running_var, Previous_running_var_tensor.devPtr}
        , {next_running_mean, Next_running_mean_tensor.devPtr}
        , {next_running_var, Next_running_var_tensor.devPtr}
        , {scale, Scale_tensor.devPtr}
        , {bias, Bias_tensor.devPtr}
        , {epsilon, &EPS_scalar}
        , {exp_avg, &EXP_AVG_scalar}
        , {accum_count, &nhw}
        , {eq_scale, eq_scale_tensor.devPtr}
        , {eq_bias, eq_bias_tensor.devPtr}
        , {workspace, workspace_tensor.devPtr}
    };
    REQUIRE(fe::error_t::OK == graph.execute(handle, variant_pack));

    cudnnDestroy(handle);
}

TEST_CASE("SGBN Add Relu Graph", "[batchnorm][graph]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph("SGBN_Add_Relu");
    graph.set_io_data_type(fe::DataType_t::HALF)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);
    
    fe::graph::Batchnorm::Inputs inputs;
    auto X = graph.tensor(fe::graph::Tensor("X").set_dim({4,32,16,16}));
    X->generateStrides(CUDNN_TENSOR_NHWC);
    auto prev_running_mean = graph.tensor(fe::graph::Tensor("prev_running_mean").set_data_type(fe::DataType_t::FLOAT));
    auto prev_running_var = graph.tensor(fe::graph::Tensor("prev_running_var").set_data_type(fe::DataType_t::FLOAT));
    auto scale = graph.tensor(fe::graph::Tensor("scale").set_data_type(fe::DataType_t::FLOAT));
    auto bias = graph.tensor(fe::graph::Tensor("bias").set_data_type(fe::DataType_t::FLOAT));

    inputs.X = X;
    inputs.SCALE = scale;
    inputs.BIAS = bias;
    inputs.PREV_RUNNING_MEAN = prev_running_mean;
    inputs.PREV_RUNNING_VAR = prev_running_var;
    
    auto batchnorm_options = fe::graph::Batchnorm("batchnorm").set_forward_phase(fe::NormFwdPhase_t::TRAINING).set_epsilon(1.0e-5).set_momentum(0.1);
    auto [bn_output, mean, inv_variance, next_running_mean, next_running_var] = graph.batchnorm(inputs, batchnorm_options);
    bn_output->set_is_virtual(true);
    
    auto A = graph.tensor(fe::graph::Tensor("A").set_dim({4,32,16,16}).set_data_type(fe::DataType_t::HALF));
    A->generateStrides(CUDNN_TENSOR_NHWC);
    auto add_options = fe::graph::Pointwise("add").set_mode(fe::PointwiseMode_t::ADD);
    auto add_output = graph.pointwise(bn_output, A, add_options);
    add_output->set_is_virtual(true);

    auto relu_options = fe::graph::Pointwise("relu").set_mode(fe::PointwiseMode_t::RELU_FWD);
    auto Y = graph.pointwise(add_output, relu_options);

    #if (CUDNN_VERSION < 8700)
        SKIP("single GPU BN is not supported in cudnn versions prior to 8.7");
    #endif

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    
    REQUIRE(fe::error_t::OK == graph.build(handle));
    auto plans = graph.get_execution_plan_list(fe::HeurMode_t::HEUR_MODE_FALLBACK)
                    .build_plans(handle);

    REQUIRE(fe::error_t::OK == graph.set_executor(plans));

    Surface<half> X_tensor(4*32*16*16, false);
    Surface<float> Mean_tensor(32, false);
    Surface<float> Var_tensor(32, false);
    Surface<float> Previous_running_mean_tensor(32, false);
    Surface<float> Previous_running_var_tensor(32, false);
    Surface<float> Next_running_mean_tensor(32, false);
    Surface<float> Next_running_var_tensor(32, false);
    Surface<float> Scale_tensor(32, false);
    Surface<float> Bias_tensor(32, false);
    Surface<half> A_tensor(4*32*16*16, false);
    Surface<half> Y_tensor(4*32*16*16, false);

    auto workspace = graph.tensor(fe::graph::Tensor("workspace"));
    Surface<int8_t> workspace_tensor(graph.get_workspace_size(), false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor>, void*> variant_pack = {
        {X, X_tensor.devPtr}
        , {mean, Mean_tensor.devPtr}
        , {inv_variance, Var_tensor.devPtr}
        , {prev_running_mean, Previous_running_mean_tensor.devPtr}
        , {prev_running_var, Previous_running_var_tensor.devPtr}
        , {next_running_mean, Next_running_mean_tensor.devPtr}
        , {next_running_var, Next_running_var_tensor.devPtr}
        , {scale, Scale_tensor.devPtr}
        , {bias, Bias_tensor.devPtr}
        , {A, A_tensor.devPtr}
        , {Y, Y_tensor.devPtr}
        , {workspace, workspace_tensor.devPtr}
    };
    REQUIRE(fe::error_t::OK == graph.execute(handle, variant_pack));

    cudnnDestroy(handle);
}

TEST_CASE("SBRCS", "[conv][genstats][graph]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph("genstats");
    graph.set_io_data_type(fe::DataType_t::HALF)
         .set_intermediate_data_type(fe::DataType_t::HALF)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(fe::graph::Tensor("image").set_dim({4, 64, 16, 16}));
    X->generateStrides(CUDNN_TENSOR_NHWC);
    auto S = graph.tensor(fe::graph::Tensor("scale").set_dim({1, 64, 1, 1}));
    S->generateStrides(CUDNN_TENSOR_NHWC);

    auto scale_options = fe::graph::Pointwise("scale").set_mode(fe::PointwiseMode_t::MUL);
    auto scale_output = graph.pointwise(X, S, scale_options);
    scale_output->set_is_virtual(true);

    auto B = graph.tensor(fe::graph::Tensor("bias").set_dim({1, 64, 1, 1}));
    B->generateStrides(CUDNN_TENSOR_NHWC);
    auto bias_options = fe::graph::Pointwise("bias").set_mode(fe::PointwiseMode_t::ADD);
    auto bias_output = graph.pointwise(scale_output, B, bias_options);
    bias_output->set_is_virtual(true);
    
    auto relu_options = fe::graph::Pointwise("relu").set_mode(fe::PointwiseMode_t::RELU_FWD);
    auto relu_output = graph.pointwise(bias_output, relu_options);
    relu_output->set_is_virtual(true);

    auto W = graph.tensor(fe::graph::Tensor("weight").set_dim({32, 64, 3, 3}));
    W->generateStrides(CUDNN_TENSOR_NHWC);
    auto conv_options = fe::graph::Conv_fprop("conv").set_padding({1,1}).set_stride({1,1}).set_dilation({1,1});
    auto Y = graph.conv_fprop(relu_output, W, conv_options);

    auto genstats_options = fe::graph::Genstats("genstats");
    auto [SUM, SQ_SUM] = graph.genstats(Y, genstats_options);

    #if (CUDNN_VERSION < 8800)
        SKIP("ConvBNFprop requires cudnn 8.8 and up");
    #endif

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    REQUIRE(fe::error_t::OK == graph.build(handle));

    auto plans = graph.get_execution_plan_list(fe::HeurMode_t::HEUR_MODE_A)
                    .build_plans(handle);

    REQUIRE(fe::error_t::OK == graph.set_executor(plans));

    Surface<half> x_tensor(4*64*16*16, false);
    Surface<half> s_tensor(64, false);
    Surface<half> b_tensor(64, false);
    Surface<half> w_tensor(32*64*3*3, false);
    Surface<half> y_tensor(4*32*16*16, false);
    Surface<float> sum_tensor(32, false);
    Surface<float> sq_sum_tensor(32, false);
    
    auto workspace = graph.tensor(fe::graph::Tensor("workspace"));
    Surface<int8_t> workspace_tensor(graph.get_workspace_size(), false);
    
    std::unordered_map<std::shared_ptr<fe::graph::Tensor>, void*> variant_pack = {
        {X, x_tensor.devPtr}
        , {S, s_tensor.devPtr}
        , {B, b_tensor.devPtr}
        , {W, w_tensor.devPtr}
        , {Y, y_tensor.devPtr}
        , {SUM, sum_tensor.devPtr}
        , {SQ_SUM, sq_sum_tensor.devPtr}
        , {workspace, workspace_tensor.devPtr}
    };
    REQUIRE(fe::error_t::OK == graph.execute(handle, variant_pack));
    cudnnDestroy(handle);
}

TEST_CASE("DBARCS", "[conv][genstats][graph]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph("genstats");
    graph.set_io_data_type(fe::DataType_t::HALF)
         .set_intermediate_data_type(fe::DataType_t::HALF)
         .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(fe::graph::Tensor("image").set_dim({4, 64, 16, 16}));
    X->generateStrides(CUDNN_TENSOR_NHWC);
    auto S = graph.tensor(fe::graph::Tensor("scale").set_dim({1, 64, 1, 1}));
    S->generateStrides(CUDNN_TENSOR_NHWC);

    auto scale_options = fe::graph::Pointwise("scale").set_mode(fe::PointwiseMode_t::MUL);
    auto scale_output = graph.pointwise(X, S, scale_options);
    scale_output->set_is_virtual(true);

    auto B = graph.tensor(fe::graph::Tensor("bias").set_dim({1, 64, 1, 1}));
    B->generateStrides(CUDNN_TENSOR_NHWC);
    auto bias_options = fe::graph::Pointwise("bias").set_mode(fe::PointwiseMode_t::ADD);
    auto bias_output = graph.pointwise(scale_output, B, bias_options);
    bias_output->set_is_virtual(true);

    auto DUAL_X = graph.tensor(fe::graph::Tensor("dual_image").set_dim({4, 64, 16, 16}));
    DUAL_X->generateStrides(CUDNN_TENSOR_NHWC);
    auto DUAL_S = graph.tensor(fe::graph::Tensor("dual_scale").set_dim({1, 64, 1, 1}));
    DUAL_S->generateStrides(CUDNN_TENSOR_NHWC);

    auto dual_scale_options = fe::graph::Pointwise("dual_scale").set_mode(fe::PointwiseMode_t::MUL);
    auto dual_scale_output = graph.pointwise(DUAL_X, DUAL_S, dual_scale_options);
    dual_scale_output->set_is_virtual(true);

    auto DUAL_B = graph.tensor(fe::graph::Tensor("dual_bias").set_dim({1, 64, 1, 1}));
    DUAL_B->generateStrides(CUDNN_TENSOR_NHWC);
    auto dual_bias_options = fe::graph::Pointwise("dual_bias").set_mode(fe::PointwiseMode_t::ADD);
    auto dual_bias_output = graph.pointwise(dual_scale_output, DUAL_B, dual_bias_options);
    dual_bias_output->set_is_virtual(true);

    auto add_options = fe::graph::Pointwise("add").set_mode(fe::PointwiseMode_t::ADD);
    auto add_output = graph.pointwise(bias_output, dual_bias_output, add_options);
    add_output->set_is_virtual(true);
    
    auto relu_options = fe::graph::Pointwise("relu").set_mode(fe::PointwiseMode_t::RELU_FWD);
    auto relu_output = graph.pointwise(add_output, relu_options);

    auto W = graph.tensor(fe::graph::Tensor("weight").set_dim({32, 64, 1, 1}));
    W->generateStrides(CUDNN_TENSOR_NHWC);
    auto conv_options = fe::graph::Conv_fprop("conv").set_padding({0,0}).set_stride({1,1}).set_dilation({1,1});
    auto Y = graph.conv_fprop(relu_output, W, conv_options);

    auto genstats_options = fe::graph::Genstats("genstats");
    auto [SUM, SQ_SUM] = graph.genstats(Y, genstats_options);

    #if (CUDNN_VERSION < 8900)
        SKIP("DBARCS requires cudnn 8.9 and up");
    #endif

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    REQUIRE(fe::error_t::OK == graph.build(handle));

    auto plans = graph.get_execution_plan_list(fe::HeurMode_t::HEUR_MODE_A)
                    .build_plans(handle);

    REQUIRE(fe::error_t::OK == graph.set_executor(plans));

    Surface<half> x_tensor(4*64*16*16, false);
    Surface<half> s_tensor(64, false);
    Surface<half> b_tensor(64, false);
    Surface<half> dual_x_tensor(4*64*16*16, false);
    Surface<half> dual_s_tensor(64, false);
    Surface<half> dual_b_tensor(64, false);
    Surface<half> relu_output_tensor(4*64*16*16, false);
    Surface<half> w_tensor(32*64*1*1, false);
    Surface<half> y_tensor(4*32*16*16, false);
    Surface<float> sum_tensor(32, false);
    Surface<float> sq_sum_tensor(32, false);
    
    auto workspace = graph.tensor(fe::graph::Tensor("workspace"));
    Surface<int8_t> workspace_tensor(graph.get_workspace_size(), false);
    
    std::unordered_map<std::shared_ptr<fe::graph::Tensor>, void*> variant_pack = {
        {X, x_tensor.devPtr}
        , {S, s_tensor.devPtr}
        , {B, b_tensor.devPtr}
        , {DUAL_X, x_tensor.devPtr}
        , {DUAL_S, s_tensor.devPtr}
        , {DUAL_B, b_tensor.devPtr}
        , {relu_output, relu_output_tensor.devPtr}
        , {W, w_tensor.devPtr}
        , {Y, y_tensor.devPtr}
        , {SUM, sum_tensor.devPtr}
        , {SQ_SUM, sq_sum_tensor.devPtr}
        , {workspace, workspace_tensor.devPtr}
    };
    REQUIRE(fe::error_t::OK == graph.execute(handle, variant_pack));
    cudnnDestroy(handle);
}
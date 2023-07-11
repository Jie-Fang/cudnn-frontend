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

    fe::graph::BN_finalize_attributes::Inputs inputs;
    auto sum = graph.tensor(fe::graph::Tensor_attributes("sum").set_dim({1,32,1,1}));
    sum->generateStrides(CUDNN_TENSOR_NHWC);
    auto sq_sum = graph.tensor(fe::graph::Tensor_attributes("sq_sum"));
    auto mean = graph.tensor(fe::graph::Tensor_attributes("mean"));
    auto inv_variance = graph.tensor(fe::graph::Tensor_attributes("inv_variance"));
    auto prev_running_mean = graph.tensor(fe::graph::Tensor_attributes("prev_running_mean"));
    auto prev_running_var = graph.tensor(fe::graph::Tensor_attributes("prev_running_var"));
    auto scale = graph.tensor(fe::graph::Tensor_attributes("scale"));
    auto bias = graph.tensor(fe::graph::Tensor_attributes("bias"));
    auto epsilon = graph.tensor(fe::graph::Tensor_attributes("epsilon").set_is_pass_by_value(true));
    auto exp_avg = graph.tensor(fe::graph::Tensor_attributes("exp_avg").set_is_pass_by_value(true));
    auto accum_count = graph.tensor(fe::graph::Tensor_attributes("accum_count").set_is_pass_by_value(true).set_data_type(fe::DataType_t::INT64));

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

    auto bn_finalize_options = fe::graph::BN_finalize_attributes("bn_finalize");
    auto [eq_scale, eq_bias, next_running_mean, next_running_var] = graph.bn_finalize(inputs, bn_finalize_options);

    #if (CUDNN_VERSION < 8400)
        SKIP("BNFinalize requires cudnn 8.4 and up");
    #endif

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    REQUIRE(graph.build_operation_graph(handle).is_good());
    auto plans = graph.get_execution_plan_list(fe::HeurMode_t::HEUR_MODE_FALLBACK)
                    .build_plans(handle);

    REQUIRE(graph.set_execution_plans(plans).is_good());

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
    float EPS_scalar = 0.001f;
    float EXP_AVG_scalar = 0.001f;
    int64_t nhw = 64;

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
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
    };
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    cudnnDestroy(handle);
}

TEST_CASE("SGBN Add Relu Graph", "[batchnorm][graph]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph("SGBN_Add_Relu");
    graph.set_io_data_type(fe::DataType_t::HALF)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);
    
    fe::graph::Batchnorm_attributes::Inputs inputs;
    auto X = graph.tensor(fe::graph::Tensor_attributes("X").set_dim({4,32,16,16}));
    X->generateStrides(CUDNN_TENSOR_NHWC);
    auto prev_running_mean = graph.tensor(fe::graph::Tensor_attributes("prev_running_mean").set_data_type(fe::DataType_t::FLOAT));
    auto prev_running_var = graph.tensor(fe::graph::Tensor_attributes("prev_running_var").set_data_type(fe::DataType_t::FLOAT));
    auto scale = graph.tensor(fe::graph::Tensor_attributes("scale").set_data_type(fe::DataType_t::FLOAT));
    auto bias = graph.tensor(fe::graph::Tensor_attributes("bias").set_data_type(fe::DataType_t::FLOAT));

    inputs.X = X;
    inputs.SCALE = scale;
    inputs.BIAS = bias;
    inputs.PREV_RUNNING_MEAN = prev_running_mean;
    inputs.PREV_RUNNING_VAR = prev_running_var;

    auto epsilon = graph.tensor(fe::graph::Tensor_attributes("epsilon").set_data_type(fe::DataType_t::FLOAT));
    auto momentum = graph.tensor(fe::graph::Tensor_attributes("momentum").set_data_type(fe::DataType_t::FLOAT));
    
    auto batchnorm_options = fe::graph::Batchnorm_attributes("batchnorm").set_forward_phase(fe::NormFwdPhase_t::TRAINING).set_epsilon(epsilon).set_momentum(momentum);
    auto [bn_output, mean, inv_variance, next_running_mean, next_running_var] = graph.batchnorm(inputs, batchnorm_options);
    bn_output->set_is_virtual(true);

    mean->set_data_type(fe::DataType_t::FLOAT);
    inv_variance->set_data_type(fe::DataType_t::FLOAT);
    next_running_mean->set_data_type(fe::DataType_t::FLOAT);
    next_running_var->set_data_type(fe::DataType_t::FLOAT);
    
    auto A = graph.tensor(fe::graph::Tensor_attributes("A").set_dim({4,32,16,16}).set_data_type(fe::DataType_t::HALF));
    A->generateStrides(CUDNN_TENSOR_NHWC);
    auto add_options = fe::graph::Pointwise_attributes("add").set_mode(fe::PointwiseMode_t::ADD);
    auto add_output = graph.pointwise(bn_output, A, add_options);
    add_output->set_is_virtual(true);

    auto relu_options = fe::graph::Pointwise_attributes("relu").set_mode(fe::PointwiseMode_t::RELU_FWD);
    auto Y = graph.pointwise(add_output, relu_options);

    #if (CUDNN_VERSION < 8700)
        SKIP("single GPU BN is not supported in cudnn versions prior to 8.7");
    #endif

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    
    REQUIRE(graph.build_operation_graph(handle).is_good());
    auto plans = graph.get_execution_plan_list(fe::HeurMode_t::HEUR_MODE_FALLBACK)
                    .build_plans(handle);

    REQUIRE(graph.set_execution_plans(plans).is_good());

    Surface<half> X_tensor(4*32*16*16, false);
    Surface<float> Mean_tensor(32, false);
    Surface<float> Var_tensor(32, false);
    Surface<float> Previous_running_mean_tensor(32, false);
    Surface<float> Previous_running_var_tensor(32, false);
    Surface<float> Next_running_mean_tensor(32, false);
    Surface<float> Next_running_var_tensor(32, false);
    Surface<float> Scale_tensor(32, false);
    Surface<float> Bias_tensor(32, false);
    float epsilon_cpu = 1e-05f;
    float momentum_cpu = 1e-01f;
    Surface<half> A_tensor(4*32*16*16, false);
    Surface<half> Y_tensor(4*32*16*16, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr}
        , {mean, Mean_tensor.devPtr}
        , {inv_variance, Var_tensor.devPtr}
        , {prev_running_mean, Previous_running_mean_tensor.devPtr}
        , {prev_running_var, Previous_running_var_tensor.devPtr}
        , {next_running_mean, Next_running_mean_tensor.devPtr}
        , {next_running_var, Next_running_var_tensor.devPtr}
        , {scale, Scale_tensor.devPtr}
        , {bias, Bias_tensor.devPtr}
        , {epsilon, &epsilon_cpu}
        , {momentum, &momentum_cpu}
        , {A, A_tensor.devPtr}
        , {Y, Y_tensor.devPtr}
    };
    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    cudnnDestroy(handle);
}

TEST_CASE("DBN Add Relu Graph", "[BN][graph][backward]") {
    namespace fe = cudnn_frontend;
    fe::graph::Graph graph("DBN_Add_Relu");
    graph.set_io_data_type(fe::DataType_t::HALF)
         .set_intermediate_data_type(fe::DataType_t::FLOAT)
         .set_compute_data_type(fe::DataType_t::FLOAT);
    
    auto DY = graph.tensor(fe::graph::Tensor_attributes("DY").set_dim({4,32,16,16}));
    DY->generateStrides(CUDNN_TENSOR_NHWC);

    auto input_mask = graph.tensor(fe::graph::Tensor_attributes("Mask").set_dim({4,32,16,16}).set_data_type(fe::DataType_t::BOOLEAN));
    input_mask->generateStrides(CUDNN_TENSOR_NHWC);

    auto mul_options = fe::graph::Pointwise_attributes("drelu").set_mode(fe::PointwiseMode_t::MUL);
    auto DX_drelu = graph.pointwise(DY, input_mask, mul_options);

    // NOTE: Toggle DADD output by toggling DX_DRELU virtualness
    bool is_dx_drelu_virtual = true;
    DX_drelu->set_is_virtual(is_dx_drelu_virtual);

    fe::graph::DBN_attributes::Inputs inputs;
    auto X = graph.tensor(fe::graph::Tensor_attributes("X").set_dim({4,32,16,16}));
    X->generateStrides(CUDNN_TENSOR_NHWC);
    
    auto scale = graph.tensor(fe::graph::Tensor_attributes("scale").set_data_type(fe::DataType_t::FLOAT));
    auto mean = graph.tensor(fe::graph::Tensor_attributes("mean").set_data_type(fe::DataType_t::FLOAT));
    auto inv_variance = graph.tensor(fe::graph::Tensor_attributes("inv_variance").set_data_type(fe::DataType_t::FLOAT));

    inputs.X = X;
    inputs.DY = DX_drelu;
    inputs.SCALE = scale;
    inputs.MEAN = mean;
    inputs.INV_VARIANCE = inv_variance;

    auto epsilon = graph.tensor(fe::graph::Tensor_attributes("epsilon").set_dim({1,1,1,1}).set_stride({1,1,1,1}).set_data_type(fe::DataType_t::FLOAT));

    auto DBN_options = fe::graph::DBN_attributes("DBN").set_epsilon(epsilon);
    auto [DX, dscale, dbias] = graph.batchnorm_backward(inputs, DBN_options);
    DX->set_is_virtual(false);
    dscale->set_is_virtual(false);
    dbias->set_is_virtual(false);

    #if (CUDNN_VERSION < 8900)
        SKIP("single GPU BN is not supported in cudnn versions prior to 8.7");
    #endif

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    
    REQUIRE(graph.build_operation_graph(handle).is_good());
    auto plans = graph.get_execution_plan_list(fe::HeurMode_t::HEUR_MODE_FALLBACK)
                    .build_plans(handle);

    REQUIRE(graph.set_execution_plans(plans).is_good());

    Surface<half> X_tensor(4*32*16*16, false);
    Surface<int8_t> Mask_tensor(4*32*16*16 / 8, false);
    Surface<half> DY_tensor(4*32*16*16, false);
    Surface<float> Mean_tensor(32, false);
    Surface<float> Inv_variance_tensor(32, false);
    Surface<float> Scale_tensor(32, false);
    Surface<float> Dscale_tensor(32, false);
    Surface<float> Dbias_tensor(32, false);
    Surface<half> DX_tensor(4*32*16*16, false);

    Surface<int8_t> workspace(graph.get_workspace_size(), false);
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {X, X_tensor.devPtr}
        , {input_mask, Mask_tensor.devPtr}
        , {DY, DY_tensor.devPtr}
        , {scale, DX_tensor.devPtr}
        , {mean, Mean_tensor.devPtr}
        , {inv_variance, Inv_variance_tensor.devPtr}
        , {scale, Scale_tensor.devPtr}
        , {dscale, Dscale_tensor.devPtr}
        , {dbias, Dbias_tensor.devPtr}
        , {DX, DX_tensor.devPtr}
    };

    // If is_dx_drelu_virtual, DADD output required
    Surface<half> DADD_tensor(4*32*16*16, false);
    if(false == is_dx_drelu_virtual) {
        variant_pack[DX_drelu] = DADD_tensor.devPtr;
    }

    REQUIRE(graph.execute(handle, variant_pack, workspace.devPtr).is_good());

    cudnnDestroy(handle);
}

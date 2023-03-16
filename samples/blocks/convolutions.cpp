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

#include <graphs/cudnn_frontend_node_batchnorm.h>
#include <graphs/cudnn_frontend_node_convolution.h>
#include <graphs/cudnn_frontend_node_pointwise.h>
#include <graphs/cudnn_frontend_node_reduction.h>
#include <graphs/cudnn_frontend_node_matmul.h>

#include "convolution_fp8_node.h"
#include "convolution_pointwise_node.h"

#include "convolutions.h"

#include "../helpers.h"

void
run_matmul_node() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::MatMulNode matmul_node{"matmul_node"};
    
    auto props = std::make_shared<cudnn_frontend::matmul_properties>("matmul_prop");
    props->set_tensor_data_type(CUDNN_DATA_HALF);
    props->set_compute_type(CUDNN_DATA_FLOAT);
    props->set_port_names({
        {cudnn_frontend::matmul_properties::PORTS::X, "tensor0"} 
        , {cudnn_frontend::matmul_properties::PORTS::W, "tensor1"}
        , {cudnn_frontend::matmul_properties::PORTS::Y, "tensor2"}
    });
    matmul_node.set_properties("matmul_node", props);

    cudnn_frontend::tensor_properties tensor0{"tensor0"};
    tensor0.set_dim({1, 32, 16});
    matmul_node.add_tensor("tensor0", tensor0);

    cudnn_frontend::tensor_properties tensor1{"tensor1"};
    tensor1.set_dim({1, 16, 32});
    matmul_node.add_tensor("tensor1", tensor1);

    cudnn_frontend::tensor_properties tensor2{"tensor2"};
    tensor2.set_dim({1, 32, 32});
    matmul_node.add_tensor("tensor2", tensor2);

    REQUIRE(cudnn_frontend::error_t::OK == matmul_node.build());

    Surface<half> x_tensor(matmul_node.tensor_props.at("tensor0")->get_tensor_size(), false);
    Surface<half> w_tensor(matmul_node.tensor_props.at("tensor1")->get_tensor_size(), false);
    Surface<half> y_tensor(matmul_node.tensor_props.at("tensor2")->get_tensor_size(), false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"tensor0", x_tensor.devPtr}
        , {"tensor1", w_tensor.devPtr}
        , {"tensor2", y_tensor.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == matmul_node.execute(variant_pack));
}


void
run_convolution_node() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::ConvolutionNode convolution_node{"conv_node"};
    
    auto props = std::make_shared<cudnn_frontend::convolution_properties>("conv_prop");
    props->set_padding({1, 1});
    props->set_stride({1, 1});
    props->set_dilation({1, 1});
    props->set_tensor_data_type(CUDNN_DATA_HALF);
    props->set_compute_type(CUDNN_DATA_FLOAT);
    props->set_port_names({
        {cudnn_frontend::convolution_properties::PORTS::X, "tensor0"} 
        , {cudnn_frontend::convolution_properties::PORTS::W, "tensor1"}
        , {cudnn_frontend::convolution_properties::PORTS::Y, "tensor2"}
    });
    convolution_node.set_properties("conv_node", props);

    cudnn_frontend::tensor_properties tensor0{"tensor0"};
    tensor0.set_dim({4, 32, 16, 16});
    convolution_node.add_tensor("tensor0", tensor0);

    cudnn_frontend::tensor_properties tensor1{"tensor1"};
    tensor1.set_dim({64, 32, 3, 3});
    convolution_node.add_tensor("tensor1", tensor1);

    cudnn_frontend::tensor_properties tensor2{"tensor2"};
    tensor2.set_dim({4, 64, 16, 16});
    convolution_node.add_tensor("tensor2", tensor2);

    REQUIRE(cudnn_frontend::error_t::OK == convolution_node.build());

    Surface<half> x_tensor(convolution_node.tensor_props.at("tensor0")->get_tensor_size(), false);
    Surface<half> w_tensor(convolution_node.tensor_props.at("tensor1")->get_tensor_size(), false);
    Surface<half> y_tensor(convolution_node.tensor_props.at("tensor2")->get_tensor_size(), false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"tensor0", x_tensor.devPtr}
        , {"tensor1", w_tensor.devPtr}
        , {"tensor2", y_tensor.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == convolution_node.execute(variant_pack));
}

void
run_pointwise_node() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::PointwiseNode pointwise_node{"pointwise_node"};

    auto props = std::make_shared<cudnn_frontend::pointwise_properties>("pointwise_prop");
    props->set_mode(cudnn_frontend::PointwiseMode_t::ADD);
    props->set_tensor_data_type(CUDNN_DATA_HALF);
    props->set_compute_type(CUDNN_DATA_FLOAT);
    props->set_port_names({
        {cudnn_frontend::pointwise_properties::PORTS::X, "tensor0"} 
        , {cudnn_frontend::pointwise_properties::PORTS::B, "tensor1"}
        , {cudnn_frontend::pointwise_properties::PORTS::Y, "tensor2"}
    });
    pointwise_node.props = props;
    
    cudnn_frontend::tensor_properties tensor0{"tensor0"};
    tensor0.set_dim({4, 32, 16, 16});
    pointwise_node.add_tensor("tensor0", tensor0);

    cudnn_frontend::tensor_properties tensor1{"tensor1"};
    tensor1.set_dim({1, 32, 1, 1});
    pointwise_node.add_tensor("tensor1", tensor1);

    cudnn_frontend::tensor_properties tensor2{"tensor2"};
    tensor2.set_dim({4, 32, 16, 16});
    pointwise_node.add_tensor("tensor2", tensor2);

    REQUIRE(cudnn_frontend::error_t::OK == pointwise_node.build());

    Surface<half> x_tensor(pointwise_node.tensor_props.at("tensor0")->get_tensor_size(), false);
    Surface<half> b_tensor(pointwise_node.tensor_props.at("tensor1")->get_tensor_size(), false);
    Surface<half> y_tensor(pointwise_node.tensor_props.at("tensor2")->get_tensor_size(), false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"tensor0", x_tensor.devPtr}
        , {"tensor1", b_tensor.devPtr}
        , {"tensor2", y_tensor.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == pointwise_node.execute(variant_pack));
}

void
run_reduction_node() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::ReductionNode reduction_node{"reduction_node"};

    auto props = std::make_shared<cudnn_frontend::reduction_properties>("reduction_prop");
    props->set_mode(CUDNN_REDUCE_TENSOR_ADD);
    props->set_tensor_data_type(CUDNN_DATA_HALF);
    props->set_compute_type(CUDNN_DATA_FLOAT);
    props->set_port_names({
        {cudnn_frontend::reduction_properties::PORTS::X, "tensor0"}
        , {cudnn_frontend::reduction_properties::PORTS::Y, "tensor1"}
    });
    reduction_node.set_properties("reduction_node", props);
    
    cudnn_frontend::tensor_properties tensor0{"tensor0"};
    tensor0.set_dim({4, 32, 16, 16});
    reduction_node.add_tensor("tensor0", tensor0);

    cudnn_frontend::tensor_properties tensor1{"tensor1"};
    tensor1.set_dim({1, 32, 1, 1});
    tensor1.set_data_type(CUDNN_DATA_FLOAT);
    reduction_node.add_tensor("tensor1", tensor1);

    REQUIRE(cudnn_frontend::error_t::OK == reduction_node.build());

    Surface<half> x_tensor(reduction_node.tensor_props.at("tensor0")->get_tensor_size(), false);
    Surface<float> y_tensor(reduction_node.tensor_props.at("tensor1")->get_tensor_size(), false);
    std::unordered_map<std::string, void*> variant_pack = {
        {"tensor0", x_tensor.devPtr}
        , {"tensor1", y_tensor.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == reduction_node.execute(variant_pack));
}

void run_batchnorm_node() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::BatchNormNode batchnorm_node{"batchnorm_node"};

    auto props = std::make_shared<cudnn_frontend::batchnorm_properties>("batchnorm_prop");
    props->set_tensor_data_type(CUDNN_DATA_HALF);
    props->set_compute_type(CUDNN_DATA_FLOAT);
    props->set_port_names({
        {cudnn_frontend::batchnorm_properties::PORTS::X, "input"} 
        , {cudnn_frontend::batchnorm_properties::PORTS::Mean, "mean"}
        , {cudnn_frontend::batchnorm_properties::PORTS::Var, "variance"}
        , {cudnn_frontend::batchnorm_properties::PORTS::Scale, "scale"}
        , {cudnn_frontend::batchnorm_properties::PORTS::Bias, "bias"}
        , {cudnn_frontend::batchnorm_properties::PORTS::Previous_running_mean, "in_running_mean"}
        , {cudnn_frontend::batchnorm_properties::PORTS::Previous_running_var, "in_running_variance"}
        , {cudnn_frontend::batchnorm_properties::PORTS::Next_running_mean, "out_running_mean"}
        , {cudnn_frontend::batchnorm_properties::PORTS::Next_running_var, "out_running_variance"}
        , {cudnn_frontend::batchnorm_properties::PORTS::Y, "output"}
        , {cudnn_frontend::batchnorm_properties::PORTS::EPS, "epsilon"}
        , {cudnn_frontend::batchnorm_properties::PORTS::EXP_AVG, "exp_avg"}
    });
    batchnorm_node.set_properties("batchnorm_node", props);
    
    cudnn_frontend::tensor_properties input{"input"};
    input.set_dim({4, 32, 16, 16});
    batchnorm_node.add_tensor("input", input);

    cudnn_frontend::tensor_properties mean{"mean"};
    mean.set_dim({1, 32, 1, 1});
    mean.set_data_type(CUDNN_DATA_FLOAT);
    batchnorm_node.add_tensor("mean", mean);
    
    cudnn_frontend::tensor_properties variance{"variance"};
    variance.set_dim({1, 32, 1, 1});
    variance.set_data_type(CUDNN_DATA_FLOAT);
    batchnorm_node.add_tensor("variance", variance);

    cudnn_frontend::tensor_properties in_running_mean{"in_running_mean"};
    in_running_mean.set_dim({1, 32, 1, 1});
    in_running_mean.set_data_type(CUDNN_DATA_FLOAT);
    batchnorm_node.add_tensor("in_running_mean", in_running_mean);
    
    cudnn_frontend::tensor_properties in_running_variance{"in_running_variance"};
    in_running_variance.set_dim({1, 32, 1, 1});
    in_running_variance.set_data_type(CUDNN_DATA_FLOAT);
    batchnorm_node.add_tensor("in_running_variance", in_running_variance);

    cudnn_frontend::tensor_properties out_running_mean{"out_running_mean"};
    out_running_mean.set_dim({1, 32, 1, 1});
    out_running_mean.set_data_type(CUDNN_DATA_FLOAT);
    batchnorm_node.add_tensor("out_running_mean", out_running_mean);
    
    cudnn_frontend::tensor_properties out_running_variance{"out_running_variance"};
    out_running_variance.set_dim({1, 32, 1, 1});
    out_running_variance.set_data_type(CUDNN_DATA_FLOAT);
    batchnorm_node.add_tensor("out_running_variance", out_running_variance);
    
    cudnn_frontend::tensor_properties scale{"scale"};
    scale.set_dim({1, 32, 1, 1});
    scale.set_data_type(CUDNN_DATA_FLOAT);
    batchnorm_node.add_tensor("scale", scale);
    
    cudnn_frontend::tensor_properties bias{"bias"};
    bias.set_dim({1, 32, 1, 1});
    bias.set_data_type(CUDNN_DATA_FLOAT);
    batchnorm_node.add_tensor("bias", bias);

    cudnn_frontend::tensor_properties epsilon{"epsilon"};
    epsilon.set_dim({1, 1, 1, 1});
    epsilon.set_is_pass_by_value(true);
    epsilon.set_data_type(CUDNN_DATA_FLOAT);
    batchnorm_node.add_tensor("epsilon", epsilon);
    
    cudnn_frontend::tensor_properties exp_avg{"exp_avg"};
    exp_avg.set_dim({1, 1, 1, 1});
    exp_avg.set_is_pass_by_value(true);
    exp_avg.set_data_type(CUDNN_DATA_FLOAT);
    batchnorm_node.add_tensor("exp_avg", exp_avg);
    
    cudnn_frontend::tensor_properties output{"output"};
    output.set_dim({4, 32, 16, 16});
    batchnorm_node.add_tensor("output", output);

    #if (CUDNN_VERSION >= 8700)
        REQUIRE(cudnn_frontend::error_t::OK == batchnorm_node.build());
    #elif (CUDNN_VERSION >= 8500)
        SKIP("Only multi-GPU batch norm is supported in cudnn versions prior to 8.7 and above 8.5.");
    #else
        SKIP("Batch Norm is not supported in cudnn versions prior to 8.5.");
    #endif

    int64_t workspace_size = 0;
    batchnorm_node.get_workspace_size(workspace_size);
    Surface<int8_t> workspace(workspace_size, false);

    Surface<half> x_tensor(batchnorm_node.tensor_props.at("input")->get_tensor_size(), false);
    Surface<float> b_tensor(batchnorm_node.tensor_props.at("bias")->get_tensor_size(), false);
    Surface<float> s_tensor(batchnorm_node.tensor_props.at("scale")->get_tensor_size(), false);
    Surface<float> m_tensor(batchnorm_node.tensor_props.at("mean")->get_tensor_size(), false);
    Surface<float> v_tensor(batchnorm_node.tensor_props.at("variance")->get_tensor_size(), false);
    Surface<float> eps_tensor(batchnorm_node.tensor_props.at("epsilon")->get_tensor_size(), false);
    Surface<float> exp_avg_tensor(batchnorm_node.tensor_props.at("exp_avg")->get_tensor_size(), false);
    Surface<float> prm_tensor(batchnorm_node.tensor_props.at("in_running_mean")->get_tensor_size(), false);
    Surface<float> prv_tensor(batchnorm_node.tensor_props.at("in_running_variance")->get_tensor_size(), false);
    Surface<float> nrm_tensor(batchnorm_node.tensor_props.at("out_running_mean")->get_tensor_size(), false);
    Surface<float> nrv_tensor(batchnorm_node.tensor_props.at("out_running_variance")->get_tensor_size(), false);
    Surface<half> y_tensor(batchnorm_node.tensor_props.at("output")->get_tensor_size(), false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"input", x_tensor.devPtr}
        , {"bias", b_tensor.devPtr}
        , {"scale", s_tensor.devPtr}
        , {"mean", m_tensor.devPtr}
        , {"variance", v_tensor.devPtr}
        , {"epsilon", &(eps_tensor.devPtr)}
        , {"exp_avg", &(exp_avg_tensor.devPtr)}
        , {"in_running_mean", prm_tensor.devPtr}
        , {"in_running_variance", prv_tensor.devPtr}
        , {"out_running_mean", nrm_tensor.devPtr}
        , {"out_running_variance", nrv_tensor.devPtr}
        , {"output", y_tensor.devPtr}
        , {"workspace", workspace.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == batchnorm_node.execute(variant_pack));
}

void
run_convolution_fp8_node() {
#if (CUDNN_VERSION >= 8700)
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::ConvolutionFP8Node convolution_fp8_node{"conv_fp8"};
    
    auto props = std::make_shared<cudnn_frontend::convolution_properties>("conv_prop");
    props->set_padding({1, 1});
    props->set_stride({1, 1});
    props->set_dilation({1, 1});
    // TODO: remove setting tensor data type in operation properties.
    props->set_tensor_data_type(CUDNN_DATA_FP8_E4M3);
    props->set_compute_type(CUDNN_DATA_FLOAT);
    props->set_port_names({
        {cudnn_frontend::convolution_properties::PORTS::X, "tensor0"} 
        , {cudnn_frontend::convolution_properties::PORTS::W, "tensor1"}
        , {cudnn_frontend::convolution_properties::PORTS::Y, "tensor2"}
    });
    convolution_fp8_node.set_properties("conv", props);

    cudnn_frontend::tensor_properties tensor0{"tensor0"};
    tensor0.set_dim({4, 32, 16, 16});
    convolution_fp8_node.add_tensor("tensor0", tensor0);

    cudnn_frontend::tensor_properties tensor1{"tensor1"};
    tensor1.set_dim({64, 32, 3, 3});
    convolution_fp8_node.add_tensor("tensor1", tensor1);

    cudnn_frontend::tensor_properties tensor2{"tensor2"};
    tensor2.set_dim({4, 64, 16, 16});
    tensor2.set_data_type(CUDNN_DATA_FLOAT);
    convolution_fp8_node.add_tensor("tensor2", tensor2);

    auto X_DQ_props = std::make_shared<cudnn_frontend::pointwise_properties>("x_dq_prop");
    X_DQ_props->set_mode(cudnn_frontend::PointwiseMode_t::MUL);
    // TODO: remove setting tensor data type in operation properties.
    X_DQ_props->set_tensor_data_type(CUDNN_DATA_FLOAT);
    X_DQ_props->set_compute_type(CUDNN_DATA_FLOAT);
    X_DQ_props->set_port_names({
        {cudnn_frontend::pointwise_properties::PORTS::X, "tensor2"} 
        , {cudnn_frontend::pointwise_properties::PORTS::B, "tensor3"}
        , {cudnn_frontend::pointwise_properties::PORTS::Y, "tensor4"}
    });
    convolution_fp8_node.set_properties("X_DQ", X_DQ_props);

    cudnn_frontend::tensor_properties tensor3{"tensor3"};
    tensor3.set_dim({1, 1, 1, 1});
    convolution_fp8_node.add_tensor("tensor3", tensor3);

    cudnn_frontend::tensor_properties tensor4{"tensor4"};
    tensor4.set_dim({4, 64, 16, 16});
    convolution_fp8_node.add_tensor("tensor4", tensor4);

    auto W_DQ_props = std::make_shared<cudnn_frontend::pointwise_properties>("w_dq_props");
    W_DQ_props->set_mode(cudnn_frontend::PointwiseMode_t::MUL);
    W_DQ_props->set_tensor_data_type(CUDNN_DATA_FLOAT);
    W_DQ_props->set_compute_type(CUDNN_DATA_FLOAT);
    W_DQ_props->set_port_names({
        {cudnn_frontend::pointwise_properties::PORTS::X, "tensor4"} 
        , {cudnn_frontend::pointwise_properties::PORTS::B, "tensor5"}
        , {cudnn_frontend::pointwise_properties::PORTS::Y, "tensor6"}
    });
    convolution_fp8_node.set_properties("W_DQ", W_DQ_props);

    cudnn_frontend::tensor_properties tensor5{"tensor5"};
    tensor5.set_dim({1, 1, 1, 1});
    convolution_fp8_node.add_tensor("tensor5", tensor5);

    cudnn_frontend::tensor_properties tensor6{"tensor6"};
    tensor6.set_dim({4, 64, 16, 16});
    convolution_fp8_node.add_tensor("tensor6", tensor6);

    auto Y_Q_props = std::make_shared<cudnn_frontend::pointwise_properties>("y_q_prop");
    Y_Q_props->set_mode(cudnn_frontend::PointwiseMode_t::MUL);
    // TODO: remove setting tensor data type in operation properties.
    Y_Q_props->set_tensor_data_type(CUDNN_DATA_FLOAT);
    Y_Q_props->set_compute_type(CUDNN_DATA_FLOAT);
    Y_Q_props->set_port_names({
        {cudnn_frontend::pointwise_properties::PORTS::X, "tensor6"} 
        , {cudnn_frontend::pointwise_properties::PORTS::B, "tensor7"}
        , {cudnn_frontend::pointwise_properties::PORTS::Y, "tensor8"}
    });
    convolution_fp8_node.set_properties("Y_Q", Y_Q_props);

    cudnn_frontend::tensor_properties tensor7{"tensor7"};
    tensor7.set_dim({1, 1, 1, 1});
    convolution_fp8_node.add_tensor("tensor7", tensor7);

    cudnn_frontend::tensor_properties tensor8{"tensor8"};
    tensor8.set_dim({4, 64, 16, 16});
    tensor8.set_data_type(CUDNN_DATA_FP8_E4M3);
    convolution_fp8_node.add_tensor("tensor8", tensor8);

    auto amax_props = std::make_shared<cudnn_frontend::reduction_properties>("amax_prop");
    amax_props->set_mode(CUDNN_REDUCE_TENSOR_AMAX);
    // TODO: remove setting tensor data type in operation properties.
    amax_props->set_tensor_data_type(CUDNN_DATA_FLOAT);
    amax_props->set_compute_type(CUDNN_DATA_FLOAT);
    amax_props->set_port_names({
        {cudnn_frontend::reduction_properties::PORTS::X, "tensor6"} 
        , {cudnn_frontend::reduction_properties::PORTS::Y, "tensor9"}
    });
    convolution_fp8_node.set_properties("amax", amax_props);

    cudnn_frontend::tensor_properties tensor9{"tensor9"};
    tensor9.set_dim({1, 1, 1, 1});
    convolution_fp8_node.add_tensor("tensor9", tensor9);

    if (check_device_arch_newer_than("hopper")) {
        REQUIRE(cudnn_frontend::error_t::OK == convolution_fp8_node.build());
    }
    else {
        SKIP("Architextures below hopper do not support fp8.");
    }

    Surface<float> x_dq_tensor(convolution_fp8_node.tensor_props.at("tensor3")->get_tensor_size(), false);
    Surface<float> w_dq_tensor(convolution_fp8_node.tensor_props.at("tensor5")->get_tensor_size(), false);
    Surface<int8_t> x_tensor(convolution_fp8_node.tensor_props.at("tensor0")->get_tensor_size(), false);
    Surface<int8_t> w_tensor(convolution_fp8_node.tensor_props.at("tensor1")->get_tensor_size(), false);
    Surface<int8_t> y_tensor(convolution_fp8_node.tensor_props.at("tensor8")->get_tensor_size(), false);
    Surface<float> y_q_tensor(convolution_fp8_node.tensor_props.at("tensor7")->get_tensor_size(), false);
    Surface<float> amax_tensor(convolution_fp8_node.tensor_props.at("tensor9")->get_tensor_size(), false);
    
    std::unordered_map<std::string, void*> variant_pack = {
        {"tensor3", x_dq_tensor.devPtr}
        , {"tensor5", w_dq_tensor.devPtr}
        , {"tensor0", x_tensor.devPtr}
        , {"tensor1", w_tensor.devPtr}
        , {"tensor8", y_tensor.devPtr}
        , {"tensor7", y_q_tensor.devPtr}
        , {"tensor9", amax_tensor.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == convolution_fp8_node.execute(variant_pack));
#endif
}

void
run_convolution_pointwise_node() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::ConvolutionPointwiseNode convolution_pointwise_node{"convolution_pointwise"};

    auto conv_props = std::make_shared<cudnn_frontend::convolution_properties>("conv_prop");
    conv_props->set_padding({1, 1});
    conv_props->set_stride({1, 1});
    conv_props->set_dilation({1, 1});
    // TODO: remove setting tensor data type in operation properties.
    conv_props->set_tensor_data_type(CUDNN_DATA_HALF);
    conv_props->set_compute_type(CUDNN_DATA_FLOAT);
    conv_props->set_port_names({
        {cudnn_frontend::convolution_properties::PORTS::X, "tensor0"} 
        , {cudnn_frontend::convolution_properties::PORTS::W, "tensor1"}
        , {cudnn_frontend::convolution_properties::PORTS::Y, "tensor2"}
    });
    convolution_pointwise_node.set_properties("conv", conv_props);

    cudnn_frontend::tensor_properties tensor0{"tensor0"};
    tensor0.set_dim({4, 32, 16, 16});
    convolution_pointwise_node.add_tensor("tensor0", tensor0);
    
    cudnn_frontend::tensor_properties tensor1{"tensor1"};
    tensor1.set_dim({64, 32, 3, 3});
    convolution_pointwise_node.add_tensor("tensor1", tensor1);
    
    cudnn_frontend::tensor_properties tensor2{"tensor2"};
    tensor2.set_dim({4, 64, 16, 16});
    convolution_pointwise_node.add_tensor("tensor2", tensor2);
    
    auto pointwise_props = std::make_shared<cudnn_frontend::pointwise_properties>("pointwise_prop");
    pointwise_props->set_mode(cudnn_frontend::PointwiseMode_t::ADD);
    pointwise_props->set_tensor_data_type(CUDNN_DATA_HALF);
    pointwise_props->set_compute_type(CUDNN_DATA_FLOAT);
    pointwise_props->set_port_names({
        {cudnn_frontend::pointwise_properties::PORTS::X, "tensor2"} 
        , {cudnn_frontend::pointwise_properties::PORTS::B, "tensor3"}
        , {cudnn_frontend::pointwise_properties::PORTS::Y, "tensor4"}
    });
    convolution_pointwise_node.set_properties("pointwise", pointwise_props);

    cudnn_frontend::tensor_properties tensor3{"tensor3"};
    tensor3.set_dim({1, 64, 1, 1});
    convolution_pointwise_node.add_tensor("tensor3", tensor3);

    cudnn_frontend::tensor_properties tensor4{"tensor4"};
    tensor4.set_dim({4, 64, 16, 16});
    convolution_pointwise_node.add_tensor("tensor4", tensor4);

    REQUIRE(cudnn_frontend::error_t::OK == convolution_pointwise_node.build());
    
    Surface<half> x_tensor(convolution_pointwise_node.tensor_props.at("tensor0")->get_tensor_size(), false);
    Surface<half> w_tensor(convolution_pointwise_node.tensor_props.at("tensor1")->get_tensor_size(), false);
    Surface<half> b_tensor(convolution_pointwise_node.tensor_props.at("tensor3")->get_tensor_size(), false);
    Surface<half> y_tensor(convolution_pointwise_node.tensor_props.at("tensor4")->get_tensor_size(), false);
    std::unordered_map<std::string, void*> variant_pack = {
        {"tensor0", x_tensor.devPtr}
        , {"tensor1", w_tensor.devPtr}
        , {"tensor3", b_tensor.devPtr}
        , {"tensor4", y_tensor.devPtr}
    };
    REQUIRE(cudnn_frontend::error_t::OK == convolution_pointwise_node.execute(variant_pack));
}

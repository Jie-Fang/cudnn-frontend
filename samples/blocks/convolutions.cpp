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

#include <graphs/cudnn_frontend_convolution_block.h>
#include <graphs/cudnn_frontend_pointwise_block.h>
#include <graphs/cudnn_frontend_reduction_block.h>
#include <graphs/cudnn_frontend_matmul_block.h>

#include "convolution_fp8_block.h"
#include "convolution_pointwise_block.h"

#include "convolutions.h"

#include "../helpers.h"

void
run_matmul_block() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::MatMulBlock matmul_block{"matmul_block"};
    
    cudnn_frontend::matmul_node props{"matmul_prop"};
    props.set_tensor_data_type(CUDNN_DATA_HALF);
    props.set_compute_type(CUDNN_DATA_FLOAT);
    props.set_port_names({
        {cudnn_frontend::matmul_node::PORTS::X, "tensor0"} 
        , {cudnn_frontend::matmul_node::PORTS::W, "tensor1"}
        , {cudnn_frontend::matmul_node::PORTS::Y, "tensor2"}
    });
    matmul_block.set_properties("matmul_block", props);

    cudnn_frontend::tensor_properties tensor0{"tensor0"};
    tensor0.set_dim({32, 32, 16});
    matmul_block.add_tensor("tensor0", tensor0);

    cudnn_frontend::tensor_properties tensor1{"tensor1"};
    tensor1.set_dim({32, 16, 32});
    matmul_block.add_tensor("tensor1", tensor1);

    cudnn_frontend::tensor_properties tensor2{"tensor2"};
    tensor2.set_dim({32, 32, 32});
    matmul_block.add_tensor("tensor2", tensor2);

    matmul_block.build(handle);

    Surface<half> x_tensor(matmul_block.tensor_props.at("tensor0")->get_tensor_size(), false);
    Surface<half> w_tensor(matmul_block.tensor_props.at("tensor1")->get_tensor_size(), false);
    Surface<half> y_tensor(matmul_block.tensor_props.at("tensor2")->get_tensor_size(), false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"tensor0", x_tensor.devPtr}
        , {"tensor1", w_tensor.devPtr}
        , {"tensor2", y_tensor.devPtr}
    };
    matmul_block.execute(handle, variant_pack);
}


void
run_convolution_block() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::ConvolutionBlock convolution_block{"conv_block"};
    
    cudnn_frontend::convolution_node props{"conv_prop"};
    props.set_padding({1, 1});
    props.set_stride({1, 1});
    props.set_dilation({1, 1});
    props.set_tensor_data_type(CUDNN_DATA_HALF);
    props.set_compute_type(CUDNN_DATA_FLOAT);
    props.set_port_names({
        {cudnn_frontend::convolution_node::PORTS::X, "tensor0"} 
        , {cudnn_frontend::convolution_node::PORTS::W, "tensor1"}
        , {cudnn_frontend::convolution_node::PORTS::Y, "tensor2"}
    });
    convolution_block.set_properties("conv_block", props);

    cudnn_frontend::tensor_properties tensor0{"tensor0"};
    tensor0.set_dim({4, 32, 16, 16});
    convolution_block.add_tensor("tensor0", tensor0);

    cudnn_frontend::tensor_properties tensor1{"tensor1"};
    tensor1.set_dim({64, 32, 3, 3});
    convolution_block.add_tensor("tensor1", tensor1);

    cudnn_frontend::tensor_properties tensor2{"tensor2"};
    tensor2.set_dim({4, 64, 16, 16});
    convolution_block.add_tensor("tensor2", tensor2);

    convolution_block.build(handle);

    Surface<half> x_tensor(convolution_block.tensor_props.at("tensor0")->get_tensor_size(), false);
    Surface<half> w_tensor(convolution_block.tensor_props.at("tensor1")->get_tensor_size(), false);
    Surface<half> y_tensor(convolution_block.tensor_props.at("tensor2")->get_tensor_size(), false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"tensor0", x_tensor.devPtr}
        , {"tensor1", w_tensor.devPtr}
        , {"tensor2", y_tensor.devPtr}
    };
    convolution_block.execute(handle, variant_pack);
}

void
run_pointwise_block() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::PointwiseBlock pointwise_block{"pointwise_block"};

    cudnn_frontend::pointwise_node props{"pointwise_prop"};
    props.set_mode(CUDNN_POINTWISE_ADD);
    props.set_tensor_data_type(CUDNN_DATA_HALF);
    props.set_compute_type(CUDNN_DATA_FLOAT);
    props.set_port_names({
        {cudnn_frontend::pointwise_node::PORTS::X, "tensor0"} 
        , {cudnn_frontend::pointwise_node::PORTS::B, "tensor1"}
        , {cudnn_frontend::pointwise_node::PORTS::Y, "tensor2"}
    });
    pointwise_block.props = props;
    
    cudnn_frontend::tensor_properties tensor0{"tensor0"};
    tensor0.set_dim({4, 32, 16, 16});
    pointwise_block.add_tensor("tensor0", tensor0);

    cudnn_frontend::tensor_properties tensor1{"tensor1"};
    tensor1.set_dim({1, 32, 1, 1});
    pointwise_block.add_tensor("tensor1", tensor1);

    cudnn_frontend::tensor_properties tensor2{"tensor2"};
    tensor2.set_dim({4, 32, 16, 16});
    pointwise_block.add_tensor("tensor2", tensor2);

    pointwise_block.build(handle);

    Surface<half> x_tensor(pointwise_block.tensor_props.at("tensor0")->get_tensor_size(), false);
    Surface<half> b_tensor(pointwise_block.tensor_props.at("tensor1")->get_tensor_size(), false);
    Surface<half> y_tensor(pointwise_block.tensor_props.at("tensor2")->get_tensor_size(), false);

    std::unordered_map<std::string, void*> variant_pack = {
        {"tensor0", x_tensor.devPtr}
        , {"tensor1", b_tensor.devPtr}
        , {"tensor2", y_tensor.devPtr}
    };
    pointwise_block.execute(handle, variant_pack);
}

void
run_reduction_block() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::ReductionBlock reduction_block{"reduction_block"};

    cudnn_frontend::reduction_node props{"reduction_prop"};
    props.set_mode(CUDNN_REDUCE_TENSOR_ADD);
    props.set_tensor_data_type(CUDNN_DATA_HALF);
    props.set_compute_type(CUDNN_DATA_FLOAT);
    props.set_port_names({
        {cudnn_frontend::reduction_node::PORTS::X, "tensor0"}
        , {cudnn_frontend::reduction_node::PORTS::Y, "tensor1"}
    });
    reduction_block.set_properties("reduction_block", props);
    
    cudnn_frontend::tensor_properties tensor0{"tensor0"};
    tensor0.set_dim({4, 32, 16, 16});
    reduction_block.add_tensor("tensor0", tensor0);

    cudnn_frontend::tensor_properties tensor1{"tensor1"};
    tensor1.set_dim({1, 32, 1, 1});
    tensor1.set_data_type(CUDNN_DATA_FLOAT);
    reduction_block.add_tensor("tensor1", tensor1);

    reduction_block.build(handle);

    Surface<half> x_tensor(reduction_block.tensor_props.at("tensor0")->get_tensor_size(), false);
    Surface<float> y_tensor(reduction_block.tensor_props.at("tensor1")->get_tensor_size(), false);
    std::unordered_map<std::string, void*> variant_pack = {
        {"tensor0", x_tensor.devPtr}
        , {"tensor1", y_tensor.devPtr}
    };
    reduction_block.execute(handle, variant_pack);
}

void
run_convolution_fp8_block() {
#if (CUDNN_VERSION >= 8700)
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::ConvolutionFP8Block convolution_fp8_block{"conv_fp8_block"};
    
    cudnn_frontend::convolution_node props{"conv_prop"};
    props.set_padding({1, 1});
    props.set_stride({1, 1});
    props.set_dilation({1, 1});
    // TODO: remove setting tensor data type in operation properties.
    props.set_tensor_data_type(CUDNN_DATA_FP8_E4M3);
    props.set_compute_type(CUDNN_DATA_FLOAT);
    props.set_port_names({
        {cudnn_frontend::convolution_node::PORTS::X, "tensor0"} 
        , {cudnn_frontend::convolution_node::PORTS::W, "tensor1"}
        , {cudnn_frontend::convolution_node::PORTS::Y, "tensor2"}
    });
    convolution_fp8_block.set_properties("conv_block", props);

    cudnn_frontend::tensor_properties tensor0{"tensor0"};
    tensor0.set_dim({4, 32, 16, 16});
    convolution_fp8_block.add_tensor("tensor0", tensor0);

    cudnn_frontend::tensor_properties tensor1{"tensor1"};
    tensor1.set_dim({64, 32, 3, 3});
    convolution_fp8_block.add_tensor("tensor1", tensor1);

    cudnn_frontend::tensor_properties tensor2{"tensor2"};
    tensor2.set_dim({4, 64, 16, 16});
    tensor2.set_data_type(CUDNN_DATA_FLOAT);
    convolution_fp8_block.add_tensor("tensor2", tensor2);

    cudnn_frontend::pointwise_node X_DQ_props{"x_dq_prop"};
    X_DQ_props.set_mode(CUDNN_POINTWISE_MUL);
    // TODO: remove setting tensor data type in operation properties.
    X_DQ_props.set_tensor_data_type(CUDNN_DATA_FLOAT);
    X_DQ_props.set_compute_type(CUDNN_DATA_FLOAT);
    X_DQ_props.set_port_names({
        {cudnn_frontend::pointwise_node::PORTS::X, "tensor2"} 
        , {cudnn_frontend::pointwise_node::PORTS::B, "tensor3"}
        , {cudnn_frontend::pointwise_node::PORTS::Y, "tensor4"}
    });
    convolution_fp8_block.set_properties("X_DQ_block", X_DQ_props);

    cudnn_frontend::tensor_properties tensor3{"tensor3"};
    tensor3.set_dim({1, 1, 1, 1});
    convolution_fp8_block.add_tensor("tensor3", tensor3);

    cudnn_frontend::tensor_properties tensor4{"tensor4"};
    tensor4.set_dim({4, 64, 16, 16});
    convolution_fp8_block.add_tensor("tensor4", tensor4);

    cudnn_frontend::pointwise_node W_DQ_props{"w_dq_props"};
    W_DQ_props.set_mode(CUDNN_POINTWISE_MUL);
    W_DQ_props.set_tensor_data_type(CUDNN_DATA_FLOAT);
    W_DQ_props.set_compute_type(CUDNN_DATA_FLOAT);
    W_DQ_props.set_port_names({
        {cudnn_frontend::pointwise_node::PORTS::X, "tensor4"} 
        , {cudnn_frontend::pointwise_node::PORTS::B, "tensor5"}
        , {cudnn_frontend::pointwise_node::PORTS::Y, "tensor6"}
    });
    convolution_fp8_block.set_properties("W_DQ_block", W_DQ_props);

    cudnn_frontend::tensor_properties tensor5{"tensor5"};
    tensor5.set_dim({1, 1, 1, 1});
    convolution_fp8_block.add_tensor("tensor5", tensor5);

    cudnn_frontend::tensor_properties tensor6{"tensor6"};
    tensor6.set_dim({4, 64, 16, 16});
    convolution_fp8_block.add_tensor("tensor6", tensor6);

    cudnn_frontend::pointwise_node Y_Q_props{"y_q_prop"};
    Y_Q_props.set_mode(CUDNN_POINTWISE_MUL);
    // TODO: remove setting tensor data type in operation properties.
    Y_Q_props.set_tensor_data_type(CUDNN_DATA_FLOAT);
    Y_Q_props.set_compute_type(CUDNN_DATA_FLOAT);
    Y_Q_props.set_port_names({
        {cudnn_frontend::pointwise_node::PORTS::X, "tensor6"} 
        , {cudnn_frontend::pointwise_node::PORTS::B, "tensor7"}
        , {cudnn_frontend::pointwise_node::PORTS::Y, "tensor8"}
    });
    convolution_fp8_block.set_properties("Y_Q_block", Y_Q_props);

    cudnn_frontend::tensor_properties tensor7{"tensor7"};
    tensor7.set_dim({1, 1, 1, 1});
    convolution_fp8_block.add_tensor("tensor7", tensor7);

    cudnn_frontend::tensor_properties tensor8{"tensor8"};
    tensor8.set_dim({4, 64, 16, 16});
    tensor8.set_data_type(CUDNN_DATA_FP8_E4M3);
    convolution_fp8_block.add_tensor("tensor8", tensor8);

    cudnn_frontend::reduction_node amax_props{"amax_prop"};
    amax_props.set_mode(CUDNN_REDUCE_TENSOR_AMAX);
    // TODO: remove setting tensor data type in operation properties.
    amax_props.set_tensor_data_type(CUDNN_DATA_FLOAT);
    amax_props.set_compute_type(CUDNN_DATA_FLOAT);
    amax_props.set_port_names({
        {cudnn_frontend::reduction_node::PORTS::X, "tensor6"} 
        , {cudnn_frontend::reduction_node::PORTS::Y, "tensor9"}
    });
    convolution_fp8_block.set_properties("amax_block", amax_props);

    cudnn_frontend::tensor_properties tensor9{"tensor9"};
    tensor9.set_dim({1, 1, 1, 1});
    convolution_fp8_block.add_tensor("tensor9", tensor9);

    convolution_fp8_block.build(handle);
    
    Surface<float> x_dq_tensor(convolution_fp8_block.tensor_props.at("tensor3")->get_tensor_size(), false);
    Surface<float> w_dq_tensor(convolution_fp8_block.tensor_props.at("tensor5")->get_tensor_size(), false);
    Surface<int8_t> x_tensor(convolution_fp8_block.tensor_props.at("tensor0")->get_tensor_size(), false);
    Surface<int8_t> w_tensor(convolution_fp8_block.tensor_props.at("tensor1")->get_tensor_size(), false);
    Surface<int8_t> y_tensor(convolution_fp8_block.tensor_props.at("tensor8")->get_tensor_size(), false);
    Surface<float> y_q_tensor(convolution_fp8_block.tensor_props.at("tensor7")->get_tensor_size(), false);
    Surface<float> amax_tensor(convolution_fp8_block.tensor_props.at("tensor9")->get_tensor_size(), false);
    
    std::unordered_map<std::string, void*> variant_pack = {
        {"tensor3", x_dq_tensor.devPtr}
        , {"tensor4", w_dq_tensor.devPtr}
        , {"tensor0", x_tensor.devPtr}
        , {"tensor1", w_tensor.devPtr}
        , {"tensor8", y_tensor.devPtr}
        , {"tensor7", y_q_tensor.devPtr}
        , {"tensor9", amax_tensor.devPtr}
    };
    convolution_fp8_block.execute(handle, variant_pack);
#endif
}

void
run_convolution_pointwise_block() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::ConvolutionPointwiseBlock convolution_pointwise_block{"convolution_pointwise_block"};

    cudnn_frontend::convolution_node conv_props{"conv_prop"};
    conv_props.set_padding({1, 1});
    conv_props.set_stride({1, 1});
    conv_props.set_dilation({1, 1});
    // TODO: remove setting tensor data type in operation properties.
    conv_props.set_tensor_data_type(CUDNN_DATA_HALF);
    conv_props.set_compute_type(CUDNN_DATA_FLOAT);
    conv_props.set_port_names({
        {cudnn_frontend::convolution_node::PORTS::X, "tensor0"} 
        , {cudnn_frontend::convolution_node::PORTS::W, "tensor1"}
        , {cudnn_frontend::convolution_node::PORTS::Y, "tensor2"}
    });
    convolution_pointwise_block.set_properties("conv_block", conv_props);

    cudnn_frontend::tensor_properties tensor0{"tensor0"};
    tensor0.set_dim({4, 32, 16, 16});
    convolution_pointwise_block.add_tensor("tensor0", tensor0);
    
    cudnn_frontend::tensor_properties tensor1{"tensor1"};
    tensor1.set_dim({64, 32, 3, 3});
    convolution_pointwise_block.add_tensor("tensor1", tensor1);
    
    cudnn_frontend::tensor_properties tensor2{"tensor2"};
    tensor2.set_dim({4, 64, 16, 16});
    convolution_pointwise_block.add_tensor("tensor2", tensor2);
    
    cudnn_frontend::pointwise_node pointwise_props {"pointwise_prop"};
    pointwise_props.set_mode(CUDNN_POINTWISE_ADD);
    pointwise_props.set_tensor_data_type(CUDNN_DATA_HALF);
    pointwise_props.set_compute_type(CUDNN_DATA_FLOAT);
    pointwise_props.set_port_names({
        {cudnn_frontend::pointwise_node::PORTS::X, "tensor2"} 
        , {cudnn_frontend::pointwise_node::PORTS::B, "tensor3"}
        , {cudnn_frontend::pointwise_node::PORTS::Y, "tensor4"}
    });
    convolution_pointwise_block.set_properties("pointwise_block", pointwise_props);

    cudnn_frontend::tensor_properties tensor3{"tensor3"};
    tensor3.set_dim({1, 64, 1, 1});
    convolution_pointwise_block.add_tensor("tensor3", tensor3);

    cudnn_frontend::tensor_properties tensor4{"tensor4"};
    tensor4.set_dim({4, 64, 16, 16});
    convolution_pointwise_block.add_tensor("tensor4", tensor4);

    convolution_pointwise_block.build(handle);
    
    Surface<half> x_tensor(convolution_pointwise_block.tensor_props.at("tensor0")->get_tensor_size(), false);
    Surface<half> w_tensor(convolution_pointwise_block.tensor_props.at("tensor1")->get_tensor_size(), false);
    Surface<half> b_tensor(convolution_pointwise_block.tensor_props.at("tensor3")->get_tensor_size(), false);
    Surface<half> y_tensor(convolution_pointwise_block.tensor_props.at("tensor4")->get_tensor_size(), false);
    std::unordered_map<std::string, void*> variant_pack = {
        {"tensor0", x_tensor.devPtr}
        , {"tensor1", w_tensor.devPtr}
        , {"tensor3", b_tensor.devPtr}
        , {"tensor4", y_tensor.devPtr}
    };
    convolution_pointwise_block.execute(handle, variant_pack);
}
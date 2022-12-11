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

#include "convolution_fp8_block.h"
#include "convolution_pointwise_block.h"

#include "convolutions.h"

#include "../helpers.h"

void
run_convolution_block() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::ConvolutionBlock convolution_block;
    
    cudnn_frontend::convolution_node props{""};
    props.set_padding({1, 1});
    props.set_stride({1, 1});
    props.set_dilation({1, 1});
    props.set_tensor_data_type(CUDNN_DATA_HALF);
    props.set_compute_type(CUDNN_DATA_FLOAT);
    convolution_block.props = props;

    convolution_block.update_uids();

    convolution_block.tensor_props.at(cudnn_frontend::convolution_node::PORTS::X).set_dim({4, 32, 16, 16});
    convolution_block.tensor_props.at(cudnn_frontend::convolution_node::PORTS::W).set_dim({64, 32, 3, 3});
    convolution_block.tensor_props.at(cudnn_frontend::convolution_node::PORTS::Y).set_dim({4, 64, 16, 16});

    convolution_block.build(handle);

    Surface<half> x_tensor(convolution_block.tensor_props.at(cudnn_frontend::convolution_node::PORTS::X).get_tensor_size(), false);
    Surface<half> w_tensor(convolution_block.tensor_props.at(cudnn_frontend::convolution_node::PORTS::W).get_tensor_size(), false);
    Surface<half> y_tensor(convolution_block.tensor_props.at(cudnn_frontend::convolution_node::PORTS::Y).get_tensor_size(), false);

    std::unordered_map<int64_t, void*> variant_pack = {
        {convolution_block.tensor_props.at(cudnn_frontend::convolution_node::PORTS::X).get_uid(), x_tensor.devPtr}
        , {convolution_block.tensor_props.at(cudnn_frontend::convolution_node::PORTS::W).get_uid(), w_tensor.devPtr}
        , {convolution_block.tensor_props.at(cudnn_frontend::convolution_node::PORTS::Y).get_uid(), y_tensor.devPtr}
    };
    convolution_block.execute(handle, variant_pack);
}

void
run_pointwise_block() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::PointwiseBlock pointwise_block;

    cudnn_frontend::pointwise_node props{""};
    props.set_mode(CUDNN_POINTWISE_ADD);
    props.set_tensor_data_type(CUDNN_DATA_HALF);
    props.set_compute_type(CUDNN_DATA_FLOAT);
    pointwise_block.props = props;

    pointwise_block.update_uids();
    pointwise_block.tensor_props.at(cudnn_frontend::pointwise_node::PORTS::X).set_dim({4, 32, 16, 16});
    pointwise_block.tensor_props.at(cudnn_frontend::pointwise_node::PORTS::B).set_dim({1, 32, 1, 1});
    pointwise_block.tensor_props.at(cudnn_frontend::pointwise_node::PORTS::Y).set_dim({4, 32, 16, 16});

    pointwise_block.build(handle);

    Surface<half> x_tensor(pointwise_block.tensor_props.at(cudnn_frontend::pointwise_node::PORTS::X).get_tensor_size(), false);
    Surface<half> b_tensor(pointwise_block.tensor_props.at(cudnn_frontend::pointwise_node::PORTS::B).get_tensor_size(), false);
    Surface<half> y_tensor(pointwise_block.tensor_props.at(cudnn_frontend::pointwise_node::PORTS::Y).get_tensor_size(), false);

    std::unordered_map<int64_t, void*> variant_pack = {
        {pointwise_block.tensor_props.at(cudnn_frontend::pointwise_node::PORTS::X).get_uid(), x_tensor.devPtr}
        , {pointwise_block.tensor_props.at(cudnn_frontend::pointwise_node::PORTS::B).get_uid(), b_tensor.devPtr}
        , {pointwise_block.tensor_props.at(cudnn_frontend::pointwise_node::PORTS::Y).get_uid(), y_tensor.devPtr}
    };
    pointwise_block.execute(handle, variant_pack);
}

void
run_reduction_block() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::ReductionBlock reduction_block;

    cudnn_frontend::reduction_node props{""};
    props.set_mode(CUDNN_REDUCE_TENSOR_ADD);
    props.set_tensor_data_type(CUDNN_DATA_HALF);
    props.set_compute_type(CUDNN_DATA_FLOAT);
    reduction_block.props = props;

    reduction_block.update_uids();
    reduction_block.tensor_props.at(cudnn_frontend::reduction_node::PORTS::X).set_dim({4, 32, 16, 16});
    reduction_block.tensor_props.at(cudnn_frontend::reduction_node::PORTS::Y).set_dim({1, 32, 1, 1});
    reduction_block.tensor_props.at(cudnn_frontend::reduction_node::PORTS::Y).set_data_type(CUDNN_DATA_FLOAT);

    reduction_block.build(handle);

    Surface<half> x_tensor(reduction_block.tensor_props.at(cudnn_frontend::reduction_node::PORTS::X).get_tensor_size(), false);
    Surface<float> y_tensor(reduction_block.tensor_props.at(cudnn_frontend::reduction_node::PORTS::Y).get_tensor_size(), false);

    std::unordered_map<int64_t, void*> variant_pack = {
        {reduction_block.tensor_props.at(cudnn_frontend::reduction_node::PORTS::X).get_uid(), x_tensor.devPtr}
        , {reduction_block.tensor_props.at(cudnn_frontend::reduction_node::PORTS::Y).get_uid(), y_tensor.devPtr}
    };
    reduction_block.execute(handle, variant_pack);
}

void
run_convolution_fp8_block() {
#if (CUDNN_VERSION >= 8700)
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::ConvolutionFP8Block convolution_fp8_block;

    std::shared_ptr<cudnn_frontend::ConvolutionBlock> convolution_block = std::dynamic_pointer_cast<cudnn_frontend::ConvolutionBlock>(convolution_fp8_block.sub_blocks["conv_block"]);
    
    cudnn_frontend::convolution_node props{""};
    props.set_padding({1, 1});
    props.set_stride({1, 1});
    props.set_dilation({1, 1});
    props.set_tensor_data_type(CUDNN_DATA_FP8_E4M3);
    props.set_compute_type(CUDNN_DATA_FLOAT);
    convolution_block->props = props;

    convolution_block->update_uids();
    convolution_block->tensor_props.at(cudnn_frontend::convolution_node::PORTS::X).set_dim({4, 32, 16, 16});
    convolution_block->tensor_props.at(cudnn_frontend::convolution_node::PORTS::W).set_dim({64, 32, 3, 3});
    convolution_block->tensor_props.at(cudnn_frontend::convolution_node::PORTS::Y).set_dim({4, 64, 16, 16});

    std::shared_ptr<cudnn_frontend::PointwiseBlock> X_DQ_block = std::dynamic_pointer_cast<cudnn_frontend::PointwiseBlock>(convolution_fp8_block.sub_blocks["X_DQ_block"]);

    cudnn_frontend::pointwise_node X_DQ_props{""};
    X_DQ_props.set_mode(CUDNN_POINTWISE_MUL);
    X_DQ_props.set_tensor_data_type(CUDNN_DATA_FP8_E4M3);
    X_DQ_props.set_compute_type(CUDNN_DATA_FLOAT);
    X_DQ_block->props = X_DQ_props;

    X_DQ_block->update_uids();
    X_DQ_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::X).set_dim({4, 64, 16, 16});
    X_DQ_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::B).set_dim({1, 1, 1, 1});
    X_DQ_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::Y).set_dim({4, 64, 16, 16});

    std::shared_ptr<cudnn_frontend::PointwiseBlock> W_DQ_block = std::dynamic_pointer_cast<cudnn_frontend::PointwiseBlock>(convolution_fp8_block.sub_blocks["W_DQ_block"]);

    cudnn_frontend::pointwise_node W_DQ_props{""};
    W_DQ_props.set_mode(CUDNN_POINTWISE_MUL);
    W_DQ_props.set_tensor_data_type(CUDNN_DATA_FP8_E4M3);
    W_DQ_props.set_compute_type(CUDNN_DATA_FLOAT);
    W_DQ_block->props = W_DQ_props;

    W_DQ_block->update_uids();
    W_DQ_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::X).set_dim({4, 64, 16, 16});
    W_DQ_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::B).set_dim({1, 1, 1, 1});
    W_DQ_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::Y).set_dim({4, 64, 16, 16});

    std::shared_ptr<cudnn_frontend::PointwiseBlock> Y_Q_block = std::dynamic_pointer_cast<cudnn_frontend::PointwiseBlock>(convolution_fp8_block.sub_blocks["Y_Q_block"]);

    cudnn_frontend::pointwise_node Y_Q_props{""};
    Y_Q_props.set_mode(CUDNN_POINTWISE_MUL);
    Y_Q_props.set_tensor_data_type(CUDNN_DATA_FP8_E4M3);
    Y_Q_props.set_compute_type(CUDNN_DATA_FLOAT);
    Y_Q_block->props = Y_Q_props;

    Y_Q_block->update_uids();
    Y_Q_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::X).set_dim({4, 64, 16, 16});
    Y_Q_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::B).set_dim({1, 1, 1, 1});
    Y_Q_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::Y).set_dim({4, 64, 16, 16});

    std::shared_ptr<cudnn_frontend::ReductionBlock> amax_block = std::dynamic_pointer_cast<cudnn_frontend::ReductionBlock>(convolution_fp8_block.sub_blocks["amax_block"]);

    cudnn_frontend::reduction_node amax_props{""};
    amax_props.set_mode(CUDNN_REDUCE_TENSOR_AMAX);
    amax_props.set_tensor_data_type(CUDNN_DATA_FLOAT);
    amax_props.set_compute_type(CUDNN_DATA_FLOAT);
    amax_block->props = amax_props;

    amax_block->update_uids();
    amax_block->tensor_props.at(cudnn_frontend::reduction_node::PORTS::X).set_dim({4, 32, 16, 16});
    amax_block->tensor_props.at(cudnn_frontend::reduction_node::PORTS::Y).set_dim({1, 1, 1, 1});

    convolution_fp8_block.build(handle);
    
    Surface<float> x_dq_tensor(X_DQ_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::B).get_tensor_size(), false);
    Surface<float> w_dq_tensor(W_DQ_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::B).get_tensor_size(), false);
    Surface<int8_t> x_tensor(convolution_block->tensor_props.at(cudnn_frontend::convolution_node::PORTS::X).get_tensor_size(), false);
    Surface<int8_t> w_tensor(convolution_block->tensor_props.at(cudnn_frontend::convolution_node::PORTS::W).get_tensor_size(), false);
    Surface<int8_t> y_tensor(Y_Q_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::Y).get_tensor_size(), false);
    Surface<float> y_q_tensor(Y_Q_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::B).get_tensor_size(), false);
    Surface<float> amax_tensor(amax_block->tensor_props.at(cudnn_frontend::reduction_node::PORTS::Y).get_tensor_size(), false);
    
    std::unordered_map<int64_t, void*> variant_pack = {
        {X_DQ_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::B).get_uid(), x_dq_tensor.devPtr}
        , {W_DQ_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::B).get_uid(), w_dq_tensor.devPtr}
        , {convolution_block->tensor_props.at(cudnn_frontend::convolution_node::PORTS::X).get_uid(), x_tensor.devPtr}
        , {convolution_block->tensor_props.at(cudnn_frontend::convolution_node::PORTS::W).get_uid(), w_tensor.devPtr}
        , {Y_Q_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::Y).get_uid(), y_tensor.devPtr}
        , {Y_Q_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::B).get_uid(), y_q_tensor.devPtr}
        , {amax_block->tensor_props.at(cudnn_frontend::reduction_node::PORTS::Y).get_uid(), amax_tensor.devPtr}
    };
    convolution_fp8_block.execute(handle, variant_pack);
#endif
}

void
run_convolution_pointwise_block() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::ConvolutionPointwiseBlock convolution_pointwise_block;

    std::shared_ptr<cudnn_frontend::ConvolutionBlock> convolution_block = std::dynamic_pointer_cast<cudnn_frontend::ConvolutionBlock>(convolution_pointwise_block.sub_blocks["conv_block"]);
    
    cudnn_frontend::convolution_node props{""};
    props.set_padding({1, 1});
    props.set_stride({1, 1});
    props.set_dilation({1, 1});
    props.set_tensor_data_type(CUDNN_DATA_HALF);
    props.set_compute_type(CUDNN_DATA_FLOAT);
    convolution_block->props = props;

    convolution_block->update_uids();
    convolution_block->tensor_props.at(cudnn_frontend::convolution_node::PORTS::X).set_dim({4, 32, 16, 16});
    convolution_block->tensor_props.at(cudnn_frontend::convolution_node::PORTS::W).set_dim({64, 32, 3, 3});
    convolution_block->tensor_props.at(cudnn_frontend::convolution_node::PORTS::Y).set_dim({4, 64, 16, 16});

    std::shared_ptr<cudnn_frontend::PointwiseBlock> pointwise_block = std::dynamic_pointer_cast<cudnn_frontend::PointwiseBlock>(convolution_pointwise_block.sub_blocks["pointwise_block"]);

    cudnn_frontend::pointwise_node pointwise_props {""};
    pointwise_props.set_mode(CUDNN_POINTWISE_ADD);
    pointwise_props.set_tensor_data_type(CUDNN_DATA_HALF);
    pointwise_props.set_compute_type(CUDNN_DATA_FLOAT);
    pointwise_block->props = pointwise_props;

    pointwise_block->update_uids();
    pointwise_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::X).set_dim({4, 64, 16, 16});
    pointwise_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::B).set_dim({1, 64, 1, 1});
    pointwise_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::Y).set_dim({4, 64, 16, 16});

    convolution_pointwise_block.build(handle);
    
    Surface<half> x_tensor(convolution_block->tensor_props.at(cudnn_frontend::convolution_node::PORTS::X).get_tensor_size(), false);
    Surface<half> w_tensor(convolution_block->tensor_props.at(cudnn_frontend::convolution_node::PORTS::W).get_tensor_size(), false);
    Surface<half> b_tensor(pointwise_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::B).get_tensor_size(), false);
    Surface<half> y_tensor(pointwise_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::Y).get_tensor_size(), false);
    std::unordered_map<int64_t, void*> variant_pack = {
        {convolution_block->tensor_props.at(cudnn_frontend::convolution_node::PORTS::X).get_uid(), x_tensor.devPtr}
        , {convolution_block->tensor_props.at(cudnn_frontend::convolution_node::PORTS::W).get_uid(), w_tensor.devPtr}
        , {pointwise_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::B).get_uid(), b_tensor.devPtr}
        , {pointwise_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::Y).get_uid(), y_tensor.devPtr}
    };
    convolution_pointwise_block.execute(handle, variant_pack);
}
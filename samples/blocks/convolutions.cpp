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

#include <cudnn_frontend/blocks/convolution_blocks.h>
#include <cudnn_frontend/blocks/pointwise_block.h>
#include <cudnn_frontend/blocks/convolution_pointwise_block.h>

#include "convolutions.h"

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

    convolution_block.tensor_props.at(cudnn_frontend::convolution_node::PORTS::X).set_dim({4, 32, 16, 16});
    convolution_block.tensor_props.at(cudnn_frontend::convolution_node::PORTS::W).set_dim({64, 32, 3, 3});
    convolution_block.tensor_props.at(cudnn_frontend::convolution_node::PORTS::Y).set_dim({4, 64, 16, 16});

    convolution_block.build(handle);
    convolution_block.execute(handle);
}

void
run_pointwise_block() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::PointwiseBlock pointwise_block;

    cudnn_frontend::pointwise_node props {""};
    props.set_mode(CUDNN_POINTWISE_ADD);
    props.set_tensor_data_type(CUDNN_DATA_HALF);
    props.set_compute_type(CUDNN_DATA_FLOAT);
    pointwise_block.props = props;

    pointwise_block.tensor_props.at(cudnn_frontend::pointwise_node::PORTS::X).set_dim({4, 32, 16, 16});
    pointwise_block.tensor_props.at(cudnn_frontend::pointwise_node::PORTS::B).set_dim({1, 32, 1, 1});
    pointwise_block.tensor_props.at(cudnn_frontend::pointwise_node::PORTS::Y).set_dim({4, 32, 16, 16});

    pointwise_block.build(handle);
    pointwise_block.execute(handle);
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

    convolution_block->tensor_props.at(cudnn_frontend::convolution_node::PORTS::X).set_dim({4, 32, 16, 16});
    convolution_block->tensor_props.at(cudnn_frontend::convolution_node::PORTS::W).set_dim({64, 32, 3, 3});
    convolution_block->tensor_props.at(cudnn_frontend::convolution_node::PORTS::Y).set_dim({4, 64, 16, 16});

    std::shared_ptr<cudnn_frontend::PointwiseBlock> pointwise_block = std::dynamic_pointer_cast<cudnn_frontend::PointwiseBlock>(convolution_pointwise_block.sub_blocks["pointwise_block"]);

    cudnn_frontend::pointwise_node pointwise_props {""};
    pointwise_props.set_mode(CUDNN_POINTWISE_ADD);
    pointwise_props.set_tensor_data_type(CUDNN_DATA_HALF);
    pointwise_props.set_compute_type(CUDNN_DATA_FLOAT);
    pointwise_block->props = pointwise_props;

    pointwise_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::X).set_dim({4, 64, 16, 16});
    pointwise_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::B).set_dim({1, 64, 1, 1});
    pointwise_block->tensor_props.at(cudnn_frontend::pointwise_node::PORTS::Y).set_dim({4, 64, 16, 16});

    convolution_pointwise_block.build(handle);
    convolution_pointwise_block.execute(handle);
}
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

#include "convolutions.h"

void
run_convolution_block() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnn_frontend::convolution_properties props;
    props.dim_count = 4;

    props.input_dim[0] = 4;
    props.input_dim[1] = 32;
    props.input_dim[2] = 16;
    props.input_dim[3] = 16;

    props.weight_dim[0] = 64;
    props.weight_dim[1] = 32;
    props.weight_dim[2] = 3;
    props.weight_dim[3] = 3;

    props.output_dim[0] = 4;
    props.output_dim[1] = 64;
    props.output_dim[2] = 16;
    props.output_dim[3] = 16;

    props.padding[0] = 1;
    props.padding[1] = 1;

    props.stride[0] = 1;
    props.stride[1] = 1;

    props.dilation[0] = 1;
    props.dilation[1] = 1;

    props.tensor_data_type = CUDNN_DATA_HALF;
    props.compute_type = CUDNN_DATA_FLOAT;

    props.update_uids(1);

    cudnn_frontend::ConvolutionBlock convolution_block(props);
    convolution_block.build(handle);
    convolution_block.execute(handle);
}
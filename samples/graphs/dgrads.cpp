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
/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include <cudnn_frontend.h>

TEST_CASE("Tensor attributes", "[tensor][serialize]") {
    namespace fe = cudnn_frontend;

    auto tensor_attributes = fe::graph::Tensor_attributes()
                                 .set_name("image")
                                 .set_dim({4, 32, 16, 16})
                                 .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
                                 .set_is_virtual(true)
                                 .set_is_pass_by_value(true)
                                 .set_uid(12312)
                                 .set_reordering_type(fe::TensorReordering_t::F16x16)
                                 .set_data_type(fe::DataType_t::HALF);

    json j                              = tensor_attributes;
    auto tensor_attributes_deserialized = j;

    REQUIRE(tensor_attributes_deserialized == tensor_attributes);
}

TEST_CASE("Conv fprop attributes", "[conv_fprop][serialize]") {
    namespace fe = cudnn_frontend;

    auto x = std::make_shared<fe::graph::Tensor_attributes>();
    x->set_name("image")
        .set_dim({4, 32, 16, 16})
        .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
        .set_is_virtual(true)
        .set_is_pass_by_value(true)
        .set_uid(12312)
        .set_reordering_type(fe::TensorReordering_t::F16x16)
        .set_data_type(fe::DataType_t::HALF);

    auto conv_fprop_attributes = fe::graph::Conv_fprop_attributes()
                                     .set_name("conv_fprop")
                                     .set_padding({1, 1})
                                     .set_stride({1, 1})
                                     .set_dilation({1, 1})
                                     .set_compute_data_type(fe::DataType_t::FLOAT);

    json j                                  = conv_fprop_attributes;
    auto conv_fprop_attributes_deserialized = j;

    REQUIRE(conv_fprop_attributes_deserialized == conv_fprop_attributes);
}

TEST_CASE("Graph key", "[serialize]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;
    graph.set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto X = graph.tensor(
        fe::graph::Tensor_attributes().set_name("image").set_dim({4, 16, 64}).set_stride({16 * 64, 1, 16}));
    auto Y = graph.tensor(
        fe::graph::Tensor_attributes().set_name("filter").set_dim({4, 64, 32}).set_stride({32 * 64, 1, 64}));

    fe::graph::Matmul_attributes matmul;
    auto Z = graph.matmul(X, Y, matmul);

    auto scale_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::MUL);
    auto S             = graph.tensor(
        fe::graph::Tensor_attributes().set_name("scale").set_dim({4, 16, 32}).set_stride({16 * 32, 32, 1}));
    auto scale_output = graph.pointwise(Z, S, scale_options);

    auto bias_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::ADD);
    auto B =
        graph.tensor(fe::graph::Tensor_attributes().set_name("bias").set_dim({4, 16, 32}).set_stride({16 * 32, 32, 1}));
    auto bias_output = graph.pointwise(scale_output, B, bias_options);

    auto relu_options = fe::graph::Pointwise_attributes().set_mode(fe::PointwiseMode_t::RELU_FWD);
    auto O            = graph.pointwise(bias_output, relu_options);
    O->set_output(true);

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    REQUIRE(graph.validate().is_good());
    auto key = graph.key();

    REQUIRE(graph.build_operation_graph(handle).is_good());
    REQUIRE(key == graph.key());

    REQUIRE(graph.create_execution_plans({fe::HeurMode_t::A}).is_good());
    REQUIRE(key == graph.key());

    REQUIRE(graph.check_support(handle).is_good());
    REQUIRE(key == graph.key());

    REQUIRE(graph.build_plans(handle).is_good());
    REQUIRE(key == graph.key());
}

TEST_CASE("conv graph serialization", "[graph][serialize]") {
    namespace fe = cudnn_frontend;

    fe::graph::Graph graph;

    auto x = graph.tensor(fe::graph::Tensor_attributes());
    x->set_name("image")
        .set_dim({4, 32, 16, 16})
        .set_stride({32 * 16 * 16, 1, 32 * 16, 32})
        .set_is_virtual(false)
        .set_is_pass_by_value(false)
        .set_reordering_type(fe::TensorReordering_t::NONE)
        .set_data_type(fe::DataType_t::HALF);

    auto w = graph.tensor(fe::graph::Tensor_attributes());
    w->set_name("weight")
        .set_dim({64, 32, 3, 3})
        .set_stride({32 * 3 * 3, 1, 32 * 3, 32})
        .set_is_virtual(false)
        .set_is_pass_by_value(false)
        .set_reordering_type(fe::TensorReordering_t::NONE)
        .set_data_type(fe::DataType_t::HALF);

    auto conv_fprop_attributes = fe::graph::Conv_fprop_attributes()
                                     .set_name("conv_fprop")
                                     .set_padding({1, 1})
                                     .set_stride({1, 1})
                                     .set_dilation({1, 1})
                                     .set_compute_data_type(fe::DataType_t::FLOAT);

    auto y = graph.conv_fprop(x, w, conv_fprop_attributes);

    auto b = graph.tensor(fe::graph::Tensor_attributes());
    b->set_name("bias")
        .set_dim({1, 32, 1, 1})
        .set_stride({32, 1, 32, 32})
        .set_is_virtual(false)
        .set_is_pass_by_value(false)
        .set_reordering_type(fe::TensorReordering_t::NONE)
        .set_data_type(fe::DataType_t::HALF);

    auto pointwise_attributes = fe::graph::Pointwise_attributes()
                                    .set_name("bias")
                                    .set_mode(fe::PointwiseMode_t::ADD)
                                    .set_compute_data_type(fe::DataType_t::FLOAT);

    auto o = graph.pointwise(y, b, pointwise_attributes);

    auto reduction_attributes = fe::graph::Reduction_attributes()
                                    .set_name("reduction")
                                    .set_mode(fe::ReductionMode_t::ADD)
                                    .set_compute_data_type(fe::DataType_t::FLOAT);
    auto r = graph.reduction(o, reduction_attributes);

    r->set_output(true).set_data_type(fe::DataType_t::HALF);

    REQUIRE(graph.validate().is_good());

    json j = graph;
    fe::graph::Graph graph_deserialized;
    REQUIRE(graph_deserialized.deserialize(j).is_good());
    json j2 = graph_deserialized;

    REQUIRE(j == j2);
}
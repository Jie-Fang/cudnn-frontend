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
#include "../helpers.h"

#include <cudnn_frontend.h>

cudnn_frontend::graph::Graph build_BMM1_graph(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d) {
    std::vector<int64_t> q_dim = {b, h, s_q, d};
    std::vector<int64_t> q_stride(q_dim.size());
    generateMHAStrides(b, h, s_q, s_kv, d, q_stride.data(), MHA_Layout::QKV_INTERLEAVED, MHA_Matrix::Q_Matrix);

    std::vector<int64_t> k_dim =  {b, h, d, s_kv};
    std::vector<int64_t> k_stride(k_dim.size());
    generateMHAStrides(b, h, s_q, s_kv, d, k_stride.data(), MHA_Layout::QKV_INTERLEAVED, MHA_Matrix::K_Matrix_Transpose);

    std::vector<int64_t> s_dim = {b, h, s_q, s_kv};
    std::vector<int64_t> s_stride(s_dim.size());
    generateMHAStrides(b, h, s_q, s_kv, d, s_stride.data(), MHA_Layout::QKV_INTERLEAVED, MHA_Matrix::S_Matrix);

    std::vector<int64_t> seqlen_dim =  {b, 1, 1, 1};
    std::vector<int64_t> seqlen_stride = {1, 1, 1, 1};

    cudnn_frontend::graph::Graph bmm1("bmm1");
    bmm1.set_io_data_type(cudnn_frontend::DataType_t::HALF)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

    bmm1.insert_tensor(cudnn_frontend::graph::Tensor("Q").set_dim(q_dim).set_stride(q_stride));
    bmm1.insert_tensor(cudnn_frontend::graph::Tensor("K").set_dim(k_dim).set_stride(k_stride));
    bmm1.insert_tensor(cudnn_frontend::graph::Tensor("P").set_dim(s_dim).set_stride(s_stride).set_is_virtual(true));
    bmm1.insert_tensor(cudnn_frontend::graph::Tensor("SQ").set_dim(seqlen_dim).set_stride(seqlen_stride).set_data_type(cudnn_frontend::DataType_t::INT32));
    bmm1.insert_tensor(cudnn_frontend::graph::Tensor("SK").set_dim(seqlen_dim).set_stride(seqlen_stride).set_data_type(cudnn_frontend::DataType_t::INT32));
    
    auto matmul = cudnn_frontend::graph::Matmul("bmm1")
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Matmul::PORTS::A, "Q"}
                        , {cudnn_frontend::graph::Matmul::PORTS::B, "K"}
                        , {cudnn_frontend::graph::Matmul::PORTS::C, "P"}
                        , {cudnn_frontend::graph::Matmul::PORTS::A_OVERRIDE, "SQ"}
                        , {cudnn_frontend::graph::Matmul::PORTS::B_OVERRIDE, "SK"}
                    });
    bmm1.insert_node(matmul);

    return bmm1;
}

cudnn_frontend::graph::Graph build_mask_graph(int64_t b, int64_t h, int64_t s_q, int64_t s_kv) {
    std::vector<int64_t> afterBMM1_dim = {b, h, s_q, s_kv};
    std::vector<int64_t> afterBMM1_stride = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    std::vector<int64_t> seqlen_dim = {b, 1, 1, 1};
    std::vector<int64_t> seqlen_stride = {1, 1, 1, 1};

    std::vector<int64_t> maskVal_dim = {1, 1, 1, 1};
    std::vector<int64_t> maskVal_stride = {1, 1, 1, 1};

    cudnn_frontend::graph::Graph mask("mask");
    mask.set_io_data_type(cudnn_frontend::DataType_t::INT32)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

    mask.insert_tensor(cudnn_frontend::graph::Tensor("P").set_dim(afterBMM1_dim).set_stride(afterBMM1_stride).set_is_virtual(true));
    mask.insert_tensor(cudnn_frontend::graph::Tensor("R").set_dim(afterBMM1_dim).set_stride(afterBMM1_stride).set_is_virtual(true));
    mask.insert_tensor(cudnn_frontend::graph::Tensor("C").set_dim(afterBMM1_dim).set_stride(afterBMM1_stride).set_is_virtual(true));
    mask.insert_tensor(cudnn_frontend::graph::Tensor("SQ").set_dim(seqlen_dim).set_stride(seqlen_stride));
    mask.insert_tensor(cudnn_frontend::graph::Tensor("SK").set_dim(seqlen_dim).set_stride(seqlen_stride));
    mask.insert_tensor(cudnn_frontend::graph::Tensor("L_R").set_dim(afterBMM1_dim).set_stride(afterBMM1_stride).set_is_virtual(true).set_data_type(cudnn_frontend::DataType_t::BOOLEAN));
    mask.insert_tensor(cudnn_frontend::graph::Tensor("L_C").set_dim(afterBMM1_dim).set_stride(afterBMM1_stride).set_is_virtual(true).set_data_type(cudnn_frontend::DataType_t::BOOLEAN));
    mask.insert_tensor(cudnn_frontend::graph::Tensor("P_M").set_dim(afterBMM1_dim).set_stride(afterBMM1_stride).set_is_virtual(true).set_data_type(cudnn_frontend::DataType_t::BOOLEAN));
    mask.insert_tensor(cudnn_frontend::graph::Tensor("VAL").set_dim(maskVal_dim).set_stride(maskVal_stride).set_is_pass_by_value(true));
    mask.insert_tensor(cudnn_frontend::graph::Tensor("M_O").set_dim(afterBMM1_dim).set_stride(afterBMM1_stride).set_is_virtual(true));

    auto row_index = cudnn_frontend::graph::Pointwise("row_index")
                    .set_mode(cudnn_frontend::PointwiseMode_t::GEN_INDEX)
                    .set_axis(2)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, "P"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "R"}
                    });
    mask.insert_node(row_index);

    auto col_index = cudnn_frontend::graph::Pointwise("col_index")
                    .set_mode(cudnn_frontend::PointwiseMode_t::GEN_INDEX)
                    .set_axis(3)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, "P"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "C"}
                    });
    mask.insert_node(col_index);

    auto less_than_row = cudnn_frontend::graph::Pointwise("less_than_row")
                    .set_mode(cudnn_frontend::PointwiseMode_t::CMP_LT)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, "R"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "SQ"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "L_R"}
                    });
    mask.insert_node(less_than_row);

    auto less_than_col = cudnn_frontend::graph::Pointwise("less_than_col")
                    .set_mode(cudnn_frontend::PointwiseMode_t::CMP_LT)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, "C"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "SK"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "L_C"}
                    });
    mask.insert_node(less_than_col);

    auto logical_and = cudnn_frontend::graph::Pointwise("logical_and")
                    .set_mode(cudnn_frontend::PointwiseMode_t::LOGICAL_AND)
                    .set_compute_data_type(cudnn_frontend::DataType_t::BOOLEAN)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, "L_R"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "L_C"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "P_M"}
                    });
    mask.insert_node(logical_and);
    
    auto selection = cudnn_frontend::graph::Pointwise("selection")
                    .set_mode(cudnn_frontend::PointwiseMode_t::BINARY_SELECT)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, "P"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "VAL"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_2, "P_M"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "M_O"}
                    });
    mask.insert_node(selection);

    return mask;
}

cudnn_frontend::graph::Graph build_softmax_graph(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d, bool should_dump_output) {
    std::vector<int64_t> afterBMM1_dim = {b, h, s_q, s_kv};
    std::vector<int64_t> afterBMM1_stride = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    std::vector<int64_t> afterReduction_dim = {b, h, s_q, 1};
    std::vector<int64_t> afterReduction_stride = {h * s_q, s_q, 1, 1};

    std::vector<int64_t> s_dim = {b, h, s_q, s_kv};
    std::vector<int64_t> s_stride(s_dim.size());
    generateMHAStrides(b, h, s_q, s_kv, d, s_stride.data(), MHA_Layout::QKV_INTERLEAVED, MHA_Matrix::S_Matrix);

    cudnn_frontend::graph::Graph softmax("softmax");
    softmax.set_io_data_type(cudnn_frontend::DataType_t::HALF)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

    softmax.insert_tensor(cudnn_frontend::graph::Tensor("P").set_dim(s_dim).set_stride(s_stride).set_is_virtual(true));
    softmax.insert_tensor(cudnn_frontend::graph::Tensor("MAX").set_dim(afterReduction_dim).set_stride(afterReduction_stride).set_is_virtual(true));
    softmax.insert_tensor(cudnn_frontend::graph::Tensor("P_MAX").set_dim(afterBMM1_dim).set_stride(afterBMM1_stride).set_is_virtual(true));
    softmax.insert_tensor(cudnn_frontend::graph::Tensor("E").set_dim(afterBMM1_dim).set_stride(afterBMM1_stride).set_is_virtual(true));
    softmax.insert_tensor(cudnn_frontend::graph::Tensor("SUM").set_dim(afterReduction_dim).set_stride(afterReduction_stride).set_is_virtual(true));
    
    auto softmax_output = cudnn_frontend::graph::Tensor("S").set_dim(afterBMM1_dim).set_stride(afterBMM1_stride).set_data_type(cudnn_frontend::DataType_t::HALF).set_is_virtual(true);
    if(should_dump_output) {
        softmax_output.set_is_virtual(false).set_reordering_type(cudnn_frontend::TensorReordering_t::F16x16);
    }
    softmax.insert_tensor(softmax_output);
    
    auto max = cudnn_frontend::graph::Reduction("max")
                    .set_mode(cudnn_frontend::ReductionMode_t::MAX)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Reduction::PORTS::X, "P"}
                        , {cudnn_frontend::graph::Reduction::PORTS::Y, "MAX"}
                    });
    softmax.insert_node(max);

    auto sub = cudnn_frontend::graph::Pointwise("sub")
                    .set_mode(cudnn_frontend::PointwiseMode_t::SUB)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, "P"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "MAX"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "P_MAX"}
                    });
    softmax.insert_node(sub);

    auto exp = cudnn_frontend::graph::Pointwise("exp")
                    .set_mode(cudnn_frontend::PointwiseMode_t::EXP)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, "P_MAX"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "E"}
                    });
    softmax.insert_node(exp);

    auto sum = cudnn_frontend::graph::Reduction("sum")
                    .set_mode(cudnn_frontend::ReductionMode_t::ADD)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Reduction::PORTS::X, "E"}
                        , {cudnn_frontend::graph::Reduction::PORTS::Y, "SUM"}
                    });
    softmax.insert_node(sum);

    auto div = cudnn_frontend::graph::Pointwise("div")
                    .set_mode(cudnn_frontend::PointwiseMode_t::DIV)
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Pointwise::PORTS::IN_0, "E"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::IN_1, "SUM"}
                        , {cudnn_frontend::graph::Pointwise::PORTS::OUT_0, "S"}
                    });
    softmax.insert_node(div);

    return softmax;
}

cudnn_frontend::graph::Graph build_BMM2_graph(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d, bool should_read_from_gmem) {
    std::vector<int64_t> seqlen_dim =  {b, 1, 1, 1};
    std::vector<int64_t> seqlen_stride = {1, 1, 1, 1};
    
    std::vector<int64_t> s_dim = {b, h, s_q, s_kv};
    std::vector<int64_t> s_stride(s_dim.size());
    generateMHAStrides(b, h, s_q, s_kv, d, s_stride.data(), MHA_Layout::QKV_INTERLEAVED, MHA_Matrix::S_Matrix);

    std::vector<int64_t> v_dim =  {b, h, s_kv, d};
    std::vector<int64_t> v_stride(v_dim.size());
    generateMHAStrides(b, h, s_q, s_kv, d, v_stride.data(), MHA_Layout::QKV_INTERLEAVED, MHA_Matrix::V_Matrix);

    std::vector<int64_t> o_dim =  {b, h, s_q, d};
    std::vector<int64_t> o_stride(o_dim.size());
    generateMHAStrides(b, h, s_q, s_kv, d, o_stride.data(), MHA_Layout::QKV_INTERLEAVED, MHA_Matrix::O_Matrix);

    cudnn_frontend::graph::Graph bmm2("bmm2");
    bmm2.set_io_data_type(cudnn_frontend::DataType_t::HALF)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::HALF)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

    auto a_tensor = cudnn_frontend::graph::Tensor("S").set_dim(s_dim).set_stride(s_stride).set_is_virtual(true);
    if(should_read_from_gmem) {
        a_tensor.set_is_virtual(false).set_reordering_type(cudnn_frontend::TensorReordering_t::F16x16);
    }
    bmm2.insert_tensor(a_tensor);
    bmm2.insert_tensor(cudnn_frontend::graph::Tensor("V").set_dim(v_dim).set_stride(v_stride));
    bmm2.insert_tensor(cudnn_frontend::graph::Tensor("O").set_dim(o_dim).set_stride(o_stride));
    bmm2.insert_tensor(cudnn_frontend::graph::Tensor("SQ").set_dim(seqlen_dim).set_stride(seqlen_stride).set_data_type(cudnn_frontend::DataType_t::INT32));
    bmm2.insert_tensor(cudnn_frontend::graph::Tensor("SK").set_dim(seqlen_dim).set_stride(seqlen_stride).set_data_type(cudnn_frontend::DataType_t::INT32));

    auto matmul = cudnn_frontend::graph::Matmul("bmm2")
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Matmul::PORTS::A, "S"}
                        , {cudnn_frontend::graph::Matmul::PORTS::B, "V"}
                        , {cudnn_frontend::graph::Matmul::PORTS::C, "O"}
                        , {cudnn_frontend::graph::Matmul::PORTS::A_OVERRIDE, "SQ"}
                        , {cudnn_frontend::graph::Matmul::PORTS::C_OVERRIDE, "SK"}
                    });
    bmm2.insert_node(matmul);

    return bmm2;
}


TEST_CASE("MHA Fprop Graphs", "[graph][mha]") {
    int64_t b = 32;  // batch size
    int64_t h = 16;  // head dim
    int64_t s_q = 512; // q tensor is padded to this seq length
    int64_t s_kv = 512; // k and v tensor is padded to this seq length
    int64_t d = 64;  // hidden dim
    bool should_mask = false;
    bool should_dump_softmax_output = true;

    cudnn_frontend::graph::Graph mha_graph("mha");
    auto BMM1_graph = build_BMM1_graph(b, h, s_q, s_kv, d);
    // auto mask_graph = build_mask_graph(b, h, s_q, s_kv);
    auto softmax_graph = build_softmax_graph(b, h, s_q, s_kv, d, should_dump_softmax_output);
    bool should_read_from_gmem = should_dump_softmax_output;
    auto BMM2_graph = build_BMM2_graph(b, h, s_q, s_kv, d, should_read_from_gmem);

    std::unordered_map<std::string, std::string> connections;
    
    connections.clear();
    mha_graph.insert_graph(BMM1_graph, connections);
    
    // connections.clear();
    // connections["P"] = "P";
    // mha_graph.insert_graph(mask_graph, connections);

    connections.clear();
    connections["P"] = "P";
    mha_graph.insert_graph(softmax_graph, connections);
    
    connections.clear();
    connections["S"] = "S";
    mha_graph.insert_graph(BMM2_graph, connections);

    #if (CUDNN_VERSION < 8900)
        SKIP("MHA Graph requires cudnn 8.9 and up");
        return;
    #endif
    if (check_device_arch_newer_than("hopper") == false) {
        SKIP("MHA Graph requires Hopper or above arch.");
        return;
    }

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    REQUIRE(cudnn_frontend::error_t::OK == mha_graph.build(handle));

    auto plans = mha_graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_A)
                    .build_plans(handle);

    REQUIRE(cudnn_frontend::error_t::OK == mha_graph.set_executor(plans));

    //// Build variant pack
    Surface<half> qkvTensor(b * s_q * 3 * h * d, false);
    Surface<half> oTensor(b * s_q * h * d, false);
    void* devPtrQ = qkvTensor.devPtr;
    void* devPtrK = (qkvTensor.devPtr + h * d);
    void* devPtrV = (qkvTensor.devPtr + 2 * h * d);
    void* devPtrO = oTensor.devPtr;

    Surface<int32_t> devActualSeqlenQ(b, false);
    Surface<int32_t> devActualSeqlenK(b, false);
    std::vector<int32_t> hostActualSeqlenQ(b, 20);
    std::vector<int32_t> hostActualSeqlenK(b, 20);

    checkCudaErr(cudaMemcpy(devActualSeqlenQ.devPtr, hostActualSeqlenQ.data(), sizeof(hostActualSeqlenQ[0]) * b, cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devActualSeqlenK.devPtr, hostActualSeqlenK.data(), sizeof(hostActualSeqlenK[0]) * b, cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());

    std::unordered_map<std::string, void*> variant_pack = {
        {"Q", devPtrQ}
        , {"K", devPtrK}
        , {"SQ", devActualSeqlenQ.devPtr}
        , {"SK", devActualSeqlenK.devPtr}
        , {"V", devPtrV}
        , {"O", devPtrO}
    };

    if(should_mask) {
        float negInfinity = std::numeric_limits<float>::min();
        variant_pack["VAL"] = &negInfinity;
    }
    
    Surface<half> sTensor(b * h * s_q * s_kv, false);
    if(should_dump_softmax_output) {
        variant_pack["S"] = sTensor.devPtr;
    }
    
    REQUIRE(cudnn_frontend::error_t::OK == mha_graph.execute(handle, variant_pack));

    checkCudaErr(cudaDeviceSynchronize());

    cudnnDestroy(handle);
}
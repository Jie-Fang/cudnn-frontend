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
         .set_intermediate_data_type(cudnn_frontend::DataType_t::HALF)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

    bmm1.insert_tensor(cudnn_frontend::graph::Tensor("Q").set_dim(q_dim).set_stride(q_stride));
    bmm1.insert_tensor(cudnn_frontend::graph::Tensor("K").set_dim(k_dim).set_stride(k_stride));
    bmm1.insert_tensor(cudnn_frontend::graph::Tensor("S").set_dim(s_dim).set_stride(s_stride).set_is_virtual(true));

    // auto seqlenQTensor = tensor_create(CUDNN_DATA_INT32, Q_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);
    // auto seqlenKTensor = tensor_create(CUDNN_DATA_INT32, K_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);

                            // .setmOverrideDesc(seqlenQTensor)
                            // .setnOverrideDesc(seqlenKTensor)
    
    auto matmul = cudnn_frontend::graph::Matmul("bmm1")
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Matmul::PORTS::A, "Q"}
                        , {cudnn_frontend::graph::Matmul::PORTS::B, "K"}
                        , {cudnn_frontend::graph::Matmul::PORTS::C, "S"}
                    });
    bmm1.insert_node(matmul);

    return bmm1;
}

cudnn_frontend::graph::Graph build_BMM2_graph(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d) {
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

    // auto seqlenQTensor = tensor_create(CUDNN_DATA_INT32, Q_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);
    // auto seqlenKTensor = tensor_create(CUDNN_DATA_INT32, K_SEQLEN_ID, seqlen_dim, seqlen_stride, false, false);

    cudnn_frontend::graph::Graph bmm2("bmm2");
    bmm2.set_io_data_type(cudnn_frontend::DataType_t::HALF)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::HALF)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);

    bmm2.insert_tensor(cudnn_frontend::graph::Tensor("S").set_dim(s_dim).set_stride(s_stride).set_is_virtual(true));
    bmm2.insert_tensor(cudnn_frontend::graph::Tensor("V").set_dim(v_dim).set_stride(v_stride));
    bmm2.insert_tensor(cudnn_frontend::graph::Tensor("O").set_dim(o_dim).set_stride(o_stride));

    auto matmul = cudnn_frontend::graph::Matmul("bmm2")
                    .map_port_to_tensor({
                        {cudnn_frontend::graph::Matmul::PORTS::A, "S"}
                        , {cudnn_frontend::graph::Matmul::PORTS::B, "V"}
                        , {cudnn_frontend::graph::Matmul::PORTS::C, "O"}
                    });
    bmm2.insert_node(matmul);

    return bmm2;

                            // .setmOverrideDesc(seqlenQTensor)
                            // .setkOverrideDesc(seqlenKTensor)
}


TEST_CASE("MHA Fprop Graphs", "[graph][mha]") {
    int64_t b = 32;  // batch size
    int64_t h = 16;  // head dim
    int64_t s_q = 512; // q tensor is padded to this seq length
    int64_t s_kv = 512; // k and v tensor is padded to this seq length
    int64_t d = 64;  // hidden dim

    cudnn_frontend::graph::Graph mha_graph("mha");
    auto BMM1_graph = build_BMM1_graph(b, h, s_q, s_kv, d);
    auto BMM2_graph = build_BMM2_graph(b, h, s_q, s_kv, d);

    std::unordered_map<std::string, std::string> connections;
    mha_graph.insert_graph(BMM1_graph, connections);
    connections["S"] = "S";
    mha_graph.insert_graph(BMM2_graph, connections);

    #if (CUDNN_VERSION < 8700)
        SKIP("fmha patterns are not supported in cudnn versions prior to 8.7");
    #endif

    cudnnHandle_t handle;
    checkCudnnErr(cudnnCreate(&handle));
    REQUIRE(cudnn_frontend::error_t::OK == mha_graph.build(handle));

    auto plans = mha_graph.get_execution_plan_list(cudnn_frontend::HeurMode_t::HEUR_MODE_A)
                    .build_plans(handle);
    cudnnDestroy(handle);

    REQUIRE(cudnn_frontend::error_t::OK == mha_graph.set_executor(plans));

    int* devActualSeqlenQ = nullptr; // actual seqlen Q
    int* devActualSeqlenK = nullptr; // actual seqlen K

    int* hostActualSeqlenQ = nullptr;
    int* hostActualSeqlenK = nullptr;

    Surface<half> qkvTensor(b * s_q * 3 * h * d, false);
    Surface<half> oTensor(b * s_q * h * d, false);
    void* devPtrQ = qkvTensor.devPtr;
    void* devPtrK = (qkvTensor.devPtr + h * d);
    void* devPtrV = (qkvTensor.devPtr + 2 * h * d);
    void* devPtrO = oTensor.devPtr;

    // setup of actual seqlen Q and seqlen K
    checkCudaErr(cudaMalloc((void**)&(devActualSeqlenQ), (b) * sizeof(devActualSeqlenQ[0])));
    hostActualSeqlenQ = (int*) calloc(b, sizeof(hostActualSeqlenQ[0]));

    for (int i = 0; i < b; i++) {
        hostActualSeqlenQ[i] = 128;
    }

    checkCudaErr(cudaMemcpy(devActualSeqlenQ, hostActualSeqlenQ, sizeof(hostActualSeqlenQ[0]) * b, cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());

    checkCudaErr(cudaMalloc((void**)&(devActualSeqlenK), (b) * sizeof(devActualSeqlenK[0])));
    hostActualSeqlenK = (int*) calloc(b, sizeof(hostActualSeqlenK[0]));

    for (int i = 0; i < b; i++) {
        hostActualSeqlenK[i] = 128;
    }

    checkCudaErr(cudaMemcpy(devActualSeqlenK, hostActualSeqlenK, sizeof(hostActualSeqlenK[0]) * b, cudaMemcpyHostToDevice));
    checkCudaErr(cudaDeviceSynchronize());

    std::unordered_map<std::string, void*> variant_pack = {
        {"Q", devPtrQ}
        , {"K", devPtrK}
        , {"V", devPtrV}
        , {"O", devPtrO}
    };
    REQUIRE(cudnn_frontend::error_t::OK == mha_graph.execute(variant_pack));

    checkCudaErr(cudaDeviceSynchronize());

    if (devActualSeqlenQ) cudaFree(devActualSeqlenQ);
    if (hostActualSeqlenQ) free(hostActualSeqlenQ);

    if (devActualSeqlenK) cudaFree(devActualSeqlenK);
    if (hostActualSeqlenK) free(hostActualSeqlenK);
}
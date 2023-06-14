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

TEST_CASE("Flash", "[graph][mha][flash][forward]") {
    int64_t b = 1;  // batch size
    int64_t h = 2;  // head dim
    int64_t s_q = 2048; // q tensor is padded to this seq length
    int64_t s_kv = 2048; // k and v tensor is padded to this seq length
    int64_t d = 128;  // hidden dim
    bool is_inference = false;
    float dropout_probability = 0.2f;

    namespace fe = cudnn_frontend;
    fe::graph::Graph mha_graph("mha");
    mha_graph.set_io_data_type(fe::DataType_t::HALF)
             .set_intermediate_data_type(fe::DataType_t::FLOAT)
             .set_compute_data_type(fe::DataType_t::FLOAT);

    fe::graph::Scaled_dot_product_flash_attention::Inputs inputs;
    inputs.Q = mha_graph.tensor(fe::graph::Tensor("Q").set_dim({b, h, s_q , d}).set_stride({3*h*d   , 3*d, 3*b*h*d, 1}));
    inputs.K = mha_graph.tensor(fe::graph::Tensor("K").set_dim({b, h, d   , s_kv}).set_stride({3*h*d, 3*d, 1      , 3*b*h*d}));
    inputs.V = mha_graph.tensor(fe::graph::Tensor("V").set_dim({b, h, s_kv, d}).set_stride({3*h*d   , 3*d, 3*b*h*d, 1}));

    auto seed = mha_graph.tensor(fe::graph::Tensor("Seed").set_dim({1,1,1,1}).set_stride({1,1,1,1}));
    auto offset = mha_graph.tensor(fe::graph::Tensor("Offset").set_dim({1,1,1,1}).set_stride({1,1,1,1}));
    auto scaled_dot_product_flash_attention_options = fe::graph::Scaled_dot_product_flash_attention("mha")
                                                    .set_is_inference(is_inference)
                                                    .use_causal_mask()
                                                    .set_scale_k(0.5f)
                                                    .set_dropout(dropout_probability, seed, offset);

    auto outputs = mha_graph.scaled_dot_product_flash_attention(inputs, scaled_dot_product_flash_attention_options);

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

    REQUIRE(fe::error_t::OK == mha_graph.validate());
    REQUIRE(fe::error_t::OK == mha_graph.is_supported());
    REQUIRE(fe::error_t::OK == mha_graph.build(handle));

    auto plans = mha_graph.get_execution_plan_list(fe::HeurMode_t::HEUR_MODE_A)
                    .build_plans(handle);

    REQUIRE(fe::error_t::OK == mha_graph.set_executor(plans));

    //// Build variant pack
    Surface<half> qkvTensor(b * s_q * 3 * h * d, false);
    Surface<half> oTensor(b * s_q * h * d, false);
    void* devPtrQ = qkvTensor.devPtr;
    void* devPtrK = (qkvTensor.devPtr + d);
    void* devPtrV = (qkvTensor.devPtr + 2 * d);
    void* devPtrO = oTensor.devPtr;

    int64_t scaleSize = 1;
    int64_t seed_value = 123456;
    Surface<int64_t> dropoutSeed(scaleSize, false, seed_value);
    Surface<int64_t> dropoutOffset(scaleSize, false, (int64_t)1);
    
    std::unordered_map<std::shared_ptr<fe::graph::Tensor>, void*> variant_pack = {
        {inputs.Q, devPtrQ}
        , {inputs.K, devPtrK}
        , {inputs.V, devPtrV}
        , {seed, dropoutSeed.devPtr}
        , {offset, dropoutOffset.devPtr}
        , {outputs.O, devPtrO}
    };

    Surface<float> statsTensor(b * h * s_q * 1, false);
    if(is_inference == false) {
        variant_pack[outputs.Stats] = statsTensor.devPtr;
    }
    
    Surface<int8_t> workspace(mha_graph.get_workspace_size(), false);
    REQUIRE(fe::error_t::OK == mha_graph.execute(handle, variant_pack, workspace.devPtr));

    checkCudaErr(cudaDeviceSynchronize());

    cudnnDestroy(handle);
}

TEST_CASE("Scaled dot product Graphs with Rng", "[graph][mha][non_flash][forward]") {
    int64_t b = 32;  // batch size
    int64_t h = 16;  // head dim
    int64_t s_q = 512; // q tensor is padded to this seq length
    int64_t s_kv = 512; // k and v tensor is padded to this seq length
    int64_t d = 64;  // hidden dim
    bool is_inference = true;
    float dropout_probability = 0.2f;
    int64_t seed = 123456;

    namespace fe = cudnn_frontend;
    fe::graph::Graph mha_graph("mha");
    mha_graph.set_io_data_type(fe::DataType_t::HALF)
             .set_intermediate_data_type(fe::DataType_t::FLOAT)
             .set_compute_data_type(fe::DataType_t::FLOAT);

    fe::graph::Scaled_dot_product_attention::Inputs inputs;
    inputs.Q = mha_graph.tensor(fe::graph::Tensor("Q").set_dim({b,h,s_q,d}).set_stride({s_q*3*h*d,d,3*h*d,1}));
    inputs.K = mha_graph.tensor(fe::graph::Tensor("K").set_dim({b,h,d,s_kv}).set_stride({s_kv*3*h*d,d,1,3*h*d}));
    inputs.V = mha_graph.tensor(fe::graph::Tensor("V").set_dim({b,h,s_kv,d}).set_stride({s_kv*3*h*d,d,3*h*d,1}));
    inputs.SEQ_LEN_Q = mha_graph.tensor(fe::graph::Tensor("SEQ_LEN_Q").set_dim({b,1,1,1}).set_stride({1,1,1,1}).set_data_type(fe::DataType_t::INT32));
    inputs.SEQ_LEN_K = mha_graph.tensor(fe::graph::Tensor("SEQ_LEN_K").set_dim({b,1,1,1}).set_stride({1,1,1,1}).set_data_type(fe::DataType_t::INT32));

    auto bias = mha_graph.tensor(fe::graph::Tensor("Bias").set_dim({1,h,s_q,s_kv}).set_stride({h*s_q*s_kv,s_q*s_kv,s_kv,1}));

    auto scaled_dot_product_attention_options = fe::graph::Scaled_dot_product_attention("mha")
                                                    .set_is_inference(is_inference)
                                                    .set_bias(bias)
                                                    .use_padding_mask()
                                                    .use_causal_mask()
                                                    .set_scale_k(0.5f)
                                                    .set_dropout(dropout_probability, seed);

    auto outputs = mha_graph.scaled_dot_product_attention(inputs, scaled_dot_product_attention_options);

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

    REQUIRE(fe::error_t::OK == mha_graph.validate());
    REQUIRE(fe::error_t::OK == mha_graph.is_supported());
    REQUIRE(fe::error_t::OK == mha_graph.build(handle));

    auto plans = mha_graph.get_execution_plan_list(fe::HeurMode_t::HEUR_MODE_A)
                    .build_plans(handle);

    REQUIRE(fe::error_t::OK == mha_graph.set_executor(plans));

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

    Surface<half> bTensor(1 * h * s_q * s_kv, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor>, void*> variant_pack = {
        {inputs.Q, devPtrQ}
        , {inputs.K, devPtrK}
        , {inputs.SEQ_LEN_Q, devActualSeqlenQ.devPtr}
        , {inputs.SEQ_LEN_K, devActualSeqlenK.devPtr}
        , {bias, bTensor.devPtr}
        , {inputs.V, devPtrV}
        , {outputs.O, devPtrO}
    };

    Surface<half> sTensor(b * h * s_q * s_kv, false);
    if(is_inference == false) {
        variant_pack[outputs.S] = sTensor.devPtr;
    }
    
    Surface<int8_t> workspace(mha_graph.get_workspace_size(), false);
    REQUIRE(fe::error_t::OK == mha_graph.execute(handle, variant_pack, workspace.devPtr));

    checkCudaErr(cudaDeviceSynchronize());

    cudnnDestroy(handle);
}

TEST_CASE("Scaled dot product Graphs with No Dropout", "[graph][mha][non_flash][forward]") {
    int64_t b = 32;  // batch size
    int64_t h = 16;  // head dim
    int64_t s_q = 512; // q tensor is padded to this seq length
    int64_t s_kv = 512; // k and v tensor is padded to this seq length
    int64_t d = 64;  // hidden dim
    bool is_inference = true;

    namespace fe = cudnn_frontend;
    fe::graph::Graph mha_graph("mha");
    mha_graph.set_io_data_type(fe::DataType_t::HALF)
             .set_intermediate_data_type(fe::DataType_t::FLOAT)
             .set_compute_data_type(fe::DataType_t::FLOAT);

    fe::graph::Scaled_dot_product_attention::Inputs inputs;
    inputs.Q = mha_graph.tensor(fe::graph::Tensor("Q").set_dim({b,h,s_q,d}).set_stride({s_q*3*h*d,d,3*h*d,1}));
    inputs.K = mha_graph.tensor(fe::graph::Tensor("K").set_dim({b,h,d,s_kv}).set_stride({s_kv*3*h*d,d,1,3*h*d}));
    inputs.V = mha_graph.tensor(fe::graph::Tensor("V").set_dim({b,h,s_kv,d}).set_stride({s_kv*3*h*d,d,3*h*d,1}));
    inputs.SEQ_LEN_Q = mha_graph.tensor(fe::graph::Tensor("SEQ_LEN_Q").set_dim({b,1,1,1}).set_stride({1,1,1,1}).set_data_type(fe::DataType_t::INT32));
    inputs.SEQ_LEN_K = mha_graph.tensor(fe::graph::Tensor("SEQ_LEN_K").set_dim({b,1,1,1}).set_stride({1,1,1,1}).set_data_type(fe::DataType_t::INT32));

    auto bias = mha_graph.tensor(fe::graph::Tensor("Bias").set_dim({1,h,s_q,s_kv}).set_stride({h*s_q*s_kv,s_q*s_kv,s_kv,1}));
    
    auto scaled_dot_product_attention_options = fe::graph::Scaled_dot_product_attention("mha")
                                                    .set_is_inference(is_inference)
                                                    .set_bias(bias)
                                                    .use_padding_mask()
                                                    .set_scale_k(0.5f);
    auto outputs = mha_graph.scaled_dot_product_attention(inputs, scaled_dot_product_attention_options);

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
    REQUIRE(fe::error_t::OK == mha_graph.build(handle));

    auto plans = mha_graph.get_execution_plan_list(fe::HeurMode_t::HEUR_MODE_A)
                    .build_plans(handle);

    REQUIRE(fe::error_t::OK == mha_graph.set_executor(plans));

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

    Surface<half> bTensor(1 * h * s_q * s_kv, false);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor>, void*> variant_pack = {
        {inputs.Q, devPtrQ}
        , {inputs.K, devPtrK}
        , {inputs.SEQ_LEN_Q, devActualSeqlenQ.devPtr}
        , {inputs.SEQ_LEN_K, devActualSeqlenK.devPtr}
        , {bias, bTensor.devPtr}
        , {inputs.V, devPtrV}
        , {outputs.O, devPtrO}
    };

    Surface<half> sTensor(b * h * s_q * s_kv, false);
    if(is_inference == false) {
        variant_pack[outputs.S] = sTensor.devPtr;
    }
    
    Surface<int8_t> workspace(mha_graph.get_workspace_size(), false);
    REQUIRE(fe::error_t::OK == mha_graph.execute(handle, variant_pack, workspace.devPtr));

    checkCudaErr(cudaDeviceSynchronize());

    cudnnDestroy(handle);
}

TEST_CASE("Scaled dot product Graphs with Dropout Mask", "[graph][mha][non_flash][forward]") {
    int64_t b = 32;  // batch size
    int64_t h = 16;  // head dim
    int64_t s_q = 512; // q tensor is padded to this seq length
    int64_t s_kv = 512; // k and v tensor is padded to this seq length
    int64_t d = 64;  // hidden dim
    bool is_inference = false;

    namespace fe = cudnn_frontend;
    fe::graph::Graph mha_graph("mha");
    mha_graph.set_io_data_type(fe::DataType_t::HALF)
             .set_intermediate_data_type(fe::DataType_t::FLOAT)
             .set_compute_data_type(fe::DataType_t::FLOAT);

    fe::graph::Scaled_dot_product_attention::Inputs inputs;
    inputs.Q = mha_graph.tensor(fe::graph::Tensor("Q").set_dim({b,h,s_q,d}).set_stride({s_q*3*h*d,d,3*h*d,1}));
    inputs.K = mha_graph.tensor(fe::graph::Tensor("K").set_dim({b,h,d,s_kv}).set_stride({s_kv*3*h*d,d,1,3*h*d}));
    inputs.V = mha_graph.tensor(fe::graph::Tensor("V").set_dim({b,h,s_kv,d}).set_stride({s_kv*3*h*d,d,3*h*d,1}));
    inputs.SEQ_LEN_Q = mha_graph.tensor(fe::graph::Tensor("SEQ_LEN_Q").set_dim({b,1,1,1}).set_stride({1,1,1,1}).set_data_type(fe::DataType_t::INT32));
    inputs.SEQ_LEN_K = mha_graph.tensor(fe::graph::Tensor("SEQ_LEN_K").set_dim({b,1,1,1}).set_stride({1,1,1,1}).set_data_type(fe::DataType_t::INT32));

    auto dropout_mask = mha_graph.tensor(fe::graph::Tensor("Dropout_mask").set_dim({b,h,s_q,s_kv}).set_stride({s_q*s_kv*h,s_q*s_kv,s_kv,1}));
    float dropout_scale = 0.5f;

    auto scaled_dot_product_attention_options = fe::graph::Scaled_dot_product_attention("mha")
                                                    .set_is_inference(is_inference)
                                                    .set_dropout(dropout_mask, dropout_scale)
                                                    .set_scale_k(0.5f);
    auto outputs = mha_graph.scaled_dot_product_attention(inputs, scaled_dot_product_attention_options);

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
    REQUIRE(fe::error_t::OK == mha_graph.build(handle));

    auto plans = mha_graph.get_execution_plan_list(fe::HeurMode_t::HEUR_MODE_A)
                    .build_plans(handle);

    REQUIRE(fe::error_t::OK == mha_graph.set_executor(plans));

    //// Build variant pack
    Surface<half> qkvTensor(b * s_q * 3 * h * d, false);
    Surface<half> dropoutMaskTensor(b * s_q * h * s_kv, false);
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

    std::unordered_map<std::shared_ptr<fe::graph::Tensor>, void*> variant_pack = {
        {inputs.Q, devPtrQ}
        , {inputs.K, devPtrK}
        , {inputs.SEQ_LEN_Q, devActualSeqlenQ.devPtr}
        , {inputs.SEQ_LEN_K, devActualSeqlenK.devPtr}
        , {dropout_mask, dropoutMaskTensor.devPtr}
        , {inputs.V, devPtrV}
        , {outputs.O, devPtrO}
    };

    Surface<half> sTensor(b * h * s_q * s_kv, false);
    if(is_inference == false) {
        variant_pack[outputs.S] = sTensor.devPtr;
    }
    
    Surface<int8_t> workspace(mha_graph.get_workspace_size(), false);
    REQUIRE(fe::error_t::OK == mha_graph.execute(handle, variant_pack, workspace.devPtr));

    checkCudaErr(cudaDeviceSynchronize());

    cudnnDestroy(handle);
}
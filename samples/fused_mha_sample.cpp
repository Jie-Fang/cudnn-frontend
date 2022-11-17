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

#include "fused_mha_sample.h"
#include <cudnn_frontend.h>
#include "error_util.h"

static bool
allowAllConfig(cudnnBackendDescriptor_t engine_config) {
    (void)engine_config;
    return false;
}

static cudnn_frontend::ExecutionPlan
get_execplan_from_heuristics(cudnn_frontend::OperationGraph&& opGraph, cudnnHandle_t handle_) {
#if (CUDNN_VERSION >= 8200)
    {
        auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                              .setOperationGraph(opGraph)
                              .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                              .build();

        std::cout << "Heuristic has " << heuristics.getEngineConfigCount() << " configurations " << std::endl;
        auto& engine_config = heuristics.getEngineConfig(heuristics.getEngineConfigCount());

        // Try engine configs returned by the heuristics and pick up the first one that works.
        for (auto& ecfg : engine_config) {
            try {
                auto plan = cudnn_frontend::ExecutionPlanBuilder()
                                .setHandle(handle_)
                                .setEngineConfig(ecfg, opGraph.getTag())
                                .build();
                return plan;
            } catch (cudnn_frontend::cudnnException& e) {
                continue;
            }
        }
    }
#endif

    {
        cudnn_frontend::EngineConfigList filtered_configs;
        auto statuses = 
            cudnn_frontend::get_heuristics_list<1>({
            "heuristics_fallback"
            }, opGraph,::allowAllConfig, filtered_configs, true);
        
        std::cout << "get_heuristics_list Statuses: ";
        for (auto status : statuses) {
            std::cout << cudnn_frontend::to_string(status) << " ";
        }
        std::cout << std::endl;
        std::cout << "Filter config list has " << filtered_configs.size() << " configurations " << std::endl;

        return cudnn_frontend::ExecutionPlanBuilder().setHandle(handle_).setEngineConfig(filtered_configs[0], opGraph.getTag()).build();
    }
}

#if (CUDNN_VERSION >= 8700)
void
run_b2b_batch_gemm(int64_t* q_dim,
                    int64_t* k_dim,
                    int64_t* s_dim,
                    int64_t* v_dim,
                    int64_t* o_dim,
                    void* devPtrQ,
                    void* devPtrK,
                    void* devPtrV,
                    void* devPtrO,
                    cudnnDataType_t tensorType,
                    int32_t nbDims,                         
                    int64_t* q_stride,
                    int64_t* k_stride,
                    int64_t* s_stride,
                    int64_t* v_stride,
                    int64_t* o_stride ) {
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        // Creates the necessary tensor descriptors
        auto qTensor = cudnn_frontend::TensorBuilder()
                           .setDim(nbDims, q_dim)
                           .setStrides(nbDims, q_stride)
                           .setId('q')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(tensorType)
                           .build();

        auto kTensor = cudnn_frontend::TensorBuilder()
                           .setDim(nbDims, k_dim)
                           .setStrides(nbDims, k_stride)
                           .setId('k')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(tensorType)
                           .build();

        // first GEMM output
        auto sTensor = cudnn_frontend::TensorBuilder()
                           .setDim(nbDims, s_dim)
                           .setStrides(nbDims, s_stride)
                           .setId('s')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(tensorType)
                           .setVirtual() // first GEMM output is virtual
                           .build();

        auto vTensor = cudnn_frontend::TensorBuilder()
                           .setDim(nbDims, v_dim)
                           .setStrides(nbDims, v_stride)
                           .setId('v')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(tensorType)
                           .build();

        // second GEMM output
        auto oTensor = cudnn_frontend::TensorBuilder()
                           .setDim(nbDims, o_dim)
                           .setStrides(nbDims, o_stride)
                           .setId('o')
                           .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                           .setDataType(tensorType)
                           .build();
        

        std::cout << qTensor.describe() << std::endl;
        std::cout << kTensor.describe() << std::endl;
        std::cout << sTensor.describe() << std::endl;
        std::cout << vTensor.describe() << std::endl;
        std::cout << oTensor.describe() << std::endl;

        // Define the matmul 1 desc
        auto matmul_1_Desc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();
        std::cout << matmul_1_Desc.describe() << std::endl;

        // Create a matmul 1 Node
        auto matmul_1_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                             .setaMatDesc(qTensor)
                             .setbMatDesc(kTensor)
                             .setcMatDesc(sTensor)
                             .setmatmulDesc(matmul_1_Desc)
                             .build();
        std::cout << matmul_1_op.describe() << std::endl;

        // Define the matmul 2 desc
        auto matmul_2_Desc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();
        std::cout << matmul_2_Desc.describe() << std::endl;

        // Create a matmul 2 Node
        auto matmul_2_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                             .setaMatDesc(sTensor)
                             .setbMatDesc(vTensor)
                             .setcMatDesc(oTensor)
                             .setmatmulDesc(matmul_2_Desc)
                             .build();
        std::cout << matmul_2_op.describe() << std::endl;

        // Create an Operation Graph. In this case it is gemm-gemm
        std::array<cudnn_frontend::Operation const*, 2> ops = {&matmul_1_op, &matmul_2_op};
        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(ops.size(), ops.data())
                           .build();

        auto plan = get_execplan_from_heuristics(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
        }
        void* data_ptrs[] = {devPtrQ, devPtrK, devPtrV, devPtrO};
        int64_t uids[]    = {'q', 'k', 'v', 'o'};
        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(4, data_ptrs)
                               .setUids(4, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }

        checkCudnnErr(cudnnDestroy(handle_));

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties( &prop, 0 ));
        
        // this example is only for GA100 cards
        if (!(prop.major == 8 && prop.minor == 0) && (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH || e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
            std::cout << "Example is only supported for GA100 Ampere GPUs" << std::endl; 
        }  else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
        }
    }
}

static void 
createScale(int64_t b, 
            int64_t h, 
            int64_t s_q,
            int64_t s_kv,
            int64_t d,
            MHA_Layout layout, 
            cudnnDataType_t tensorType,
            std::vector<cudnn_frontend::Operation>& ops) {
    int nbDims = 4;

    // scale
    int64_t scale_dim [4] = {1, 1, 1, 1};
    int64_t scale_stride [4] = {1, 1, 1, 1};
    auto scaleTensor = cudnn_frontend::TensorBuilder()
                            .setDim(nbDims, scale_dim)
                            .setStrides(nbDims, scale_stride)
                            .setId('C')
                            .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                            .setDataType(tensorType)
                            .setByValue(true)
                            .build();
    std::cout << scaleTensor.describe() << std::endl;

    int64_t k_dim [4] =  {b, h, d, s_kv};
    int64_t k_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, k_stride, layout, MHA_Matrix::K_Matrix);
    auto kTensor = cudnn_frontend::TensorBuilder()
                        .setDim(nbDims, k_dim)
                        .setStrides(nbDims, k_stride)
                        .setId('K')
                        .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                        .setDataType(tensorType)
                        .build();
    std::cout << kTensor.describe() << std::endl;

    auto afterScaleKTensor = cudnn_frontend::TensorBuilder()
                        .setDim(nbDims, k_dim)
                        .setStrides(nbDims, k_stride)
                        .setId('q')
                        .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                        .setDataType(tensorType)
                        .setVirtual()
                        .build();
    std::cout << afterScaleKTensor.describe() << std::endl;

    // Define the scale descriptor
    auto scaleDesc = cudnn_frontend::PointWiseDescBuilder()
                        .setMode(CUDNN_POINTWISE_MUL)
                        .setComputeType(CUDNN_DATA_FLOAT)
                        .build();
    std::cout << scaleDesc.describe() << std::endl;

    // Create a Scale Node.
    auto scale_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(kTensor)
                        .setbDesc(scaleTensor)
                        .setyDesc(afterScaleKTensor)
                        .setpwDesc(scaleDesc)
                        .build();

    std::cout << scale_op.describe() << std::endl;

    ops.push_back(std::move(scale_op));
}

static void
createBMM1(int64_t b, 
           int64_t h, 
           int64_t s_q,
           int64_t s_kv,
           int64_t d,
           MHA_Layout layout,
           cudnnDataType_t tensorType,
           std::vector<cudnn_frontend::Operation>& ops) {
    int nbDims = 4;

    // Creates the necessary tensor descriptors
    int64_t q_dim [4] = {b, h, s_q, d};
    int64_t q_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, q_stride, layout, MHA_Matrix::Q_Matrix);
    auto qTensor = cudnn_frontend::TensorBuilder()
                        .setDim(nbDims, q_dim)
                        .setStrides(nbDims, q_stride)
                        .setId('Q')
                        .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                        .setDataType(tensorType)
                        .build();

    int64_t k_dim [4] =  {b, h, d, s_kv};
    int64_t k_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, k_stride, layout, MHA_Matrix::K_Matrix);
    auto afterScaleKTensor = cudnn_frontend::TensorBuilder()
                        .setDim(nbDims, k_dim)
                        .setStrides(nbDims, k_stride)
                        .setId('q')
                        .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                        .setDataType(tensorType)
                        .setVirtual()
                        .build();

    int64_t p_dim [4] = {b, h, s_q, s_kv};
    int64_t p_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, p_stride, layout, MHA_Matrix::S_Matrix);
    // first GEMM output
    auto pTensor = cudnn_frontend::TensorBuilder()
                        .setDim(nbDims, p_dim)
                        .setStrides(nbDims, p_stride)
                        .setId('P')
                        .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                        .setDataType(CUDNN_DATA_FLOAT)
                        .setVirtual() // first GEMM output is virtual
                        .build();

    int64_t seqlenQ_dim [4] =  {b, 1, 1, 1};
    int64_t seqlenQ_stride [4] = {1, 1, 1, 1};
    auto seqlenQTensor = cudnn_frontend::TensorBuilder()
                            .setDim(nbDims, seqlenQ_dim)
                            .setStrides(nbDims, seqlenQ_stride)
                            .setId('I')
                            .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                            .setDataType(CUDNN_DATA_INT32)
                            .build();

    int64_t seqlenK_dim [4] =  {b, 1, 1, 1};
    int64_t seqlenK_stride [4] = {1, 1, 1, 1};
    auto seqlenKTensor = cudnn_frontend::TensorBuilder()
                            .setDim(nbDims, seqlenK_dim)
                            .setStrides(nbDims, seqlenK_stride)
                            .setId('J')
                            .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                            .setDataType(CUDNN_DATA_INT32)
                            .build();

    std::cout << qTensor.describe() << std::endl;
    std::cout << pTensor.describe() << std::endl;
    std::cout << seqlenQTensor.describe() << std::endl;
    std::cout << seqlenKTensor.describe() << std::endl;

    // Define the matmul 1 desc
    auto matmul_1_Desc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();
    std::cout << matmul_1_Desc.describe() << std::endl;

    // Create a matmul 1 Node
    auto matmul_op1 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(qTensor)
                            .setbMatDesc(afterScaleKTensor)
                            .setcMatDesc(pTensor)
                            .setmOverrideDesc(seqlenQTensor)
                            .setnOverrideDesc(seqlenKTensor)
                            .setmatmulDesc(matmul_1_Desc)
                            .build();

    std::cout << matmul_op1.describe() << std::endl;

    ops.push_back(std::move(matmul_op1));
}

static void
createBias(int64_t b, 
           int64_t h, 
           int64_t s_q,
           int64_t s_kv,
           int64_t d,
           MHA_Layout layout,
           cudnnDataType_t tensorType,
           std::vector<cudnn_frontend::Operation>& ops) {
    int nbDims = 4;

    cudnn_frontend::throw_if(ops.size() == 0, "Bias op constructed incorrectly as the first one", CUDNN_STATUS_BAD_PARAM);

    int64_t b_dim [4] = {1, h, s_q, s_kv};
    int64_t b_stride [4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};
    // bias
    auto bTensor = cudnn_frontend::TensorBuilder()
                        .setDim(nbDims, b_dim)
                        .setStrides(nbDims, b_stride)
                        .setId('B')
                        .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                        .setDataType(tensorType)
                        .build();

    int64_t afterBias_dim [4] = {b, h, s_q, s_kv};
    int64_t afterBias_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, afterBias_stride, layout, MHA_Matrix::S_Matrix);
    // output
    auto afterBiasTensor = cudnn_frontend::TensorBuilder()
                        .setDim(nbDims, afterBias_dim)
                        .setStrides(nbDims, afterBias_stride)
                        .setId('a')
                        .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                        .setDataType(CUDNN_DATA_FLOAT)
                        .setVirtual()
                        .build();

    std::cout << bTensor.describe() << std::endl;
    std::cout << afterBiasTensor.describe() << std::endl;

    // Define the bias descriptor
    auto biasDesc = cudnn_frontend::PointWiseDescBuilder()
                        .setMode(CUDNN_POINTWISE_ADD)
                        .setComputeType(CUDNN_DATA_FLOAT)
                        .build();
    std::cout << biasDesc.describe() << std::endl;

    // Create a Bias Node.
    auto bias_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(ops[ops.size() - 1].getOutputTensor())
                        .setbDesc(bTensor)
                        .setyDesc(afterBiasTensor)
                        .setpwDesc(biasDesc)
                        .build();

    std::cout << bias_op.describe() << std::endl;

    ops.push_back(std::move(bias_op));
}

static void
createMask(int64_t b, 
           int64_t h, 
           int64_t s_q,
           int64_t s_kv,
           int64_t d,
           MHA_Layout layout,
           bool is_causal_masking,
           cudnnDataType_t tensorType,
           std::vector<cudnn_frontend::Operation>& ops) {

    CUDNN_FRONTEND_UNUSED(d);
    CUDNN_FRONTEND_UNUSED(layout);
    CUDNN_FRONTEND_UNUSED(tensorType);
    int nbDims = 4;

    cudnn_frontend::throw_if(ops.size() == 0, "Padding Mask constructed incorrectly as the first one", CUDNN_STATUS_BAD_PARAM);

    auto tensor_create = [&nbDims](cudnnDataType_t type,
                                int64_t id, int64_t const * dim, int64_t const * stride) {
            return cudnn_frontend::TensorBuilder()
                   .setDim(nbDims, dim)
                   .setStride(nbDims, stride)
                   .setId(id) 
                   .setAlignment(16) // 16B alignment is needed to run a tensor core engine
                   .setDataType(type)
                   .setVirtual()
                   .build();
    };

    auto prevBlockOutputTensor = ops[ops.size() - 1].getOutputTensor();

    // subtraction output
    int64_t afterBMM1_dim [4] = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride [4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t seqlenQ_dim [4] =  {b, 1, 1, 1};
    int64_t seqlenQ_stride [4] = {1, 1, 1, 1};
    auto seqlenQTensor = cudnn_frontend::TensorBuilder()
                            .setDim(nbDims, seqlenQ_dim)
                            .setStrides(nbDims, seqlenQ_stride)
                            .setId('I')
                            .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                            .setDataType(CUDNN_DATA_INT32)
                            .build();

    int64_t seqlenK_dim [4] =  {b, 1, 1, 1};
    int64_t seqlenK_stride [4] = {1, 1, 1, 1};
    auto seqlenKTensor = cudnn_frontend::TensorBuilder()
                            .setDim(nbDims, seqlenK_dim)
                            .setStrides(nbDims, seqlenK_stride)
                            .setId('J')
                            .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                            .setDataType(CUDNN_DATA_INT32)
                            .build();

    // gen index row output
    auto rowIndexTensor = tensor_create(CUDNN_DATA_FLOAT, 'g', afterBMM1_dim, afterBMM1_stride);
    std::cout << rowIndexTensor.describe() << std::endl;

    // Define the gen index for row descriptor
    auto genIndexRowDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_GEN_INDEX)
                            .setAxis(2)
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();
    std::cout << genIndexRowDesc.describe() << std::endl;

    // Create a gen index Node.
    auto genIndexRow_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(prevBlockOutputTensor)
                        .setyDesc(rowIndexTensor)
                        .setpwDesc(genIndexRowDesc)
                        .build();

    std::cout << genIndexRow_op.describe() << std::endl;

    ops.push_back(std::move(genIndexRow_op));

    // gen index column output
    auto columnIndexTensor = tensor_create(CUDNN_DATA_FLOAT, 'h', afterBMM1_dim, afterBMM1_stride);
    std::cout << columnIndexTensor.describe() << std::endl;

    // Define the gen index for row descriptor
    auto genIndexColumnDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_GEN_INDEX)
                            .setAxis(3)
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();
    std::cout << genIndexColumnDesc.describe() << std::endl;

    // Create a gen index Node.
    auto genIndexColumn_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(prevBlockOutputTensor)
                        .setyDesc(columnIndexTensor)
                        .setpwDesc(genIndexColumnDesc)
                        .build();

    std::cout << genIndexColumn_op.describe() << std::endl;

    ops.push_back(std::move(genIndexColumn_op));

    // less than row output
    auto lessThanRowTensor = tensor_create(CUDNN_DATA_BOOLEAN, 'i', afterBMM1_dim, afterBMM1_stride);
    std::cout << lessThanRowTensor.describe() << std::endl;

    // Define the less than comparison for row descriptor
    auto lessThanRowDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_CMP_LT)
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();
    std::cout << lessThanRowDesc.describe() << std::endl;

    // Create a less than comparison for row Node.
    auto lessThanRow_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(rowIndexTensor)
                        .setbDesc(seqlenQTensor)
                        .setyDesc(lessThanRowTensor)
                        .setpwDesc(lessThanRowDesc)
                        .build();

    std::cout << lessThanRow_op.describe() << std::endl;

    ops.push_back(std::move(lessThanRow_op));

    // less than column output
    auto lessThanColTensor = tensor_create(CUDNN_DATA_BOOLEAN, 'j', afterBMM1_dim, afterBMM1_stride);
    std::cout << lessThanColTensor.describe() << std::endl;

    // Define the less than comparison for column descriptor
    auto lessThanColDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_CMP_LT)
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();
    std::cout << lessThanColDesc.describe() << std::endl;

    // Create a less than comparison for col Node.
    auto lessThanCol_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(columnIndexTensor)
                        .setbDesc(seqlenKTensor)
                        .setyDesc(lessThanColTensor)
                        .setpwDesc(lessThanColDesc)
                        .build();

    std::cout << lessThanCol_op.describe() << std::endl;

    ops.push_back(std::move(lessThanCol_op));
    
    // padding mask
    auto paddingMaskTensor = tensor_create(CUDNN_DATA_BOOLEAN, 'k', afterBMM1_dim, afterBMM1_stride);
    std::cout << paddingMaskTensor.describe() << std::endl;

    // Define the less than comparison for column descriptor
    auto paddingMaskAndDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_LOGICAL_AND)
                            .setComputeType(CUDNN_DATA_BOOLEAN)
                            .build();
    std::cout << paddingMaskAndDesc.describe() << std::endl;

    // Create a less than comparison for col Node.
    auto paddingMaskAnd_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(lessThanRowTensor)
                        .setbDesc(lessThanColTensor)
                        .setyDesc(paddingMaskTensor)
                        .setpwDesc(paddingMaskAndDesc)
                        .build();

    std::cout << paddingMaskAnd_op.describe() << std::endl;

    ops.push_back(std::move(paddingMaskAnd_op));

    // row >= col check for causal mask
    auto rowGreaterColTensor = tensor_create(CUDNN_DATA_BOOLEAN, 'l', afterBMM1_dim, afterBMM1_stride);
    std::cout << rowGreaterColTensor.describe() << std::endl;

    // Define the greater than equal to comparison descriptor
    auto rowGreaterColDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_CMP_GE)
                            .setComputeType(CUDNN_DATA_BOOLEAN)
                            .build();
    std::cout << rowGreaterColDesc.describe() << std::endl;

    // Create a greater than equal to Node.
    auto rowGreaterCol_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(rowIndexTensor)
                        .setbDesc(columnIndexTensor)
                        .setyDesc(rowGreaterColTensor)
                        .setpwDesc(rowGreaterColDesc)
                        .build();

    std::cout << rowGreaterCol_op.describe() << std::endl;

    if (is_causal_masking) ops.push_back(std::move(rowGreaterCol_op));

    // creeate causal mask
    auto causalMaskTensor = tensor_create(CUDNN_DATA_BOOLEAN, 'm', afterBMM1_dim, afterBMM1_stride);
    std::cout << causalMaskTensor.describe() << std::endl;

    // Define the and to create causal mask descriptor
    auto causalMaskAndDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_LOGICAL_AND)
                            .setComputeType(CUDNN_DATA_BOOLEAN)
                            .build();
    std::cout << causalMaskAndDesc.describe() << std::endl;

    // Create a causal Mask Node.
    auto causalMaskAnd_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(paddingMaskTensor)
                        .setbDesc(rowGreaterColTensor)
                        .setyDesc(causalMaskTensor)
                        .setpwDesc(causalMaskAndDesc)
                        .build();

    std::cout << causalMaskAnd_op.describe() << std::endl;

    if (is_causal_masking) ops.push_back(std::move(causalMaskAnd_op));

    /////////////////// Apply the mask //////////////////////////

    auto maskTensor = (is_causal_masking) ? std::move(causalMaskTensor) : std::move(paddingMaskTensor);
    auto maskOutputTensor = tensor_create(CUDNN_DATA_FLOAT, 'n', afterBMM1_dim, afterBMM1_stride);

    int64_t negInf_dim [4] =  {1, 1, 1, 1};
    int64_t negInf_stride [4] = {1, 1, 1, 1};

    auto negInfTensor = cudnn_frontend::TensorBuilder()
                   .setDim(nbDims, negInf_dim)
                   .setStride(nbDims, negInf_stride)
                   .setId('N') 
                   .setAlignment(16) // 16B alignment is needed to run a tensor core engine
                   .setDataType(CUDNN_DATA_FLOAT)
                   .setByValue(true)
                   .build();
    std::cout << negInfTensor.describe() << std::endl;

    // Define the binary select to perform masking descriptor
    auto maskDesc = cudnn_frontend::PointWiseDescBuilder()
                            .setMode(CUDNN_POINTWISE_BINARY_SELECT)
                            .setComputeType(CUDNN_DATA_FLOAT)
                            .build();
    std::cout << maskDesc.describe() << std::endl;

    // Create a binary select Node.
    auto mask_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(prevBlockOutputTensor)
                        .setbDesc(negInfTensor)
                        .settDesc(maskTensor)
                        .setyDesc(maskOutputTensor)
                        .setpwDesc(maskDesc)
                        .build();

    std::cout << mask_op.describe() << std::endl;

    ops.push_back(std::move(mask_op));
}

static void
createSoftmaxForward(int64_t b, 
                     int64_t h, 
                     int64_t s_q,
                     int64_t s_kv,
                     int64_t d,
                     MHA_Layout layout,
                     bool enable_dropout,
                     cudnnDataType_t tensorType,
                     std::vector<cudnn_frontend::Operation>& ops) {
    CUDNN_FRONTEND_UNUSED(d);
    CUDNN_FRONTEND_UNUSED(layout);
    int nbDims = 4;

    cudnn_frontend::throw_if(ops.size() == 0, "Softmax DAG constructed incorrectly as the first one", CUDNN_STATUS_BAD_PARAM);

    auto tensor_create = [&nbDims](cudnnDataType_t type,
                                int64_t id, int64_t const * dim, int64_t const * stride) {
            return cudnn_frontend::TensorBuilder()
                   .setDim(nbDims, dim)
                   .setStride(nbDims, stride)
                   .setId(id) 
                   .setAlignment(16) // 16B alignment is needed to run a tensor core engine
                   .setDataType(type)
                   .setVirtual()
                   .build();
    };

    int64_t afterBMM1_dim [4] = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride [4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    int64_t afterReduction_dim [4] = {b, h, s_q, 1};
    int64_t afterReduction_stride [4] = {h * s_q, s_q, 1, 1};

    auto afterMaxReductionTensor = tensor_create(CUDNN_DATA_FLOAT, 'b', afterReduction_dim, afterReduction_stride);
    std::cout << afterMaxReductionTensor.describe() << std::endl;

    // Define the reduction descriptor
    auto reductionMaxDesc = cudnn_frontend::ReductionDescBuilder()
                                .setComputeType(CUDNN_DATA_FLOAT)
                                .setReductionOp(CUDNN_REDUCE_TENSOR_MAX)
                                .build();
    std::cout << reductionMaxDesc.describe() << std::endl;

    // Create a reduction max Node.
    auto reductionMax_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(ops[ops.size() - 1].getOutputTensor())
                                .setyDesc(afterMaxReductionTensor)
                                .setreductionDesc(reductionMaxDesc)
                                .build();

    std::cout << reductionMax_op.describe() << std::endl;

    ops.push_back(std::move(reductionMax_op));

    // subtract output
    auto afterSubtractionTensor = tensor_create(CUDNN_DATA_FLOAT, 'c', afterBMM1_dim, afterBMM1_stride);
    std::cout << afterSubtractionTensor.describe() << std::endl;

    // Define the subtract descriptor
    auto subtractDesc = cudnn_frontend::PointWiseDescBuilder()
                        .setMode(CUDNN_POINTWISE_SUB)
                        .setComputeType(CUDNN_DATA_FLOAT)
                        .build();
    std::cout << subtractDesc.describe() << std::endl;

    // Create a subtract Node.
    auto subtract_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(ops[ops.size() - 2].getOutputTensor())
                        .setbDesc(afterMaxReductionTensor)
                        .setyDesc(afterSubtractionTensor)
                        .setpwDesc(subtractDesc)
                        .build();

    std::cout << subtract_op.describe() << std::endl;

    ops.push_back(std::move(subtract_op));

    // exponent output
    auto afterExponentTensor = tensor_create(CUDNN_DATA_FLOAT, 'd', afterBMM1_dim, afterBMM1_stride);
    std::cout << afterExponentTensor.describe() << std::endl;

    // Define the exponent descriptor
    auto exponentDesc = cudnn_frontend::PointWiseDescBuilder()
                        .setMode(CUDNN_POINTWISE_EXP)
                        .setComputeType(CUDNN_DATA_FLOAT)
                        .build();
    std::cout << exponentDesc.describe() << std::endl;

    // Create a exponent Node.
    auto exponent_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(afterSubtractionTensor)
                        .setyDesc(afterExponentTensor)
                        .setpwDesc(exponentDesc)
                        .build();

    std::cout << exponent_op.describe() << std::endl;

    ops.push_back(std::move(exponent_op));

    // add reduction result
    auto afterAddReductionTensor = tensor_create(CUDNN_DATA_FLOAT, 'e', afterReduction_dim, afterReduction_stride);
    std::cout << afterAddReductionTensor.describe() << std::endl;

    // Define the reduction descriptor
    auto reductionAddDesc = cudnn_frontend::ReductionDescBuilder()
                                .setComputeType(CUDNN_DATA_FLOAT)
                                .setReductionOp(CUDNN_REDUCE_TENSOR_ADD)
                                .build();
    std::cout << reductionAddDesc.describe() << std::endl;

    // Create a reduction add Node.
    auto reductionAdd_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                .setxDesc(afterExponentTensor)
                                .setyDesc(afterAddReductionTensor)
                                .setreductionDesc(reductionAddDesc)
                                .build();

    std::cout << reductionAdd_op.describe() << std::endl;

    ops.push_back(std::move(reductionAdd_op));

    cudnnDataType_t softmaxOutputType = (enable_dropout) ? CUDNN_DATA_FLOAT : tensorType;
    auto afterDivisionTensor = tensor_create(softmaxOutputType, 'f', afterBMM1_dim, afterBMM1_stride);
    std::cout << afterDivisionTensor.describe() << std::endl;

    // Define the division descriptor
    auto divisionDesc = cudnn_frontend::PointWiseDescBuilder()
                        .setMode(CUDNN_POINTWISE_DIV)
                        .setComputeType(CUDNN_DATA_FLOAT)
                        .build();
    std::cout << divisionDesc.describe() << std::endl;

    // Create a subtract Node.
    auto division_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(afterExponentTensor)
                        .setbDesc(afterAddReductionTensor)
                        .setyDesc(afterDivisionTensor)
                        .setpwDesc(divisionDesc)
                        .build();

    std::cout << division_op.describe() << std::endl;

    ops.push_back(std::move(division_op));
}

static void
createDropout(int64_t b, 
              int64_t h, 
              int64_t s_q,
              int64_t s_kv,
              int64_t d,
              int64_t seed,
              double probability,
              cudnnDataType_t tensorType,
              std::vector<cudnn_frontend::Operation>& ops) {
    
    CUDNN_FRONTEND_UNUSED(d);
    int nbDims = 4;

    cudnn_frontend::throw_if(ops.size() == 0, "Dropout DAG constructed incorrectly as the first one", CUDNN_STATUS_BAD_PARAM);

    auto tensor_create = [&nbDims](cudnnDataType_t type,
                                int64_t id, int64_t const * dim, int64_t const * stride) {
            return cudnn_frontend::TensorBuilder()
                   .setDim(nbDims, dim)
                   .setStride(nbDims, stride)
                   .setId(id) 
                   .setAlignment(16) // 16B alignment is needed to run a tensor core engine
                   .setDataType(type)
                   .setVirtual()
                   .build();
    };

    int64_t afterBMM1_dim [4] = {b, h, s_q, s_kv};
    int64_t afterBMM1_stride [4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};

    // mask for the dropout
    auto dropoutMaskTensor = tensor_create(CUDNN_DATA_FLOAT, 'o', afterBMM1_dim, afterBMM1_stride);
    std::cout << dropoutMaskTensor.describe() << std::endl;

    // Define the reduction descriptor
    auto rngDesc = cudnn_frontend::RngDescBuilder()
                                .setRngDistribution(CUDNN_RNG_DISTRIBUTION_BERNOULLI)
                                .setBernoulliDistProbability(1.0 - probability)
                                .build();
    std::cout << rngDesc.describe() << std::endl;

    // Create a rng Node.
    auto rng_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_RNG_DESCRIPTOR)
                                .setyDesc(dropoutMaskTensor)
                                .setSeed(seed)
                                .setRngDesc(rngDesc)
                                .build();

    std::cout << rng_op.describe() << std::endl;

    ops.push_back(std::move(rng_op));

     // after dropout tensor
    auto afterDropoutTensor = cudnn_frontend::TensorBuilder()
                            .setDim(nbDims, afterBMM1_dim)
                            .setStrides(nbDims, afterBMM1_stride)
                            .setId('S')
                            .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                            .setDataType(tensorType)
                            .build();

    std::cout << afterDropoutTensor.describe() << std::endl;

    // Define the multiply mask descriptor
    auto maskMulDesc = cudnn_frontend::PointWiseDescBuilder()
                        .setMode(CUDNN_POINTWISE_MUL)
                        .setComputeType(CUDNN_DATA_FLOAT)
                        .build();
    std::cout << maskMulDesc.describe() << std::endl;

    // Create a multiply mask Node.
    auto maskMul_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(ops[ops.size() - 2].getOutputTensor())
                        .setbDesc(dropoutMaskTensor)
                        .setyDesc(afterDropoutTensor)
                        .setpwDesc(maskMulDesc)
                        .build();

    std::cout << maskMul_op.describe() << std::endl;

    ops.push_back(std::move(maskMul_op));

    // scale after dropout
    int64_t scale_dim [4] = {1, 1, 1, 1};
    int64_t scale_stride [4] = {1, 1, 1, 1};
    auto scaleDropoutTensor = cudnn_frontend::TensorBuilder()
                            .setDim(nbDims, scale_dim)
                            .setStrides(nbDims, scale_stride)
                            .setId('D')
                            .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                            .setDataType(tensorType)
                            .setByValue(true)
                            .build();
    std::cout << scaleDropoutTensor.describe() << std::endl;

    // after Scale
    auto afterScaleTensor = tensor_create(tensorType, 'p', afterBMM1_dim, afterBMM1_stride);
    std::cout << afterScaleTensor.describe() << std::endl;

    // Define the multiply scale descriptor
    auto scaleMulDesc = cudnn_frontend::PointWiseDescBuilder()
                        .setMode(CUDNN_POINTWISE_MUL)
                        .setComputeType(CUDNN_DATA_FLOAT)
                        .build();
    std::cout << scaleMulDesc.describe() << std::endl;

    // Create a multiply mask Node.
    auto scaleMul_op = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                        .setxDesc(afterDropoutTensor)
                        .setbDesc(scaleDropoutTensor)
                        .setyDesc(afterScaleTensor)
                        .setpwDesc(scaleMulDesc)
                        .build();

    std::cout << scaleMul_op.describe() << std::endl;

    ops.push_back(std::move(scaleMul_op));
    
}

static void
createBMM2(int64_t b, 
           int64_t h, 
           int64_t s_q,
           int64_t s_kv,
           int64_t d,
           MHA_Layout layout,
           cudnnDataType_t tensorType,
           std::vector<cudnn_frontend::Operation>& ops) {
    int nbDims = 4;

    cudnn_frontend::throw_if(ops.size() == 0, "BMM2 op constructed incorrectly as the first one", CUDNN_STATUS_BAD_PARAM);

    int64_t seqlenQ_dim [4] =  {b, 1, 1, 1};
    int64_t seqlenQ_stride [4] = {1, 1, 1, 1};
    auto seqlenQTensor = cudnn_frontend::TensorBuilder()
                            .setDim(nbDims, seqlenQ_dim)
                            .setStrides(nbDims, seqlenQ_stride)
                            .setId('I')
                            .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                            .setDataType(CUDNN_DATA_INT32)
                            .build();

    int64_t seqlenK_dim [4] =  {b, 1, 1, 1};
    int64_t seqlenK_stride [4] = {1, 1, 1, 1};
    auto seqlenKTensor = cudnn_frontend::TensorBuilder()
                            .setDim(nbDims, seqlenK_dim)
                            .setStrides(nbDims, seqlenK_stride)
                            .setId('J')
                            .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                            .setDataType(CUDNN_DATA_INT32)
                            .build();

    int64_t v_dim [4] =  {b, h, s_kv, d};
    int64_t v_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, v_stride, layout, MHA_Matrix::V_Matrix);
    auto vTensor = cudnn_frontend::TensorBuilder()
                        .setDim(nbDims, v_dim)
                        .setStrides(nbDims, v_stride)
                        .setId('V')
                        .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                        .setDataType(tensorType)
                        .build();

        
    int64_t o_dim [4] =  {b, h, s_q, d};
    int64_t o_stride [4];
    generateMHAStrides(b, h, s_q, s_kv, d, o_stride, layout, MHA_Matrix::O_Matrix);
    // second GEMM output
    auto oTensor = cudnn_frontend::TensorBuilder()
                        .setDim(nbDims, o_dim)
                        .setStrides(nbDims, o_stride)
                        .setId('O')
                        .setAlignment(16)  // 16B alignment is needed to run a tensor core engine
                        .setDataType(tensorType)
                        .build();

    std::cout << vTensor.describe() << std::endl;
    std::cout << oTensor.describe() << std::endl;

    // Define the matmul 2 desc
    auto matmul_2_Desc = cudnn_frontend::MatMulDescBuilder().setComputeType(CUDNN_DATA_FLOAT).build();
    std::cout << matmul_2_Desc.describe() << std::endl;

    // Create a matmul 2 Node
    auto matmul_op2 = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                            .setaMatDesc(ops[ops.size() - 1].getOutputTensor())
                            .setbMatDesc(vTensor)
                            .setcMatDesc(oTensor)
                            .setmOverrideDesc(seqlenQTensor)
                            .setkOverrideDesc(seqlenKTensor)
                            .setmatmulDesc(matmul_2_Desc)
                            .build();

    std::cout << matmul_op2.describe() << std::endl;

    ops.push_back(std::move(matmul_op2));
}

void 
run_mha_fprop(int64_t b, 
              int64_t h, 
              int64_t s_q,
              int64_t s_kv,
              int64_t d,
              int64_t seed,
              MHA_Layout layout,
              half1 scaling_factor,
              double dropout_probability,
              MHA_Bias_Type bias_type,
              bool is_causal_masking,
              void* devPtrQ, 
              void* devPtrK,   
              void* devPtrV,   
              void* devPtrS,
              void* devPtrO,
              void* devPtrBias,
              void* devActualSeqlenQ,
              void* devActualSeqlenK,
              cudnnDataType_t tensorType) {
                
    cudnnHandle_t handle_;
    try {
        // Create cudnn handle
        checkCudnnErr(cudnnCreate(&handle_));

        std::vector<cudnn_frontend::Operation const*> all_ops;
        std::vector<cudnn_frontend::Operation> ops;

        createScale(b, h, s_q, s_kv, d, layout, tensorType, ops);

        createBMM1(b, h, s_q, s_kv, d, layout, tensorType, ops);

        if (bias_type != MHA_Bias_Type::NO_BIAS) {
            createBias(b, h, s_q, s_kv, d, layout, tensorType, ops);
        }

        float negInfinity = -1.0E+20; // change this if you have access to float_min
        createMask(b, h, s_q, s_kv, d, layout, is_causal_masking, tensorType, ops);

        bool enable_dropout = (dropout_probability != 0.0f);
        cudnn_frontend::throw_if(dropout_probability == 1.0f, "Dropout probability cannot be 1.0", CUDNN_STATUS_BAD_PARAM);

        // needs to be bf16 (Please change)
        half1 scale_dropout = cpu_float2half_rn(1/(1 - dropout_probability));

        createSoftmaxForward(b, h, s_q, s_kv, d, layout, enable_dropout, tensorType, ops);

        if (dropout_probability != 0.0f) {
            createDropout(b, h, s_q, s_kv, d, seed, dropout_probability, tensorType, ops);
        }

        createBMM2(b, h, s_q, s_kv, d, layout, tensorType, ops);

        std::cout << "Total ops created: " << ops.size() << std::endl;

        for (unsigned int i = 0; i < ops.size(); i++) {
            all_ops.push_back(&ops[i]);
        }

        // Create an Operation Graph
        auto opGraph = cudnn_frontend::OperationGraphBuilder()
                           .setHandle(handle_)
                           .setOperationGraph(all_ops.size(), all_ops.data())
                           .build();


        auto plan = get_execplan_from_heuristics(std::move(opGraph), handle_);

        std::cout << "Plan tag: " << plan.getTag() << std::endl;

        auto workspace_size = plan.getWorkspaceSize();
        std::cout << plan.describe() << " requires workspace " << workspace_size << std::endl;

        void* workspace_ptr = nullptr;
        if (workspace_size > 0) {
            checkCudaErr(cudaMalloc(&workspace_ptr, workspace_size));
        }

        void* data_ptrs[] = {devPtrQ, devPtrK, devPtrV, devPtrO, 
                            devActualSeqlenQ, devActualSeqlenK, &negInfinity,
                            &scaling_factor, devPtrBias, devPtrS, &scale_dropout};
        int64_t uids[]    = {'Q', 'K', 'V', 'O', 'I', 'J', 'N', 'C', 'B', 'S', 'D'};
        int64_t number_of_tensors_variant_pack;

        if (bias_type != MHA_Bias_Type::NO_BIAS) {
            if (enable_dropout) {
                number_of_tensors_variant_pack = 11;
            } else {
                number_of_tensors_variant_pack = 9;
            }
        } else {
            if (enable_dropout) {
                data_ptrs[8] = &scale_dropout; 
                uids[8] = 'D';
                number_of_tensors_variant_pack = 10;
            } else {
                number_of_tensors_variant_pack = 8;
            }
        }

        auto variantPack  = cudnn_frontend::VariantPackBuilder()
                               .setWorkspacePointer(workspace_ptr)
                               .setDataPointers(number_of_tensors_variant_pack, data_ptrs)
                               .setUids(number_of_tensors_variant_pack, uids)
                               .build();
        std::cout << "variantPack " << variantPack.describe() << std::endl;
        cudnnStatus_t status = cudnnBackendExecute(handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
        if (workspace_size > 0) {
            checkCudaErr(cudaFree(workspace_ptr));
        }

        checkCudnnErr(cudnnDestroy(handle_));

        cudnn_frontend::throw_if([status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Plan execute error", status);

    } catch (cudnn_frontend::cudnnException& e) {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties( &prop, 0 ));
        
        // this example is only for GA100 cards
        if (!(prop.major == 8 && prop.minor == 0) && (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH || e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
            std::cout << "Example is only supported for GA100 Ampere GPUs" << std::endl; 
        }  else {
            std::cout << "[ERROR] Exception " << e.what() << std::endl;
            CHECK(false);
        }
    }
}
#endif
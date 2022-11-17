#pragma once

#include <memory>
#include <unordered_map>

#include <cudnn_frontend_Tensor.h>
#include <cudnn_frontend_Operation.h>
#include <cudnn_frontend_OperationGraph.h>
#include <cudnn_frontend_ExecutionPlan.h>
#include <cudnn_frontend_VariantPack.h>

namespace cudnn_frontend {

// Interface for all blocks to follow.
class IBlock {
private:

protected:

    // Tensors belonging to each block.
    // Connecting blocks can modify and delete tensors in this container.
    std::unordered_map<std::string, tensor_properties> tensor_props;

    std::unordered_map<std::string, std::shared_ptr<cudnn_frontend::Tensor>> tensors;
    std::unordered_map<std::string, std::shared_ptr<cudnn_frontend::Operation>> operations;
    std::unordered_map<std::string, std::shared_ptr<OperationGraph>> operation_graphs;
    std::unordered_map<std::string, std::shared_ptr<ExecutionPlan>> execution_plans;
    std::unordered_map<std::string, std::shared_ptr<VariantPack>> variant_packs;


    // Type of each block. Blocks can either be a composite (value BLOCK) or
    // one of the other primitive types. Primitives types are nothing but 
    // cudnn operations.
    enum class Type {
        BLOCK
        , CONVOLUTION
        , MATMUL
        , POINTWISE
        , REDUCTION
        , RESAMPLE
    };
    Type tag;
    
    virtual Type getType() = 0;
    
    virtual int validate() = 0;

    virtual int createTensors() = 0;
    virtual int createDescritpors() = 0;
    virtual int createOperations() = 0;

    virtual int partition(cudnnHandle_t& handle) = 0;

    virtual int createExecutionPlan(cudnnHandle_t& handle) = 0;

public:
    
    virtual int build(cudnnHandle_t& handle) = 0;
    
    virtual int execute(cudnnHandle_t& handle) = 0;

};

} // namespace cudnn_frontend
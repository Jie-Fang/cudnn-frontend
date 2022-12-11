#pragma once

#include <memory>
#include <vector>
#include <unordered_map>

#include "cudnn_frontend_Tensor.h"
#include "cudnn_frontend_Operation.h"
#include "cudnn_frontend_OperationGraph.h"
#include "cudnn_frontend_ExecutionPlan.h"
#include "cudnn_frontend_VariantPack.h"

#include "cudnn_frontend_ICudnn.h"

namespace cudnn_frontend {

// Interface for all blocks to follow.
class IBlock: public ICudnn {

    friend class ConvolutionFP8Block;
    friend class ConvolutionPointwiseBlock;

private:

protected:
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

    virtual int partition(cudnnHandle_t& handle) = 0;

    virtual int createTensors() {
        for(auto const& sub_block: sub_blocks) {
            sub_block.second->createTensors();
        }
        return 0;
    }
    
    virtual int createDescritpors() {
        for(auto const& sub_block: sub_blocks) {
            sub_block.second->createDescritpors();
        }
        return 0;
    }

    virtual int createOperations() {
        for(auto const& sub_block: sub_blocks) {
            sub_block.second->createOperations();
        }
        return 0;
    }
    
public:
    // Tensors belonging to each block.
    // Connecting blocks can modify and delete tensors in this container.
    std::unordered_map<int64_t, tensor_properties> tensor_props;

    std::unordered_map <std::string, std::shared_ptr<IBlock>> sub_blocks;

    virtual int build(cudnnHandle_t& handle) = 0;
    
    virtual int execute(cudnnHandle_t& handle, std::unordered_map<int64_t, void*> const& tensor_uid_to_pointer_map) = 0;

    virtual ~IBlock() {};
};

} // namespace cudnn_frontend
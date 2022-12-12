#pragma once

#include <memory>
#include <vector>
#include <unordered_map>

#include "cudnn_frontend_Tensor.h"
#include "cudnn_frontend_Operation.h"
#include "cudnn_frontend_OperationGraph.h"
#include "cudnn_frontend_ExecutionPlan.h"
#include "cudnn_frontend_VariantPack.h"

#include "graphs/cudnn_frontend_ICudnn.h"


#include "graphs/cudnn_frontend_nodes.h"


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
    std::string name;
    int offset = 1;

    // Tensors belonging to each block.
    // Connecting blocks can modify and delete tensors in this container.
    std::unordered_map<std::string, std::shared_ptr<tensor_properties>> tensor_props;

    IBlock* parent_block;
    std::unordered_map <std::string, std::shared_ptr<IBlock>> sub_blocks;

    virtual int build(cudnnHandle_t& handle) = 0;
    
    virtual int execute(cudnnHandle_t& handle, std::unordered_map<std::string, void*> const& tensor_uid_to_pointer_map) = 0;

    IBlock(std::string const& name, int64_t const offset) : name(name), offset(offset) {}

    virtual ~IBlock() {};

    int add_tensor(std::string const& name, tensor_properties& properties) {
        tensor_props.emplace(name, std::make_shared<tensor_properties>(properties));
        return 0;
    }
    
    std::shared_ptr<tensor_properties> get_tensor_props(std::string const& name) {
        if(tensor_props.count(name)) {
            return tensor_props.at(name);
        }
        if(parent_block == nullptr) {
            return nullptr;
        }
        tensor_props[name] = parent_block->get_tensor_props(name);
        return tensor_props.at(name);
    }
};

} // namespace cudnn_frontend
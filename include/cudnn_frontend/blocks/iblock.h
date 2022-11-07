#pragma once

#include <memory>
#include <unordered_map>

#include <cudnn_frontend_Tensor.h>

// Interface for all blocks to follow.
// It contains a container for all tensors and operations involved.
class IBlock {
private:
    
    // Type of each block. Blocks can either be a composite (value BLOCK) or
    // one of the other primitive types. Primitives types are nothing but 
    // cudnn operations.
    enum class Type {
        BLOCK,
        CONVOLUTION,
        MATMUL,
        POINTWISE,
        REDUCTION,
        RESAMPLE
    };

protected:

    // Tensors belonging to each block.
    // Connecting blocks can modify and delete tensors in this container.
    std::unordered_map <std::string, std::shared_ptr<cudnn_frontend::Tensor>> tensors;

    Type tag;

public:

    virtual Type getType() = 0;

    virtual int createTensors() = 0;
    virtual int createDescritpors() = 0;
    virtual int createOperations() = 0;
    
    virtual int build() = 0;
    
    virtual int execute() = 0;

};
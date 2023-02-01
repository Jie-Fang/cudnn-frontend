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

    virtual cudnn_frontend_error_t partition(cudnnHandle_t& handle) = 0;

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
    
    virtual int infer_properties() = 0;
    
    virtual int validate() const = 0;

    virtual cudnn_frontend_error_t build(cudnnHandle_t& handle) = 0;
    
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

class CompositeBlock : public IBlock {

protected:
    Type
    getType() override {
        return Type::BLOCK;
    }

    cudnn_frontend_error_t partition(cudnnHandle_t& handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partioning CompositeBlock..." << std::endl;

        std::vector<Operation const*> operation_graph{};

        for (auto block : sub_blocks) {
            getLogger() << "Getting the operation from " << block.first << std::endl;
            for (auto &operation : block.second->get_operations()) {
                operation_graph.push_back(operation.second.get());
            }
        }

        getLogger() << "Operation Graph has " << operation_graph.size() << " operations." << std::endl;

        auto composite_graph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(operation_graph.size(), operation_graph.data()).build();
        operation_graphs.push_back(std::make_shared<OperationGraph>(std::move(composite_graph)));

        int status = createExecutionPlan(handle);
        if(status) {
            getLogger() << "[cudnn_frontend] INFO: " << "Failed to create execution plans for graph partitioning in ConvolutionBlock." << std::endl;
            return cudnn_frontend_error_t::GRAPH_PARTITION_EXECUTION_PLAN_CREATION_FAILED;
        }

        getLogger() << "[cudnn_frontend] INFO: Partitioned CompositeBlock." << std::endl;
        return cudnn_frontend_error_t::OK;
    }

public:
    int infer_properties() override {
        for(auto const& sub_block: sub_blocks) {
            sub_block.second->infer_properties();
        }
        return 0;
    }

    int validate() const override {return 0;}

    cudnn_frontend_error_t build(cudnnHandle_t& handle) override {
        infer_properties();
        createTensors();
        createDescritpors();
        createOperations();
        partition(handle);
        return cudnn_frontend_error_t::OK;
    }

    int
    execute(cudnnHandle_t& handle, std::unordered_map<std::string, void*> const& tensor_to_pointer_map) override {
        std::vector<int64_t> uids;
        std::vector<void *> device_ptrs;
        for (auto & item : tensor_to_pointer_map) {
            device_ptrs.push_back(item.second);
            uids.push_back(tensor_props.at(item.first)->get_uid());
        }
        run_execution_plans(handle, device_ptrs, uids);
        return 0;
    }

    CompositeBlock(std::string const& name, int64_t const offset) : IBlock(name, offset) {}

    ~CompositeBlock() {};
};

} // namespace cudnn_frontend
#pragma once

#include <memory>
#include <vector>
#include <unordered_map>

#include "cudnn_frontend_Tensor.h"
#include "cudnn_frontend_Operation.h"
#include "cudnn_frontend_OperationGraph.h"
#include "cudnn_frontend_ExecutionPlan.h"
#include "cudnn_frontend_VariantPack.h"

#include "graphs/cudnn_frontend_cudnn_interface.h"


#include "graphs/cudnn_frontend_graph_properties.h"


namespace cudnn_frontend {

// Interface for all nodes to follow.
class INode: public ICudnn {

    friend class ConvolutionFP8Node;
    friend class ConvolutionPointwiseNode;

private:

protected:
    // Type of each node. Nodes can either be a composite (value COMPOSITE) or
    // one of the other primitive types. Primitives types are nothing but 
    // cudnn operations.
    enum class Type {
        COMPOSITE
        , CONVOLUTION
        , MATMUL
        , POINTWISE
        , REDUCTION
        , RESAMPLE
    };
    Type tag;
    
    virtual Type getType() = 0;

    virtual error_t partition(cudnnHandle_t& handle) = 0;

    virtual int createTensors() {
        for(auto const& sub_node: sub_nodes) {
            sub_node.second->createTensors();
        }
        return 0;
    }
    
    virtual int createDescritpors() {
        for(auto const& sub_node: sub_nodes) {
            sub_node.second->createDescritpors();
        }
        return 0;
    }

    virtual int createOperations() {
        for(auto const& sub_node: sub_nodes) {
            sub_node.second->createOperations();
        }
        return 0;
    }
    
public:
    std::string name;
    int offset = 1;

    // Tensors belonging to each node.
    // Connecting nodes can modify and delete tensors in this container.
    std::unordered_map<std::string, std::shared_ptr<tensor_properties>> tensor_props;

    INode* parent_node;
    std::unordered_map <std::string, std::shared_ptr<INode>> sub_nodes;
    
    virtual int infer_properties() = 0;
    
    virtual int validate() const = 0;

    virtual error_t build(cudnnHandle_t& handle) = 0;
    
    virtual error_t execute(cudnnHandle_t& handle, std::unordered_map<std::string, void*> const& tensor_uid_to_pointer_map) = 0;

    INode(std::string const& name, int64_t const offset) : name(name), offset(offset) {}

    virtual ~INode() {};

    int add_tensor(std::string const& name, tensor_properties& properties) {
        tensor_props.emplace(name, std::make_shared<tensor_properties>(properties));
        return 0;
    }
    
    std::shared_ptr<tensor_properties> get_tensor_props(std::string const& name) {
        if(tensor_props.count(name)) {
            return tensor_props.at(name);
        }
        if(parent_node == nullptr) {
            return nullptr;
        }
        tensor_props[name] = parent_node->get_tensor_props(name);
        return tensor_props.at(name);
    }
};

class CompositeNode : public INode {

protected:
    Type
    getType() override {
        return Type::COMPOSITE;
    }

    error_t partition(cudnnHandle_t& handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partioning CompositeNode..." << std::endl;

        std::vector<Operation const*> operation_graph{};

        for (auto node : sub_nodes) {
            getLogger() << "Getting the operation from " << node.first << std::endl;
            for (auto &operation : node.second->get_operations()) {
                operation_graph.push_back(operation.second.get());
            }
        }

        getLogger() << "Operation Graph has " << operation_graph.size() << " operations." << std::endl;

        auto composite_graph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(operation_graph.size(), operation_graph.data()).build();
        operation_graphs.push_back(std::make_shared<OperationGraph>(std::move(composite_graph)));

        int status = createExecutionPlan(handle);
        if(status) {
            getLogger() << "[cudnn_frontend] INFO: " << "Failed to create execution plans for graph partitioning in ConvolutionNode." << std::endl;
            return error_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED;
        }

        getLogger() << "[cudnn_frontend] INFO: Partitioned CompositeNode." << std::endl;
        return error_t::OK;
    }

public:
    int infer_properties() override {
        for(auto const& sub_node: sub_nodes) {
            sub_node.second->infer_properties();
        }
        return 0;
    }

    int validate() const override {return 0;}

    error_t build(cudnnHandle_t& handle) override {
        infer_properties();
        createTensors();
        createDescritpors();
        createOperations();
        partition(handle);
        return error_t::OK;
    }

    error_t
    execute(cudnnHandle_t& handle, std::unordered_map<std::string, void*> const& tensor_to_pointer_map) override {
        std::vector<int64_t> uids;
        std::vector<void *> device_ptrs;
        for (auto & item : tensor_to_pointer_map) {
            device_ptrs.push_back(item.second);
            uids.push_back(tensor_props.at(item.first)->get_uid());
        }
        auto status = run_execution_plans(handle, device_ptrs, uids);
        return status;
    }

    CompositeNode(std::string const& name, int64_t const offset) : INode(name, offset) {}

    ~CompositeNode() {};
};

} // namespace cudnn_frontend
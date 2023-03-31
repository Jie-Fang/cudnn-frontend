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

protected:
    // Type of each node. Nodes can either be a composite (value COMPOSITE) or
    // one of the other primitive types. Primitives types are nothing but 
    // cudnn operations.
    enum class Type {
        COMPOSITE
        , BATCHNORM
        , CONVOLUTION
        , MATMUL
        , POINTWISE
        , REDUCTION
        , RESAMPLE
    };
    Type tag;
    
    virtual Type getType() = 0;

    virtual error_t partition() = 0;

    virtual error_t createTensors() {
        for(auto const& sub_node: sub_nodes) {
            auto status = sub_node.second->createTensors();
            if(status != error_t::OK) {
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to create tensors in " << name << std::endl;
                return status;
            }
        }
        return error_t::OK;
    }

    virtual error_t createOperations() {
        for(auto const& sub_node: sub_nodes) {
            auto status = sub_node.second->createOperations();
            if(status != error_t::OK) {
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to create operation for " << sub_node.first << " in " << name << std::endl;
                return status;
            }

            // Roll up operations to parent node, so that parent can too partition operation graphs.
            for (auto const &item : sub_node.second->get_operations()) {
                operations.emplace(item.first, item.second);
            }
            for (auto const &item : sub_node.second->tensors_in_operations) {
                tensors_in_operations.emplace(item.first, item.second);
            }
        }
        return error_t::OK;
    }
    
public:
    std::string name;
    int offset = 1;

    // Tensors belonging to each node.
    // Connecting nodes can modify and delete tensors in this container.
    std::unordered_map<std::string, std::shared_ptr<graph::Tensor>> tensor_props;

    INode* parent_node;
    std::unordered_map <std::string, std::shared_ptr<INode>> sub_nodes;
    
    virtual error_t infer_properties() {
        for(auto const& sub_node: sub_nodes) {
            auto status = sub_node.second->infer_properties();
            if(status != error_t::OK) {
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to infer properties in " << name << std::endl;
                return status;
            }
        }
        return error_t::OK;
    }
    
    virtual error_t validate() const {
        for(auto const& sub_node: sub_nodes) {
            auto status = sub_node.second->validate();
            if(status != error_t::OK) {
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to validate in " << name << std::endl;
                return status;
            }
        }
        return error_t::OK;
    }

    error_t build() {
        auto status = infer_properties();
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to build in " << name << std::endl;
            return status;
        }

        status = validate();
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to build in " << name << std::endl;
            return status;
        }

        status = createTensors();
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to build in " << name << std::endl;
            return status;
        }

        status = createOperations();
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to build in " << name << std::endl;
            return status;
        }

        status = partition();
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to build in " << name << std::endl;
            return status;
        }

        return error_t::OK;
    }

    error_t execute(std::unordered_map<std::string, void*> const& tensor_name_to_pointer_map) {
        std::unordered_map<int64_t, void*> tensor_uid_to_pointer_map;
        void* workspace_ptr = nullptr;

        for (auto const &item : tensor_name_to_pointer_map) {
            if(item.first == "workspace") {
                workspace_ptr = item.second;
            }
            else {
                tensor_uid_to_pointer_map.emplace(get_tensor_props(item.first)->get_uid(), item.second);
            }
        }
        
        auto status = execute_cudnn_plans(tensor_uid_to_pointer_map, workspace_ptr);
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Execution failed in " << name << std::endl;
            return status;
        }
        
        return status;
    }
    
    INode(std::string const& name, int64_t const offset) : name(name), offset(offset), parent_node(nullptr) {}

    virtual ~INode() {};

    int insert_tensor(std::string const& name, graph::Tensor& properties) {
        tensor_props.emplace(name, std::make_shared<graph::Tensor>(properties));
        return 0;
    }
    
    std::shared_ptr<graph::Tensor> get_tensor_props(std::string const& name) {
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

    error_t partition() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partitioning CompositeNode..." << std::endl;

        // Currently just make one large graph of operations from all sub nodes.
        std::vector<std::string> operation_names;
        for (auto node : sub_nodes) {
            getLogger() << "Getting the operation from " << node.first << std::endl;
            for (auto &operation : node.second->get_operations()) {
                operation_names.push_back(operation.first);
            }
        }

        getLogger() << "Operation Graph has " << operation_names.size() << " operations." << std::endl;

        auto status = create_cudnn_execution_plan({operation_names});
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to create execution plans for graph partitioning in CompositeNode." << std::endl;
            return status;
        }

        getLogger() << "[cudnn_frontend] INFO: Partitioned CompositeNode." << std::endl;
        return error_t::OK;
    }

public:
    CompositeNode(std::string const& name, int64_t const offset) : INode(name, offset) {}

    ~CompositeNode() {};
};

} // namespace cudnn_frontend
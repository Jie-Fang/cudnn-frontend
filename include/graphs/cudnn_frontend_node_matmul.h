#pragma once

#include <cudnn_frontend_MatMulDesc.h>
#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

class MatMulNode : public INode {
private:

protected:

public:
    std::shared_ptr<matmul_properties> props;

    MatMulNode(std::string const& name, int64_t offset = 1)  : INode (name, offset) {}

    Type getType() override final {
        return Type::MATMUL;
    }

    int set_properties(std::string const& INode_name, std::shared_ptr<matmul_properties> properties) {
        if(sub_nodes.size() != 0) {
            return 1;
        }
        if(INode_name != name) {
            return 1;
        }
        
        props = properties;
        return 0;
    }

    error_t infer_properties() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for matmul node named " << name << "." << std::endl;

        props->update_uids(offset);

        for(size_t i = 0; i < matmul_properties::PORTS::COUNT; ++i) {
            auto tensor_prop = get_tensor_props(props->get_port_name(static_cast<matmul_properties::PORTS>(i)));
            if(tensor_prop->is_uid_set)
                props->uids[i] = tensor_prop->get_uid();
            tensor_prop->set_properties_from_context(CUDNN_TENSOR_NCHW, props->get_tensor_data_type(), props->uids[i]);
        }

        return error_t::OK;
    }
    
    error_t validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating MatMulNode..." << std::endl;

        // TODO: check all properties of this operation and its tensor are correct
        // Like do dim count match dim/stride
        // Do dim and corresponding stride match

        getLogger() << "[cudnn_frontend] INFO: " << "Validated MatMulNode." << std::endl;
        return error_t::OK;
    }

    error_t createTensors() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building MatMulNode tensors..." << std::endl;

        create_cudnn_tensor(get_tensor_props(props->get_port_name(matmul_properties::PORTS::X)));
        create_cudnn_tensor(get_tensor_props(props->get_port_name(matmul_properties::PORTS::W)));
        create_cudnn_tensor(get_tensor_props(props->get_port_name(matmul_properties::PORTS::Y)));

        getLogger() << "[cudnn_frontend] INFO: " << "Built MatMulNode tensors." << std::endl;

        return error_t::OK;
    }
    
    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building MatMulNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        // matmul descriptor
        auto matmul_descriptor = cudnn_frontend::MatMulDescBuilder()
                                                        .setComputeType(props->get_compute_type())
                                                        .build();

        // Create the matmul operation.
        auto matmul_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                                        .setaMatDesc(*(tensors.at(props->uids[matmul_properties::PORTS::X])))
                                        .setbMatDesc(*(tensors.at(props->uids[matmul_properties::PORTS::W])))
                                        .setcMatDesc(*(tensors.at(props->uids[matmul_properties::PORTS::Y])))
                                        .setmatmulDesc(matmul_descriptor)
                                        .build();
        operations.emplace(name, std::make_shared<Operation>(std::move(matmul_operation)));

        // Push all real tensors as required for operation execution.
        auto const& tensor_props_involved_in_operation = {
            get_tensor_props(props->get_port_name(matmul_properties::PORTS::X))
            , get_tensor_props(props->get_port_name(matmul_properties::PORTS::W))
            , get_tensor_props(props->get_port_name(matmul_properties::PORTS::Y))
        };
        for(auto const& tensor_props: tensor_props_involved_in_operation) {
            if(tensor_props->get_is_virtual() == false) {
                tensors_in_operations[name].emplace_back(tensor_props->get_uid());
            }
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Built MatMulNode operation." << std::endl;

        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnn_frontend::cudnnException &e) {
            throw cudnnException(e.what(), e.getCudnnStatus());
        }
        #endif
        
        return error_t::OK;
    }

    error_t partition() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partitioning MatMulNode..." << std::endl;

        auto status = create_cudnn_execution_plan({{name}});
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to create execution plans for graph partitioning in MatMulNode." << std::endl;
            return status;
        }

        getLogger() << "[cudnn_frontend] INFO: Partitioned MatMulNode." << std::endl;
        return error_t::OK;
    }
};

} // namespace cudnn_frontend

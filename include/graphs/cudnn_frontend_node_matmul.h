#pragma once

#include <cudnn_frontend_MatMulDesc.h>
#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {

class MatMulNode : public INode {
private:

protected:

public:
    std::shared_ptr<matmul> props;

    MatMulNode(std::string const& name, int64_t offset = 1)  : INode (name, offset) {}

    Type getType() override final {
        return Type::MATMUL;
    }

    int set_properties(std::string const& INode_name, std::shared_ptr<matmul> properties) {
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

        // TODO: Only inferrencing from (X, W) -> Y works today.
        auto x_tensor_prop = get_tensor_props(props->get_port_name(matmul::PORTS::X));
        auto w_tensor_prop = get_tensor_props(props->get_port_name(matmul::PORTS::W));
        auto y_tensor_prop = get_tensor_props(props->get_port_name(matmul::PORTS::Y));
        
        auto const& x_tensor_dim = x_tensor_prop->get_dim();
        auto const& w_tensor_dim = w_tensor_prop->get_dim();
        auto& y_tensor_dim = y_tensor_prop->get_dim();
        if(x_tensor_dim.size() != w_tensor_dim.size()) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
            getLogger() << "[cudnn_frontend] ERROR: " << status << "  Tensor dimensionality mismatch at X and W ports of " << name << "." << std::endl;
            return status;
        }
        
        if(y_tensor_dim.empty()) {
            y_tensor_dim.resize(x_tensor_dim.size());
            y_tensor_dim[0] = x_tensor_dim[0]; // B
            y_tensor_dim[1] = x_tensor_dim[1]; // M
            y_tensor_dim[2] = w_tensor_dim[2]; // N
        } else {
            if(x_tensor_dim.size() != y_tensor_dim.size()) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at X and Y ports of " << name << "." << std::endl;
                return status;
            }
        }

        for(size_t i = 0; i < matmul::PORTS::COUNT; ++i) {
            auto tensor_prop = get_tensor_props(props->get_port_name(static_cast<matmul::PORTS>(i)));
            if(tensor_prop->is_uid_set)
                props->uids[i] = tensor_prop->get_uid();
            tensor_prop->set_properties_from_context(CUDNN_TENSOR_NHWC, props->get_tensor_data_type(), props->uids[i]);
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

        create_cudnn_tensor(get_tensor_props(props->get_port_name(matmul::PORTS::X)));
        create_cudnn_tensor(get_tensor_props(props->get_port_name(matmul::PORTS::W)));
        create_cudnn_tensor(get_tensor_props(props->get_port_name(matmul::PORTS::Y)));

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
                                        .setaMatDesc(*(tensors.at(props->uids[matmul::PORTS::X])))
                                        .setbMatDesc(*(tensors.at(props->uids[matmul::PORTS::W])))
                                        .setcMatDesc(*(tensors.at(props->uids[matmul::PORTS::Y])))
                                        .setmatmulDesc(matmul_descriptor)
                                        .build();
        operations.emplace(name, std::make_shared<Operation>(std::move(matmul_operation)));

        // Push all real tensors as required for operation execution.
        auto const& tensor_props_involved_in_operation = {
            get_tensor_props(props->get_port_name(matmul::PORTS::X))
            , get_tensor_props(props->get_port_name(matmul::PORTS::W))
            , get_tensor_props(props->get_port_name(matmul::PORTS::Y))
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

} // namespace graph

} // namespace cudnn_frontend

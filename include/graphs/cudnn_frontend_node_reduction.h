#pragma once

#include "cudnn_frontend_ReductionDesc.h"
#include "cudnn_frontend_Logging.h"

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {

class ReductionNode : public INode {
private:

protected:

public:
    std::shared_ptr<reduction> props;

    ReductionNode(std::string const& name, int64_t const offset = 1)  : INode (name, offset) {}

    Type getType() override final {
        return Type::REDUCTION;
    }

    int set_properties(std::string const& INode_name, std::shared_ptr<reduction> properties) {
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
        props->update_uids(offset);
        
        for(size_t i = 0; i < reduction::PORTS::COUNT; ++i) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(static_cast<reduction::PORTS>(i)));
            if(tensor_prop->is_uid_set)
                props->uids[i] = tensor_prop->get_uid();
            tensor_prop->set_properties_from_context(CUDNN_TENSOR_NHWC, props->get_tensor_data_type(), props->uids[i]);
        }
        return error_t::OK;
    }

    error_t validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating ReductionNode..." << std::endl;

        getLogger() << "[cudnn_frontend] INFO: " << "Validated ReductionNode." << std::endl;
        return error_t::OK;
    }

    error_t createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Building ReductionNode tensors..." << std::endl;

        create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(reduction::PORTS::X)));
        create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(reduction::PORTS::Y)));

        getLogger() << "[cudnn_frontend] INFO: " << "Built ReductionNode tensors." << std::endl;

        return error_t::OK;
    }

    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building ReductionNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        auto reduction_descriptor = cudnn_frontend::ReductionDescBuilder()
                                                        .setComputeType(props->get_compute_type())
                                                        .setReductionOp(props->get_mode())
                                                        .build();

        auto reduction_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                        .setxDesc(*(tensors.at(props->uids[reduction::PORTS::X])))
                                        .setyDesc(*(tensors.at(props->uids[reduction::PORTS::Y])))
                                        .setreductionDesc(reduction_descriptor)
                                        .build();
        
        operations.emplace(name, std::make_shared<Operation>(std::move(reduction_operation)));

        // Push all real tensors as required for operation execution.
        auto const& tensor_props_involved_in_operation = {
            get_tensor_props(props->get_tensor_at_port(reduction::PORTS::X))
            , get_tensor_props(props->get_tensor_at_port(reduction::PORTS::Y))
        };
        for(auto const& tensor_props: tensor_props_involved_in_operation) {
            if(tensor_props->get_is_virtual() == false) {
                tensors_in_operations[name].emplace_back(tensor_props->get_uid());
            }
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Built ReductionNode operation." << std::endl;

        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnn_frontend::cudnnException &e) {
            throw cudnnException(e.what(), e.getCudnnStatus());
        }
        #endif
        
        return error_t::OK;
    }

    error_t partition() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partitioning ReductionNode..." << std::endl;
        
        auto status = create_cudnn_execution_plan({{name}});
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to create execution plans for graph partitioning in ReductionNode." << std::endl;
            return status;
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Partitioned ReductionNode." << std::endl;
        return error_t::OK;
    }
};

} // namespace graph

} // namespace cudnn_frontend
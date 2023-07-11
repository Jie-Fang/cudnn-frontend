#pragma once

#include "cudnn_frontend_ReductionDesc.h"
#include "cudnn_frontend_Logging.h"

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend::graph {

class ReductionNode : public INode {
    Reduction_attributes options;
public:

    ReductionNode(std::string const& name, Reduction_attributes&& options_, detail::Context const& context)  : INode (name, context), options(std::move(options_)) {
        options.fill_from_context(get_context());
    }
    
    Type getType() override final {
        return Type::REDUCTION;
    }

    error_t infer_properties_node() override final {
        // Only inferrencing from IN_0 to OUT_0 works today.
        auto x_tensor = options.inputs.X;
        auto y_tensor = options.outputs.Y;
        
        auto const& x_tensor_dim = x_tensor->get_dim();
        auto y_tensor_dim = y_tensor->get_dim();
        if(y_tensor_dim.empty()) {
            y_tensor->set_dim(x_tensor_dim).generateStrides(CUDNN_TENSOR_NHWC);
        } else {
            if(y_tensor_dim.size() != x_tensor_dim.size()) {
                auto status = error_code_t::SHAPE_DEDUCTION_FAILED;
                std::string message = "[cudnn_frontend] ERROR: Tensor dimensionality mismatch at X and Y ports of " + name;
                return {status, message};
            }
        }

        return {error_code_t::OK, ""};
    }

    error_t assign_uids_node() override final {
        options.inputs.X->set_uid(ICudnn::create_new_uid());
        options.outputs.Y->set_uid(ICudnn::create_new_uid());
        return {error_code_t::OK, ""};
    }

    error_t createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Building ReductionNode tensors..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.X));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.Y));

        getLogger() << "[cudnn_frontend] INFO: " << "Built ReductionNode tensors." << std::endl;

        return {error_code_t::OK, ""};
    }

    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building ReductionNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        auto reduction_descriptor = cudnn_frontend::ReductionDescBuilder()
                                                        .setComputeType(options.get_compute_data_type())
                                                        .setReductionOp(options.get_mode().value())
                                                        .build();

        auto reduction_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                        .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
                                        .setyDesc(*(tensors.at(options.outputs.Y->get_uid())))
                                        .setreductionDesc(reduction_descriptor)
                                        .build();
        
        operations.emplace(name, std::make_shared<Operation_v8>(std::move(reduction_operation)));

        // Push all real tensors as required for operation execution.
        auto const& tensors_involved_in_operation = {
            options.inputs.X
            , options.outputs.Y
        };
        auto& tensors_in_operation = tensors_in_operations[name];
        for(auto const& tensor: tensors_involved_in_operation) {
            if(tensor && tensor->get_is_virtual() == false) {
                tensors_in_operation.emplace_back(tensor->get_uid());
            }
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Built ReductionNode operation." << std::endl;

        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnn_frontend::cudnnException &e) {
            throw cudnnException(e.what(), e.getCudnnStatus());
        }
        #endif
        
        return {error_code_t::OK, ""};
    }

    error_t createOperationGraphs(cudnnHandle_t) override final {
        return {error_code_t::OK, ""};
    }

    virtual void serialize(json& j) const override final {
        j = options;
    }
};

} // namespace cudnn_frontend
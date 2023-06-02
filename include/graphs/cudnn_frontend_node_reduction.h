#pragma once

#include "cudnn_frontend_ReductionDesc.h"
#include "cudnn_frontend_Logging.h"

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend::graph {

class ReductionNode : public INode {
    std::shared_ptr<Reduction> options;
public:

    ReductionNode(std::string const& name, std::shared_ptr<Reduction> const options, int64_t const offset = 1)  : INode (name, offset), options(options) {}

    Type getType() override final {
        return Type::REDUCTION;
    }

    error_t infer_properties() override final {

        // Merge with ancestor's context
        fill_missing_context();

        options->fill_from_context(get_context());
        
        // Only inferrencing from IN_0 to OUT_0 works today.
        auto x_tensor = options->inputs.X;
        auto y_tensor = options->outputs.Y;
        
        auto const& x_tensor_dim = x_tensor->get_dim();
        auto y_tensor_dim = y_tensor->get_dim();
        if(y_tensor_dim.empty()) {
            y_tensor->set_dim(x_tensor_dim).generateStrides(CUDNN_TENSOR_NHWC);
        } else {
            if(y_tensor_dim.size() != x_tensor_dim.size()) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at X and Y ports of " << name << "." << std::endl;
                return status;
            }
        }

        // TODO: gather all tensors and assign them uids at once using a counter. So no need to keep uids in properties.
        // But for the time being doing it here manually.
        if(x_tensor->is_uid_set == false) {
            x_tensor->set_uid(offset + 1);
        }
        if(y_tensor->is_uid_set == false) {
            y_tensor->set_uid(offset + 2);
        }

        return error_t::OK;
    }

    error_t createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Building ReductionNode tensors..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->inputs.X));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->outputs.Y));

        getLogger() << "[cudnn_frontend] INFO: " << "Built ReductionNode tensors." << std::endl;

        return error_t::OK;
    }

    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building ReductionNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        auto reduction_descriptor = cudnn_frontend::ReductionDescBuilder()
                                                        .setComputeType(options->get_compute_data_type())
                                                        .setReductionOp(options->get_mode().value())
                                                        .build();

        auto reduction_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                        .setxDesc(*(tensors.at(options->inputs.X->get_uid())))
                                        .setyDesc(*(tensors.at(options->outputs.Y->get_uid())))
                                        .setreductionDesc(reduction_descriptor)
                                        .build();
        
        operations.emplace(name, std::make_shared<Operation_v8>(std::move(reduction_operation)));

        // Push all real tensors as required for operation execution.
        auto const& tensors_involved_in_operation = {
            options->inputs.X
            , options->outputs.Y
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
        
        return error_t::OK;
    }

    error_t createOperationGraphs(cudnnHandle_t) override final {
        return error_t::OK;
    }

    error_t createExecutionPlans(cudnnHandle_t) override final {
        return error_t::OK;
    }
};

} // namespace cudnn_frontend
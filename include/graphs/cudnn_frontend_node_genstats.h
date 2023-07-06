#pragma once

#include "cudnn_frontend_Logging.h"

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {

class GenstatsNode : public INode {
    Genstats options;
public:
    GenstatsNode(std::string const& name, Genstats&& options_, detail::Context const& context)  : INode (name, context), options(std::move(options_)) {
        options.fill_from_context(get_context());
        
        // outputs should be float type
        options.outputs.SUM->set_data_type(DataType_t::FLOAT);
        options.outputs.SQ_SUM->set_data_type(DataType_t::FLOAT);
    }

    Type getType() override final {
        return Type::GENSTATS;
    }

    error_t infer_properties_node() override final {
        // Only inferrencing from X works today.
        auto X = options.inputs.X;
        auto SUM = options.outputs.SUM;
        auto SQ_SUM = options.outputs.SQ_SUM;
        
        auto const x_tensor_dim = X->get_dim();
        auto sum_tensor_dim = SUM->get_dim();
        auto sq_sum_tensor_dim = SQ_SUM->get_dim();
        
        if(sum_tensor_dim.empty()) {
            sum_tensor_dim.resize(x_tensor_dim.size(), 1);
            sum_tensor_dim[1] = x_tensor_dim[1];
            SUM->set_dim(sum_tensor_dim).generateStrides(CUDNN_TENSOR_NHWC);
        } else {
            if(x_tensor_dim.size() != sum_tensor_dim.size()) {
                auto status = error_code_t::SHAPE_DEDUCTION_FAILED;
                std::string message = "[cudnn_frontend] ERROR: Tensor dimensionality mismatch at X and SUM ports of " + name;
                return {status, message};
            }
        }
        
        if(sq_sum_tensor_dim.empty()) {
            sq_sum_tensor_dim.resize(x_tensor_dim.size(), 1);
            sq_sum_tensor_dim[1] = x_tensor_dim[1];
            SQ_SUM->set_dim(sq_sum_tensor_dim).generateStrides(CUDNN_TENSOR_NHWC);
        } else {
            if(x_tensor_dim.size() != sq_sum_tensor_dim.size()) {
                auto status = error_code_t::SHAPE_DEDUCTION_FAILED;
                std::string message = "[cudnn_frontend] ERROR: Tensor dimensionality mismatch at X and SQSUM ports of " + name;
                return {status, message};
            }
        }

        return {error_code_t::OK, ""};
    }

    error_t assign_uids_node() override final {
        options.inputs.X->set_uid(ICudnn::create_new_uid());
        options.outputs.SUM->set_uid(ICudnn::create_new_uid());
        options.outputs.SQ_SUM->set_uid(ICudnn::create_new_uid());
        return {error_code_t::OK, ""};
    }

    error_t createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Building GenstatsNode tensors..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.X));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.SUM));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.SQ_SUM));

        getLogger() << "[cudnn_frontend] INFO: " << "Built GenstatsNode tensors." << std::endl;

        return {error_code_t::OK, ""};
    }

    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building GenstatsNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        auto genstats_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR)
                                        .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
                                        .setGenStatsMode(CUDNN_GENSTATS_SUM_SQSUM)
                                        .setSumDesc(*(tensors.at(options.outputs.SUM->get_uid())))
                                        .setSqSumDesc(*(tensors.at(options.outputs.SQ_SUM->get_uid())))
                                        .build();
        
        operations.emplace(name, std::make_shared<Operation_v8>(std::move(genstats_operation)));

        // Push all real tensors as required for operation execution.
        auto const& tensors_involved_in_operation = {
            options.inputs.X
            , options.outputs.SUM
            , options.outputs.SQ_SUM
        };
        auto& tensors_in_operation = tensors_in_operations[name];
        for(auto const& tensor: tensors_involved_in_operation) {
            if(tensor && tensor->get_is_virtual() == false) {
                tensors_in_operation.emplace_back(tensor->get_uid());
            }
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Built GenstatsNode operation." << std::endl;

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

    error_t createExecutionPlans(cudnnHandle_t) override final {
        return {error_code_t::OK, ""};
    }
};

} // namespace graph

} // namespace cudnn_frontend
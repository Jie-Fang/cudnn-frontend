#pragma once

#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {

class BatchNormFinalizeNode : public INode {
    BN_finalize options;
public:

    BatchNormFinalizeNode(std::string const& name, BN_finalize&& options_, detail::Context const& context)  : INode (name, context), options(std::move(options_)) {
        options.fill_from_context(get_context());
    }

    Type getType() override final {
        return Type::BN_FINALIZE;
    }

    error_t infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for batchnorm finalize node named " << name << "." << std::endl;
        
        // TODO: Only inferencing from SUM works today.
        auto SUM = options.inputs.SUM;
        auto const sum_tensor_dim = SUM->get_dim();

        // Set channel length tensors
        auto infer_per_channel_tensors = [&sum_tensor_dim] (std::shared_ptr<Tensor>& T) {
            auto tensor_dim = T->get_dim();
            if(tensor_dim.empty()) {
                tensor_dim = sum_tensor_dim;
                T->set_dim(tensor_dim).generateStrides(CUDNN_TENSOR_NHWC);
            }
        };
        infer_per_channel_tensors(options.inputs.SQ_SUM);
        infer_per_channel_tensors(options.inputs.MEAN);
        infer_per_channel_tensors(options.inputs.INV_VARIANCE);
        infer_per_channel_tensors(options.inputs.SCALE);
        infer_per_channel_tensors(options.inputs.BIAS);
        infer_per_channel_tensors(options.inputs.PREV_RUNNING_MEAN);
        infer_per_channel_tensors(options.inputs.PREV_RUNNING_VAR);
        infer_per_channel_tensors(options.outputs.EQ_BIAS);
        infer_per_channel_tensors(options.outputs.EQ_SCALE);
        infer_per_channel_tensors(options.outputs.NEXT_RUNNING_MEAN);
        infer_per_channel_tensors(options.outputs.NEXT_RUNNING_VAR);

        // Set scalars
        auto infer_scalars = [&sum_tensor_dim] (std::shared_ptr<Tensor>& T) {
            auto tensor_dim = T->get_dim();
            if(tensor_dim.empty()) {
                tensor_dim.resize(sum_tensor_dim.size(), 1);
                T->set_dim(tensor_dim).generateStrides(CUDNN_TENSOR_NHWC);
            }
        };
        infer_scalars(options.inputs.EPSILON);
        infer_scalars(options.inputs.EXP_AVG);
        infer_scalars(options.inputs.ACCUM_COUNT);

        return {error_code_t::OK, ""};
    }
    
    error_t validate_node() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating BatchNormFinalizeNode..." << std::endl;

        auto SUM = options.inputs.SUM;
        auto const sum_tensor_dim = SUM->get_dim();

        auto validate_per_channel_tensors = [this, &sum_tensor_dim] (std::shared_ptr<Tensor> const& T) {
            error_t status = {error_code_t::OK, ""};
            auto tensor_dim = T->get_dim();
            if(sum_tensor_dim != tensor_dim) {
                status.code = error_code_t::SHAPE_DEDUCTION_FAILED;
                status.err_msg = "[cudnn_frontend] ERROR: Tensor dimensionality mismatch at SUM and Y ports of " + name + ".";
                return status;
            }
            return status;
        };
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.inputs.SQ_SUM));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.inputs.MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.inputs.INV_VARIANCE));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.inputs.SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.inputs.BIAS));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.inputs.PREV_RUNNING_MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.inputs.PREV_RUNNING_VAR));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.outputs.EQ_BIAS));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.outputs.EQ_SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.outputs.NEXT_RUNNING_MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.outputs.NEXT_RUNNING_VAR));

        auto validate_scalars = [] (std::shared_ptr<Tensor> const& T) {
            error_t status = {error_code_t::OK, ""};
            auto tensor_dim = T->get_dim();
            bool allOnes = std::all_of(tensor_dim.begin(), tensor_dim.end(), [](auto element) {
                return element == 1;
            });
            if(!allOnes) {
                status.code = error_code_t::SHAPE_DEDUCTION_FAILED;
                status.err_msg = "[cudnn_frontend] ERROR: Tensor dimensionality mismatch at SUM and Y ports.";
                return status;
            }
            return status;
        };
        CHECK_CUDNN_FRONTEND_ERROR(validate_scalars(options.inputs.EPSILON));
        CHECK_CUDNN_FRONTEND_ERROR(validate_scalars(options.inputs.EXP_AVG));
        CHECK_CUDNN_FRONTEND_ERROR(validate_scalars(options.inputs.ACCUM_COUNT));

        getLogger() << "[cudnn_frontend] INFO: " << "Validated BatchNormFinalizeNode." << std::endl;
        return {error_code_t::OK, ""};
    }
    
    error_t assign_uids_node() override final {
        options.inputs.SUM->set_uid(ICudnn::create_new_uid());
        options.inputs.SQ_SUM->set_uid(ICudnn::create_new_uid());
        options.inputs.MEAN->set_uid(ICudnn::create_new_uid());
        options.inputs.INV_VARIANCE->set_uid(ICudnn::create_new_uid());
        options.inputs.SCALE->set_uid(ICudnn::create_new_uid());
        options.inputs.BIAS->set_uid(ICudnn::create_new_uid());
        options.inputs.PREV_RUNNING_MEAN->set_uid(ICudnn::create_new_uid());
        options.inputs.PREV_RUNNING_VAR->set_uid(ICudnn::create_new_uid());
        options.inputs.EPSILON->set_uid(ICudnn::create_new_uid());
        options.inputs.EXP_AVG->set_uid(ICudnn::create_new_uid());
        options.inputs.ACCUM_COUNT->set_uid(ICudnn::create_new_uid());
        options.outputs.EQ_BIAS->set_uid(ICudnn::create_new_uid());
        options.outputs.EQ_SCALE->set_uid(ICudnn::create_new_uid());
        options.outputs.NEXT_RUNNING_MEAN->set_uid(ICudnn::create_new_uid());
        options.outputs.NEXT_RUNNING_VAR->set_uid(ICudnn::create_new_uid());
        return {error_code_t::OK, ""};
    }

    error_t createTensors() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building BatchNormFinalizeNode tensors..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.SUM));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.SQ_SUM));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.INV_VARIANCE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.BIAS));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.PREV_RUNNING_MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.PREV_RUNNING_VAR));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.EPSILON));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.EXP_AVG));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.ACCUM_COUNT));

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.EQ_BIAS));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.EQ_SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.NEXT_RUNNING_MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.NEXT_RUNNING_VAR));

        getLogger() << "[cudnn_frontend] INFO: " << "Built BatchNormFinalizeNode tensors." << std::endl;

        return {error_code_t::OK, ""};
    }
    
    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building BatchNormFinalizeNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        // Create the batchnorm operation.
        auto batchnorm_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR)
                                        .setComputeType(CUDNN_DATA_FLOAT)
                                        .setBNFinalizeMode(CUDNN_BN_FINALIZE_STATISTICS_TRAINING)
                                        .setSumDesc(*(tensors.at(options.inputs.SUM->get_uid())))
                                        .setSqSumDesc(*(tensors.at(options.inputs.SQ_SUM->get_uid())))
                                        .setEqScaleAndBias(*(tensors.at(options.outputs.EQ_SCALE->get_uid())), *(tensors.at(options.outputs.EQ_BIAS->get_uid())))
                                        .setSavedMeanAndInvVar(*(tensors.at(options.inputs.MEAN->get_uid())), *(tensors.at(options.inputs.INV_VARIANCE->get_uid())))
                                        .setScaleAndBias(*(tensors.at(options.inputs.SCALE->get_uid())), *(tensors.at(options.inputs.BIAS->get_uid())))
                                        .setPrevRunningMeanAndVar(*(tensors.at(options.inputs.PREV_RUNNING_MEAN->get_uid())), *(tensors.at(options.inputs.PREV_RUNNING_VAR->get_uid())))
                                        .setNextRunningMeanAndVar(*(tensors.at(options.outputs.NEXT_RUNNING_MEAN->get_uid())), *(tensors.at(options.outputs.NEXT_RUNNING_VAR->get_uid())))
                                        .setEpsilonTensor(*(tensors.at(options.inputs.EPSILON->get_uid())))
                                        .setExpDecayFactorTensor(*(tensors.at(options.inputs.EXP_AVG->get_uid())))
                                        .setAccumCountTensor(*(tensors.at(options.inputs.ACCUM_COUNT->get_uid())))
                                        .build();
        operations.emplace(name, std::make_shared<Operation_v8>(std::move(batchnorm_operation)));
        
        // Push all real tensors as required for operation execution.
        auto const& tensors_involved_in_operation = {
            options.inputs.SUM
            , options.inputs.SQ_SUM
            , options.inputs.MEAN
            , options.inputs.INV_VARIANCE
            , options.inputs.PREV_RUNNING_MEAN
            , options.inputs.PREV_RUNNING_VAR
            , options.inputs.EPSILON
            , options.inputs.EXP_AVG
            , options.inputs.ACCUM_COUNT
            , options.inputs.SCALE
            , options.inputs.BIAS
            , options.outputs.EQ_BIAS
            , options.outputs.EQ_SCALE
            , options.outputs.NEXT_RUNNING_MEAN
            , options.outputs.NEXT_RUNNING_VAR
        };
        for(auto const& tensor: tensors_involved_in_operation) {
            if(tensor && tensor->get_is_virtual() == false) {
                tensors_in_operations[name].emplace_back(tensor->get_uid());
            }
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Built BatchNormFinalizeNode operation." << std::endl;

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

} // namespace graph

} // namespace cudnn_frontend
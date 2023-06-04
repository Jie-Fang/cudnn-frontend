#pragma once

#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {

class BatchNormNode : public INode {
    std::shared_ptr<Batchnorm> options;
public:

    BatchNormNode(std::string const& name, std::shared_ptr<Batchnorm> const options, int64_t offset = 1)  : INode (name, offset), options(options) {
        // outputs should be float type
        options->outputs.MEAN->set_data_type(DataType_t::FLOAT);
        options->outputs.INV_VARIANCE->set_data_type(DataType_t::FLOAT);
        options->outputs.NEXT_RUNNING_MEAN->set_data_type(DataType_t::FLOAT);
        options->outputs.NEXT_RUNNING_VAR->set_data_type(DataType_t::FLOAT);
    }

    Type getType() override final {
        return Type::BATCHNORM;
    }

    error_t infer_properties() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for batchnorm node named " << name << "." << std::endl;

        // Merge with ancestor's context
        fill_missing_context();

        options->fill_from_context(get_context());

        // TODO: Only inferencing from X works today.
        auto X = options->inputs.X;
        auto const x_tensor_dim = X->get_dim();

        auto Y = options->outputs.Y;
        auto y_tensor_dim = Y->get_dim();
        if(y_tensor_dim.empty()) {
            y_tensor_dim.resize(x_tensor_dim.size());
            Y->set_dim(x_tensor_dim).generateStrides(CUDNN_TENSOR_NHWC);
        }

        // Set channel length tensors
        auto infer_per_channel_tensors = [&x_tensor_dim] (std::shared_ptr<Tensor>& T) {
            auto tensor_dim = T->get_dim();
            if(tensor_dim.empty()) {
                tensor_dim.resize(x_tensor_dim.size(), 1);
                tensor_dim[1] = x_tensor_dim[1];
                T->set_dim(tensor_dim).generateStrides(CUDNN_TENSOR_NHWC);
            }
        };
        infer_per_channel_tensors(options->outputs.MEAN);
        infer_per_channel_tensors(options->outputs.INV_VARIANCE);
        infer_per_channel_tensors(options->outputs.NEXT_RUNNING_MEAN);
        infer_per_channel_tensors(options->outputs.NEXT_RUNNING_VAR);
        infer_per_channel_tensors(options->inputs.PREV_RUNNING_MEAN);
        infer_per_channel_tensors(options->inputs.PREV_RUNNING_VAR);
        infer_per_channel_tensors(options->inputs.SCALE);
        infer_per_channel_tensors(options->inputs.BIAS);

        // Set scalars
        auto infer_scalars = [&x_tensor_dim] (std::shared_ptr<Tensor>& T) {
            auto tensor_dim = T->get_dim();
            if(tensor_dim.empty()) {
                tensor_dim.resize(x_tensor_dim.size(), 1);
                T->set_dim(tensor_dim).generateStrides(CUDNN_TENSOR_NHWC);
            }
        };
        infer_scalars(options->inputs.EPSILON);
        infer_scalars(options->inputs.EXP_AVG);

        return error_t::OK;
    }
    
    error_t validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating BatchNormNode..." << std::endl;

        auto X = options->inputs.X;
        auto const x_tensor_dim = X->get_dim();

        auto Y = options->outputs.Y;
        auto const y_tensor_dim = Y->get_dim();
        if(x_tensor_dim != y_tensor_dim) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at X and Y ports of " << name << "." << std::endl;
            return status;
        }

        auto validate_per_channel_tensors = [this, &x_tensor_dim] (std::shared_ptr<Tensor>& T) {
            if(x_tensor_dim[1] != T->get_dim()[1]) {
                auto status = error_t::SHAPE_DEDUCTION_FAILED;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at X and Y ports of " << name << "." << std::endl;
                return status;
            }
            return error_t::OK;
        };
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options->outputs.MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options->outputs.INV_VARIANCE));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options->outputs.NEXT_RUNNING_MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options->outputs.NEXT_RUNNING_VAR));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options->inputs.PREV_RUNNING_MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options->inputs.PREV_RUNNING_VAR));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options->inputs.SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options->inputs.BIAS));

        auto validate_scalars = [this] (std::shared_ptr<Tensor>& T) {
            auto tensor_dim = T->get_dim();
            bool allOnes = std::all_of(tensor_dim.begin(), tensor_dim.end(), [](float const element) {
                return element == 1;
            });
            if(!allOnes) {
                auto status = error_t::SHAPE_DEDUCTION_FAILED;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at X and Y ports of " << name << "." << std::endl;
                return status;
            }
            return error_t::OK;
        };
        CHECK_CUDNN_FRONTEND_ERROR(validate_scalars(options->inputs.EPSILON));
        CHECK_CUDNN_FRONTEND_ERROR(validate_scalars(options->inputs.EXP_AVG));

        getLogger() << "[cudnn_frontend] INFO: " << "Validated BatchNormNode." << std::endl;
        return error_t::OK;
    }
    
    error_t assignUids_() override final {
        options->inputs.X->set_uid(ICudnn::create_new_uid());
        options->inputs.SCALE->set_uid(ICudnn::create_new_uid());
        options->inputs.BIAS->set_uid(ICudnn::create_new_uid());
        options->inputs.PREV_RUNNING_MEAN->set_uid(ICudnn::create_new_uid());
        options->inputs.PREV_RUNNING_VAR->set_uid(ICudnn::create_new_uid());
        options->inputs.EPSILON->set_uid(ICudnn::create_new_uid());
        options->inputs.EXP_AVG->set_uid(ICudnn::create_new_uid());
        options->outputs.Y->set_uid(ICudnn::create_new_uid());
        options->outputs.MEAN->set_uid(ICudnn::create_new_uid());
        options->outputs.INV_VARIANCE->set_uid(ICudnn::create_new_uid());
        options->outputs.NEXT_RUNNING_MEAN->set_uid(ICudnn::create_new_uid());
        options->outputs.NEXT_RUNNING_VAR->set_uid(ICudnn::create_new_uid());
        return error_t::OK;
    }

    error_t createTensors() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building BatchNormNode tensors..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->inputs.X));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->inputs.PREV_RUNNING_MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->inputs.PREV_RUNNING_VAR));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->inputs.EPSILON));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->inputs.EXP_AVG));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->inputs.SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->inputs.BIAS));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->outputs.Y));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->outputs.MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->outputs.INV_VARIANCE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->outputs.NEXT_RUNNING_MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options->outputs.NEXT_RUNNING_VAR));

        getLogger() << "[cudnn_frontend] INFO: " << "Built BatchNormNode tensors." << std::endl;

        return error_t::OK;
    }
    
    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building BatchNormNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        // Create the batchnorm operation.
        auto batchnorm_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                                        .setNormalizationMode(NormMode_t::BATCH_NORM)
                                        .setNormFwdPhase(NormFwdPhase_t::TRAINING)
                                        .setxDesc(*(tensors.at(options->inputs.X->get_uid())))
                                        .setSavedMeanAndInvVar(*(tensors.at(options->outputs.MEAN->get_uid())), *(tensors.at(options->outputs.INV_VARIANCE->get_uid())))
                                        .setScaleAndBias(*(tensors.at(options->inputs.SCALE->get_uid())), *(tensors.at(options->inputs.BIAS->get_uid())))
                                        .setPrevRunningMeanAndVar(*(tensors.at(options->inputs.PREV_RUNNING_MEAN->get_uid())), *(tensors.at(options->inputs.PREV_RUNNING_VAR->get_uid())))
                                        .setNextRunningMeanAndVar(*(tensors.at(options->outputs.NEXT_RUNNING_MEAN->get_uid())), *(tensors.at(options->outputs.NEXT_RUNNING_VAR->get_uid())))
                                        .setEpsilonTensor(*(tensors.at(options->inputs.EPSILON->get_uid())))
                                        .setExpDecayFactorTensor(*(tensors.at(options->inputs.EXP_AVG->get_uid())))
                                        .setyDesc(*(tensors.at(options->outputs.Y->get_uid())))
                                        .build();
        operations.emplace(name, std::make_shared<Operation_v8>(std::move(batchnorm_operation)));
        
        // Push all real tensors as required for operation execution.
        auto const& tensors_involved_in_operation = {
            options->inputs.X
            , options->inputs.PREV_RUNNING_MEAN
            , options->inputs.PREV_RUNNING_VAR
            , options->inputs.EPSILON
            , options->inputs.EXP_AVG
            , options->inputs.SCALE
            , options->inputs.BIAS
            , options->outputs.Y
            , options->outputs.MEAN
            , options->outputs.INV_VARIANCE
            , options->outputs.NEXT_RUNNING_MEAN
            , options->outputs.NEXT_RUNNING_VAR
        };
        for(auto const& tensor: tensors_involved_in_operation) {
            if(tensor && tensor->get_is_virtual() == false) {
                tensors_in_operations[name].emplace_back(tensor->get_uid());
            }
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Built BatchNormNode operation." << std::endl;

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

} // namespace graph

} // namespace cudnn_frontend
#pragma once

#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {

class BatchNormNode : public INode {
public:
    Batchnorm options;

    BatchNormNode(std::string const& name, Batchnorm&& options_, detail::Context const& context)  : INode (name, context), options(std::move(options_)) {
        options.fill_from_context(get_context());
    }

    Type getType() override final {
        return Type::BATCHNORM;
    }

    error_t infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for batchnorm node named " << name << "." << std::endl;

        // TODO: Only inferencing from X works today.
        auto X = options.inputs.X;
        auto const x_tensor_dim = X->get_dim();

        auto Y = options.outputs.Y;
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
        infer_per_channel_tensors(options.outputs.MEAN);
        infer_per_channel_tensors(options.outputs.INV_VARIANCE);
        infer_per_channel_tensors(options.outputs.NEXT_RUNNING_MEAN);
        infer_per_channel_tensors(options.outputs.NEXT_RUNNING_VAR);
        infer_per_channel_tensors(options.inputs.PREV_RUNNING_MEAN);
        infer_per_channel_tensors(options.inputs.PREV_RUNNING_VAR);
        infer_per_channel_tensors(options.inputs.SCALE);
        infer_per_channel_tensors(options.inputs.BIAS);

        // Set scalar tensors
        auto infer_scalar_tensors = [&x_tensor_dim] (std::shared_ptr<Tensor>& T) {
            auto tensor_dim = T->get_dim();
            if(tensor_dim.empty()) {
                tensor_dim.resize(x_tensor_dim.size(), 1);
                T->set_dim(tensor_dim).generateStrides(CUDNN_TENSOR_NHWC);
            }
        };
        infer_scalar_tensors(options.inputs.EPSILON);
        infer_scalar_tensors(options.inputs.MOMENTUM);

        return {error_code_t::OK, ""};
    }
    
    error_t validate_node() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating BatchNormNode..." << std::endl;

        // Norm forward phase should be set
        if(options.forward_phase == NormFwdPhase_t::NOT_SET) {
            auto status = error_code_t::ATTRIBUTE_NOT_SET;
            std::string message = "[cudnn_frontend] ERROR: Forward phase not set of batchnorm node named " + name + ".";
            return {status, message};
        }

        auto X = options.inputs.X;
        auto const x_tensor_dim = X->get_dim();

        auto Y = options.outputs.Y;
        auto const y_tensor_dim = Y->get_dim();
        if(x_tensor_dim != y_tensor_dim) {
            std::string message = "[cudnn_frontend] ERROR: Tensor dimensionality mismatch at X and Y ports of " + name + ".";
            return {error_code_t::SHAPE_DEDUCTION_FAILED, message};
        }

        auto validate_per_channel_tensors = [this, &x_tensor_dim] (std::shared_ptr<Tensor> const& T) {
            error_t status = {error_code_t::OK, ""};
            if(x_tensor_dim[1] != T->get_dim()[1]) {
                status.code = error_code_t::SHAPE_DEDUCTION_FAILED;
                status.err_msg = "[cudnn_frontend] ERROR: Tensor dimensionality mismatch at X and Y ports of " + name + ".";
            }
            return status;
        };

        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.outputs.MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.outputs.INV_VARIANCE));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.outputs.NEXT_RUNNING_MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.outputs.NEXT_RUNNING_VAR));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.inputs.PREV_RUNNING_MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.inputs.PREV_RUNNING_VAR));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.inputs.SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.inputs.BIAS));

        auto validate_scalars = [this] (std::shared_ptr<Tensor> const& T) {
            error_t status = {error_code_t::OK, ""};
            auto tensor_dim = T->get_dim();
            bool allOnes = std::all_of(tensor_dim.begin(), tensor_dim.end(), [](float const element) {
                return element == 1;
            });
            if(!allOnes) {
                status.code = error_code_t::SHAPE_DEDUCTION_FAILED;
                status.err_msg = "[cudnn_frontend] ERROR: Tensor dimensionality mismatch at X and Y ports of " + name + ".";
                return status;
            }
            return status;
        };
        CHECK_CUDNN_FRONTEND_ERROR(validate_scalars(options.inputs.EPSILON));
        CHECK_CUDNN_FRONTEND_ERROR(validate_scalars(options.inputs.MOMENTUM));

        getLogger() << "[cudnn_frontend] INFO: " << "Validated BatchNormNode." << std::endl;
        return {error_code_t::OK, ""};
    }
    
    error_t assign_uids_node() override final {
        options.inputs.X->set_uid(ICudnn::create_new_uid());
        options.inputs.SCALE->set_uid(ICudnn::create_new_uid());
        options.inputs.BIAS->set_uid(ICudnn::create_new_uid());
        options.inputs.PREV_RUNNING_MEAN->set_uid(ICudnn::create_new_uid());
        options.inputs.PREV_RUNNING_VAR->set_uid(ICudnn::create_new_uid());
        options.inputs.EPSILON->set_uid(ICudnn::create_new_uid());
        options.inputs.MOMENTUM->set_uid(ICudnn::create_new_uid());
        options.outputs.Y->set_uid(ICudnn::create_new_uid());
        options.outputs.MEAN->set_uid(ICudnn::create_new_uid());
        options.outputs.INV_VARIANCE->set_uid(ICudnn::create_new_uid());
        options.outputs.NEXT_RUNNING_MEAN->set_uid(ICudnn::create_new_uid());
        options.outputs.NEXT_RUNNING_VAR->set_uid(ICudnn::create_new_uid());
        return {error_code_t::OK, ""};
    }

    error_t createTensors() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building BatchNormNode tensors..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.X));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.PREV_RUNNING_MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.PREV_RUNNING_VAR));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.EPSILON));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.MOMENTUM));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.BIAS));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.Y));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.INV_VARIANCE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.NEXT_RUNNING_MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.NEXT_RUNNING_VAR));

        getLogger() << "[cudnn_frontend] INFO: " << "Built BatchNormNode tensors." << std::endl;

        return {error_code_t::OK, ""};
    }
    
    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building BatchNormNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        // Create the batchnorm operation.
        auto batchnorm_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_FORWARD_DESCRIPTOR)
                                        .setNormalizationMode(NormMode_t::BATCH_NORM)
                                        .setNormFwdPhase(options.forward_phase)
                                        .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
                                        .setSavedMeanAndInvVar(*(tensors.at(options.outputs.MEAN->get_uid())), *(tensors.at(options.outputs.INV_VARIANCE->get_uid())))
                                        .setScaleAndBias(*(tensors.at(options.inputs.SCALE->get_uid())), *(tensors.at(options.inputs.BIAS->get_uid())))
                                        .setPrevRunningMeanAndVar(*(tensors.at(options.inputs.PREV_RUNNING_MEAN->get_uid())), *(tensors.at(options.inputs.PREV_RUNNING_VAR->get_uid())))
                                        .setNextRunningMeanAndVar(*(tensors.at(options.outputs.NEXT_RUNNING_MEAN->get_uid())), *(tensors.at(options.outputs.NEXT_RUNNING_VAR->get_uid())))
                                        .setEpsilonTensor(*(tensors.at(options.inputs.EPSILON->get_uid())))
                                        .setExpDecayFactorTensor(*(tensors.at(options.inputs.MOMENTUM->get_uid())))
                                        .setyDesc(*(tensors.at(options.outputs.Y->get_uid())))
                                        .build();
        operations.emplace(name, std::make_shared<Operation_v8>(std::move(batchnorm_operation)));
        
        // Push all real tensors as required for operation execution.
        auto const& tensors_involved_in_operation = {
            options.inputs.X
            , options.inputs.PREV_RUNNING_MEAN
            , options.inputs.PREV_RUNNING_VAR
            , options.inputs.EPSILON
            , options.inputs.MOMENTUM
            , options.inputs.SCALE
            , options.inputs.BIAS
            , options.outputs.Y
            , options.outputs.MEAN
            , options.outputs.INV_VARIANCE
            , options.outputs.NEXT_RUNNING_MEAN
            , options.outputs.NEXT_RUNNING_VAR
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

        return {error_code_t::OK, ""};
    }

    error_t createOperationGraphs(cudnnHandle_t) override final {
        return {error_code_t::OK, ""};
    }

};

} // namespace graph

} // namespace cudnn_frontend
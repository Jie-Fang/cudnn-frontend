#pragma once

#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {

class DBNNode : public INode {
public:
    DBN options;

    DBNNode(std::string const& name, DBN&& options_, detail::Context const& context)  : INode (name, context), options(std::move(options_)) {
        options.fill_from_context(get_context());

        options.outputs.DBIAS->set_data_type(DataType_t::FLOAT);
        options.outputs.DSCALE->set_data_type(DataType_t::FLOAT);
        options.outputs.DX->set_data_type(DataType_t::FLOAT);

        // User does not create tensor for epsilon/momentum, so create it internally
        // Data type is i/o type
        // epsilon = std::make_shared<Tensor>("epsilon");
        // epsilon->set_dim({1,1,1,1}).set_stride({1,1,1,1}).set_is_pass_by_value(true).set_data_type(DataType_t::FLOAT);
    }

    Type getType() override final {
        return Type::DBN;
    }

    error_t infer_properties_node() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for DBN node named " << name << "." << std::endl;

        // TODO: Only inferencing from X works today.
        auto X = options.inputs.X;
        auto const x_tensor_dim = X->get_dim();

        auto DY = options.inputs.DY;
        auto dy_tensor_dim = DY->get_dim();
        if(dy_tensor_dim.empty()) {
            dy_tensor_dim.resize(x_tensor_dim.size());
            DY->set_dim(x_tensor_dim).generateStrides(CUDNN_TENSOR_NHWC);
        }
        
        auto DX = options.outputs.DX;
        auto dx_tensor_dim = DX->get_dim();
        if(dx_tensor_dim.empty()) {
            dx_tensor_dim.resize(x_tensor_dim.size());
            DX->set_dim(x_tensor_dim).generateStrides(CUDNN_TENSOR_NHWC);
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
        infer_per_channel_tensors(options.inputs.MEAN);
        infer_per_channel_tensors(options.inputs.INV_VARIANCE);
        infer_per_channel_tensors(options.inputs.SCALE);
        infer_per_channel_tensors(options.outputs.DSCALE);
        infer_per_channel_tensors(options.outputs.DBIAS);

        return error_t::OK;
    }
    
    error_t validate_node() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating DBNNode..." << std::endl;

        auto X = options.inputs.X;
        auto const x_tensor_dim = X->get_dim();

        auto DY = options.inputs.DY;
        auto const dy_tensor_dim = DY->get_dim();
        if(x_tensor_dim != dy_tensor_dim) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at X and DY ports of " << name << "." << std::endl;
            return status;
        }
        
        auto DX = options.outputs.DX;
        auto const dx_tensor_dim = DX->get_dim();
        if(x_tensor_dim != dx_tensor_dim) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at X and DX ports of " << name << "." << std::endl;
            return status;
        }

        auto validate_per_channel_tensors = [this, &x_tensor_dim] (std::shared_ptr<Tensor> const& T) {
            if(x_tensor_dim[1] != T->get_dim()[1]) {
                auto status = error_t::SHAPE_DEDUCTION_FAILED;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at X and Y ports of " << name << "." << std::endl;
                return status;
            }
            return error_t::OK;
        };
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.inputs.MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.inputs.INV_VARIANCE));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.inputs.SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.outputs.DSCALE));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(options.outputs.DBIAS));

        // auto validate_scalars = [this] (std::shared_ptr<Tensor> const& T) {
        //     auto tensor_dim = T->get_dim();
        //     bool allOnes = std::all_of(tensor_dim.begin(), tensor_dim.end(), [](float const element) {
        //         return element == 1;
        //     });
        //     if(!allOnes) {
        //         auto status = error_t::SHAPE_DEDUCTION_FAILED;
        //         getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at X and Y ports of " << name << "." << std::endl;
        //         return status;
        //     }
        //     return error_t::OK;
        // };
        // CHECK_CUDNN_FRONTEND_ERROR(validate_scalars(epsilon));

        getLogger() << "[cudnn_frontend] INFO: " << "Validated DBNNode." << std::endl;
        return error_t::OK;
    }
    
    error_t assign_uids_node() override final {
        options.inputs.X->set_uid(ICudnn::create_new_uid());
        options.inputs.DY->set_uid(ICudnn::create_new_uid());
        options.inputs.SCALE->set_uid(ICudnn::create_new_uid());
        options.inputs.MEAN->set_uid(ICudnn::create_new_uid());
        options.inputs.INV_VARIANCE->set_uid(ICudnn::create_new_uid());
        // epsilon->set_uid(ICudnn::create_new_uid());
        options.outputs.DX->set_uid(ICudnn::create_new_uid());
        options.outputs.DSCALE->set_uid(ICudnn::create_new_uid());
        options.outputs.DBIAS->set_uid(ICudnn::create_new_uid());
        return error_t::OK;
    }

    error_t createTensors() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building DBNNode tensors..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.X));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.DY));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.inputs.INV_VARIANCE));
        // CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(epsilon));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.DX));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.DSCALE));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.DBIAS));

        getLogger() << "[cudnn_frontend] INFO: " << "Built DBNNode tensors." << std::endl;

        return error_t::OK;
    }
    
    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building DBNNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        // Create the DBN operation.
        auto DBN_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_NORM_BACKWARD_DESCRIPTOR)
                                        .setNormalizationMode(NormMode_t::BATCH_NORM)
                                        .setxDesc(*(tensors.at(options.inputs.X->get_uid())))
                                        .setdyDesc(*(tensors.at(options.inputs.DY->get_uid())))
                                        .setScale(*(tensors.at(options.inputs.SCALE->get_uid())))
                                        .setSavedMeanAndInvVar(*(tensors.at(options.inputs.MEAN->get_uid())), *(tensors.at(options.inputs.INV_VARIANCE->get_uid())))
                                        .setDScaleAndDBias(*(tensors.at(options.outputs.DSCALE->get_uid())), *(tensors.at(options.outputs.DBIAS->get_uid())))
                                        // .setEpsilonTensor(*(tensors.at(epsilon->get_uid())))
                                        .setdxDesc(*(tensors.at(options.outputs.DX->get_uid())))
                                        .build();
        operations.emplace(name, std::make_shared<Operation_v8>(std::move(DBN_operation)));
        
        // Push all real tensors as required for operation execution.
        auto const& tensors_involved_in_operation = {
            options.inputs.X
            , options.inputs.DY
            , options.inputs.SCALE
            , options.inputs.MEAN
            , options.inputs.INV_VARIANCE
            // , epsilon
            , options.outputs.DX
            , options.outputs.DSCALE
            , options.outputs.DBIAS
        };
        for(auto const& tensor: tensors_involved_in_operation) {
            if(tensor && tensor->get_is_virtual() == false) {
                tensors_in_operations[name].emplace_back(tensor->get_uid());
            }
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Built DBNNode operation." << std::endl;

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

    // virtual error_t pass_by_value_tensors_(std::unordered_map<std::shared_ptr<Tensor>, pass_by_values_t>& tensor_to_pass_by_value) override {
    //     float epsilon_value = options.get_epsilon().value();
    //     tensor_to_pass_by_value.emplace(epsilon, epsilon_value);

    //     return error_t::OK;
    // }
};

} // namespace graph

} // namespace cudnn_frontend
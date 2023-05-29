#pragma once

#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {

class BatchNormNode : public INode {
private:

protected:

public:
    std::shared_ptr<Batchnorm> props;

    BatchNormNode(std::string const& name, int64_t offset = 1)  : INode (name, offset) {}

    Type getType() override final {
        return Type::BATCHNORM;
    }

    int set_properties(std::string const& INode_name, std::shared_ptr<Batchnorm> properties) {
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
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for batchnorm node named " << name << "." << std::endl;
        props->update_uids(offset);

        // Merge with ancestor's context
        fill_missing_context();

        if(props->get_compute_data_type() == DataType_t::NOT_SET) {
            props->set_compute_data_type(context.get_compute_data_type());
        }
        // TODO: Only inferencing from X works today.
        auto x_tensor_prop = get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::X));
        auto const x_tensor_dim = x_tensor_prop->get_dim();

        auto y_tensor_prop = get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Y));
        auto y_tensor_dim = y_tensor_prop->get_dim();
        if(y_tensor_dim.empty()) {
            y_tensor_dim.resize(x_tensor_dim.size());
            y_tensor_prop->set_dim(x_tensor_dim);
        }

        // Set channel length tensors
        auto infer_per_channel_tensors = [this, &x_tensor_dim] (Batchnorm::PORTS const port) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(port));
            auto tensor_dim = tensor_prop->get_dim();
            if(tensor_dim.empty()) {
                tensor_dim.resize(x_tensor_dim.size(), 1);
                tensor_dim[1] = x_tensor_dim[1];
                tensor_prop->set_dim(tensor_dim);
            }
        };
        infer_per_channel_tensors(Batchnorm::PORTS::Mean);
        infer_per_channel_tensors(Batchnorm::PORTS::Var);
        infer_per_channel_tensors(Batchnorm::PORTS::Next_running_mean);
        infer_per_channel_tensors(Batchnorm::PORTS::Next_running_var);
        infer_per_channel_tensors(Batchnorm::PORTS::Previous_running_mean);
        infer_per_channel_tensors(Batchnorm::PORTS::Previous_running_var);
        infer_per_channel_tensors(Batchnorm::PORTS::Scale);
        infer_per_channel_tensors(Batchnorm::PORTS::Bias);

        // Set scalars
        auto infer_scalars = [this, &x_tensor_dim] (Batchnorm::PORTS const port) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(port));
            auto tensor_dim = tensor_prop->get_dim();
            if(tensor_dim.empty()) {
                tensor_dim.resize(x_tensor_dim.size(), 1);
                tensor_prop->set_dim(tensor_dim);
            }
        };
        infer_scalars(Batchnorm::PORTS::EPS);
        infer_scalars(Batchnorm::PORTS::EXP_AVG);

        for(size_t i = 0; i < Batchnorm::PORTS::COUNT; ++i) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(static_cast<Batchnorm::PORTS>(i)));

            tensor_prop->fill_from_context(get_context());

            if(tensor_prop->is_uid_set)
                props->uids[i] = tensor_prop->get_uid();

            tensor_prop->set_properties_from_context(CUDNN_TENSOR_NHWC, props->uids[i]);
        }

        return error_t::OK;
    }
    
    error_t validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating BatchNormNode..." << std::endl;

        auto x_tensor_prop = get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::X));
        auto const x_tensor_dim = x_tensor_prop->get_dim();

        auto y_tensor_prop = get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Y));
        auto const y_tensor_dim = y_tensor_prop->get_dim();
        if(x_tensor_dim != y_tensor_dim) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at X and Y ports of " << name << "." << std::endl;
            return status;
        }

        auto validate_per_channel_tensors = [this, &x_tensor_dim] (Batchnorm::PORTS const port) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(port));
            auto tensor_dim = tensor_prop->get_dim();
            if(x_tensor_dim[1] != tensor_dim[1]) {
                auto status = error_t::SHAPE_DEDUCTION_FAILED;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at X and Y ports of " << name << "." << std::endl;
                return status;
            }
            return error_t::OK;
        };
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm::PORTS::Mean));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm::PORTS::Var));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm::PORTS::Next_running_mean));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm::PORTS::Next_running_var));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm::PORTS::Previous_running_mean));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm::PORTS::Previous_running_var));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm::PORTS::Scale));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm::PORTS::Bias));

        auto validate_scalars = [this] (Batchnorm::PORTS const port) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(port));
            auto tensor_dim = tensor_prop->get_dim();
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
        CHECK_CUDNN_FRONTEND_ERROR(validate_scalars(Batchnorm::PORTS::EPS));
        CHECK_CUDNN_FRONTEND_ERROR(validate_scalars(Batchnorm::PORTS::EXP_AVG));

        getLogger() << "[cudnn_frontend] INFO: " << "Validated BatchNormNode." << std::endl;
        return error_t::OK;
    }

    error_t createTensors() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building BatchNormNode tensors..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::X))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Mean))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Var))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Previous_running_mean))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Previous_running_var))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Next_running_mean))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Next_running_var))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::EPS))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::EXP_AVG))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Scale))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Bias))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Y))));

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
                                        .setxDesc(*(tensors.at(props->uids[Batchnorm::PORTS::X])))
                                        .setSavedMeanAndInvVar(*(tensors.at(props->uids[Batchnorm::PORTS::Mean])), *(tensors.at(props->uids[Batchnorm::PORTS::Var])))
                                        .setScaleAndBias(*(tensors.at(props->uids[Batchnorm::PORTS::Scale])), *(tensors.at(props->uids[Batchnorm::PORTS::Bias])))
                                        .setPrevRunningMeanAndVar(*(tensors.at(props->uids[Batchnorm::PORTS::Previous_running_mean])), *(tensors.at(props->uids[Batchnorm::PORTS::Previous_running_var])))
                                        .setNextRunningMeanAndVar(*(tensors.at(props->uids[Batchnorm::PORTS::Next_running_mean])), *(tensors.at(props->uids[Batchnorm::PORTS::Next_running_var])))
                                        .setEpsilonTensor(*(tensors.at(props->uids[Batchnorm::PORTS::EPS])))
                                        .setExpDecayFactorTensor(*(tensors.at(props->uids[Batchnorm::PORTS::EXP_AVG])))
                                        .setyDesc(*(tensors.at(props->uids[Batchnorm::PORTS::Y])))
                                        .build();
        operations.emplace(name, std::make_shared<Operation_v8>(std::move(batchnorm_operation)));
        
        // Push all real tensors as required for operation execution.
        auto const& tensor_props_involved_in_operation = {
            get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::X))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Mean))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Var))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Previous_running_mean))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Previous_running_var))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Next_running_mean))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Next_running_var))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::EPS))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::EXP_AVG))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Scale))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Bias))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Y))
        };
        for(auto const& tensor_props: tensor_props_involved_in_operation) {
            if(tensor_props->get_is_virtual() == false) {
                tensors_in_operations[name].emplace_back(tensor_props->get_uid());
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
#pragma once

#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {

class BatchNormFinalizeNode : public INode {
private:

protected:

public:
    std::shared_ptr<Batchnorm_finalize> props;

    BatchNormFinalizeNode(std::string const& name, int64_t offset = 1)  : INode (name, offset) {}

    Type getType() override final {
        return Type::BATCHNORM_FINALIZE;
    }

    error_t infer_properties() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for batchnorm finalize node named " << name << "." << std::endl;
        props->update_uids(offset);

        // Merge with ancestor's context
        fill_missing_context();

        if(props->get_compute_data_type() == DataType_t::NOT_SET) {
            props->set_compute_data_type(context.get_compute_data_type());
        }
        // TODO: Only inferencing from SUM works today.
        auto sum_tensor_prop = get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::SUM));
        auto const sum_tensor_dim = sum_tensor_prop->get_dim();

        // Set channel length tensors
        auto infer_per_channel_tensors = [this, &sum_tensor_dim] (Batchnorm_finalize::PORTS const port) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(port));
            auto tensor_dim = tensor_prop->get_dim();
            if(tensor_dim.empty()) {
                tensor_dim = sum_tensor_dim;
                tensor_prop->set_dim(tensor_dim);
            }
        };
        infer_per_channel_tensors(Batchnorm_finalize::PORTS::SQUARE_SUM);
        infer_per_channel_tensors(Batchnorm_finalize::PORTS::MEAN);
        infer_per_channel_tensors(Batchnorm_finalize::PORTS::INV_VARIANCE);
        infer_per_channel_tensors(Batchnorm_finalize::PORTS::SCALE);
        infer_per_channel_tensors(Batchnorm_finalize::PORTS::BIAS);
        infer_per_channel_tensors(Batchnorm_finalize::PORTS::Next_running_mean);
        infer_per_channel_tensors(Batchnorm_finalize::PORTS::Next_running_var);
        infer_per_channel_tensors(Batchnorm_finalize::PORTS::Previous_running_mean);
        infer_per_channel_tensors(Batchnorm_finalize::PORTS::Previous_running_var);
        infer_per_channel_tensors(Batchnorm_finalize::PORTS::EQUIVALENT_BIAS);
        infer_per_channel_tensors(Batchnorm_finalize::PORTS::EQUIVALENT_SCALE);

        // Set scalars
        auto infer_scalars = [this, &sum_tensor_dim] (Batchnorm_finalize::PORTS const port) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(port));
            auto tensor_dim = tensor_prop->get_dim();
            if(tensor_dim.empty()) {
                tensor_dim.resize(sum_tensor_dim.size(), 1);
                tensor_prop->set_dim(tensor_dim);
            }
        };
        infer_scalars(Batchnorm_finalize::PORTS::EPSILON);
        infer_scalars(Batchnorm_finalize::PORTS::EXP_AVG);
        infer_scalars(Batchnorm_finalize::PORTS::ACCUMULATION_COUNT);

        for(size_t i = 0; i < Batchnorm_finalize::PORTS::COUNT; ++i) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(static_cast<Batchnorm_finalize::PORTS>(i)));

            tensor_prop->fill_from_context(get_context());

            if(tensor_prop->is_uid_set)
                props->uids[i] = tensor_prop->get_uid();

            tensor_prop->set_properties_from_context(CUDNN_TENSOR_NHWC, props->uids[i]);
        }

        return error_t::OK;
    }
    
    error_t validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating BatchNormFinalizeNode..." << std::endl;

        auto sum_tensor_prop = get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::SUM));
        auto const sum_tensor_dim = sum_tensor_prop->get_dim();

        auto validate_per_channel_tensors = [this, &sum_tensor_dim] (Batchnorm_finalize::PORTS const port) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(port));
            auto tensor_dim = tensor_prop->get_dim();
            if(sum_tensor_dim != tensor_dim) {
                auto status = error_t::SHAPE_DEDUCTION_FAILED;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at SUM and Y ports of " << name << "." << std::endl;
                return status;
            }
            return error_t::OK;
        };
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm_finalize::PORTS::SQUARE_SUM));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm_finalize::PORTS::MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm_finalize::PORTS::INV_VARIANCE));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm_finalize::PORTS::SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm_finalize::PORTS::BIAS));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm_finalize::PORTS::Next_running_mean));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm_finalize::PORTS::Next_running_var));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm_finalize::PORTS::Previous_running_mean));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm_finalize::PORTS::Previous_running_var));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm_finalize::PORTS::EQUIVALENT_BIAS));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm_finalize::PORTS::EQUIVALENT_SCALE));

        auto validate_scalars = [this] (Batchnorm_finalize::PORTS const port) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(port));
            auto tensor_dim = tensor_prop->get_dim();
            bool allOnes = std::all_of(tensor_dim.begin(), tensor_dim.end(), [](float const element) {
                return element == 1;
            });
            if(!allOnes) {
                auto status = error_t::SHAPE_DEDUCTION_FAILED;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at SUM and Y ports of " << name << "." << std::endl;
                return status;
            }
            return error_t::OK;
        };
        CHECK_CUDNN_FRONTEND_ERROR(validate_scalars(Batchnorm_finalize::PORTS::EPSILON));
        CHECK_CUDNN_FRONTEND_ERROR(validate_scalars(Batchnorm_finalize::PORTS::EXP_AVG));
        CHECK_CUDNN_FRONTEND_ERROR(validate_scalars(Batchnorm_finalize::PORTS::ACCUMULATION_COUNT));

        getLogger() << "[cudnn_frontend] INFO: " << "Validated BatchNormFinalizeNode." << std::endl;
        return error_t::OK;
    }

    error_t createTensors() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building BatchNormFinalizeNode tensors..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::SUM))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::SQUARE_SUM))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::MEAN))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::INV_VARIANCE))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::SCALE))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::BIAS))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::Previous_running_mean))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::Previous_running_var))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::Next_running_mean))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::Next_running_var))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::EPSILON))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::EXP_AVG))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::ACCUMULATION_COUNT))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::EQUIVALENT_BIAS))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::EQUIVALENT_SCALE))));

        getLogger() << "[cudnn_frontend] INFO: " << "Built BatchNormFinalizeNode tensors." << std::endl;

        return error_t::OK;
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
                                        .setSumDesc(*(tensors.at(props->uids[Batchnorm_finalize::PORTS::SUM])))
                                        .setSqSumDesc(*(tensors.at(props->uids[Batchnorm_finalize::PORTS::SQUARE_SUM])))
                                        .setEqScaleAndBias(*(tensors.at(props->uids[Batchnorm_finalize::PORTS::EQUIVALENT_SCALE])), *(tensors.at(props->uids[Batchnorm_finalize::PORTS::EQUIVALENT_BIAS])))
                                        .setSavedMeanAndInvVar(*(tensors.at(props->uids[Batchnorm_finalize::PORTS::MEAN])), *(tensors.at(props->uids[Batchnorm_finalize::PORTS::INV_VARIANCE])))
                                        .setScaleAndBias(*(tensors.at(props->uids[Batchnorm_finalize::PORTS::SCALE])), *(tensors.at(props->uids[Batchnorm_finalize::PORTS::BIAS])))
                                        .setPrevRunningMeanAndVar(*(tensors.at(props->uids[Batchnorm_finalize::PORTS::Previous_running_mean])), *(tensors.at(props->uids[Batchnorm_finalize::PORTS::Previous_running_var])))
                                        .setNextRunningMeanAndVar(*(tensors.at(props->uids[Batchnorm_finalize::PORTS::Next_running_mean])), *(tensors.at(props->uids[Batchnorm_finalize::PORTS::Next_running_var])))
                                        .setEpsilonTensor(*(tensors.at(props->uids[Batchnorm_finalize::PORTS::EPSILON])))
                                        .setExpDecayFactorTensor(*(tensors.at(props->uids[Batchnorm_finalize::PORTS::EXP_AVG])))
                                        .setAccumCountTensor(*(tensors.at(props->uids[Batchnorm_finalize::PORTS::ACCUMULATION_COUNT])))
                                        .build();
        operations.emplace(name, std::make_shared<Operation_v8>(std::move(batchnorm_operation)));
        
        // Push all real tensors as required for operation execution.
        auto const& tensor_props_involved_in_operation = {
            get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::SUM))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::SQUARE_SUM))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::MEAN))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::INV_VARIANCE))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::Previous_running_mean))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::Previous_running_var))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::Next_running_mean))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::Next_running_var))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::EPSILON))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::EXP_AVG))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::ACCUMULATION_COUNT))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::SCALE))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::BIAS))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::EQUIVALENT_BIAS))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_finalize::PORTS::EQUIVALENT_SCALE))
        };
        for(auto const& tensor_props: tensor_props_involved_in_operation) {
            if(tensor_props->get_is_virtual() == false) {
                tensors_in_operations[name].emplace_back(tensor_props->get_uid());
            }
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Built BatchNormFinalizeNode operation." << std::endl;

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
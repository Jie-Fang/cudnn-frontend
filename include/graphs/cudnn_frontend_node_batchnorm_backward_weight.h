#pragma once

#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {

class BatchnormBackwardWeightNode : public INode {
private:

protected:

public:
    std::shared_ptr<Batchnorm_backward_weight> props;

    BatchnormBackwardWeightNode(std::string const& name, int64_t offset = 1)  : INode (name, offset) {}

    Type getType() override final {
        return Type::BATCHNORM_BACKWARD_WEIGHT;
    }

    error_t infer_properties() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferencing properties for batchnorm finalize node named " << name << "." << std::endl;
        props->update_uids(offset);

        // Merge with ancestor's context
        fill_missing_context();

        if(props->get_compute_data_type() == DataType_t::NOT_SET) {
            props->set_compute_data_type(context.get_compute_data_type());
        }
        // TODO: Only inferencing from DY works today.
        auto dy_tensor_prop = get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::DY));
        auto const dy_tensor_dim = dy_tensor_prop->get_dim();

        auto x_tensor_prop = get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::X));
        auto x_tensor_dim = x_tensor_prop->get_dim();
        if(x_tensor_dim.empty()) {
            x_tensor_dim.resize(dy_tensor_dim.size());
            x_tensor_prop->set_dim(dy_tensor_dim);
        }

        // Set channel length tensors
        auto infer_per_channel_tensors = [this, &dy_tensor_dim] (Batchnorm_backward_weight::PORTS const port) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(port));
            auto tensor_dim = tensor_prop->get_dim();
            if(tensor_dim.empty()) {
                tensor_dim = dy_tensor_dim;
                tensor_prop->set_dim(tensor_dim);
            }
        };
        infer_per_channel_tensors(Batchnorm_backward_weight::PORTS::MEAN);
        infer_per_channel_tensors(Batchnorm_backward_weight::PORTS::INV_VARIANCE);
        infer_per_channel_tensors(Batchnorm_backward_weight::PORTS::SCALE);
        infer_per_channel_tensors(Batchnorm_backward_weight::PORTS::DBIAS);
        infer_per_channel_tensors(Batchnorm_backward_weight::PORTS::DSCALE);
        infer_per_channel_tensors(Batchnorm_backward_weight::PORTS::EQUIVALENT_BIAS);
        infer_per_channel_tensors(Batchnorm_backward_weight::PORTS::EQUIVALENT_SCALE_DY);
        infer_per_channel_tensors(Batchnorm_backward_weight::PORTS::EQUIVALENT_SCALE_X);

        for(size_t i = 0; i < Batchnorm_backward_weight::PORTS::COUNT; ++i) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(static_cast<Batchnorm_backward_weight::PORTS>(i)));

            tensor_prop->fill_from_context(get_context());
        }

        return error_t::OK;
    }
    
    error_t validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating BatchnormBackwardWeightNode..." << std::endl;

        auto dy_tensor_prop = get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::DY));
        auto const dy_tensor_dim = dy_tensor_prop->get_dim();

        auto x_tensor_prop = get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::X));
        auto const x_tensor_dim = x_tensor_prop->get_dim();
        if(dy_tensor_dim != x_tensor_dim) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at DY and X ports of " << name << "." << std::endl;
            return status;
        }
        
        auto validate_per_channel_tensors = [this, &dy_tensor_dim] (Batchnorm_backward_weight::PORTS const port) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(port));
            auto tensor_dim = tensor_prop->get_dim();
            if(dy_tensor_dim[1] != tensor_dim[1]) {
                auto status = error_t::SHAPE_DEDUCTION_FAILED;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at DY and Y ports of " << name << "." << std::endl;
                return status;
            }
            return error_t::OK;
        };
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm_backward_weight::PORTS::MEAN));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm_backward_weight::PORTS::INV_VARIANCE));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm_backward_weight::PORTS::SCALE));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm_backward_weight::PORTS::DBIAS));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm_backward_weight::PORTS::DSCALE));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm_backward_weight::PORTS::EQUIVALENT_BIAS));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm_backward_weight::PORTS::EQUIVALENT_SCALE_DY));
        CHECK_CUDNN_FRONTEND_ERROR(validate_per_channel_tensors(Batchnorm_backward_weight::PORTS::EQUIVALENT_SCALE_X));

        getLogger() << "[cudnn_frontend] INFO: " << "Validated BatchnormBackwardWeightNode." << std::endl;
        return error_t::OK;
    }

    error_t createTensors() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building BatchnormBackwardWeightNode tensors..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::X))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::DY))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::MEAN))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::INV_VARIANCE))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::SCALE))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::DSCALE))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::DBIAS))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::EQUIVALENT_BIAS))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::EQUIVALENT_SCALE_DY))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::EQUIVALENT_SCALE_X))));

        getLogger() << "[cudnn_frontend] INFO: " << "Built BatchnormBackwardWeightNode tensors." << std::endl;

        return error_t::OK;
    }
    
    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building BatchnormBackwardWeightNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        // Create the batchnorm operation.
        auto batchnorm_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR)
                                        .setComputeType(CUDNN_DATA_FLOAT)
                                        .setEqScalesAndBias(*(tensors.at(props->uids[Batchnorm_backward_weight::PORTS::EQUIVALENT_SCALE_DY])), *(tensors.at(props->uids[Batchnorm_backward_weight::PORTS::EQUIVALENT_SCALE_X])), *(tensors.at(props->uids[Batchnorm_backward_weight::PORTS::EQUIVALENT_BIAS])))
                                        .setSavedMeanAndInvVar(*(tensors.at(props->uids[Batchnorm_backward_weight::PORTS::MEAN])), *(tensors.at(props->uids[Batchnorm_backward_weight::PORTS::INV_VARIANCE])))
                                        .setScale(*(tensors.at(props->uids[Batchnorm_backward_weight::PORTS::SCALE])))
                                        .setxDesc(*(tensors.at(props->uids[Batchnorm_backward_weight::PORTS::X])))
                                        .setdyDesc(*(tensors.at(props->uids[Batchnorm_backward_weight::PORTS::DY])))
                                        .setDScaleAndDBias(*(tensors.at(props->uids[Batchnorm_backward_weight::PORTS::DSCALE])), *(tensors.at(props->uids[Batchnorm_backward_weight::PORTS::DBIAS])))
                                        .build();
        operations.emplace(name, std::make_shared<Operation_v8>(std::move(batchnorm_operation)));
        
        // Push all real tensors as required for operation execution.
        auto const& tensor_props_involved_in_operation = {
            get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::X))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::DY))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::MEAN))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::INV_VARIANCE))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::SCALE))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::DBIAS))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::DSCALE))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::EQUIVALENT_BIAS))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::EQUIVALENT_SCALE_DY))
            , get_tensor_props(props->get_tensor_at_port(Batchnorm_backward_weight::PORTS::EQUIVALENT_SCALE_X))
        };
        for(auto const& tensor_props: tensor_props_involved_in_operation) {
            if(tensor_props->get_is_virtual() == false) {
                tensors_in_operations[name].emplace_back(tensor_props->get_uid());
            }
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Built BatchnormBackwardWeightNode operation." << std::endl;

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
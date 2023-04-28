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
        props->update_uids(offset);

        for(size_t i = 0; i < Batchnorm::PORTS::COUNT; ++i) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(static_cast<Batchnorm::PORTS>(i)));
            if(tensor_prop->is_uid_set)
                props->uids[i] = tensor_prop->get_uid();
            tensor_prop->set_properties_from_context(CUDNN_TENSOR_NHWC, props->uids[i]);
        }

        return error_t::OK;
    }
    
    error_t validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating BatchNormNode..." << std::endl;

        // TODO: check all properties of this operation and its tensor are correct
        // Like do dim count match dim/stride
        // Do dim and corresponding stride match

        getLogger() << "[cudnn_frontend] INFO: " << "Validated BatchNormNode." << std::endl;
        return error_t::OK;
    }

    error_t createTensors() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building BatchNormNode tensors..." << std::endl;

        create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::X)));
        create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Mean)));
        create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Var)));
        create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Previous_running_mean)));
        create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Previous_running_var)));
        create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Next_running_mean)));
        create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Next_running_var)));
        create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::EPS)));
        create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::EXP_AVG)));
        create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Scale)));
        create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Bias)));
        create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Batchnorm::PORTS::Y)));

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

    error_t partition() override final {
        getLogger() << "[cudnn_frontend] INFO: Partitioning BatchNormNode..." << std::endl;

        auto status = create_cudnn_execution_plan({{name}});
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to create execution plans for graph partitioning in BatchNormNode." << std::endl;
            return status;
        }

        getLogger() << "[cudnn_frontend] INFO: Partitioned BatchNormNode." << std::endl;
        return error_t::OK;
    }
};

} // namespace graph

} // namespace cudnn_frontend
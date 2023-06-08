#pragma once

#include "cudnn_frontend_Rng.h"
#include "cudnn_frontend_Logging.h"

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend::graph {

class RngNode : public INode {
    Rng options;
public:

    RngNode(std::string const& name, Rng&& options_, detail::Context const& context)  : INode (name, context), options(std::move(options_)) {
        options.fill_from_context(get_context());
    }

    Type getType() override final {
        return Type::RNG;
    }

    error_t infer_properties() override final {
        getLogger() << "[cudnn_frontend] INFO: Inferrencing properties for Scaled_dot_product_attention node named " << name << "." << std::endl;

        return error_t::OK;
    }

    error_t assignUids_() override final {
        options.outputs.Y->set_uid(ICudnn::create_new_uid());
        return error_t::OK;
    }

    error_t createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Building RngNode tensors..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(options.outputs.Y));

        getLogger() << "[cudnn_frontend] INFO: " << "Built RngNode tensors." << std::endl;

        return error_t::OK;
    }

    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building RngNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

            if(options.get_distribution() == RngDistribution_t::BERNOULLI) {
                auto rng_descriptor = cudnn_frontend::RngDescBuilder()
                                                    .setRngDistribution(options.get_distribution())
                                                    .setBernoulliDistProbability(options.get_bernoulli_probability().value())
                                                    .build();
                
                auto Rng_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_RNG_DESCRIPTOR)
                                                .setyDesc(*(tensors.at(options.outputs.Y->get_uid())))
                                                .setRngDesc(rng_descriptor)
                                                .setSeed(options.get_seed().value())
                                                .build();
                operations.emplace(name, std::make_shared<Operation_v8>(std::move(Rng_operation)));
            }

        // Push all real tensors as required for operation execution.
        auto const& tensors_involved_in_operation = {
            options.outputs.Y
        };
        auto& tensors_in_operation = tensors_in_operations[name];
        for(auto const& tensor: tensors_involved_in_operation) {
            if(tensor && tensor->get_is_virtual() == false) {
                tensors_in_operation.emplace_back(tensor->get_uid());
            }
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Built RngNode operation." << std::endl;

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
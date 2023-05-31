#pragma once

#include "cudnn_frontend_Logging.h"

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

namespace graph {

class GenstatsNode : public INode {
private:

protected:

public:
    std::shared_ptr<Genstats> props;

    GenstatsNode(std::string const& name, int64_t const offset = 1)  : INode (name, offset) {}

    Type getType() override final {
        return Type::GENSTATS;
    }

    int set_properties(std::string const& INode_name, std::shared_ptr<Genstats> properties) {
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

        // Merge with ancestor's context
        fill_missing_context();

        props->fill_from_context(get_context());

        // TODO: Only inferrencing from (X, W) -> Y works today.
        auto x_tensor_prop = get_tensor_props(props->get_tensor_at_port(Genstats::PORTS::X));
        auto sum_tensor_prop = get_tensor_props(props->get_tensor_at_port(Genstats::PORTS::SUM));
        auto sq_sum_tensor_prop = get_tensor_props(props->get_tensor_at_port(Genstats::PORTS::SQ_SUM));
        
        auto const x_tensor_dim = x_tensor_prop->get_dim();
        auto sum_tensor_dim = sum_tensor_prop->get_dim();
        auto sq_sum_tensor_dim = sq_sum_tensor_prop->get_dim();
        
        if(sum_tensor_dim.empty()) {
            sum_tensor_dim.resize(x_tensor_dim.size(), 1);
            sum_tensor_dim[1] = x_tensor_dim[1];
            sum_tensor_prop->set_dim(sum_tensor_dim);
        } else {
            if(x_tensor_dim.size() != sum_tensor_dim.size()) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at X and SUM ports of " << name << "." << std::endl;
                return status;
            }
        }
        
        if(sq_sum_tensor_dim.empty()) {
            sq_sum_tensor_dim.resize(x_tensor_dim.size(), 1);
            sq_sum_tensor_dim[1] = x_tensor_dim[1];
            sq_sum_tensor_prop->set_dim(sq_sum_tensor_dim);
        } else {
            if(x_tensor_dim.size() != sq_sum_tensor_dim.size()) {
            auto status = error_t::SHAPE_DEDUCTION_FAILED;
                getLogger() << "[cudnn_frontend] ERROR: " << status << " Tensor dimensionality mismatch at X and SQ_SUM ports of " << name << "." << std::endl;
                return status;
            }
        }
        
        for(size_t i = 0; i < Genstats::PORTS::COUNT; ++i) {
            auto tensor_prop = get_tensor_props(props->get_tensor_at_port(static_cast<Genstats::PORTS>(i)));

            tensor_prop->fill_from_context(get_context());
            
            if(tensor_prop->is_uid_set)
                props->uids[i] = tensor_prop->get_uid();
            tensor_prop->set_properties_from_context(CUDNN_TENSOR_NHWC, props->uids[i]);
        }
        return error_t::OK;
    }

    error_t validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating GenstatsNode..." << std::endl;

        getLogger() << "[cudnn_frontend] INFO: " << "Validated GenstatsNode." << std::endl;
        return error_t::OK;
    }

    error_t createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Building GenstatsNode tensors..." << std::endl;

        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Genstats::PORTS::X))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Genstats::PORTS::SUM))));
        CHECK_CUDNN_FRONTEND_ERROR(create_cudnn_tensor(get_tensor_props(props->get_tensor_at_port(Genstats::PORTS::SQ_SUM))));

        getLogger() << "[cudnn_frontend] INFO: " << "Built GenstatsNode tensors." << std::endl;

        return error_t::OK;
    }

    error_t createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building GenstatsNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        auto genstats_operation = cudnn_frontend::OperationBuilder(DescriptorType_t::OPERATION_GEN_STATS_DESCRIPTOR)
                                        .setxDesc(*(tensors.at(props->uids[Genstats::PORTS::X])))
                                        .setGenStatsMode(CUDNN_GENSTATS_SUM_SQSUM)
                                        .setSumDesc(*(tensors.at(props->uids[Genstats::PORTS::SUM])))
                                        .setSqSumDesc(*(tensors.at(props->uids[Genstats::PORTS::SQ_SUM])))
                                        .build();
        
        operations.emplace(name, std::make_shared<Operation_v8>(std::move(genstats_operation)));

        // Push all real tensors as required for operation execution.
        auto const& tensor_props_involved_in_operation = {
            get_tensor_props(props->get_tensor_at_port(Genstats::PORTS::X))
            , get_tensor_props(props->get_tensor_at_port(Genstats::PORTS::SUM))
            , get_tensor_props(props->get_tensor_at_port(Genstats::PORTS::SQ_SUM))
        };
        auto& tensors_in_operation = tensors_in_operations[name];
        for(auto const& tensor_props: tensor_props_involved_in_operation) {
            if(tensor_props->get_is_virtual() == false) {
                tensors_in_operation.emplace_back(tensor_props->get_uid());
            }
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Built GenstatsNode operation." << std::endl;

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
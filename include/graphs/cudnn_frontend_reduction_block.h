#pragma once

#include "cudnn_frontend_ReductionDesc.h"
#include "cudnn_frontend_Logging.h"

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_IBlock.h"

namespace cudnn_frontend {

class ReductionBlock : public IBlock {
private:

protected:

public:
    reduction_node props;

    ReductionBlock(std::string const& name, int64_t const offset = 1)  : IBlock (name, offset), props(name) {
    }

    Type getType() override final {
        return Type::REDUCTION;
    }

    int set_properties(std::string const& IBlock_name, reduction_node const& properties) {
        if(sub_blocks.size() != 0) {
            return 1;
        }
        if(IBlock_name != name) {
            return 1;
        }

        props = properties;
        return 0;
    }

    int infer_properties() override final {
        props.update_uids(offset);
        
        for(size_t i = 0; i < reduction_node::PORTS::COUNT; ++i) {
            auto tensor_prop = get_tensor_props(props.port_to_name.at(static_cast<reduction_node::PORTS>(i)));
            tensor_prop->set_properties_from_context(CUDNN_TENSOR_NHWC, props.get_tensor_data_type(), props.uids[i]);
        }
        return 0;
    }

    int validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating ReductionBlock..." << std::endl;

        getLogger() << "[cudnn_frontend] INFO: " << "Validated ReductionBlock." << std::endl;
        return 0;
    }

    int createTensors() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Building ReductionBlock tensors..." << std::endl;

        auto x_tensor = get_tensor_props(props.port_to_name.at(reduction_node::PORTS::X));
        size_t const dim_count = x_tensor->get_stride().size();
        auto input  = cudnn_frontend::TensorBuilder()
                        .setDim(dim_count, x_tensor->get_dim().data())
                        .setStrides(dim_count, x_tensor->get_stride().data())
                        .setId(x_tensor->get_uid())
                        .setAlignment(16)
                        .setDataType(x_tensor->get_data_type())
                        .setVirtual(x_tensor->get_is_virtual())
                        .setByValue(x_tensor->get_is_pass_by_value())
                        .build();
        tensors.emplace(reduction_node::PORTS::X, std::make_shared<Tensor>(std::move(input)));

        auto y_tensor = get_tensor_props(props.port_to_name.at(reduction_node::PORTS::Y));
        auto output = cudnn_frontend::TensorBuilder()
                        .setDim(dim_count, y_tensor->get_dim().data())
                        .setStrides(dim_count, y_tensor->get_stride().data())
                        .setId(y_tensor->get_uid())
                        .setAlignment(16)
                        .setDataType(y_tensor->get_data_type())
                        .setVirtual(y_tensor->get_is_virtual())
                        .setByValue(y_tensor->get_is_pass_by_value())
                        .build();
        tensors.emplace(reduction_node::PORTS::Y, std::make_shared<Tensor>(std::move(output)));

        getLogger() << "[cudnn_frontend] INFO: " << "Built ReductionBlock tensors." << std::endl;

        return 0;
    }
    
    int createDescritpors() override final {
        return 0;
    }

    int createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building ReductionBlock operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        auto reduction_descriptor = cudnn_frontend::ReductionDescBuilder()
                                                        .setComputeType(props.get_compute_type())
                                                        .setReductionOp(props.get_mode())
                                                        .build();

        auto reduction_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR)
                                        .setxDesc(*(tensors.at(reduction_node::PORTS::X)))
                                        .setyDesc(*(tensors.at(reduction_node::PORTS::Y)))
                                        .setreductionDesc(reduction_descriptor)
                                        .build();
        
        operations.emplace("reduction", std::make_shared<Operation>(std::move(reduction_operation)));

        getLogger() << "[cudnn_frontend] INFO: " << "Built ReductionBlock operation." << std::endl;

        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnn_frontend::cudnnException &e) {
            throw cudnnException(e.what(), e.getCudnnStatus());
        }
        #endif
        
        return 0;
    }

    cudnn_frontend_error_t partition(cudnnHandle_t& handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partioning ReductionBlock..." << std::endl;

        std::vector<Operation const*> operation_graph = {operations.at("reduction").get()};
        auto reduction_graph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(operation_graph.size(), operation_graph.data()).build();
        operation_graphs.push_back(std::make_shared<OperationGraph>(std::move(reduction_graph)));
        
        int status = createExecutionPlan(handle);
        if(status) {
            getLogger() << "[cudnn_frontend] INFO: " << "Failed to create execution plans for graph partitioning in ReductionBlock." << std::endl;
            return cudnn_frontend_error_t::GRAPH_PARTITION_EXECUTION_PLAN_CREATION_FAILED;
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Partitioned ReductionBlock." << std::endl;
        return cudnn_frontend_error_t::OK;
    }

    cudnn_frontend_error_t build(cudnnHandle_t& handle) override final {

        infer_properties();
        validate();
        createTensors();
        createDescritpors();
        createOperations();
        return partition(handle);
    }
    
    cudnn_frontend_error_t execute(cudnnHandle_t& handle, std::unordered_map<std::string, void*> const& tensor_uid_to_pointer_map) override final {
        getLogger() << "[cudnn_frontend] INFO: ReductionBlock starting execution..." << std::endl;

        for(auto const& execution_plan: execution_plans) {
            getLogger() << "[cudnn_frontend] INFO: Executing " << execution_plan->getTag() << "..." << std::endl;
        
            std::vector<int64_t> uids;
            std::vector<void*> device_ptrs;

            uids.reserve(tensor_uid_to_pointer_map.size());
            device_ptrs.reserve(tensor_uid_to_pointer_map.size());

            for (auto const& p : tensor_uid_to_pointer_map) {
                uids.push_back(get_tensor_props(p.first)->get_uid());
                device_ptrs.push_back(p.second);
            }

            auto variant_pack = VariantPackBuilder()
                                .setDataPointers(device_ptrs.size(), device_ptrs.data())
                                .setUids(uids.size(), uids.data())
                                .build();

            auto status = cudnnBackendExecute(handle, execution_plan->get_raw_desc(), variant_pack.get_raw_desc());
            if (status != CUDNN_STATUS_SUCCESS) {
                return cudnn_frontend_error_t::GRAPH_EXECUTION_FAILED;
            }
            getLogger() << "[cudnn_frontend] INFO: Executed " << execution_plan->getTag() << "." << std::endl;
        }
        
        getLogger() << "[cudnn_frontend] INFO: ReductionBlock executed successfully." << std::endl;
        return cudnn_frontend_error_t::OK;
    }
};

} // namespace cudnn_frontend
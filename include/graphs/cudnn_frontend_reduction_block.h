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
    reduction_node props{""};

    ReductionBlock(int64_t const offset = 1) {
        update_uids(offset);
    }

    int update_uids(int64_t const& offset) {
        props.update_uids(offset);
        
        for(size_t i = 0; i < reduction_node::PORTS::COUNT; ++i) {
            tensor_props.insert({i, tensor_properties(props.port_to_name[static_cast<reduction_node::PORTS>(i)])});
            tensor_props.at(i).set_uid(props.uids[i]);
        }

        return 0;
    }

    Type getType() override final {
        return Type::REDUCTION;
    }

    int validate() override final {

        for(size_t i = 0; i < reduction_node::PORTS::COUNT; ++i) {            
            tensor_props.at(i).generateStrides(CUDNN_TENSOR_NHWC);
            // Only override property not set already
            if(!(tensor_props.at(i).check_if_data_type_set())) {
                tensor_props.at(i).set_data_type(props.get_tensor_data_type());
            }
        }

        return 0;
    }

    int createTensors() override final {
        
        getLogger() << "[cudnn_frontend] INFO: " << "Building ReductionBlock tensors..." << std::endl;

        auto& x_tensor = tensor_props.at(reduction_node::PORTS::X);
        size_t const dim_count = x_tensor.get_stride().size();
        auto input  = cudnn_frontend::TensorBuilder()
                        .setDim(dim_count, x_tensor.get_dim().data())
                        .setStrides(dim_count, x_tensor.get_stride().data())
                        .setId(x_tensor.get_uid())
                        .setAlignment(16)
                        .setDataType(x_tensor.get_data_type())
                        .setVirtual(x_tensor.get_is_virtual())
                        .setByValue(x_tensor.get_is_pass_by_value())
                        .build();
        tensors.emplace(reduction_node::PORTS::X, std::make_shared<Tensor>(std::move(input)));

        auto& y_tensor = tensor_props.at(reduction_node::PORTS::Y);
        auto output = cudnn_frontend::TensorBuilder()
                        .setDim(dim_count, y_tensor.get_dim().data())
                        .setStrides(dim_count, y_tensor.get_stride().data())
                        .setId(y_tensor.get_uid())
                        .setAlignment(16)
                        .setDataType(y_tensor.get_data_type())
                        .setVirtual(y_tensor.get_is_virtual())
                        .setByValue(y_tensor.get_is_pass_by_value())
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

    int partition(cudnnHandle_t& handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partioning ReductionBlock..." << std::endl;

        std::vector<Operation const*> operation_graph = {operations["reduction"].get()};
        auto reduction_graph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(operation_graph.size(), operation_graph.data()).build();
        operation_graphs.push_back(std::make_shared<OperationGraph>(std::move(reduction_graph)));

        getLogger() << "[cudnn_frontend] INFO: " << "Partitioned ReductionBlock." << std::endl;
        return 0;
    }

    int build(cudnnHandle_t& handle) override final {

        validate();
        createTensors();
        createDescritpors();
        createOperations();
        partition(handle);
        createExecutionPlan(handle);

        return 0;
    }
    
    int execute(cudnnHandle_t& handle, std::unordered_map<int64_t, void*> const& tensor_uid_to_pointer_map) override final {
        getLogger() << "[cudnn_frontend] INFO: ReductionBlock starting execution..." << std::endl;

        for(auto const& execution_plan: execution_plans) {
            getLogger() << "[cudnn_frontend] INFO: Executing " << execution_plan->getTag() << "..." << std::endl;
        
            std::vector<int64_t> uids;
            std::vector<void*> device_ptrs;

            uids.reserve(tensor_uid_to_pointer_map.size());
            device_ptrs.reserve(tensor_uid_to_pointer_map.size());

            for (auto const& p : tensor_uid_to_pointer_map) {
                uids.push_back(p.first);
                device_ptrs.push_back(p.second);
            }

            auto variant_pack = VariantPackBuilder()
                                .setDataPointers(device_ptrs.size(), device_ptrs.data())
                                .setUids(uids.size(), uids.data())
                                .build();

            auto status = cudnnBackendExecute(handle, execution_plan->get_raw_desc(), variant_pack.get_raw_desc());
            if (status != CUDNN_STATUS_SUCCESS) {
                return 1;
            }
            getLogger() << "[cudnn_frontend] INFO: Executed " << execution_plan->getTag() << "." << std::endl;
        }
        
        getLogger() << "[cudnn_frontend] INFO: ReductionBlock executed successfully." << std::endl;
        return 0;
    }
};

} // namespace cudnn_frontend
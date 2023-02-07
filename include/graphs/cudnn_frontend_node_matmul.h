#pragma once

#include <cudnn_frontend_MatMulDesc.h>
#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

class MatMulNode : public INode {
private:

protected:

public:
    std::shared_ptr<matmul_properties> props;

    MatMulNode(std::string const& name, int64_t offset = 1)  : INode (name, offset) {}

    Type getType() override final {
        return Type::MATMUL;
    }

    int set_properties(std::string const& INode_name, std::shared_ptr<matmul_properties> properties) {
        if(sub_nodes.size() != 0) {
            return 1;
        }
        if(INode_name != name) {
            return 1;
        }
        
        props = properties;
        return 0;
    }

    int infer_properties() override final {
        props->update_uids(offset);

        for(size_t i = 0; i < matmul_properties::PORTS::COUNT; ++i) {
            auto tensor_prop = get_tensor_props(props->get_port_name(static_cast<matmul_properties::PORTS>(i)));
            tensor_prop->set_properties_from_context(CUDNN_TENSOR_NCHW, props->get_tensor_data_type(), props->uids[i]);
        }

        return 0;
    }
    
    int validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating MatMulNode..." << std::endl;

        // TODO: check all properties of this operation and its tensor are correct
        // Like do dim count match dim/stride
        // Do dim and corresponding stride match

        getLogger() << "[cudnn_frontend] INFO: " << "Validated MatMulNode." << std::endl;
        return 0;
    }

    int createTensors() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building MatMulNode tensors..." << std::endl;

        getLogger() << "X: " << props->get_port_name(matmul_properties::PORTS::X);

        auto x_tensor = get_tensor_props(props->get_port_name(matmul_properties::PORTS::X));
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
        tensors.emplace(matmul_properties::PORTS::X, std::make_shared<Tensor>(std::move(input)));

        auto w_tensor = get_tensor_props(props->get_port_name(matmul_properties::PORTS::W));
        auto weight = cudnn_frontend::TensorBuilder()
                        .setDim(dim_count, w_tensor->get_dim().data())
                        .setStrides(dim_count, w_tensor->get_stride().data())
                        .setId(w_tensor->get_uid())
                        .setAlignment(16)
                        .setDataType(w_tensor->get_data_type())
                        .setVirtual(w_tensor->get_is_virtual())
                        .setByValue(w_tensor->get_is_pass_by_value())
                        .build();
        tensors.emplace(matmul_properties::PORTS::W, std::make_shared<Tensor>(std::move(weight)));

        auto y_tensor = get_tensor_props(props->get_port_name(matmul_properties::PORTS::Y));
        auto output = cudnn_frontend::TensorBuilder()
                        .setDim(dim_count, y_tensor->get_dim().data())
                        .setStrides(dim_count, y_tensor->get_stride().data())
                        .setId(y_tensor->get_uid())
                        .setAlignment(16)
                        .setDataType(y_tensor->get_data_type())
                        .setVirtual(y_tensor->get_is_virtual())
                        .setByValue(y_tensor->get_is_pass_by_value())
                        .build();
        tensors.emplace(matmul_properties::PORTS::Y, std::make_shared<Tensor>(std::move(output)));

        getLogger() << "[cudnn_frontend] INFO: " << "Built MatMulNode tensors." << std::endl;

        return 0;
    }
    
    int createDescritpors() override final {
        return 0;
    }

    int createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building MatMulNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        // matmul descriptor
        auto matmul_descriptor = cudnn_frontend::MatMulDescBuilder()
                                                        .setComputeType(props->get_compute_type())
                                                        .build();

        // Create the matmul operation.
        auto matmul_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR)
                                        .setaMatDesc(*(tensors.at(matmul_properties::PORTS::X)))
                                        .setbMatDesc(*(tensors.at(matmul_properties::PORTS::W)))
                                        .setcMatDesc(*(tensors.at(matmul_properties::PORTS::Y)))
                                        .setmatmulDesc(matmul_descriptor)
                                        .build();
        operations.emplace("matmul", std::make_shared<Operation>(std::move(matmul_operation)));

        getLogger() << "[cudnn_frontend] INFO: " << "Built MatMulNode operation." << std::endl;

        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnn_frontend::cudnnException &e) {
            throw cudnnException(e.what(), e.getCudnnStatus());
        }
        #endif
        
        return 0;
    }

    error_t partition(cudnnHandle_t& handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partioning MatMulNode..." << std::endl;

        std::vector<Operation const*> operation_graph = {operations.at("matmul").get()};
        auto matmul_graph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(operation_graph.size(), operation_graph.data()).build();
        operation_graphs.push_back(std::make_shared<OperationGraph>(std::move(matmul_graph)));

        int status = createExecutionPlan(handle);
        if(status) {
            getLogger() << "[cudnn_frontend] INFO: " << "Failed to create execution plans for graph partitioning in MatMulNode." << std::endl;
            return error_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED;
        }

        getLogger() << "[cudnn_frontend] INFO: Partitioned MatMulNode." << std::endl;
        return error_t::OK;
    }

    error_t build(cudnnHandle_t& handle) override final {

        infer_properties();
        validate();
        createTensors();
        createDescritpors();
        createOperations();
        return partition(handle);
    }
    
    error_t execute(cudnnHandle_t& handle, std::unordered_map<std::string, void*> const& tensor_uid_to_pointer_map) override final {
        getLogger() << "[cudnn_frontend] INFO: MatMulNode starting execution..." << std::endl;

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
                return error_t::GRAPH_EXECUTION_FAILED;
            }
            getLogger() << "[cudnn_frontend] INFO: Executed " << execution_plan->getTag() << "." << std::endl;
        }
        
        getLogger() << "[cudnn_frontend] INFO: MatMulNode executed successfully." << std::endl;
        return error_t::OK;
    }
};

} // namespace cudnn_frontend

#pragma once

#include <cudnn_frontend_PointWiseDesc.h>
#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include <cudnn_frontend/blocks/helpers.h>
#include <cudnn_frontend/blocks/iblock.h>

namespace cudnn_frontend {

class PointwiseBlock : public IBlock {
private:

protected:

public:
    pointwise_properties props;

    PointwiseBlock(int64_t const offset = 1) {
        // default initialize tensor properties of the block
        tensor_props["X"];
        tensor_props["B"]; 
        tensor_props["Y"];

        update_uids(offset);
    }

    int update_uids(int64_t const& offset) {
        props.update_uids(offset);
        tensor_props["X"].uid = props.uids[pointwise_properties::UIDs::X_UID];
        tensor_props["B"].uid = props.uids[pointwise_properties::UIDs::B_UID];
        tensor_props["Y"].uid = props.uids[pointwise_properties::UIDs::Y_UID];

        return 0;
    }

    Type getType() override final {
        return Type::POINTWISE;
    }

    int validate() override final {

        cudnn_frontend::generateStrides(tensor_props["X"].dim, tensor_props["X"].stride, tensor_props["X"].dim_count, CUDNN_TENSOR_NHWC);
        cudnn_frontend::generateStrides(tensor_props["B"].dim, tensor_props["B"].stride, tensor_props["B"].dim_count, CUDNN_TENSOR_NHWC);
        cudnn_frontend::generateStrides(tensor_props["Y"].dim, tensor_props["Y"].stride, tensor_props["Y"].dim_count, CUDNN_TENSOR_NHWC);

        tensor_props["X"].data_type = props.tensor_data_type;
        tensor_props["B"].data_type = props.tensor_data_type;
        tensor_props["Y"].data_type = props.tensor_data_type;

        return 0;
    }

    int createTensors() override final {
        
        getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseBlock tensors..." << std::endl;

        auto& x_tensor = tensor_props["X"];
        auto input  = cudnn_frontend::TensorBuilder()
                        .setDim(x_tensor.dim_count, x_tensor.dim)
                        .setStrides(x_tensor.dim_count, x_tensor.stride)
                        .setId(x_tensor.uid)
                        .setAlignment(16)
                        .setDataType(x_tensor.data_type)
                        .setVirtual(x_tensor.is_virtual)
                        .setByValue(x_tensor.is_pass_by_value)
                        .build();
        tensors.emplace("X", std::make_shared<Tensor>(std::move(input)));

        auto& b_tensor = tensor_props["B"];
        auto weight = cudnn_frontend::TensorBuilder()
                        .setDim(b_tensor.dim_count, b_tensor.dim)
                        .setStrides(b_tensor.dim_count, b_tensor.stride)
                        .setId(b_tensor.uid)
                        .setAlignment(16)
                        .setDataType(b_tensor.data_type)
                        .setVirtual(b_tensor.is_virtual)
                        .setByValue(b_tensor.is_pass_by_value)
                        .build();
        tensors.emplace("B", std::make_shared<Tensor>(std::move(weight)));

        auto& y_tensor = tensor_props["Y"];
        auto output = cudnn_frontend::TensorBuilder()
                        .setDim(y_tensor.dim_count, y_tensor.dim)
                        .setStrides(y_tensor.dim_count, y_tensor.stride)
                        .setId(y_tensor.uid)
                        .setAlignment(16)
                        .setDataType(y_tensor.data_type)
                        .setVirtual(y_tensor.is_virtual)
                        .setByValue(y_tensor.is_pass_by_value)
                        .build();
        tensors.emplace("Y", std::make_shared<Tensor>(std::move(output)));

        getLogger() << "[cudnn_frontend] INFO: " << "Built PointwiseBlock tensors." << std::endl;

        return 0;
    }
    
    int createDescritpors() override final {
        return 0;
    }

    int createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building PointwiseBlock operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        auto pointwise_descriptor = cudnn_frontend::PointwiseDescBuilder()
                                                        .setComputeType(props.compute_data_type)
                                                        .setMode(props.mode)
                                                        .build();

        auto pointwise_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
                                        .setxDesc(*(tensors.at("X")))
                                        .setbDesc(*(tensors.at("B")))
                                        .setyDesc(*(tensors.at("Y")))
                                        .setpwDesc(pointwise_descriptor)
                                        .build();
        
        operations.emplace("pointwise", std::make_shared<Operation>(std::move(pointwise_operation)));

        getLogger() << "[cudnn_frontend] INFO: " << "Built PointwiseBlock operation." << std::endl;

        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnn_frontend::cudnnException &e) {
            throw cudnnException(e.what(), e.getCudnnStatus());
        }
        #endif
        
        return 0;
    }

    int partition(cudnnHandle_t& handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partioning PointwiseBlock..." << std::endl;

        std::vector<Operation const*> operation_graph = {operations["pointwise"].get()};
        auto pointwise_graph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(operation_graph.size(), operation_graph.data()).build();
        operation_graphs.push_back(std::make_shared<OperationGraph>(std::move(pointwise_graph)));

        getLogger() << "[cudnn_frontend] INFO: " << "Partitioned PointwiseBlock." << std::endl;
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
    
    int execute(cudnnHandle_t& handle) override final {
        (void)handle;
        return 0;
    } 
};

} // namespace cudnn_frontend
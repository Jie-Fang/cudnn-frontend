#pragma once

#include <cudnn_frontend_ConvDesc.h>
#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include <cudnn_frontend/blocks/helpers.h>
#include <cudnn_frontend/blocks/Iblock.h>

namespace cudnn_frontend {

class ConvolutionBlock : public IBlock {
private:

protected:

public:
    convolution_properties props;

    ConvolutionBlock(int64_t const offset = 1) {
        // default initialize tensor properties of the block
        tensor_props["X"];
        tensor_props["W"]; 
        tensor_props["Y"];

        update_uids(offset);
    }

    int update_uids(int64_t const& offset) {
        props.update_uids(offset);
        tensor_props["X"].uid = props.uids[convolution_properties::UIDs::X_UID];
        tensor_props["W"].uid = props.uids[convolution_properties::UIDs::W_UID];
        tensor_props["Y"].uid = props.uids[convolution_properties::UIDs::Y_UID];

        return 0;
    }

    Type getType() override final {
        return Type::CONVOLUTION;
    }

    int validate() override final {

        cudnn_frontend::generateStrides(tensor_props["X"].dim, tensor_props["X"].stride, tensor_props["X"].dim_count, CUDNN_TENSOR_NHWC);
        cudnn_frontend::generateStrides(tensor_props["W"].dim, tensor_props["W"].stride, tensor_props["W"].dim_count, CUDNN_TENSOR_NHWC);
        cudnn_frontend::generateStrides(tensor_props["Y"].dim, tensor_props["Y"].stride, tensor_props["Y"].dim_count, CUDNN_TENSOR_NHWC);

        tensor_props["X"].data_type = props.tensor_data_type;
        tensor_props["W"].data_type = props.tensor_data_type;
        tensor_props["Y"].data_type = props.tensor_data_type;

        return 0;
    }

    int createTensors() override final {
        
        getLogger() << "[cudnn_frontend] INFO: " << "Building ConvolutionBlock tensors..." << std::endl;

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

        auto& w_tensor = tensor_props["W"];
        auto weight = cudnn_frontend::TensorBuilder()
                        .setDim(w_tensor.dim_count, w_tensor.dim)
                        .setStrides(w_tensor.dim_count, w_tensor.stride)
                        .setId(w_tensor.uid)
                        .setAlignment(16)
                        .setDataType(w_tensor.data_type)
                        .setVirtual(w_tensor.is_virtual)
                        .setByValue(w_tensor.is_pass_by_value)
                        .build();
        tensors.emplace("W", std::make_shared<Tensor>(std::move(weight)));

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

        getLogger() << "[cudnn_frontend] INFO: " << "Built ConvolutionBlock tensors." << std::endl;

        return 0;
    }
    
    int createDescritpors() override final {
        return 0;
    }

    int createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building ConvolutionBlock operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        // convolution descriptor
        int64_t const* conv_stride = props.stride;
        int64_t const* conv_padding = props.padding;
        int64_t const* conv_dilation = props.dilation;

        auto convolution_descriptor = cudnn_frontend::ConvDescBuilder()
                                                        .setComputeType(props.compute_data_type)
                                                        .setMathMode(CUDNN_CROSS_CORRELATION)
                                                        .setSpatialDimCount(props.dim_count)
                                                        .setSpatialStride(props.dim_count, conv_stride)
                                                        .setPrePadding(props.dim_count, conv_padding)
                                                        .setPostPadding(props.dim_count, conv_padding)
                                                        .setDilation(props.dim_count, conv_dilation)
                                                        .build();

        // Create the convolution operation.
        auto convolution_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                                        .setxDesc(*(tensors.at("X")))
                                        .setwDesc(*(tensors.at("W")))
                                        .setyDesc(*(tensors.at("Y")))
                                        .setcDesc(convolution_descriptor)
                                        .setAlpha(1.f)
                                        .setBeta(0.f)
                                        .build();
        
    
        operations.emplace("conv", std::make_shared<Operation>(std::move(convolution_operation)));

        getLogger() << "[cudnn_frontend] INFO: " << "Built ConvolutionBlock operation." << std::endl;

        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnn_frontend::cudnnException &e) {
            throw cudnnException(e.what(), e.getCudnnStatus());
        }
        #endif
        
        return 0;
    }

    int partition(cudnnHandle_t& handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partioning ConvolutionBlock..." << std::endl;

        std::vector<Operation const*> operation_graph = {operations["conv"].get()};
        auto convolution_graph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(operation_graph.size(), operation_graph.data()).build();
        operation_graphs.push_back(std::make_shared<OperationGraph>(std::move(convolution_graph)));

        getLogger() << "[cudnn_frontend] INFO: " << "Partitioned ConvolutionBlock." << std::endl;
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
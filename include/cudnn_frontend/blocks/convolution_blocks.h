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
        update_uids(offset);
    }

    int update_uids(int64_t const& offset) {
        props.update_uids(offset);

        for(size_t i = 0; i < convolution_properties::PORTS::COUNT; ++i) {
            tensor_props[i].name = props.port_to_name[static_cast<convolution_properties::PORTS>(i)];
            tensor_props[i].uid = props.uids[i];
        }

        return 0;
    }

    Type getType() override final {
        return Type::CONVOLUTION;
    }

    int validate() override final {

        for(size_t i = 0; i < convolution_properties::PORTS::COUNT; ++i) {
            cudnn_frontend::generateStrides(tensor_props[i].dim, tensor_props[i].stride, CUDNN_TENSOR_NHWC);
            tensor_props[i].data_type = props.tensor_data_type;
        }

        return 0;
    }

    int createTensors() override final {
        
        getLogger() << "[cudnn_frontend] INFO: " << "Building ConvolutionBlock tensors..." << std::endl;

        auto& x_tensor = tensor_props[convolution_properties::PORTS::X];
        size_t const dim_count = x_tensor.stride.size();
        auto input  = cudnn_frontend::TensorBuilder()
                        .setDim(dim_count, x_tensor.dim.data())
                        .setStrides(dim_count, x_tensor.stride.data())
                        .setId(x_tensor.uid)
                        .setAlignment(16)
                        .setDataType(x_tensor.data_type)
                        .setVirtual(x_tensor.is_virtual)
                        .setByValue(x_tensor.is_pass_by_value)
                        .build();
        tensors.emplace(convolution_properties::PORTS::X, std::make_shared<Tensor>(std::move(input)));

        auto& w_tensor = tensor_props[convolution_properties::PORTS::W];
        auto weight = cudnn_frontend::TensorBuilder()
                        .setDim(dim_count, w_tensor.dim.data())
                        .setStrides(dim_count, w_tensor.stride.data())
                        .setId(w_tensor.uid)
                        .setAlignment(16)
                        .setDataType(w_tensor.data_type)
                        .setVirtual(w_tensor.is_virtual)
                        .setByValue(w_tensor.is_pass_by_value)
                        .build();
        tensors.emplace(convolution_properties::PORTS::W, std::make_shared<Tensor>(std::move(weight)));

        auto& y_tensor = tensor_props[convolution_properties::PORTS::Y];
        auto output = cudnn_frontend::TensorBuilder()
                        .setDim(dim_count, y_tensor.dim.data())
                        .setStrides(dim_count, y_tensor.stride.data())
                        .setId(y_tensor.uid)
                        .setAlignment(16)
                        .setDataType(y_tensor.data_type)
                        .setVirtual(y_tensor.is_virtual)
                        .setByValue(y_tensor.is_pass_by_value)
                        .build();
        tensors.emplace(convolution_properties::PORTS::Y, std::make_shared<Tensor>(std::move(output)));

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
        int64_t const spatial_dim_count = props.padding.size();
        auto convolution_descriptor = cudnn_frontend::ConvDescBuilder()
                                                        .setComputeType(props.compute_data_type)
                                                        .setMathMode(CUDNN_CROSS_CORRELATION)
                                                        .setSpatialDimCount(spatial_dim_count)
                                                        .setSpatialStride(spatial_dim_count, props.stride.data())
                                                        .setPrePadding(spatial_dim_count, props.padding.data())
                                                        .setPostPadding(spatial_dim_count, props.padding.data())
                                                        .setDilation(spatial_dim_count, props.dilation.data())
                                                        .build();

        // Create the convolution operation.
        auto convolution_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                                        .setxDesc(*(tensors.at(convolution_properties::PORTS::X)))
                                        .setwDesc(*(tensors.at(convolution_properties::PORTS::W)))
                                        .setyDesc(*(tensors.at(convolution_properties::PORTS::Y)))
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
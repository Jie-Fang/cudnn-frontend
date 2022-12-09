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
    convolution_node props{""};

    ConvolutionBlock(int64_t const offset = 1) {
        update_uids(offset);
    }

    int update_uids(int64_t const& offset) {
        props.update_uids(offset);

        for(size_t i = 0; i < convolution_node::PORTS::COUNT; ++i) {
            tensor_props.insert(std::make_pair<int64_t, tensor_properties> (static_cast<convolution_node::PORTS>(i), tensor_properties(props.port_to_name[static_cast<convolution_node::PORTS>(i)])));
            tensor_props.at(i).set_uid(props.uids[i]);
        }

        return 0;
    }

    Type getType() override final {
        return Type::CONVOLUTION;
    }

    int validate() override final {

        for(size_t i = 0; i < convolution_node::PORTS::COUNT; ++i) {
            tensor_props.at(i).generateStrides(CUDNN_TENSOR_NHWC);
            tensor_props.at(i).set_data_type(props.get_tensor_data_type());
        }

        return 0;
    }

    int createTensors() override final {
        
        getLogger() << "[cudnn_frontend] INFO: " << "Building ConvolutionBlock tensors..." << std::endl;

        auto& x_tensor = tensor_props.at(convolution_node::PORTS::X);
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
        tensors.emplace(convolution_node::PORTS::X, std::make_shared<Tensor>(std::move(input)));

        auto& w_tensor = tensor_props.at(convolution_node::PORTS::W);
        auto weight = cudnn_frontend::TensorBuilder()
                        .setDim(dim_count, w_tensor.get_dim().data())
                        .setStrides(dim_count, w_tensor.get_stride().data())
                        .setId(w_tensor.get_uid())
                        .setAlignment(16)
                        .setDataType(w_tensor.get_data_type())
                        .setVirtual(w_tensor.get_is_virtual())
                        .setByValue(w_tensor.get_is_pass_by_value())
                        .build();
        tensors.emplace(convolution_node::PORTS::W, std::make_shared<Tensor>(std::move(weight)));

        auto& y_tensor = tensor_props.at(convolution_node::PORTS::Y);
        auto output = cudnn_frontend::TensorBuilder()
                        .setDim(dim_count, y_tensor.get_dim().data())
                        .setStrides(dim_count, y_tensor.get_stride().data())
                        .setId(y_tensor.get_uid())
                        .setAlignment(16)
                        .setDataType(y_tensor.get_data_type())
                        .setVirtual(y_tensor.get_is_virtual())
                        .setByValue(y_tensor.get_is_pass_by_value())
                        .build();
        tensors.emplace(convolution_node::PORTS::Y, std::make_shared<Tensor>(std::move(output)));

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
        int64_t const spatial_dim_count = props.get_padding().size();
        auto convolution_descriptor = cudnn_frontend::ConvDescBuilder()
                                                        .setComputeType(props.get_compute_type())
                                                        .setMathMode(CUDNN_CROSS_CORRELATION)
                                                        .setSpatialDimCount(spatial_dim_count)
                                                        .setSpatialStride(spatial_dim_count, props.get_stride().data())
                                                        .setPrePadding(spatial_dim_count, props.get_padding().data())
                                                        .setPostPadding(spatial_dim_count, props.get_padding().data())
                                                        .setDilation(spatial_dim_count, props.get_dilation().data())
                                                        .build();

        // Create the convolution operation.
        auto convolution_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                                        .setxDesc(*(tensors.at(convolution_node::PORTS::X)))
                                        .setwDesc(*(tensors.at(convolution_node::PORTS::W)))
                                        .setyDesc(*(tensors.at(convolution_node::PORTS::Y)))
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

        int status = createExecutionPlan(handle);
        if(status) {
            getLogger() << "[cudnn_frontend] INFO: " << "Failed to create execution plans for graph partitioning in ConvolutionBlock." << std::endl;
            return status;
        }

        getLogger() << "[cudnn_frontend] INFO: Partitioned ConvolutionBlock." << std::endl;
        return 0;
    }

    int build(cudnnHandle_t& handle) override final {

        validate();
        createTensors();
        createDescritpors();
        createOperations();
        partition(handle);

        return 0;
    }
    
    int execute(cudnnHandle_t& handle, std::unordered_map<int64_t, void*> const& tensor_uid_to_pointer_map) override final {
        getLogger() << "[cudnn_frontend] INFO: ConvolutionBlock starting execution..." << std::endl;

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
        
        getLogger() << "[cudnn_frontend] INFO: ConvolutionBlock executed successfully." << std::endl;
        return 0;
    }
};

} // namespace cudnn_frontend
#pragma once

#include <cudnn_frontend_ConvDesc.h>
#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_IBlock.h"

namespace cudnn_frontend {

class ConvolutionBlock : public IBlock {
private:

protected:

public:
    convolution_node props;

    ConvolutionBlock(std::string const& name, int64_t offset = 1)  : IBlock (name, offset), props(name) {
    }

    Type getType() override final {
        return Type::CONVOLUTION;
    }

    int set_properties(std::string const& IBlock_name, convolution_node const& properties) {
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

        for(size_t i = 0; i < convolution_node::PORTS::COUNT; ++i) {
            auto tensor_prop = get_tensor_props(props.port_to_name.at(static_cast<convolution_node::PORTS>(i)));
            tensor_prop->set_properties_from_context(CUDNN_TENSOR_NHWC, props.get_tensor_data_type(), props.uids[i]);
        }

        return 0;
    }
    
    int validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating ConvolutionBlock..." << std::endl;

        // TODO: check all properties of this operation and its tensor are correct
        // Like do dim count match dim/stride
        // Do dim and corresponding stride match

        getLogger() << "[cudnn_frontend] INFO: " << "Validated ConvolutionBlock." << std::endl;
        return 0;
    }

    int createTensors() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building ConvolutionBlock tensors..." << std::endl;

        getLogger() << "X: " << props.port_to_name.at(convolution_node::PORTS::X);

        auto x_tensor = get_tensor_props(props.port_to_name.at(convolution_node::PORTS::X));
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
        tensors.emplace(convolution_node::PORTS::X, std::make_shared<Tensor>(std::move(input)));

        auto w_tensor = get_tensor_props(props.port_to_name.at(convolution_node::PORTS::W));
        auto weight = cudnn_frontend::TensorBuilder()
                        .setDim(dim_count, w_tensor->get_dim().data())
                        .setStrides(dim_count, w_tensor->get_stride().data())
                        .setId(w_tensor->get_uid())
                        .setAlignment(16)
                        .setDataType(w_tensor->get_data_type())
                        .setVirtual(w_tensor->get_is_virtual())
                        .setByValue(w_tensor->get_is_pass_by_value())
                        .build();
        tensors.emplace(convolution_node::PORTS::W, std::make_shared<Tensor>(std::move(weight)));

        auto y_tensor = get_tensor_props(props.port_to_name.at(convolution_node::PORTS::Y));
        auto output = cudnn_frontend::TensorBuilder()
                        .setDim(dim_count, y_tensor->get_dim().data())
                        .setStrides(dim_count, y_tensor->get_stride().data())
                        .setId(y_tensor->get_uid())
                        .setAlignment(16)
                        .setDataType(y_tensor->get_data_type())
                        .setVirtual(y_tensor->get_is_virtual())
                        .setByValue(y_tensor->get_is_pass_by_value())
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

    cudnn_frontend_error_t partition(cudnnHandle_t& handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partioning ConvolutionBlock..." << std::endl;

        std::vector<Operation const*> operation_graph = {operations.at("conv").get()};
        auto convolution_graph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(operation_graph.size(), operation_graph.data()).build();
        operation_graphs.push_back(std::make_shared<OperationGraph>(std::move(convolution_graph)));

        int status = createExecutionPlan(handle);
        if(status) {
            getLogger() << "[cudnn_frontend] INFO: " << "Failed to create execution plans for graph partitioning in ConvolutionBlock." << std::endl;
            return cudnn_frontend_error_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED;
        }

        getLogger() << "[cudnn_frontend] INFO: Partitioned ConvolutionBlock." << std::endl;
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
        getLogger() << "[cudnn_frontend] INFO: ConvolutionBlock starting execution..." << std::endl;

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
        
        getLogger() << "[cudnn_frontend] INFO: ConvolutionBlock executed successfully." << std::endl;
        return cudnn_frontend_error_t::OK;
    }
};

} // namespace cudnn_frontend
#pragma once

#include <cudnn_frontend_ConvDesc.h>
#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include "cudnn_frontend_graph_helpers.h"
#include "cudnn_frontend_node_interface.h"

namespace cudnn_frontend {

class ConvolutionNode : public INode {
private:

protected:

public:
    std::shared_ptr<convolution_properties> props;

    ConvolutionNode(std::string const& name, int64_t offset = 1)  : INode (name, offset) {}

    Type getType() override final {
        return Type::CONVOLUTION;
    }

    int set_properties(std::string const& INode_name, std::shared_ptr<convolution_properties> properties) {
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

        for(size_t i = 0; i < convolution_properties::PORTS::COUNT; ++i) {
            auto tensor_prop = get_tensor_props(props->get_port_name(static_cast<convolution_properties::PORTS>(i)));
            if(tensor_prop->is_uid_set)
                props->uids[i] = tensor_prop->get_uid();
            tensor_prop->set_properties_from_context(CUDNN_TENSOR_NHWC, props->get_tensor_data_type(), props->uids[i]);
        }

        return 0;
    }
    
    int validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating ConvolutionNode..." << std::endl;

        // TODO: check all properties of this operation and its tensor are correct
        // Like do dim count match dim/stride
        // Do dim and corresponding stride match

        getLogger() << "[cudnn_frontend] INFO: " << "Validated ConvolutionNode." << std::endl;
        return 0;
    }

    int createTensors() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building ConvolutionNode tensors..." << std::endl;

        create_cudnn_tensor(get_tensor_props(props->get_port_name(convolution_properties::PORTS::X)));
        create_cudnn_tensor(get_tensor_props(props->get_port_name(convolution_properties::PORTS::W)));
        create_cudnn_tensor(get_tensor_props(props->get_port_name(convolution_properties::PORTS::Y)));

        getLogger() << "[cudnn_frontend] INFO: " << "Built ConvolutionNode tensors." << std::endl;

        return 0;
    }
    
    int createDescritpors() override final {
        return 0;
    }

    int createOperations() override final {

        getLogger() << "[cudnn_frontend] INFO: " << "Building ConvolutionNode operations..." << std::endl;
        
        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        try {
        #endif

        // convolution descriptor
        int64_t const spatial_dim_count = props->get_padding().size();
        auto convolution_descriptor = cudnn_frontend::ConvDescBuilder()
                                                        .setComputeType(props->get_compute_type())
                                                        .setMathMode(CUDNN_CROSS_CORRELATION)
                                                        .setSpatialDimCount(spatial_dim_count)
                                                        .setSpatialStride(spatial_dim_count, props->get_stride().data())
                                                        .setPrePadding(spatial_dim_count, props->get_padding().data())
                                                        .setPostPadding(spatial_dim_count, props->get_padding().data())
                                                        .setDilation(spatial_dim_count, props->get_dilation().data())
                                                        .build();

        // Create the convolution operation.
        auto convolution_operation = cudnn_frontend::OperationBuilder(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                                        .setxDesc(*(tensors.at(props->uids[convolution_properties::PORTS::X])))
                                        .setwDesc(*(tensors.at(props->uids[convolution_properties::PORTS::W])))
                                        .setyDesc(*(tensors.at(props->uids[convolution_properties::PORTS::Y])))
                                        .setcDesc(convolution_descriptor)
                                        .setAlpha(1.f)
                                        .setBeta(0.f)
                                        .build();
        operations.emplace("conv", std::make_shared<Operation>(std::move(convolution_operation)));

        getLogger() << "[cudnn_frontend] INFO: " << "Built ConvolutionNode operation." << std::endl;

        #ifndef NV_CUDNN_DISABLE_EXCEPTION
        } catch (cudnn_frontend::cudnnException &e) {
            throw cudnnException(e.what(), e.getCudnnStatus());
        }
        #endif
        
        return 0;
    }

    error_t partition(cudnnHandle_t& handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partioning ConvolutionNode..." << std::endl;

        std::vector<Operation const*> operation_graph = {operations.at("conv").get()};
        auto convolution_graph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(operation_graph.size(), operation_graph.data()).build();
        operation_graphs.push_back(std::make_shared<OperationGraph>(std::move(convolution_graph)));

        int status = createExecutionPlan(handle);
        if(status) {
            getLogger() << "[cudnn_frontend] INFO: " << "Failed to create execution plans for graph partitioning in ConvolutionNode." << std::endl;
            return error_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED;
        }

        getLogger() << "[cudnn_frontend] INFO: Partitioned ConvolutionNode." << std::endl;
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
        getLogger() << "[cudnn_frontend] INFO: ConvolutionNode starting execution..." << std::endl;

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
        
        getLogger() << "[cudnn_frontend] INFO: ConvolutionNode executed successfully." << std::endl;
        return error_t::OK;
    }
};

} // namespace cudnn_frontend
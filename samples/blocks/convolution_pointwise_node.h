#pragma once

#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include <graphs/cudnn_frontend_graph_helpers.h>
#include <graphs/cudnn_frontend_node_interface.h>
#include <graphs/cudnn_frontend_node_convolution.h>
#include <graphs/cudnn_frontend_node_pointwise.h>

namespace cudnn_frontend {

class ConvolutionPointwiseNode : public INode {
private:

protected:

public:

    ConvolutionPointwiseNode(std::string const& name, int64_t const offset = 1)  : INode (name, offset) {
        auto conv_node = std::make_shared<ConvolutionNode>("conv_node", offset);
        auto pointwise_node = std::make_shared<PointwiseNode>("pointwise_node", offset + 200);

        sub_nodes.emplace("conv_node", conv_node);
        sub_nodes.emplace("pointwise_node", pointwise_node);

        conv_node->parent_node = this;
        pointwise_node->parent_node = this;
    }

    Type getType() override final {
        return Type::COMPOSITE;
    }

    int set_properties(std::string const& INode_name, convolution_properties const& properties) {
        if(sub_nodes.count(INode_name) == 0) {
            return 1;
        }

        auto convolution_node_ptr = std::dynamic_pointer_cast<ConvolutionNode>(sub_nodes.at(INode_name));
        if(convolution_node_ptr == nullptr) {
            return 1;
        }

        convolution_node_ptr->set_properties(INode_name, properties);
        return 0;
    }

    int set_properties(std::string const& INode_name, pointwise_properties const& properties) {
        if(sub_nodes.count(INode_name) == 0) {
            return 1;
        }

        auto pointwise_node_ptr = std::dynamic_pointer_cast<PointwiseNode>(sub_nodes.at(INode_name));
        if(pointwise_node_ptr == nullptr) {
            return 1;
        }

        pointwise_node_ptr->set_properties(INode_name, properties);
        return 0;
    }
    
    int infer_properties() override final {        
        auto const& conv_node_ptr = std::dynamic_pointer_cast<ConvolutionNode>(sub_nodes.at("conv_node"));
        tensor_props.at(conv_node_ptr->props.port_to_name.at(convolution_properties::PORTS::Y))->set_is_virtual(true);

        for(auto const& sub_node: sub_nodes) {
            sub_node.second->infer_properties();
        }
        return 0;
    }

    int validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating ConvolutionPointwiseNode..." << std::endl;

        for(auto const& sub_node: sub_nodes) {
            sub_node.second->validate();
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Validated ConvolutionPointwiseNode." << std::endl;
        return 0;
    }

    error_t partition(cudnnHandle_t& handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partioning ConvolutionPointwiseNode..." << std::endl;

        std::vector<Operation const*> operation_graph = {sub_nodes.at("conv_node")->operations.at("conv").get(), sub_nodes.at("pointwise_node")->operations.at("pointwise").get()};
        auto conv_pointwise_graph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(operation_graph.size(), operation_graph.data()).build();
        operation_graphs.push_back(std::make_shared<OperationGraph>(std::move(conv_pointwise_graph)));

        int status = createExecutionPlan(handle);
        if(status) {
            getLogger() << "[cudnn_frontend] INFO: " << "Failed to create execution plans for graph partitioning in ConvolutionPointwiseNode." << std::endl;
            return error_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED;
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Partitioned ConvolutionPointwiseNode." << std::endl;
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
        getLogger() << "[cudnn_frontend] INFO: ConvolutionPointwiseNode starting execution..." << std::endl;

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
        
        getLogger() << "[cudnn_frontend] INFO: ConvolutionPointwiseNode executed successfully." << std::endl;
        return error_t::OK;
    }
};

} // namespace cudnn_frontend
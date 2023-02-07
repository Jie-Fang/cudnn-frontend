#pragma once

#include <cudnn_frontend_Logging.h>

#include <graphs/cudnn_frontend_graph_helpers.h>
#include <graphs/cudnn_frontend_node_interface.h>
#include <graphs/cudnn_frontend_node_convolution.h>
#include <graphs/cudnn_frontend_node_pointwise.h>
#include <graphs/cudnn_frontend_node_reduction.h>

namespace cudnn_frontend {

class ConvolutionFP8Node : public INode {
private:

protected:

public:

    ConvolutionFP8Node(std::string const& name, int64_t const offset = 1)  : INode (name, offset) {
        auto conv_node = std::make_shared<ConvolutionNode>("conv_node", offset);
        auto X_DQ_node = std::make_shared<PointwiseNode>("X_DQ_node", offset + 100);
        auto W_DQ_node = std::make_shared<PointwiseNode>("W_DQ_node", offset + 200);
        auto Y_Q_node = std::make_shared<PointwiseNode>("Y_Q_node", offset + 300);
        auto amax_node = std::make_shared<ReductionNode>("amax_node", offset + 400);

        sub_nodes.emplace("conv_node", conv_node);
        sub_nodes.emplace("X_DQ_node", X_DQ_node);
        sub_nodes.emplace("W_DQ_node", W_DQ_node);
        sub_nodes.emplace("Y_Q_node", Y_Q_node);
        sub_nodes.emplace("amax_node", amax_node);

        conv_node->parent_node = this;
        X_DQ_node->parent_node = this;
        W_DQ_node->parent_node = this;
        Y_Q_node->parent_node = this;
        amax_node->parent_node = this;
    }

    Type getType() override final {
        return Type::COMPOSITE;
    }

    int set_properties(std::string const& INode_name, std::shared_ptr<convolution_properties> properties) {
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

    int set_properties(std::string const& INode_name, std::shared_ptr<pointwise_properties> properties) {
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

    int set_properties(std::string const& INode_name, std::shared_ptr<reduction_properties> properties) {
        if(sub_nodes.count(INode_name) == 0) {
            return 1;
        }

        auto reduction_node_ptr = std::dynamic_pointer_cast<ReductionNode>(sub_nodes.at(INode_name));
        if(reduction_node_ptr == nullptr) {
            return 1;
        }

        reduction_node_ptr->set_properties(INode_name, properties);
        return 0;
    }

    int infer_properties() override final {        
        auto const& conv_node_ptr = std::dynamic_pointer_cast<ConvolutionNode>(sub_nodes.at("conv_node"));
        auto conv_output_tensor = get_tensor_props(conv_node_ptr->props->get_port_name(convolution_properties::PORTS::Y));
        conv_output_tensor->set_is_virtual(true);
        
        auto const& x_dq_node_ptr = std::dynamic_pointer_cast<PointwiseNode>(sub_nodes.at("X_DQ_node"));
        auto x_dq_tensor = get_tensor_props(x_dq_node_ptr->props->get_port_name(pointwise_properties::PORTS::Y));
        x_dq_tensor->set_is_virtual(true);
        
        auto const& w_dq_node_ptr = std::dynamic_pointer_cast<PointwiseNode>(sub_nodes.at("W_DQ_node"));
        auto w_dq_tensor = get_tensor_props(w_dq_node_ptr->props->get_port_name(pointwise_properties::PORTS::Y));
        w_dq_tensor->set_is_virtual(true);

        for(auto const& sub_node: sub_nodes) {
            sub_node.second->infer_properties();
        }
        return 0;
    }

    int validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating ConvolutionFP8Node..." << std::endl;

        for(auto const& sub_node: sub_nodes) {
            sub_node.second->validate();
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Validated ConvolutionFP8Node." << std::endl;
        return 0;
    }

    error_t partition(cudnnHandle_t& handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partioning ConvolutionFP8Node..." << std::endl;

        std::vector<Operation const*> operation_graph = {sub_nodes.at("conv_node")->operations.at("conv").get()
                                                        , sub_nodes.at("X_DQ_node")->operations.at("pointwise").get()
                                                        , sub_nodes.at("W_DQ_node")->operations.at("pointwise").get()
                                                        , sub_nodes.at("Y_Q_node")->operations.at("pointwise").get()
                                                        , sub_nodes.at("amax_node")->operations.at("reduction").get()};
        auto conv_pointwise_graph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(operation_graph.size(), operation_graph.data()).build();
        operation_graphs.push_back(std::make_shared<OperationGraph>(std::move(conv_pointwise_graph)));

        int status = createExecutionPlan(handle);
        if(status) {
            getLogger() << "[cudnn_frontend] INFO: " << "Failed to create execution plans for graph partitioning in ConvolutionFP8Node." << std::endl;
            return error_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED;
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Partitioned ConvolutionFP8Node." << std::endl;
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
        getLogger() << "[cudnn_frontend] INFO: ConvolutionFP8Node starting execution..." << std::endl;

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
        
        getLogger() << "[cudnn_frontend] INFO: ConvolutionFP8Node executed successfully." << std::endl;
        return error_t::OK;
    }
};

} // namespace cudnn_frontend
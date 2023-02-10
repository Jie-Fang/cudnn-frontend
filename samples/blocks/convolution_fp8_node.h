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
        auto conv_node = std::make_shared<ConvolutionNode>("conv", offset);
        auto X_DQ_node = std::make_shared<PointwiseNode>("X_DQ", offset + 100);
        auto W_DQ_node = std::make_shared<PointwiseNode>("W_DQ", offset + 200);
        auto Y_Q_node = std::make_shared<PointwiseNode>("Y_Q", offset + 300);
        auto amax_node = std::make_shared<ReductionNode>("amax", offset + 400);

        sub_nodes.emplace("conv", conv_node);
        sub_nodes.emplace("X_DQ", X_DQ_node);
        sub_nodes.emplace("W_DQ", W_DQ_node);
        sub_nodes.emplace("Y_Q", Y_Q_node);
        sub_nodes.emplace("amax", amax_node);

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

        auto convolution_node_ptr = std::static_pointer_cast<ConvolutionNode>(sub_nodes.at(INode_name));
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

        auto pointwise_node_ptr = std::static_pointer_cast<PointwiseNode>(sub_nodes.at(INode_name));
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

        auto reduction_node_ptr = std::static_pointer_cast<ReductionNode>(sub_nodes.at(INode_name));
        if(reduction_node_ptr == nullptr) {
            return 1;
        }

        reduction_node_ptr->set_properties(INode_name, properties);
        return 0;
    }

    error_t infer_properties() override final {        
        auto const& conv_node_ptr = std::static_pointer_cast<ConvolutionNode>(sub_nodes.at("conv"));
        auto conv_output_tensor = get_tensor_props(conv_node_ptr->props->get_port_name(convolution_properties::PORTS::Y));
        conv_node_ptr->props->uids[convolution_properties::PORTS::Y] = conv_output_tensor->get_uid();
        conv_output_tensor->set_is_virtual(true);
        
        auto const& x_dq_node_ptr = std::static_pointer_cast<PointwiseNode>(sub_nodes.at("X_DQ"));
        auto x_dq_tensor = get_tensor_props(x_dq_node_ptr->props->get_port_name(pointwise_properties::PORTS::Y));
        x_dq_node_ptr->props->uids[pointwise_properties::PORTS::Y] = x_dq_tensor->get_uid();
        x_dq_tensor->set_is_virtual(true);
        
        auto const& w_dq_node_ptr = std::static_pointer_cast<PointwiseNode>(sub_nodes.at("W_DQ"));
        auto w_dq_tensor = get_tensor_props(w_dq_node_ptr->props->get_port_name(pointwise_properties::PORTS::Y));
        w_dq_node_ptr->props->uids[pointwise_properties::PORTS::Y] = w_dq_tensor->get_uid();
        w_dq_tensor->set_is_virtual(true);

        for(auto const& sub_node: sub_nodes) {
            sub_node.second->infer_properties();
        }
        return error_t::OK;
    }

    error_t validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating ConvolutionFP8Node..." << std::endl;

        for(auto const& sub_node: sub_nodes) {
            sub_node.second->validate();
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Validated ConvolutionFP8Node." << std::endl;
        return error_t::OK;
    }

    error_t partition() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partitioning ConvolutionFP8Node..." << std::endl;

        // All operations in the sub-tree of nodes can be made into a single execution plan
        std::vector<std::string> operation_names;
        for (auto node : sub_nodes) {
            getLogger() << "Getting the operation from " << node.first << std::endl;
            for (auto &operation : node.second->get_operations()) {
                operation_names.push_back(operation.first);
            }
        }
        getLogger() << "[cudnn_frontend] INFO: Operation Graph has " << operation_names.size() << " operations." << std::endl;

        auto status = create_cudnn_execution_plan({operation_names});
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to create execution plans for graph partitioning in ConvolutionFP8Node." << std::endl;
            return status;
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Partitioned ConvolutionFP8Node." << std::endl;
        return error_t::OK;
    }
};

} // namespace cudnn_frontend
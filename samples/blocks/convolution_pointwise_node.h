#pragma once

#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include <graphs/cudnn_frontend_graph_helpers.h>
#include <graphs/cudnn_frontend_node_interface.h>
#include <graphs/cudnn_frontend_node_convolution.h>
#include <graphs/cudnn_frontend_node_pointwise.h>

namespace cudnn_frontend {

namespace graph {

class ConvolutionPointwiseNode : public INode {
private:

protected:

public:

    ConvolutionPointwiseNode(std::string const& name, int64_t const offset = 1)  : INode (name, offset) {
        auto conv_node = std::make_shared<ConvolutionNode>("conv", offset);
        auto pointwise_node = std::make_shared<PointwiseNode>("pointwise", offset + 200);

        sub_nodes.emplace("conv", conv_node);
        sub_nodes.emplace("pointwise", pointwise_node);

        conv_node->parent_node = this;
        pointwise_node->parent_node = this;
    }

    Type getType() override final {
        return Type::COMPOSITE;
    }

    int set_properties(std::string const& INode_name, std::shared_ptr<convolution> properties) {
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

    int set_properties(std::string const& INode_name, std::shared_ptr<pointwise> properties) {
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
    
    error_t infer_properties() override final {        
        auto const& conv_node_ptr = std::dynamic_pointer_cast<ConvolutionNode>(sub_nodes.at("conv"));
        tensor_props.at(conv_node_ptr->props->get_tensor_at_port(convolution::PORTS::Y))->set_is_virtual(true);

        for(auto const& sub_node: sub_nodes) {
            sub_node.second->infer_properties();
        }
        return error_t::OK;
    }

    error_t validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating ConvolutionPointwiseNode..." << std::endl;

        for(auto const& sub_node: sub_nodes) {
            sub_node.second->validate();
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Validated ConvolutionPointwiseNode." << std::endl;
        return error_t::OK;
    }

    error_t partition() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partitioning ConvolutionPointwiseNode..." << std::endl;

        std::vector<std::string> operation_names;
        for (auto node : sub_nodes) {
            getLogger() << "Getting the operation from " << node.first << std::endl;
            for (auto &operation : node.second->get_operations()) {
                operation_names.push_back(operation.first);
            }
        }

        getLogger() << "Operation Graph has " << operation_names.size() << " operations." << std::endl;

        auto status = create_cudnn_execution_plan({operation_names});
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to create execution plans for graph partitioning in ConvolutionPointwiseNode." << std::endl;
            return status;
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Partitioned ConvolutionPointwiseNode." << std::endl;
        return error_t::OK;
    }
};

} // namespace graph

} // namespace cudnn_frontend
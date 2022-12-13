#pragma once

#include "cudnn_frontend_context.h"

namespace cudnn_frontend {



class IGraph : public graph_properties {
protected:
    std::unordered_map<std::string, tensor_properties> all_tensors;
    std::unordered_map<std::string, convolution_node>  conv_nodes;
    std::unordered_map<std::string, pointwise_node>    pointwise_nodes;
    
public:
    IGraph(std::string const &name ) : graph_properties(name) {}
    virtual cudnn_frontend_error_t add_tensor(tensor_properties const &props) = 0;

    virtual cudnn_frontend_error_t add_node(convolution_node const &props) = 0;
    virtual cudnn_frontend_error_t add_node(pointwise_node   const &props) = 0;
};

class Graph : public IGraph {
protected:
    cuDNNFEContext ctx;
    int64_t uid_offset = 1;

public:

    Graph(std::string const &name, cuDNNFEContext const &ctx_) : IGraph(name), ctx(ctx_) {}

    cudnn_frontend_error_t
    is_valid_tensor(std::string const& name) {
        if (all_tensors.find(name) == all_tensors.end()) {
            return cudnn_frontend_error_t::UNKNOWN_TENSOR_NAME;
        } 
        return cudnn_frontend_error_t::OK;
    }

    tensor_properties &
    tensor_at(std::string const& name) {
        return all_tensors.at(name);
    }

    cudnn_frontend_error_t
    add_tensor(tensor_properties const &props) {
        
        std::string name = props.get_name();
        all_tensors.insert(std::pair<std::string, tensor_properties>(name, props));

        auto &tensor = all_tensors.at(name);

        if (tensor.is_dim_set == false) {
            return cudnn_frontend_error_t::TENSOR_DIMENSIONS_NOT_SET;
        }

        if (tensor.is_stride_set == false) {
            tensor.generateStrides(ctx.get_layout() == cuDNNFEContext::Layout::ChannelFirst ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC);
        }

        if (tensor.is_data_type_set == false) {
            tensor.set_data_type(tensor.get_is_virtual() ? ctx.get_intermediate_data_type() :  ctx.get_tensor_data_type());
        }

        return cudnn_frontend_error_t::OK;
    }

    cudnn_frontend_error_t
    add_node(pointwise_node const &props) {
                
        std::string name = props.get_name();
        pointwise_nodes.insert(std::pair<std::string, pointwise_node>(name, props));

        auto &node = pointwise_nodes.at(name);

        if (node.is_mode_set == false) {
            return cudnn_frontend_error_t::POINTWISE_MODE_NOT_SET;
        }

        if (node.is_compute_type_set == false)  {
            node.set_compute_type(ctx.get_compute_type());
        }

        if (node.is_tensor_data_type_set == false)  {
            node.set_tensor_data_type(ctx.get_intermediate_data_type());
        }

        all_tensors.insert(std::make_pair(name + "::Y", tensor_properties{name + "::Y"}));

        return cudnn_frontend_error_t::OK;
    }

    cudnn_frontend_error_t
    add_node(convolution_node const &props) {
                
        std::string name = props.get_name();
        conv_nodes.insert(std::pair<std::string, convolution_node>(name, props));

        auto &node = conv_nodes.at(name);

        if (node.is_compute_type_set == false)  {
            node.set_compute_type(ctx.get_compute_type());
        }

        if (node.is_tensor_data_type_set == false)  {
            node.set_tensor_data_type(ctx.get_intermediate_data_type());
        }

        if (node.is_padding_set == false) {
            node.set_padding(ctx.get_spatial_dims() == 2 ? std::vector<int64_t>{0, 0} : std::vector<int64_t>{0, 0, 0});
        }

        if (node.is_stride_set == false) {
            node.set_stride(ctx.get_spatial_dims() == 2 ? std::vector<int64_t>{1, 1} : std::vector<int64_t>{1, 1, 1});
        }

        all_tensors.insert(std::make_pair(name + "::Y", tensor_properties{name + "::Y"}));

        return cudnn_frontend_error_t::OK;
    }

    cudnn_frontend_error_t
    infer_shapes() {
        std::unordered_map<std::string, bool>   visited_nodes;
        std::unordered_map<std::string, bool>   visited_tensors;
        std::unordered_map<std::string, Node const*> all_nodes;
        std::unordered_map<std::string, Node const*> entrance_nodes;

        getLogger() << "Available tensors are [";
        for (auto &tensor : all_tensors) {
            getLogger() << tensor.first << ", ";
            if (tensor.first.find("::") == std::string::npos) {
                visited_tensors[tensor.first] = true;
            } else {
                visited_tensors[tensor.first] = false;
            }
        }
        getLogger() << "]" << std::endl;


        for (auto &node : conv_nodes) {
            visited_nodes[node.first] = false;
            all_nodes[node.first] = &node.second;
        }
        for (auto &node : pointwise_nodes) {
            visited_nodes[node.first] = false;
            all_nodes[node.first] = &node.second;
        }

        auto find_entrance_nodes = [&all_nodes, &entrance_nodes, &visited_tensors, &visited_nodes] () {
            for (auto &node : all_nodes) {
                if (visited_nodes.at(node.first) == false) {
                    bool is_entrance_node = 
                        std::all_of(std::begin(node.second->get_inputs()),
                                    std::end(node.second->get_inputs()),
                                    [&visited_tensors] (auto input) {
                                        return visited_tensors.at(input) == true;
                                    } );
                    if (is_entrance_node) {
                        entrance_nodes[node.first] = node.second;
                        getLogger() << "Added " << node.first << " to entrance nodes." << std::endl;
                    }
                }
            }
        };

        find_entrance_nodes();

        while (entrance_nodes.size()) {
            Node const *node = entrance_nodes.begin()->second;
            visited_nodes.at(node->get_name()) = true;

            switch (node->get_node_type()) {
                case Node::Type::Pointwise: 
                case Node::Type::Convolution: {
                    auto &tensor = all_tensors.at(node->get_name() + "::Y");
                    auto &inputs = node->get_inputs();
                    // TODO: Compute output size correctly. Right now just
                    // Copying the input tensor sizes
                    tensor.set_data_type(all_tensors.at(inputs[0]).get_data_type());
                    tensor.set_dim(all_tensors.at(inputs[0]).get_dim());
                    tensor.set_stride(all_tensors.at(inputs[0]).get_stride());
                    visited_tensors[tensor.get_name()] = true;
                    find_entrance_nodes();
                    entrance_nodes.erase(entrance_nodes.begin());
                    continue;
                }
                break;
                case Node::Type::Reduction: {
                }
                break;
            }
        }

        if (std::any_of(std::begin(visited_nodes), std::end(visited_nodes),
                        [](auto node){return node.second == false;})) {
            return cudnn_frontend_error_t::SHAPE_DEDUCTION_FAILED;
        }

        return cudnn_frontend_error_t::OK;
    }

    cudnn_frontend_error_t
    build() {
        cudnnHandle_t handle_;
        cudnnCreate(&handle_);
        for (auto &node : conv_nodes) {
            getLogger() << "Adding the conv block" << node.first << std::endl;
            auto conv_block = std::make_shared<ConvolutionBlock>(node.first, uid_offset);
            conv_block->props = node.second;
            conv_block->add_tensor(node.second.port_to_name.at(convolution_node::PORTS::X), all_tensors.at(node.second.port_to_name.at(convolution_node::PORTS::X)));
            conv_block->add_tensor(node.second.port_to_name.at(convolution_node::PORTS::W), all_tensors.at(node.second.port_to_name.at(convolution_node::PORTS::W)));
            conv_block->add_tensor(node.second.port_to_name.at(convolution_node::PORTS::Y), all_tensors.at(node.second.port_to_name.at(convolution_node::PORTS::Y)));
            getLogger() << "Conv block has " << conv_block->tensor_props.size() << " tensors [" ;
            getLogger() << node.second.port_to_name.at(convolution_node::PORTS::X) << ",";
            getLogger() << node.second.port_to_name.at(convolution_node::PORTS::W) << ",";
            getLogger() << node.second.port_to_name.at(convolution_node::PORTS::Y) << "]" << std::endl;

            conv_block->build(handle_);
            uid_offset += 100;
        }

        return cudnn_frontend_error_t::OK;
    }


    ~Graph() = default;
};

}
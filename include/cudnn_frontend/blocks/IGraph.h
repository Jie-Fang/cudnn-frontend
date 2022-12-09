#pragma once

#include "cudnn_frontend/blocks/cudnn_frontend_context.h"

namespace cudnn_frontend {



class IGraph {
protected:
    std::unordered_map<std::string, tensor_properties> all_tensors;
    std::unordered_map<std::string, convolution_node>  conv_nodes;
    std::unordered_map<std::string, pointwise_node>    pointwise_nodes;
    
public:
    virtual cudnn_frontend_error_t add_tensor(tensor_properties const &props) = 0;

    virtual cudnn_frontend_error_t add_node(convolution_node const &props) = 0;
    virtual cudnn_frontend_error_t add_node(pointwise_node   const &props) = 0;
};

class Graph : public IGraph {
    cuDNNFEContext ctx;
public:
    Graph(cuDNNFEContext const &ctx_) : ctx(ctx_) {}

    cudnn_frontend_error_t
    add_tensor(tensor_properties const &props) {
        
        std::string name = props.get_name();
        all_tensors.insert(std::pair<std::string, tensor_properties>(name, props));

        auto &tensor = all_tensors.at(name);

        if (tensor.is_dim_set == false) {
            return cudnn_frontend_error_t::TENSOR_DIMENSIONS_NOT_SET;
        }

        if (tensor.is_stride_set == false) {
            tensor.generateStrides(ctx.get_layout() == cuDNNFEContext::Layout::ChannelFirst ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW);
        }

        if (tensor.is_data_type_set == false) {
            tensor.set_data_type(tensor.get_is_virtual() ? ctx.get_tensor_data_type() : ctx.get_intermediate_data_type());
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

        return cudnn_frontend_error_t::OK;
    }

    cudnn_frontend_error_t
    infer_shapes() {
        std::unordered_map<std::string, bool>   visited_nodes;
        std::unordered_map<std::string, Node *> all_nodes;
        std::unordered_map<std::string, Node *> entrance_nodes;

        for (auto &node : conv_nodes) {
            visited_nodes[node.first] = false;
            all_nodes[node.first] = &node.second;
            bool is_entrance_node = 
                std::all_of(std::begin(node.second.get_inputs()),
                            std::end(node.second.get_inputs()),
                            [] (auto input) {
                                return input.find("::") == std::string::npos;
                            } );
            if (is_entrance_node) {
                entrance_nodes[node.first] = &node.second;
            }
        }
        for (auto &node : pointwise_nodes) {
            visited_nodes[node.first] = false;
            all_nodes[node.first] = &node.second;
            bool is_entrance_node = 
                std::all_of(std::begin(node.second.get_inputs()),
                            std::end(node.second.get_inputs()),
                            [] (auto input) {
                                return input.find("::") == std::string::npos;
                            } );
            if (is_entrance_node) {
                entrance_nodes[node.first] = &node.second;
            }
        }

        return cudnn_frontend_error_t::OK;
    }

    ~Graph() = default;
};

}
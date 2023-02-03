#pragma once

#include <unordered_map>

#include "cudnn_frontend_context.h"
#include "graphs/cudnn_frontend_pointwise_block.h"

namespace cudnn_frontend {



class IGraph : public graph_properties {
protected:
    std::unordered_map<std::string, std::shared_ptr<tensor_properties>> all_tensors;
    std::unordered_map<std::string, convolution_node>  conv_nodes;
    std::unordered_map<std::string, matmul_node>  matmul_nodes;
    std::unordered_map<std::string, pointwise_node>    pointwise_nodes;
    
public:
    IGraph(std::string const &name ) : graph_properties(name) {}
    virtual cudnn_frontend_error_t add_tensor(tensor_properties const &props) = 0;

    virtual cudnn_frontend_error_t add_node(convolution_node const &props) = 0;
    virtual cudnn_frontend_error_t add_node(matmul_node   const &props) = 0;
    virtual cudnn_frontend_error_t add_node(pointwise_node   const &props) = 0;
    
    friend std::ostream& operator<<(std::ostream& os, const IGraph& props);
};

inline std::ostream& operator<<(std::ostream& os, const IGraph& graph) {
    os << "{"
    << "\ntensors: [\n";
    for(auto const& tensor: graph.all_tensors) {
        os << *(tensor.second) << ",";
    }
    os << "],"
    << "\nconv: [\n";
    for(auto const& node: graph.conv_nodes) {
        os << (node.second) << ",";
    }
    os << "],"
    << "\npointwise: [\n";
    for(auto const& node: graph.pointwise_nodes) {
        os << (node.second) << ",";
    }
    os << "],"
    << "\nmatmul: [\n";
    for(auto const& node: graph.matmul_nodes) {
        os << (node.second) << ",";
    }
    os << "],";
    os << "}";
    return os;
}

class Graph : public IGraph {
protected:
    cuDNNFEContext ctx;
    int64_t uid_offset = 1;

    CompositeBlock block{"composite_block", 1};

public:

    Graph(std::string const &name, cuDNNFEContext const &ctx_) : IGraph(name), ctx(ctx_) {}

    cudnn_frontend_error_t
    is_valid_tensor(std::string const& name) {
        if (all_tensors.find(name) == all_tensors.end()) {
            return cudnn_frontend_error_t::INVALID_TENSOR_NAME;
        } 
        return cudnn_frontend_error_t::OK;
    }

    tensor_properties &
    tensor_at(std::string const& name) {
        return *(all_tensors.at(name));
    }

    // Add a tensor to the graph.
    // This function takes in reference to tensor properties and copies the resource.
    // So the lifetime of tensor properties is independent from its replica in graph.
    cudnn_frontend_error_t
    add_tensor(tensor_properties const &props) {
        
        std::string name = props.get_name();
        all_tensors.emplace(name, std::make_shared<tensor_properties>(props));

        auto &tensor = all_tensors.at(name);

        if (tensor->is_dim_set == false) {
            return cudnn_frontend_error_t::ATTRIBUTE_NOT_SET;
        }

        if (tensor->is_stride_set == false) {
            tensor->generateStrides(ctx.get_layout() == cuDNNFEContext::Layout::ChannelFirst ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC);
        }

        if (tensor->is_data_type_set == false) {
            tensor->set_data_type(tensor->get_is_virtual() ? ctx.get_intermediate_data_type() :  ctx.get_tensor_data_type());
        }

        return cudnn_frontend_error_t::OK;
    }

    // Add a tensor properties object with shared ownership.
    // A shared pointer is taken by value, which makes the graph an owner too.
    cudnn_frontend_error_t add_tensor(std::shared_ptr<tensor_properties> props_ptr) {
        all_tensors.emplace(props_ptr->get_name(), props_ptr);
        return cudnn_frontend_error_t::OK;
    }

    // Returns a shared pointer by value, so the caller is also an owner.
    std::shared_ptr<tensor_properties> get_tensor(std::string const& tensor_name) {
        return all_tensors.at(tensor_name);
    }

    cudnn_frontend_error_t
    add_node(pointwise_node const &props) {
                
        std::string name = props.get_name();
        pointwise_nodes.insert(std::pair<std::string, pointwise_node>(name, props));

        auto &node = pointwise_nodes.at(name);

        if (node.is_mode_set == false) {
            return cudnn_frontend_error_t::ATTRIBUTE_NOT_SET;
        }

        if (node.is_compute_type_set == false)  {
            node.set_compute_type(ctx.get_compute_type());
        }

        if (node.is_tensor_data_type_set == false)  {
            node.set_tensor_data_type(ctx.get_intermediate_data_type());
        }

        auto const& output_port_name = props.get_port_name(pointwise_node::PORTS::Y);
        all_tensors.emplace(output_port_name, std::make_shared<tensor_properties>(output_port_name));

        return cudnn_frontend_error_t::OK;
    }

    cudnn_frontend_error_t
    add_node(convolution_node const &props) {
                
        std::string name = props.get_name();
        conv_nodes.emplace(name, props);

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

        auto const& output_port_name = props.get_port_name(convolution_node::PORTS::Y);
        all_tensors.emplace(output_port_name, std::make_shared<tensor_properties>(output_port_name));

        return cudnn_frontend_error_t::OK;
    }

    cudnn_frontend_error_t
    add_node(matmul_node const &props) {
                
        std::string name = props.get_name();
        matmul_nodes.emplace(name, props);

        auto &node = matmul_nodes.at(name);

        if (node.is_compute_type_set == false)  {
            node.set_compute_type(ctx.get_compute_type());
        }

        if (node.is_tensor_data_type_set == false)  {
            node.set_tensor_data_type(ctx.get_intermediate_data_type());
        }

        auto const& output_port_name = props.get_port_name(matmul_node::PORTS::Y);
        all_tensors.emplace(output_port_name, std::make_shared<tensor_properties>(output_port_name));

        return cudnn_frontend_error_t::OK;
    }

    cudnn_frontend_error_t
    infer_shapes() {
        std::unordered_map<std::string, bool>   visited_nodes;
        std::unordered_map<std::string, bool>   visited_tensors;
        std::unordered_map<std::string, Node const*> all_nodes;
        std::vector<std::pair<std::string, Node const*>> entrance_nodes;

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
        for (auto &node : matmul_nodes) {
            visited_nodes[node.first] = false;
            all_nodes[node.first] = &node.second;
        }
        for (auto &node : pointwise_nodes) {
            visited_nodes[node.first] = false;
            all_nodes[node.first] = &node.second;
        }

        auto find_entrance_nodes = [&all_nodes, &entrance_nodes, &visited_tensors, &visited_nodes] () {
            for (auto node : all_nodes) {
                if (visited_nodes.at(node.first) == false) {
                    auto is_visited = [&visited_tensors](std::string input) -> bool {return visited_tensors.at(input) == true;};
                    auto inputs = node.second->get_inputs();
                    bool is_entrance_node = 
                        std::all_of(std::begin(inputs),
                                    std::end(inputs),
                                    is_visited );
                    if (is_entrance_node) {
                        getLogger() << "Added " << node.first << " to entrance nodes." << std::endl;
                        entrance_nodes.push_back(std::pair<std::string, Node const*>(node.first, node.second));
                    }
                }
            }
        };

        find_entrance_nodes();

        while (entrance_nodes.size()) {
            auto it = entrance_nodes.begin();
            Node const *node = it->second;
            visited_nodes.at(node->get_name()) = true;

            switch (node->get_node_type()) {
                case Node::Type::Pointwise:{
                    pointwise_node const *pw_node = dynamic_cast<pointwise_node const *>(node);
                    auto &tensor = all_tensors.at(pw_node->get_port_name(pointwise_node::PORTS::Y));
                    auto &input_x_tensor = all_tensors.at(pw_node->get_port_name(pointwise_node::PORTS::X));
                    // TODO: Compute output size correctly. Right now just
                    // Copying the input tensor sizes
                    tensor->set_data_type(input_x_tensor->get_data_type());
                    tensor->set_dim(input_x_tensor->get_dim());
                    visited_tensors[tensor->get_name()] = true;
                    find_entrance_nodes();
                    entrance_nodes.erase(entrance_nodes.begin());
                    continue;
                }
                break;
                case Node::Type::Convolution: {
                    convolution_node const *conv_node = dynamic_cast<convolution_node const *>(node);
                    auto &tensor = all_tensors.at(conv_node->get_port_name(convolution_node::PORTS::Y));
                    auto &input_x_tensor = all_tensors.at(conv_node->get_port_name(convolution_node::PORTS::X));
                    // TODO: Compute output size correctly. Right now just
                    // Copying the input tensor sizes
                    tensor->set_data_type(input_x_tensor->get_data_type());
                    tensor->set_dim(input_x_tensor->get_dim());
                    visited_tensors[tensor->get_name()] = true;
                    find_entrance_nodes();
                    entrance_nodes.erase(entrance_nodes.begin());
                    continue;
                }
                break;
                case Node::Type::Reduction: {
                }
                case Node::Type::Matmul: {
                    matmul_node const *mm_node = dynamic_cast<matmul_node const *>(node);
                    auto &tensor = all_tensors.at(mm_node->get_port_name(matmul_node::PORTS::Y));
                    auto &input_x_tensor = all_tensors.at(mm_node->get_port_name(matmul_node::PORTS::X));
                    auto &input_w_tensor = all_tensors.at(mm_node->get_port_name(matmul_node::PORTS::W));
                    // TODO: Compute output size correctly. Right now just
                    // Copying the input tensor sizes
                    tensor->set_data_type(input_x_tensor->get_data_type());
                    auto x_dim = input_x_tensor->get_dim();
                    auto w_dim = input_w_tensor->get_dim();
                    tensor->set_dim({x_dim[0], x_dim[1], w_dim[2]});
                    visited_tensors[tensor->get_name()] = true;
                    find_entrance_nodes();
                    entrance_nodes.erase(entrance_nodes.begin());
                    continue;
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

        block.tensor_props = all_tensors;
        
        for (auto &node : conv_nodes) {
            getLogger() << "Adding the conv block " << node.first << std::endl;
            auto conv_block = std::make_shared<ConvolutionBlock>(node.first, uid_offset);
            conv_block->props = node.second;
            conv_block->parent_block = &block;
            block.sub_blocks[node.first] = conv_block;
            uid_offset += 100;
        }

        for (auto &node : matmul_nodes) {
            getLogger() << "Adding the matmul block " << node.first << std::endl;
            auto matmul_block = std::make_shared<MatMulBlock>(node.first, uid_offset);
            matmul_block->props = node.second;
            matmul_block->parent_block = &block;
            block.sub_blocks[node.first] = matmul_block;
            uid_offset += 100;
        }

        for (auto &node : pointwise_nodes) {
            getLogger() << "Adding the pointwise block" << node.first << std::endl;
            auto pointwise_block = std::make_shared<PointwiseBlock>(node.first, uid_offset);
            pointwise_block->props = node.second;
            pointwise_block->parent_block = &block;
            block.sub_blocks[node.first] = pointwise_block;
            uid_offset += 100;
        }

        block.build(handle_);

        return cudnn_frontend_error_t::OK;
    }

    cudnn_frontend_error_t
    execute(std::unordered_map<std::string, void *> var_pack) {
        cudnnHandle_t handle;
        cudnnCreate(&handle);
        
        auto status = block.execute(handle, var_pack);
        return status;
    }

    ~Graph() = default;
};

}
#pragma once

#include <unordered_map>

#include "cudnn_frontend_context.h"
#include "graphs/cudnn_frontend_node_pointwise.h"

namespace cudnn_frontend {



class IGraph {
protected:
    std::string name;
    std::unordered_map<std::string, std::shared_ptr<tensor_properties>> all_tensors;
    std::unordered_map<std::string, std::shared_ptr<convolution_properties>>  conv_properties;
    std::unordered_map<std::string, std::shared_ptr<matmul_properties>>  mm_properties;
    std::unordered_map<std::string, std::shared_ptr<pointwise_properties>>    pw_properties;
    
public:
    IGraph(std::string const &name ) : name(name) {}
    
    std::string const
    get_name() const {
        return name;
    }

    virtual error_t add_tensor(std::shared_ptr<tensor_properties> props_ptr) = 0;

    virtual error_t add_node(std::shared_ptr<convolution_properties> props_ptr) = 0;
    virtual error_t add_node(std::shared_ptr<matmul_properties> props_ptr) = 0;
    virtual error_t add_node(std::shared_ptr<pointwise_properties> props_ptr) = 0;
    
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
    for(auto const& node: graph.conv_properties) {
        os << (node.second) << ",";
    }
    os << "],"
    << "\npointwise: [\n";
    for(auto const& node: graph.pw_properties) {
        os << (node.second) << ",";
    }
    os << "],"
    << "\nmatmul: [\n";
    for(auto const& node: graph.mm_properties) {
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

    CompositeNode node{"composite_node", 1};

public:

    Graph(std::string const &name, cuDNNFEContext const &ctx_) : IGraph(name), ctx(ctx_) {}

    error_t
    is_valid_tensor(std::string const& name) {
        if (all_tensors.find(name) == all_tensors.end()) {
            return error_t::INVALID_TENSOR_NAME;
        } 
        return error_t::OK;
    }

    tensor_properties &
    tensor_at(std::string const& name) {
        return *(all_tensors.at(name));
    }

    // Add a tensor properties object with shared ownership.
    // A shared pointer is taken by value, which makes the graph an owner too.
    error_t add_tensor(std::shared_ptr<tensor_properties> props_ptr) {
        all_tensors.emplace(props_ptr->get_name(), props_ptr);
        return error_t::OK;
    }

    // Returns a shared pointer by value, so the caller is also an owner.
    std::shared_ptr<tensor_properties> get_tensor(std::string const& tensor_name) {
        return all_tensors.at(tensor_name);
    }

    // Add a conv properties object with shared ownership.
    // A shared pointer is taken by value, which makes the graph an owner too.
    error_t add_node(std::shared_ptr<convolution_properties> props_ptr) {
        conv_properties.emplace(props_ptr->get_name(), props_ptr);
        return error_t::OK;
    }
    
    // Add a pointwise properties object with shared ownership.
    // A shared pointer is taken by value, which makes the graph an owner too.
    error_t add_node(std::shared_ptr<pointwise_properties> props_ptr) {
        pw_properties.emplace(props_ptr->get_name(), props_ptr);
        return error_t::OK;
    }
    
    // Add a matmul properties object with shared ownership.
    // A shared pointer is taken by value, which makes the graph an owner too.
    error_t add_node(std::shared_ptr<matmul_properties> props_ptr) {
        mm_properties.emplace(props_ptr->get_name(), props_ptr);
        return error_t::OK;
    }

    error_t
    infer_shapes() {
        std::unordered_map<std::string, bool>   visited_nodes;
        std::unordered_map<std::string, bool>   visited_tensors;
        std::unordered_map<std::string, std::shared_ptr<operation_properties const>> all_nodes;
        std::vector<std::pair<std::string, std::shared_ptr<operation_properties const>>> entrance_nodes;

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


        for (auto &node : conv_properties) {
            visited_nodes[node.first] = false;
            all_nodes[node.first] = node.second;
        }
        for (auto &node : mm_properties) {
            visited_nodes[node.first] = false;
            all_nodes[node.first] = node.second;
        }
        for (auto &node : pw_properties) {
            visited_nodes[node.first] = false;
            all_nodes[node.first] = node.second;
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
                        entrance_nodes.push_back({node.first, node.second});
                    }
                }
            }
        };

        find_entrance_nodes();

        while (entrance_nodes.size()) {
            auto it = entrance_nodes.begin();
            std::shared_ptr<operation_properties const> node = it->second;
            visited_nodes.at(node->get_name()) = true;

            switch (node->get_tag()) {
                case operation_properties::Tag::Pointwise:{
                    auto pw_node = std::static_pointer_cast<pointwise_properties const>(node);
                    auto &tensor = all_tensors.at(pw_node->get_port_name(pointwise_properties::PORTS::Y));
                    auto &input_x_tensor = all_tensors.at(pw_node->get_port_name(pointwise_properties::PORTS::X));
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
                case operation_properties::Tag::Convolution: {
                    auto conv_node = std::static_pointer_cast<convolution_properties const>(node);
                    auto &tensor = all_tensors.at(conv_node->get_port_name(convolution_properties::PORTS::Y));
                    auto &input_x_tensor = all_tensors.at(conv_node->get_port_name(convolution_properties::PORTS::X));
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
                case operation_properties::Tag::Reduction: {
                }
                case operation_properties::Tag::MatMul: {
                    auto mm_node = std::static_pointer_cast<matmul_properties const>(node);
                    auto &tensor = all_tensors.at(mm_node->get_port_name(matmul_properties::PORTS::Y));
                    auto &input_x_tensor = all_tensors.at(mm_node->get_port_name(matmul_properties::PORTS::X));
                    auto &input_w_tensor = all_tensors.at(mm_node->get_port_name(matmul_properties::PORTS::W));
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
            return error_t::SHAPE_DEDUCTION_FAILED;
        }

        return error_t::OK;
    }

    error_t
    build() {
        node.tensor_props = all_tensors;
        
        for (auto const &prop : conv_properties) {
            getLogger() << "Adding the conv node " << prop.first << std::endl;
            auto conv_node = std::make_shared<ConvolutionNode>(prop.first, uid_offset);
            conv_node->props = prop.second;
            conv_node->parent_node = &node;
            node.sub_nodes[prop.first] = conv_node;
            uid_offset += 100;
        }

        for (auto const &prop : mm_properties) {
            getLogger() << "Adding the matmul node " << prop.first << std::endl;
            auto matmul_node = std::make_shared<MatMulNode>(prop.first, uid_offset);
            matmul_node->props = prop.second;
            matmul_node->parent_node = &node;
            node.sub_nodes[prop.first] = matmul_node;
            uid_offset += 100;
        }

        for (auto const &prop : pw_properties) {
            getLogger() << "Adding the pointwise node" << prop.first << std::endl;
            auto pointwise_node = std::make_shared<PointwiseNode>(prop.first, uid_offset);
            pointwise_node->props = prop.second;
            pointwise_node->parent_node = &node;
            node.sub_nodes[prop.first] = pointwise_node;
            uid_offset += 100;
        }

        node.build();

        return error_t::OK;
    }

    error_t execute(std::unordered_map<std::string, void *> var_pack) {
        auto status = node.execute(var_pack);
        return status;
    }

    ~Graph() = default;
};

}
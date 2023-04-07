#pragma once

#include <unordered_map>

#include "cudnn_frontend_context.h"
#include "graphs/cudnn_frontend_node_pointwise.h"

namespace cudnn_frontend {

namespace graph {

class IGraph {
protected:
    std::string name;
    std::unordered_map<std::string, std::shared_ptr<graph::Tensor>> all_tensors;
    std::unordered_map<std::string, std::shared_ptr<convolution>>  conv;
    std::unordered_map<std::string, std::shared_ptr<matmul>>  mm;
    std::unordered_map<std::string, std::shared_ptr<pointwise>>    pw;
    
public:
    IGraph(std::string const &name ) : name(name) {}
    
    std::string const
    get_name() const {
        return name;
    }

    virtual error_t insert_tensor(std::shared_ptr<graph::Tensor> props_ptr) = 0;

    virtual error_t insert_node(std::shared_ptr<convolution> props_ptr) = 0;
    virtual error_t insert_node(std::shared_ptr<matmul> props_ptr) = 0;
    virtual error_t insert_node(std::shared_ptr<pointwise> props_ptr) = 0;
    
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
    for(auto const& node: graph.conv) {
        os << (node.second) << ",";
    }
    os << "],"
    << "\npointwise: [\n";
    for(auto const& node: graph.pw) {
        os << (node.second) << ",";
    }
    os << "],"
    << "\nmatmul: [\n";
    for(auto const& node: graph.mm) {
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

    CompositeNode flat_node{"composite_node", 1};

public:

    Graph(std::string const &name, cuDNNFEContext const &ctx_) : IGraph(name), ctx(ctx_) {}

    error_t
    is_valid_tensor(std::string const& name) {
        if (all_tensors.find(name) == all_tensors.end()) {
            return error_t::INVALID_TENSOR_NAME;
        } 
        return error_t::OK;
    }

    graph::Tensor &
    tensor_at(std::string const& name) {
        return *(all_tensors.at(name));
    }

    // Add a tensor properties object with shared ownership.
    // A shared pointer is taken by value, which makes the graph an owner too.
    error_t insert_tensor(std::shared_ptr<graph::Tensor> props_ptr) {
        all_tensors.emplace(props_ptr->get_name(), props_ptr);
        return error_t::OK;
    }

    // Returns a shared pointer by value, so the caller is also an owner.
    std::shared_ptr<graph::Tensor> get_tensor(std::string const& tensor_name) {
        return all_tensors.at(tensor_name);
    }

    // Add a conv properties object with shared ownership.
    // A shared pointer is taken by value, which makes the graph an owner too.
    error_t insert_node(std::shared_ptr<convolution> props_ptr) {
        conv.emplace(props_ptr->get_name(), props_ptr);
        return error_t::OK;
    }
    
    // Add a pointwise properties object with shared ownership.
    // A shared pointer is taken by value, which makes the graph an owner too.
    error_t insert_node(std::shared_ptr<pointwise> props_ptr) {
        pw.emplace(props_ptr->get_name(), props_ptr);
        return error_t::OK;
    }
    
    // Add a matmul properties object with shared ownership.
    // A shared pointer is taken by value, which makes the graph an owner too.
    error_t insert_node(std::shared_ptr<matmul> props_ptr) {
        mm.emplace(props_ptr->get_name(), props_ptr);
        return error_t::OK;
    }

    error_t infer_shapes() noexcept {
        std::unordered_map<std::string, std::shared_ptr<operation const>> all_nodes;

        // Make map of tensors to/from nodes
        std::unordered_map<std::string, std::vector<std::string>> incoming_tensors_for_nodes;
        std::unordered_map<std::string, std::vector<std::string>> outgoing_tensors_for_nodes;
        std::unordered_map<std::string, std::vector<std::string>> incoming_nodes_for_tensors;
        std::unordered_map<std::string, std::vector<std::string>> outgoing_nodes_for_tensors;
        for (auto &node : conv) {
            all_nodes[node.first] = node.second;
            auto &y_tensor = all_tensors.at(node.second->get_tensor_at_port(convolution::PORTS::Y));
            auto &x_tensor = all_tensors.at(node.second->get_tensor_at_port(convolution::PORTS::X));
            auto &w_tensor = all_tensors.at(node.second->get_tensor_at_port(convolution::PORTS::W));
        
            outgoing_nodes_for_tensors[x_tensor->get_name()].push_back(node.second->get_name());
            outgoing_nodes_for_tensors[w_tensor->get_name()].push_back(node.second->get_name());
            incoming_nodes_for_tensors[y_tensor->get_name()].push_back(node.second->get_name());
            
            incoming_tensors_for_nodes[node.second->get_name()].push_back(x_tensor->get_name());
            incoming_tensors_for_nodes[node.second->get_name()].push_back(w_tensor->get_name());
            outgoing_tensors_for_nodes[node.second->get_name()].push_back(y_tensor->get_name());
        }
        for (auto &node : mm) {
            all_nodes[node.first] = node.second;
            auto &y_tensor = all_tensors.at(node.second->get_tensor_at_port(matmul::PORTS::Y));
            auto &x_tensor = all_tensors.at(node.second->get_tensor_at_port(matmul::PORTS::X));
            auto &w_tensor = all_tensors.at(node.second->get_tensor_at_port(matmul::PORTS::W));
        
            outgoing_nodes_for_tensors[x_tensor->get_name()].push_back(node.second->get_name());
            outgoing_nodes_for_tensors[w_tensor->get_name()].push_back(node.second->get_name());
            incoming_nodes_for_tensors[y_tensor->get_name()].push_back(node.second->get_name());
            
            incoming_tensors_for_nodes[node.second->get_name()].push_back(x_tensor->get_name());
            incoming_tensors_for_nodes[node.second->get_name()].push_back(w_tensor->get_name());
            outgoing_tensors_for_nodes[node.second->get_name()].push_back(y_tensor->get_name());
        }
        for (auto &node : pw) {
            all_nodes[node.first] = node.second;
            
            auto const port_count = get_pointwise_mode_port_count(node.second->get_mode());
            if(port_count == 3) {
                auto &y_tensor = all_tensors.at(node.second->get_tensor_at_port(pointwise::PORTS::Y));
                auto &x_tensor = all_tensors.at(node.second->get_tensor_at_port(pointwise::PORTS::X));
                auto &b_tensor = all_tensors.at(node.second->get_tensor_at_port(pointwise::PORTS::B));
            
                outgoing_nodes_for_tensors[x_tensor->get_name()].push_back(node.second->get_name());
                outgoing_nodes_for_tensors[b_tensor->get_name()].push_back(node.second->get_name());
                incoming_nodes_for_tensors[y_tensor->get_name()].push_back(node.second->get_name());
            
                incoming_tensors_for_nodes[node.second->get_name()].push_back(x_tensor->get_name());
                incoming_tensors_for_nodes[node.second->get_name()].push_back(b_tensor->get_name());
                outgoing_tensors_for_nodes[node.second->get_name()].push_back(y_tensor->get_name());
            }
            else if(port_count == 2) {
                auto &y_tensor = all_tensors.at(node.second->get_tensor_at_port(pointwise::PORTS::Y));
                auto &x_tensor = all_tensors.at(node.second->get_tensor_at_port(pointwise::PORTS::X));
            
                outgoing_nodes_for_tensors[x_tensor->get_name()].push_back(node.second->get_name());
                incoming_nodes_for_tensors[y_tensor->get_name()].push_back(node.second->get_name());
                
                incoming_tensors_for_nodes[node.second->get_name()].push_back(x_tensor->get_name());
                outgoing_tensors_for_nodes[node.second->get_name()].push_back(y_tensor->get_name());
            }            
        }

        // https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
        std::vector<std::string> sorted_nodes;
        std::unordered_map<std::string, int32_t> incoming_edges_count;
        std::vector<std::string> source_nodes;
        for(auto const& itr: incoming_tensors_for_nodes) {
            incoming_edges_count[itr.first] = 0;
            for(auto const& incoming_tensors: itr.second) {
                incoming_edges_count[itr.first] += (incoming_nodes_for_tensors[incoming_tensors].size());
            }
            if(incoming_edges_count[itr.first] == 0) {
                source_nodes.push_back(itr.first);
            }
        }

        while(source_nodes.size()) {
            std::string const node = source_nodes.back(); 
            sorted_nodes.push_back(node);
            source_nodes.pop_back();
            
            for(std::string const& tensor: outgoing_tensors_for_nodes[node]) {
                for(std::string const& next_node: outgoing_nodes_for_tensors[tensor]) {
                    --incoming_edges_count[next_node];
                    if(incoming_edges_count[next_node] == 0) {
                        source_nodes.push_back(next_node);
                    }
                }
            }
        }
        
        getLogger() << "[cudnn_frontend] INFO: Sorted Nodes: ";
        for(std::string const& node: sorted_nodes) {
            getLogger() << node << " ";
        }
        getLogger() << std::endl;

        for(std::string const& node: sorted_nodes) {
            std::shared_ptr<operation const> node_props = all_nodes[node];
            
            switch (node_props->get_tag()) {
                case operation::Tag::Pointwise:{
                    auto pw_node = std::static_pointer_cast<pointwise const>(node_props);
                    flat_node.sub_nodes[node]->infer_properties();
                    break;
                }
                case operation::Tag::Convolution: {
                    auto conv_node = std::static_pointer_cast<convolution const>(node_props);
                    flat_node.sub_nodes[node]->infer_properties();
                    break;
                }
                case operation::Tag::BatchNorm: {
                }
                break;
                case operation::Tag::Reduction: {
                }
                break;
                case operation::Tag::MatMul: {
                    auto mm_node = std::static_pointer_cast<matmul const>(node_props);
                    flat_node.sub_nodes[node]->infer_properties();
                    break;
                }
            }
        }

        return error_t::OK;
    }

    error_t build() {
        auto status = error_t::OK;

        flat_node.tensor_props = all_tensors;
        
        for (auto const &prop : conv) {
            getLogger() << "[cudnn_frontend] INFO: Adding the conv node named " << prop.first << std::endl;
            auto conv_node = std::make_shared<ConvolutionNode>(prop.first, uid_offset);
            conv_node->props = prop.second;
            conv_node->parent_node = &flat_node;
            flat_node.sub_nodes[prop.first] = conv_node;
            uid_offset += 100;
        }

        for (auto const &prop : mm) {
            getLogger() << "[cudnn_frontend] INFO: Adding the matmul node named " << prop.first << std::endl;
            auto matmul_node = std::make_shared<MatMulNode>(prop.first, uid_offset);
            matmul_node->props = prop.second;
            matmul_node->parent_node = &flat_node;
            flat_node.sub_nodes[prop.first] = matmul_node;
            uid_offset += 100;
        }

        for (auto const &prop : pw) {
            getLogger() << "[cudnn_frontend] INFO: Adding the pointwise node named " << prop.first << std::endl;
            auto pointwise_node = std::make_shared<PointwiseNode>(prop.first, uid_offset);
            pointwise_node->props = prop.second;
            pointwise_node->parent_node = &flat_node;
            flat_node.sub_nodes[prop.first] = pointwise_node;
            uid_offset += 100;
        }

        status = infer_shapes();
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to build in " << name << std::endl;
            return status;
        }

        status = flat_node.build();
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to build in " << name << std::endl;
            return status;
        }

        return status;
    }

    error_t execute(std::unordered_map<std::string, void *> var_pack) {
        auto status = flat_node.execute(var_pack);
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Execution failed in " << name << std::endl;
            return status;
        }

        return status;
    }

    ~Graph() = default;
};

} // namespace graph

} // namespace cudnn_frontend
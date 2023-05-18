#pragma once

#include <unordered_map>

#include "graphs/cudnn_frontend_node_batchnorm.h"
#include "graphs/cudnn_frontend_node_convolution.h"
#include "graphs/cudnn_frontend_node_matmul.h"
#include "graphs/cudnn_frontend_node_pointwise.h"

namespace cudnn_frontend {

namespace graph {

class Graph {
private:
    std::string name;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> all_tensors;
    std::unordered_map<std::string, std::shared_ptr<Convolution>>  conv;
    std::unordered_map<std::string, std::shared_ptr<Matmul>>  mm;
    std::unordered_map<std::string, std::shared_ptr<Batchnorm>>  bn;
    std::unordered_map<std::string, std::shared_ptr<Pointwise>>    pw;


    int64_t tensor_dims  = 4;
    int64_t spatial_dims = 2;
    DataType_t compute_type           = DataType_t::FLOAT;
    DataType_t intermediate_data_type = DataType_t::FLOAT;
    DataType_t io_data_type           = DataType_t::HALF;

    int64_t uid_offset = 1;

    CompositeNode flat_node{"composite_node", 1};

    error_t infer_shapes();

public:
    Graph(std::string name);
    
    std::string const &
    get_name() const {
        return name;
    }

    Graph& set_intermediate_data_type(DataType_t type);
    Graph& set_io_data_type(DataType_t type);
    Graph& set_compute_type(DataType_t type);
    Graph& set_tensor_dims(int64_t x);
    Graph& set_spatial_dims(int64_t x);

    Graph& insert_tensor(Tensor const& props);
    std::shared_ptr<Tensor> get_tensor(std::string const& tensor_name);

    Graph& insert_node(Batchnorm const& props);
    Graph& insert_node(Convolution const& props);
    Graph& insert_node(Matmul const& props);
    Graph& insert_node(Pointwise const& props);
    
    error_t build();
    
    int64_t get_workspace_size();
    error_t execute(std::unordered_map<std::string, void *>);

    friend std::ostream& operator<<(std::ostream& os, const Graph& props);
    
    ~Graph() = default;
};

inline std::ostream& operator<<(std::ostream& os, const Graph& graph) {
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
    os << "],"
    << "\nbatchnorm: [\n";
    for(auto const& node: graph.bn) {
        os << (node.second) << ",";
    }
    os << "],";
    os << "}";
    return os;
}

inline Graph::Graph(std::string name) : name(name) {}

inline Graph & Graph::set_intermediate_data_type(DataType_t type) {
    intermediate_data_type = type;
    return *this;
}

inline Graph & Graph::set_io_data_type(DataType_t type) {
    io_data_type = type;
    return *this;
}

inline Graph & Graph::set_compute_type(DataType_t type) {
    compute_type = type;
    return *this;
}

inline Graph & Graph::set_tensor_dims(int64_t x) {
    tensor_dims = x;
    return *this;
}

inline Graph & Graph::set_spatial_dims(int64_t x) {
    spatial_dims = x;
    return *this;
}

inline Graph& Graph::insert_tensor(Tensor const& props) {
    all_tensors.emplace(props.get_name(), std::make_shared<Tensor>(props));
    return *this;
}

inline Graph& Graph::insert_node(Convolution const& props) {
    conv.emplace(props.get_name(), std::make_shared<Convolution>(props));
    return *this;
}

inline Graph& Graph::insert_node(Pointwise const &props) {
    pw.emplace(props.get_name(), std::make_shared<Pointwise>(props));
    return *this;
}

inline Graph& Graph::insert_node(Matmul const& props) {
    mm.emplace(props.get_name(), std::make_shared<Matmul>(props));
    return *this;
}

inline Graph& Graph::insert_node(Batchnorm const& props) {
    bn.emplace(props.get_name(), std::make_shared<Batchnorm>(props));
    return *this;
}

inline std::shared_ptr<Tensor> Graph::get_tensor(std::string const& tensor_name) {
    return all_tensors[tensor_name];
}

inline error_t Graph::infer_shapes() {
    std::unordered_map<std::string, std::shared_ptr<Operation const>> all_nodes;

    // Make map of tensors to/from nodes
    std::unordered_map<std::string, std::vector<std::string>> incoming_tensors_for_nodes;
    std::unordered_map<std::string, std::vector<std::string>> outgoing_tensors_for_nodes;
    std::unordered_map<std::string, std::vector<std::string>> incoming_nodes_for_tensors;
    std::unordered_map<std::string, std::vector<std::string>> outgoing_nodes_for_tensors;
    for (auto &node : conv) {
        all_nodes[node.first] = node.second;
        auto &y_tensor = all_tensors.at(node.second->get_tensor_at_port(Convolution::PORTS::Y));
        auto &x_tensor = all_tensors.at(node.second->get_tensor_at_port(Convolution::PORTS::X));
        auto &w_tensor = all_tensors.at(node.second->get_tensor_at_port(Convolution::PORTS::W));
    
        outgoing_nodes_for_tensors[x_tensor->get_name()].push_back(node.second->get_name());
        outgoing_nodes_for_tensors[w_tensor->get_name()].push_back(node.second->get_name());
        incoming_nodes_for_tensors[y_tensor->get_name()].push_back(node.second->get_name());
        
        incoming_tensors_for_nodes[node.second->get_name()].push_back(x_tensor->get_name());
        incoming_tensors_for_nodes[node.second->get_name()].push_back(w_tensor->get_name());
        outgoing_tensors_for_nodes[node.second->get_name()].push_back(y_tensor->get_name());
    }
    for (auto &node : mm) {
        all_nodes[node.first] = node.second;
        auto &y_tensor = all_tensors.at(node.second->get_tensor_at_port(Matmul::PORTS::Y));
        auto &x_tensor = all_tensors.at(node.second->get_tensor_at_port(Matmul::PORTS::X));
        auto &w_tensor = all_tensors.at(node.second->get_tensor_at_port(Matmul::PORTS::W));
    
        outgoing_nodes_for_tensors[x_tensor->get_name()].push_back(node.second->get_name());
        outgoing_nodes_for_tensors[w_tensor->get_name()].push_back(node.second->get_name());
        incoming_nodes_for_tensors[y_tensor->get_name()].push_back(node.second->get_name());
        
        incoming_tensors_for_nodes[node.second->get_name()].push_back(x_tensor->get_name());
        incoming_tensors_for_nodes[node.second->get_name()].push_back(w_tensor->get_name());
        outgoing_tensors_for_nodes[node.second->get_name()].push_back(y_tensor->get_name());
    }
    for (auto &node : bn) {
        all_nodes[node.first] = node.second;
        auto &y_tensor = all_tensors.at(node.second->get_tensor_at_port(Batchnorm::PORTS::Y));
        auto &x_tensor = all_tensors.at(node.second->get_tensor_at_port(Batchnorm::PORTS::X));
        auto &mean_tensor = all_tensors.at(node.second->get_tensor_at_port(Batchnorm::PORTS::Mean));
        auto &var_tensor = all_tensors.at(node.second->get_tensor_at_port(Batchnorm::PORTS::Var));
        auto &prev_running_mean_tensor = all_tensors.at(node.second->get_tensor_at_port(Batchnorm::PORTS::Previous_running_mean));
        auto &prev_running_var_tensor = all_tensors.at(node.second->get_tensor_at_port(Batchnorm::PORTS::Previous_running_var));
        auto &next_running_mean_tensor = all_tensors.at(node.second->get_tensor_at_port(Batchnorm::PORTS::Next_running_mean));
        auto &next_running_var_tensor = all_tensors.at(node.second->get_tensor_at_port(Batchnorm::PORTS::Next_running_var));
        auto &scale_tensor = all_tensors.at(node.second->get_tensor_at_port(Batchnorm::PORTS::Scale));
        auto &bias_tensor = all_tensors.at(node.second->get_tensor_at_port(Batchnorm::PORTS::Bias));
        auto &eps_tensor = all_tensors.at(node.second->get_tensor_at_port(Batchnorm::PORTS::EPS));
        auto &exp_avg_tensor = all_tensors.at(node.second->get_tensor_at_port(Batchnorm::PORTS::EXP_AVG));
    
        outgoing_nodes_for_tensors[x_tensor->get_name()].push_back(node.second->get_name());
        outgoing_nodes_for_tensors[prev_running_mean_tensor->get_name()].push_back(node.second->get_name());
        outgoing_nodes_for_tensors[prev_running_var_tensor->get_name()].push_back(node.second->get_name());
        outgoing_nodes_for_tensors[eps_tensor->get_name()].push_back(node.second->get_name());
        outgoing_nodes_for_tensors[exp_avg_tensor->get_name()].push_back(node.second->get_name());
        outgoing_nodes_for_tensors[scale_tensor->get_name()].push_back(node.second->get_name());
        outgoing_nodes_for_tensors[bias_tensor->get_name()].push_back(node.second->get_name());
        incoming_nodes_for_tensors[y_tensor->get_name()].push_back(node.second->get_name());
        incoming_nodes_for_tensors[mean_tensor->get_name()].push_back(node.second->get_name());
        incoming_nodes_for_tensors[var_tensor->get_name()].push_back(node.second->get_name());
        incoming_nodes_for_tensors[next_running_mean_tensor->get_name()].push_back(node.second->get_name());
        incoming_nodes_for_tensors[next_running_var_tensor->get_name()].push_back(node.second->get_name());
        
        incoming_tensors_for_nodes[node.second->get_name()].push_back(x_tensor->get_name());
        incoming_tensors_for_nodes[node.second->get_name()].push_back(prev_running_mean_tensor->get_name());
        incoming_tensors_for_nodes[node.second->get_name()].push_back(prev_running_var_tensor->get_name());
        incoming_tensors_for_nodes[node.second->get_name()].push_back(eps_tensor->get_name());
        incoming_tensors_for_nodes[node.second->get_name()].push_back(exp_avg_tensor->get_name());
        incoming_tensors_for_nodes[node.second->get_name()].push_back(scale_tensor->get_name());
        incoming_tensors_for_nodes[node.second->get_name()].push_back(bias_tensor->get_name());
        outgoing_tensors_for_nodes[node.second->get_name()].push_back(y_tensor->get_name());
        outgoing_tensors_for_nodes[node.second->get_name()].push_back(mean_tensor->get_name());
        outgoing_tensors_for_nodes[node.second->get_name()].push_back(var_tensor->get_name());
        outgoing_tensors_for_nodes[node.second->get_name()].push_back(next_running_mean_tensor->get_name());
        outgoing_tensors_for_nodes[node.second->get_name()].push_back(next_running_var_tensor->get_name());
    }
    for (auto &node : pw) {
        all_nodes[node.first] = node.second;
        
        auto const port_count = get_pointwise_mode_port_count(node.second->get_mode());
        if(port_count == 3) {
            auto &y_tensor = all_tensors.at(node.second->get_tensor_at_port(Pointwise::PORTS::Y));
            auto &x_tensor = all_tensors.at(node.second->get_tensor_at_port(Pointwise::PORTS::X));
            auto &b_tensor = all_tensors.at(node.second->get_tensor_at_port(Pointwise::PORTS::B));
        
            outgoing_nodes_for_tensors[x_tensor->get_name()].push_back(node.second->get_name());
            outgoing_nodes_for_tensors[b_tensor->get_name()].push_back(node.second->get_name());
            incoming_nodes_for_tensors[y_tensor->get_name()].push_back(node.second->get_name());
        
            incoming_tensors_for_nodes[node.second->get_name()].push_back(x_tensor->get_name());
            incoming_tensors_for_nodes[node.second->get_name()].push_back(b_tensor->get_name());
            outgoing_tensors_for_nodes[node.second->get_name()].push_back(y_tensor->get_name());
        }
        else if(port_count == 2) {
            auto &y_tensor = all_tensors.at(node.second->get_tensor_at_port(Pointwise::PORTS::Y));
            auto &x_tensor = all_tensors.at(node.second->get_tensor_at_port(Pointwise::PORTS::X));
        
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
        std::shared_ptr<Operation const> node_props = all_nodes[node];
        
        switch (node_props->get_tag()) {
            case Operation::Tag::Pointwise:{
                auto pw_node = std::static_pointer_cast<Pointwise const>(node_props);
                flat_node.sub_nodes[node]->infer_properties();
                break;
            }
            case Operation::Tag::Convolution: {
                auto conv_node = std::static_pointer_cast<Convolution const>(node_props);
                flat_node.sub_nodes[node]->infer_properties();
                break;
            }
            case Operation::Tag::BatchNorm: {
                auto bn_node = std::static_pointer_cast<Batchnorm const>(node_props);
                flat_node.sub_nodes[node]->infer_properties();
                break;
            }
            break;
            case Operation::Tag::Reduction: {
            }
            break;
            case Operation::Tag::MatMul: {
                auto mm_node = std::static_pointer_cast<Matmul const>(node_props);
                flat_node.sub_nodes[node]->infer_properties();
                break;
            }
        }
    }

    return error_t::OK;
}

inline int64_t Graph::get_workspace_size() {
    return flat_node.get_workspace_size();
}

inline error_t Graph::build() {
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

    for (auto const &prop : bn) {
        getLogger() << "[cudnn_frontend] INFO: Adding the batch norm node named " << prop.first << std::endl;
        auto batchnorm_node = std::make_shared<BatchNormNode>(prop.first, uid_offset);
        batchnorm_node->props = prop.second;
        batchnorm_node->parent_node = &flat_node;
        flat_node.sub_nodes[prop.first] = batchnorm_node;
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

inline error_t Graph::execute(std::unordered_map<std::string, void *> var_pack) {
    auto status = flat_node.execute(var_pack);
    if(status != error_t::OK) {
        getLogger() << "[cudnn_frontend] ERROR: " << status << " Execution failed in " << name << std::endl;
        return status;
    }

    return status;
}

} // namespace graph

} // namespace cudnn_frontend
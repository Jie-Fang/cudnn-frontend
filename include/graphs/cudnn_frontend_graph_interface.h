#pragma once

#include <unordered_map>

#include "graphs/cudnn_frontend_node_batchnorm.h"
#include "graphs/cudnn_frontend_node_convolution.h"
#include "graphs/cudnn_frontend_node_matmul.h"
#include "graphs/cudnn_frontend_node_pointwise.h"

#include <graphs/cudnn_frontend_graph_helpers.h>

namespace cudnn_frontend {

namespace graph {

class Plans {
    friend class Graph;
    Execution_plan_list list_of_engine_configs;
    public:
        Plans &filter_by_numeric_notes(std::vector<cudnnBackendNumericalNote_t> const &);
        Plans &filter_by_behavior_notes(std::vector<cudnnBackendBehaviorNote_t> const &);
        Plans &build_plans(cudnnHandle_t);

        int64_t get_workspace_size();
        int64_t get_max_workspace_size();
};

inline Plans& Plans::filter_by_behavior_notes(std::vector<cudnnBackendBehaviorNote_t> const &notes) {
    // TODO: The error returned is not propagate to user.
    // Should the return value be changed to error_t too?
    auto status = list_of_engine_configs.filter_by_behavior_notes(notes);
    if(status != error_t::OK) {
        getLogger() << "[cudnn_frontend] ERROR: Filtering by behavioural notes failed." << std::endl; 
    }
    return *this;
}

inline Plans& Plans::filter_by_numeric_notes(std::vector<cudnnBackendNumericalNote_t> const &notes) {
    // TODO: The error returned is not propagate to user.
    // Should the return value be changed to error_t too?
    auto status = list_of_engine_configs.filter_by_numeric_notes(notes);
    if(status != error_t::OK) {
        getLogger() << "[cudnn_frontend] ERROR: Filtering by numerical notes failed." << std::endl; 
    }
    return *this;
}

inline Plans& Plans::build_plans(cudnnHandle_t h){
    // TODO: The error returned is not propagate to user.
    // Should the return value be changed to error_t too?
    auto status = list_of_engine_configs.build_plans(h);
    if(status != error_t::OK) {
        getLogger() << "[cudnn_frontend] ERROR: Plan building failed." << std::endl; 
    }
    return *this;
}


inline int64_t Plans::get_max_workspace_size(){
    return list_of_engine_configs.get_max_workspace_size();
}


inline int64_t Plans::get_workspace_size(){
    return list_of_engine_configs.get_workspace_size();
}

class Graph {
private:
    std::string name;
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors;
    std::unordered_map<std::string, std::shared_ptr<Operation>> nodes;

    int64_t uid_offset = 1;

    FlatNode flat_node{"flat_node", 1};
    detail::Context& get_context();

    error_t infer_properties();

    Graph& insert_tensor_(std::shared_ptr<Tensor> tensor_ptr);
    Graph& insert_node_(std::shared_ptr<Operation> node_ptr);

public:
    Graph(std::string name);
    
    std::string const &
    get_name() const {
        return name;
    }

    Graph& set_intermediate_data_type(DataType_t type);
    Graph& set_io_data_type(DataType_t type);
    Graph& set_compute_data_type(DataType_t type);
    
    Graph& insert_tensor(Tensor const& props);
    std::shared_ptr<Tensor> get_tensor(std::string const& tensor_name) const;

    Graph& insert_node(Operation const& props);

    Graph& insert_graph(Graph& other_graph, std::unordered_map<std::string, std::string> const& connections);
    
    error_t build(cudnnHandle_t handle);
    
    Plans
    get_execution_plan_list(HeurMode_t mode);

    error_t set_executor(Plans const & plan) {
        if (plan.list_of_engine_configs.get_candidate() == nullptr) {
            return error_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED;
        }
        CHECK_CUDNN_FRONTEND_ERROR(flat_node.set_executor(plan.list_of_engine_configs));
        return error_t::OK;
    }

    int64_t get_workspace_size();
    error_t execute(cudnnHandle_t handle, std::unordered_map<std::string, void *>);

    friend std::ostream& operator<<(std::ostream& os, const Graph& props);
    
    ~Graph() = default;
};

inline std::ostream& operator<<(std::ostream& os, const Graph& graph) {
    os << "{tensors: [\n";
    for(auto const& tensor: graph.tensors) {
        os << *(tensor.second) << ",";
    }
    os << "],"
    << "\nnodes: [\n";
    for(auto const& node: graph.nodes) {
        os << (node.second) << ",";
    }
    os << "]}";
    return os;
}

inline Graph::Graph(std::string name) : name(name) {}

inline Plans Graph::get_execution_plan_list(HeurMode_t mode) {
    Plans plan_list;
    // TODO: The error returned is not propagate to user.
    // Should the return value be changed to error_t too?
    auto status = flat_node.get_engine_configs(mode, plan_list.list_of_engine_configs);
    if(status != error_t::OK) {
        getLogger() << "[cudnn_frontend] ERROR: Querying engine configs failed." << std::endl; 
    }
    return plan_list;
}

inline Graph& Graph::set_intermediate_data_type(DataType_t const type) {
    get_context().set_intermediate_data_type(type);
    return *this;
}

inline Graph& Graph::set_io_data_type(DataType_t const type) {
    get_context().set_io_data_type(type);
    return *this;
}

inline Graph& Graph::set_compute_data_type(DataType_t const type) {
    get_context().set_compute_data_type(type);
    return *this;
}

inline detail::Context& Graph::get_context() {
    return flat_node.get_context();
}

inline Graph& Graph::insert_tensor_(std::shared_ptr<Tensor> tensor_ptr) {
    tensors.emplace(tensor_ptr->get_name(), tensor_ptr);
    return *this;
}

inline Graph& Graph::insert_tensor(Tensor const& props) {
    insert_tensor_(std::make_shared<Tensor>(props));
    return *this;
}

inline Graph& Graph::insert_node_(std::shared_ptr<Operation> node_ptr) {    
    switch (node_ptr->get_tag()) {
        case Operation::Tag::Batchnorm:{
            nodes.emplace(node_ptr->get_name(), std::static_pointer_cast<Batchnorm>(node_ptr));
            break;
        }
        case Operation::Tag::Convolution:{
            nodes.emplace(node_ptr->get_name(), std::static_pointer_cast<Convolution>(node_ptr));
            break;
        }
        case Operation::Tag::Matmul:{
            nodes.emplace(node_ptr->get_name(), std::static_pointer_cast<Matmul>(node_ptr));
            break;
        }
        case Operation::Tag::Pointwise:{
            nodes.emplace(node_ptr->get_name(), std::static_pointer_cast<Pointwise>(node_ptr));
            break;
        }
        case Operation::Tag::Reduction:{
            nodes.emplace(node_ptr->get_name(), std::static_pointer_cast<Reduction>(node_ptr));
            break;
        }
    }

    return *this;
}

inline Graph& Graph::insert_node(Operation const& props) {    
    switch (props.get_tag()) {
        case Operation::Tag::Batchnorm:{
            insert_node_(std::static_pointer_cast<Operation>(std::make_shared<Batchnorm>((Batchnorm&)props)));
            break;
        }
        case Operation::Tag::Convolution:{
            insert_node_(std::static_pointer_cast<Operation>(std::make_shared<Convolution>((Convolution&)props)));
            break;
        }
        case Operation::Tag::Matmul:{
            insert_node_(std::static_pointer_cast<Operation>(std::make_shared<Matmul>((Matmul&)props)));
            break;
        }
        case Operation::Tag::Pointwise:{
            insert_node_(std::static_pointer_cast<Operation>(std::make_shared<Pointwise>((Pointwise&)props)));
            break;
        }
        case Operation::Tag::Reduction:{
            insert_node_(std::static_pointer_cast<Operation>(std::make_shared<Reduction>((Reduction&)props)));
            break;
        }
    }

    return *this;
}

inline std::shared_ptr<Tensor> Graph::get_tensor(std::string const& tensor_name) const {
    return tensors.at(tensor_name);
}

inline Graph& Graph::insert_graph(Graph& other_graph, std::unordered_map<std::string, std::string> const& connections) {

    // Add tensors to this graph.
    for(auto itr: other_graph.tensors) {
        if (auto search = tensors.find(itr.first); search != tensors.end()) {
            getLogger() << "[cudnn_frontend] ERROR: " << itr.first << " exists in both the graphs." << std::endl;
        }
        // Apply other_graph's properties before inserting into this graph.
        itr.second->fill_from_context(other_graph.get_context());
        insert_tensor_(itr.second);
    }

    // Look at connections and connect their nodes
    for(auto const& connection: connections) {
        auto tensor_A = get_tensor(connection.first);
        auto tensor_B = other_graph.get_tensor(connection.second);
        
        // TODO: Check all specified properties of the two connected tensors are same.
        tensor_A->fill_from_context(get_context());
        if(tensor_A->get_data_type() != cudnn_frontend::DataType_t::NOT_SET && tensor_B->get_data_type() != cudnn_frontend::DataType_t::NOT_SET) {
            if(tensor_A->get_data_type() != tensor_B->get_data_type()) {
                getLogger() << "[cudnn_frontend] ERROR: Connection between " << connection.first << " " << connection.first << " have different data types." << std::endl;
            }
        }
        if(tensor_A->get_is_virtual() != tensor_B->get_is_virtual()) {
            getLogger() << "[cudnn_frontend] ERROR: Connection between " << connection.first << " " << connection.first << " have different virtualness." << std::endl;
        }

        // Iterate over nodes and change port name to match connection
        for(auto& node: nodes) {
            switch (node.second->get_tag()) {
                case Operation::Tag::Batchnorm:{
                    for(auto& itr: std::static_pointer_cast<Batchnorm>(node.second)->port_to_name) {
                        if(itr.second == connection.first) {
                            itr.second = connection.second;
                        }
                    }
                    break;
                }
                case Operation::Tag::Convolution:{
                    for(auto& itr: std::static_pointer_cast<Convolution>(node.second)->port_to_name) {
                        if(itr.second == connection.first) {
                            itr.second = connection.second;
                        }
                    }
                    break;
                }
                case Operation::Tag::Matmul:{
                    for(auto& itr: std::static_pointer_cast<Matmul>(node.second)->port_to_name) {
                        if(itr.second == connection.first) {
                            itr.second = connection.second;
                        }
                    }
                    break;
                }
                case Operation::Tag::Pointwise:{
                    for(auto& itr: std::static_pointer_cast<Pointwise>(node.second)->port_to_name) {
                        if(itr.second == connection.first) {
                            itr.second = connection.second;
                        }
                    }
                    break;
                }
                case Operation::Tag::Reduction:{
                    for(auto& itr: std::static_pointer_cast<Reduction>(node.second)->port_to_name) {
                        if(itr.second == connection.first) {
                            itr.second = connection.second;
                        }
                    }
                    break;
                }
            }
        }
    }

    for(auto itr: other_graph.nodes) {
        // Apply other_graph's properties before inserting into this graph.
        getLogger() << "[cudnn_frontend] INFO: Adding " << itr.first << " node to larger graph." << std::endl;
        switch (itr.second->get_tag()) {
            case Operation::Tag::Batchnorm: {
                std::static_pointer_cast<Batchnorm>(itr.second)->fill_from_context(other_graph.get_context());
                break;
            }
            case Operation::Tag::Convolution: {
                std::static_pointer_cast<Convolution>(itr.second)->fill_from_context(other_graph.get_context());
                break;
            }
            case Operation::Tag::Matmul: {
                std::static_pointer_cast<Matmul>(itr.second)->fill_from_context(other_graph.get_context());
                break;
            }
            case Operation::Tag::Pointwise: {
                std::static_pointer_cast<Pointwise>(itr.second)->fill_from_context(other_graph.get_context());
                break;
            }
            case Operation::Tag::Reduction: {
                std::static_pointer_cast<Reduction>(itr.second)->fill_from_context(other_graph.get_context());
                break;
            }
        }
        insert_node_(itr.second);
    }

    return *this;
}

inline error_t Graph::infer_properties() {
    // Make map of tensors to/from nodes
    std::unordered_map<std::string, std::vector<std::string>> incoming_tensors_for_nodes;
    std::unordered_map<std::string, std::vector<std::string>> outgoing_tensors_for_nodes;
    std::unordered_map<std::string, std::vector<std::string>> incoming_nodes_for_tensors;
    std::unordered_map<std::string, std::vector<std::string>> outgoing_nodes_for_tensors;
    for (auto &node : nodes) {
        switch (node.second->get_tag()) {
            case Operation::Tag::Convolution: {
                auto conv_node = std::static_pointer_cast<Convolution>(node.second);
                auto &y_tensor = tensors.at(conv_node->get_tensor_at_port(Convolution::PORTS::Y));
                auto &x_tensor = tensors.at(conv_node->get_tensor_at_port(Convolution::PORTS::X));
                auto &w_tensor = tensors.at(conv_node->get_tensor_at_port(Convolution::PORTS::W));
            
                outgoing_nodes_for_tensors[x_tensor->get_name()].push_back(conv_node->get_name());
                outgoing_nodes_for_tensors[w_tensor->get_name()].push_back(conv_node->get_name());
                incoming_nodes_for_tensors[y_tensor->get_name()].push_back(conv_node->get_name());
                
                incoming_tensors_for_nodes[conv_node->get_name()].push_back(x_tensor->get_name());
                incoming_tensors_for_nodes[conv_node->get_name()].push_back(w_tensor->get_name());
                outgoing_tensors_for_nodes[conv_node->get_name()].push_back(y_tensor->get_name());
                break;
            }
            case Operation::Tag::Matmul: {
                auto matmul_node = std::static_pointer_cast<Matmul>(node.second);       
                auto &y_tensor = tensors.at(matmul_node->get_tensor_at_port(Matmul::PORTS::C));
                auto &x_tensor = tensors.at(matmul_node->get_tensor_at_port(Matmul::PORTS::A));
                auto &w_tensor = tensors.at(matmul_node->get_tensor_at_port(Matmul::PORTS::B));
            
                outgoing_nodes_for_tensors[x_tensor->get_name()].push_back(matmul_node->get_name());
                outgoing_nodes_for_tensors[w_tensor->get_name()].push_back(matmul_node->get_name());
                incoming_nodes_for_tensors[y_tensor->get_name()].push_back(matmul_node->get_name());
                
                incoming_tensors_for_nodes[matmul_node->get_name()].push_back(x_tensor->get_name());
                incoming_tensors_for_nodes[matmul_node->get_name()].push_back(w_tensor->get_name());
                outgoing_tensors_for_nodes[matmul_node->get_name()].push_back(y_tensor->get_name());
                break;
            }
            case Operation::Tag::Batchnorm: {
                auto batchnorm_node = std::static_pointer_cast<Batchnorm>(node.second);    
                auto &y_tensor = tensors.at(batchnorm_node->get_tensor_at_port(Batchnorm::PORTS::Y));
                auto &x_tensor = tensors.at(batchnorm_node->get_tensor_at_port(Batchnorm::PORTS::X));
                auto &mean_tensor = tensors.at(batchnorm_node->get_tensor_at_port(Batchnorm::PORTS::Mean));
                auto &var_tensor = tensors.at(batchnorm_node->get_tensor_at_port(Batchnorm::PORTS::Var));
                auto &prev_running_mean_tensor = tensors.at(batchnorm_node->get_tensor_at_port(Batchnorm::PORTS::Previous_running_mean));
                auto &prev_running_var_tensor = tensors.at(batchnorm_node->get_tensor_at_port(Batchnorm::PORTS::Previous_running_var));
                auto &next_running_mean_tensor = tensors.at(batchnorm_node->get_tensor_at_port(Batchnorm::PORTS::Next_running_mean));
                auto &next_running_var_tensor = tensors.at(batchnorm_node->get_tensor_at_port(Batchnorm::PORTS::Next_running_var));
                auto &scale_tensor = tensors.at(batchnorm_node->get_tensor_at_port(Batchnorm::PORTS::Scale));
                auto &bias_tensor = tensors.at(batchnorm_node->get_tensor_at_port(Batchnorm::PORTS::Bias));
                auto &eps_tensor = tensors.at(batchnorm_node->get_tensor_at_port(Batchnorm::PORTS::EPS));
                auto &exp_avg_tensor = tensors.at(batchnorm_node->get_tensor_at_port(Batchnorm::PORTS::EXP_AVG));
    
                outgoing_nodes_for_tensors[x_tensor->get_name()].push_back(batchnorm_node->get_name());
                outgoing_nodes_for_tensors[prev_running_mean_tensor->get_name()].push_back(batchnorm_node->get_name());
                outgoing_nodes_for_tensors[prev_running_var_tensor->get_name()].push_back(batchnorm_node->get_name());
                outgoing_nodes_for_tensors[eps_tensor->get_name()].push_back(batchnorm_node->get_name());
                outgoing_nodes_for_tensors[exp_avg_tensor->get_name()].push_back(batchnorm_node->get_name());
                outgoing_nodes_for_tensors[scale_tensor->get_name()].push_back(batchnorm_node->get_name());
                outgoing_nodes_for_tensors[bias_tensor->get_name()].push_back(batchnorm_node->get_name());
                incoming_nodes_for_tensors[y_tensor->get_name()].push_back(batchnorm_node->get_name());
                incoming_nodes_for_tensors[mean_tensor->get_name()].push_back(batchnorm_node->get_name());
                incoming_nodes_for_tensors[var_tensor->get_name()].push_back(batchnorm_node->get_name());
                incoming_nodes_for_tensors[next_running_mean_tensor->get_name()].push_back(batchnorm_node->get_name());
                incoming_nodes_for_tensors[next_running_var_tensor->get_name()].push_back(batchnorm_node->get_name());
                
                incoming_tensors_for_nodes[batchnorm_node->get_name()].push_back(x_tensor->get_name());
                incoming_tensors_for_nodes[batchnorm_node->get_name()].push_back(prev_running_mean_tensor->get_name());
                incoming_tensors_for_nodes[batchnorm_node->get_name()].push_back(prev_running_var_tensor->get_name());
                incoming_tensors_for_nodes[batchnorm_node->get_name()].push_back(eps_tensor->get_name());
                incoming_tensors_for_nodes[batchnorm_node->get_name()].push_back(exp_avg_tensor->get_name());
                incoming_tensors_for_nodes[batchnorm_node->get_name()].push_back(scale_tensor->get_name());
                incoming_tensors_for_nodes[batchnorm_node->get_name()].push_back(bias_tensor->get_name());
                outgoing_tensors_for_nodes[batchnorm_node->get_name()].push_back(y_tensor->get_name());
                outgoing_tensors_for_nodes[batchnorm_node->get_name()].push_back(mean_tensor->get_name());
                outgoing_tensors_for_nodes[batchnorm_node->get_name()].push_back(var_tensor->get_name());
                outgoing_tensors_for_nodes[batchnorm_node->get_name()].push_back(next_running_mean_tensor->get_name());
                outgoing_tensors_for_nodes[batchnorm_node->get_name()].push_back(next_running_var_tensor->get_name());
                break;
            }
            case Operation::Tag::Pointwise: {
                auto pointwise_node = std::static_pointer_cast<Pointwise>(node.second); 
                auto const port_count = get_pointwise_mode_port_count(pointwise_node->get_mode());
                
                auto &y_tensor = tensors.at(pointwise_node->get_tensor_at_port(Pointwise::PORTS::OUT_0));
                auto &x_tensor = tensors.at(pointwise_node->get_tensor_at_port(Pointwise::PORTS::IN_0));
            
                outgoing_nodes_for_tensors[x_tensor->get_name()].push_back(pointwise_node->get_name());
                incoming_nodes_for_tensors[y_tensor->get_name()].push_back(pointwise_node->get_name());
                
                incoming_tensors_for_nodes[pointwise_node->get_name()].push_back(x_tensor->get_name());
                outgoing_tensors_for_nodes[pointwise_node->get_name()].push_back(y_tensor->get_name());
                
                if(port_count >= 3) {
                    auto &b_tensor = tensors.at(pointwise_node->get_tensor_at_port(Pointwise::PORTS::IN_1));
                
                    outgoing_nodes_for_tensors[b_tensor->get_name()].push_back(pointwise_node->get_name());
                
                    incoming_tensors_for_nodes[pointwise_node->get_name()].push_back(b_tensor->get_name());
                }

                if(port_count >= 4) {
                    auto &t_tensor = tensors.at(pointwise_node->get_tensor_at_port(Pointwise::PORTS::IN_2));
                
                    outgoing_nodes_for_tensors[t_tensor->get_name()].push_back(pointwise_node->get_name());
                
                    incoming_tensors_for_nodes[pointwise_node->get_name()].push_back(t_tensor->get_name());
                }

                break;
            }
            case Operation::Tag::Reduction: {
                auto reduction_node = std::static_pointer_cast<Reduction>(node.second);       
                auto &y_tensor = tensors.at(reduction_node->get_tensor_at_port(Reduction::PORTS::Y));
                auto &x_tensor = tensors.at(reduction_node->get_tensor_at_port(Reduction::PORTS::X));
            
                outgoing_nodes_for_tensors[x_tensor->get_name()].push_back(reduction_node->get_name());
                incoming_nodes_for_tensors[y_tensor->get_name()].push_back(reduction_node->get_name());
                
                incoming_tensors_for_nodes[reduction_node->get_name()].push_back(x_tensor->get_name());
                outgoing_tensors_for_nodes[reduction_node->get_name()].push_back(y_tensor->get_name());
                break;
            }
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
        CHECK_CUDNN_FRONTEND_ERROR(flat_node.sub_nodes[node]->infer_properties());
    }

    return error_t::OK;
}

inline int64_t Graph::get_workspace_size() {
    return flat_node.get_workspace_size();
}

inline error_t Graph::build(cudnnHandle_t handle) {
    flat_node.tensor_props = tensors;
    
    for (auto &node : nodes) {
        switch (node.second->get_tag()) {
            case Operation::Tag::Convolution: {
                getLogger() << "[cudnn_frontend] INFO: Adding the conv node named " << node.first << std::endl;
                auto conv_node = std::make_shared<ConvolutionNode>(node.first, uid_offset);
                conv_node->props = std::static_pointer_cast<Convolution>(node.second);
                conv_node->parent_node = &flat_node;
                flat_node.sub_nodes[node.first] = conv_node;
                uid_offset += 100;
                break;
            }
            case Operation::Tag::Matmul: {
                getLogger() << "[cudnn_frontend] INFO: Adding the matmul node named " << node.first << std::endl;
                auto matmul_node = std::make_shared<MatMulNode>(node.first, uid_offset);
                matmul_node->props = std::static_pointer_cast<Matmul>(node.second);
                matmul_node->parent_node = &flat_node;
                flat_node.sub_nodes[node.first] = matmul_node;
                uid_offset += 100;
                break;
            }
            case Operation::Tag::Batchnorm: {
                getLogger() << "[cudnn_frontend] INFO: Adding the batch norm node named " << node.first << std::endl;
                auto batchnorm_node = std::make_shared<BatchNormNode>(node.first, uid_offset);
                batchnorm_node->props = std::static_pointer_cast<Batchnorm>(node.second);
                batchnorm_node->parent_node = &flat_node;
                flat_node.sub_nodes[node.first] = batchnorm_node;
                uid_offset += 100;
                break;
            }
            case Operation::Tag::Pointwise: {
                getLogger() << "[cudnn_frontend] INFO: Adding the pointwise node named " << node.first << std::endl;
                auto pointwise_node = std::make_shared<PointwiseNode>(node.first, uid_offset);
                pointwise_node->props = std::static_pointer_cast<Pointwise>(node.second);
                pointwise_node->parent_node = &flat_node;
                flat_node.sub_nodes[node.first] = pointwise_node;
                uid_offset += 100;
                break;
            }
            case Operation::Tag::Reduction: {
                getLogger() << "[cudnn_frontend] INFO: Adding the reduction node named " << node.first << std::endl;
                auto reduction_node = std::make_shared<ReductionNode>(node.first, uid_offset);
                reduction_node->props = std::static_pointer_cast<Reduction>(node.second);
                reduction_node->parent_node = &flat_node;
                flat_node.sub_nodes[node.first] = reduction_node;
                uid_offset += 100;
                break;
            }
        }
    }

    CHECK_CUDNN_FRONTEND_ERROR(infer_properties());    
    CHECK_CUDNN_FRONTEND_ERROR(flat_node.build(handle));

    return error_t::OK;
}

inline error_t Graph::execute(cudnnHandle_t handle, std::unordered_map<std::string, void *> var_pack) {
    CHECK_CUDNN_FRONTEND_ERROR(flat_node.execute(handle, var_pack));
    
    return error_t::OK;
}

} // namespace graph

} // namespace cudnn_frontend
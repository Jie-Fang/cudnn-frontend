#pragma once

#include <unordered_map>

#include "graphs/cudnn_frontend_node_batchnorm.h"
#include "graphs/cudnn_frontend_node_batchnorm_finalize.h"
#include "graphs/cudnn_frontend_node_convolution.h"
#include "graphs/cudnn_frontend_node_genstats.h"
#include "graphs/cudnn_frontend_node_matmul.h"
#include "graphs/cudnn_frontend_node_pointwise.h"
#include "graphs/cudnn_frontend_node_wgrad.h"
#include "graphs/cudnn_frontend_node_dgrad.h"

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

    // https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
    std::vector<std::string> sorted_nodes;

    int64_t uid_offset = 1;

    FlatNode flat_node{"flat_node", 1};
    detail::Context& get_context();

    error_t infer_properties();
    error_t run_graph_rules() const;

    Graph& insert_tensor_(std::shared_ptr<Tensor> tensor_ptr);
    Graph& insert_node_(std::shared_ptr<Operation> node_ptr);

    bool is_validated = false;

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
    error_t validate(cudnnHandle_t handle);

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
        case Operation::Tag::Batchnorm_finalize:{
            nodes.emplace(node_ptr->get_name(), std::static_pointer_cast<Batchnorm_finalize>(node_ptr));
            break;
        }
        case Operation::Tag::Convolution:{
            nodes.emplace(node_ptr->get_name(), std::static_pointer_cast<Convolution>(node_ptr));
            break;
        }
        case Operation::Tag::Dgrad:{
            nodes.emplace(node_ptr->get_name(), std::static_pointer_cast<Dgrad>(node_ptr));
            break;
        }
        case Operation::Tag::Genstats:{
            nodes.emplace(node_ptr->get_name(), std::static_pointer_cast<Genstats>(node_ptr));
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
        case Operation::Tag::Wgrad:{
            nodes.emplace(node_ptr->get_name(), std::static_pointer_cast<Wgrad>(node_ptr));
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
        case Operation::Tag::Batchnorm_finalize:{
            insert_node_(std::static_pointer_cast<Operation>(std::make_shared<Batchnorm_finalize>((Batchnorm_finalize&)props)));
            break;
        }
        case Operation::Tag::Convolution:{
            insert_node_(std::static_pointer_cast<Operation>(std::make_shared<Convolution>((Convolution&)props)));
            break;
        }
        case Operation::Tag::Dgrad:{
            insert_node_(std::static_pointer_cast<Operation>(std::make_shared<Dgrad>((Dgrad&)props)));
            break;
        }
        case Operation::Tag::Genstats:{
            insert_node_(std::static_pointer_cast<Operation>(std::make_shared<Genstats>((Genstats&)props)));
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
        case Operation::Tag::Wgrad:{
            insert_node_(std::static_pointer_cast<Operation>(std::make_shared<Wgrad>((Wgrad&)props)));
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
                case Operation::Tag::Batchnorm_finalize:{
                    for(auto& itr: std::static_pointer_cast<Batchnorm_finalize>(node.second)->port_to_name) {
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
                case Operation::Tag::Dgrad:{
                    for(auto& itr: std::static_pointer_cast<Dgrad>(node.second)->port_to_name) {
                        if(itr.second == connection.first) {
                            itr.second = connection.second;
                        }
                    }
                    break;
                }
                case Operation::Tag::Genstats:{
                    for(auto& itr: std::static_pointer_cast<Genstats>(node.second)->port_to_name) {
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
                case Operation::Tag::Wgrad:{
                    for(auto& itr: std::static_pointer_cast<Wgrad>(node.second)->port_to_name) {
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
            case Operation::Tag::Batchnorm_finalize: {
                std::static_pointer_cast<Batchnorm_finalize>(itr.second)->fill_from_context(other_graph.get_context());
                break;
            }
            case Operation::Tag::Convolution: {
                std::static_pointer_cast<Convolution>(itr.second)->fill_from_context(other_graph.get_context());
                break;
            }
            case Operation::Tag::Dgrad: {
                std::static_pointer_cast<Dgrad>(itr.second)->fill_from_context(other_graph.get_context());
                break;
            }
            case Operation::Tag::Genstats: {
                std::static_pointer_cast<Genstats>(itr.second)->fill_from_context(other_graph.get_context());
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
            case Operation::Tag::Wgrad: {
                std::static_pointer_cast<Wgrad>(itr.second)->fill_from_context(other_graph.get_context());
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
            case Operation::Tag::Dgrad: {
                auto dgrad_node = std::static_pointer_cast<Dgrad>(node.second);
                auto &dy_tensor = tensors.at(dgrad_node->get_tensor_at_port(Dgrad::PORTS::DY));
                auto &w_tensor = tensors.at(dgrad_node->get_tensor_at_port(Dgrad::PORTS::W));
                auto &dx_tensor = tensors.at(dgrad_node->get_tensor_at_port(Dgrad::PORTS::DX));

                outgoing_nodes_for_tensors[dy_tensor->get_name()].push_back(dgrad_node->get_name());
                outgoing_nodes_for_tensors[w_tensor->get_name()].push_back(dgrad_node->get_name());
                incoming_nodes_for_tensors[dx_tensor->get_name()].push_back(dgrad_node->get_name());

                incoming_tensors_for_nodes[dgrad_node->get_name()].push_back(dy_tensor->get_name());
                incoming_tensors_for_nodes[dgrad_node->get_name()].push_back(w_tensor->get_name());
                outgoing_tensors_for_nodes[dgrad_node->get_name()].push_back(dx_tensor->get_name());
                break;
            }
            case Operation::Tag::Genstats: {
                auto genstats_node = std::static_pointer_cast<Genstats>(node.second);
                auto &x_tensor = tensors.at(genstats_node->get_tensor_at_port(Genstats::PORTS::X));
                auto &sum_tensor = tensors.at(genstats_node->get_tensor_at_port(Genstats::PORTS::SUM));
                auto &sq_sum_tensor = tensors.at(genstats_node->get_tensor_at_port(Genstats::PORTS::SQ_SUM));

                outgoing_nodes_for_tensors[x_tensor->get_name()].push_back(genstats_node->get_name());
                incoming_nodes_for_tensors[sum_tensor->get_name()].push_back(genstats_node->get_name());
                incoming_nodes_for_tensors[sq_sum_tensor->get_name()].push_back(genstats_node->get_name());

                incoming_tensors_for_nodes[genstats_node->get_name()].push_back(x_tensor->get_name());
                outgoing_tensors_for_nodes[genstats_node->get_name()].push_back(sum_tensor->get_name());
                outgoing_tensors_for_nodes[genstats_node->get_name()].push_back(sq_sum_tensor->get_name());
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
            case Operation::Tag::Batchnorm_finalize: {
                auto batchnorm_finalize_node = std::static_pointer_cast<Batchnorm_finalize>(node.second);
                auto &sum_tensor = tensors.at(batchnorm_finalize_node->get_tensor_at_port(Batchnorm_finalize::PORTS::SUM));
                auto &square_sum_tensor = tensors.at(batchnorm_finalize_node->get_tensor_at_port(Batchnorm_finalize::PORTS::SQUARE_SUM));
                auto &mean_tensor = tensors.at(batchnorm_finalize_node->get_tensor_at_port(Batchnorm_finalize::PORTS::MEAN));
                auto &inv_var_tensor = tensors.at(batchnorm_finalize_node->get_tensor_at_port(Batchnorm_finalize::PORTS::INV_VARIANCE));
                auto &prev_running_mean_tensor = tensors.at(batchnorm_finalize_node->get_tensor_at_port(Batchnorm_finalize::PORTS::Previous_running_mean));
                auto &prev_running_var_tensor = tensors.at(batchnorm_finalize_node->get_tensor_at_port(Batchnorm_finalize::PORTS::Previous_running_var));
                auto &next_running_mean_tensor = tensors.at(batchnorm_finalize_node->get_tensor_at_port(Batchnorm_finalize::PORTS::Next_running_mean));
                auto &next_running_var_tensor = tensors.at(batchnorm_finalize_node->get_tensor_at_port(Batchnorm_finalize::PORTS::Next_running_var));
                auto &scale_tensor = tensors.at(batchnorm_finalize_node->get_tensor_at_port(Batchnorm_finalize::PORTS::SCALE));
                auto &bias_tensor = tensors.at(batchnorm_finalize_node->get_tensor_at_port(Batchnorm_finalize::PORTS::BIAS));
                auto &eq_scale_tensor = tensors.at(batchnorm_finalize_node->get_tensor_at_port(Batchnorm_finalize::PORTS::EQUIVALENT_SCALE));
                auto &eq_bias_tensor = tensors.at(batchnorm_finalize_node->get_tensor_at_port(Batchnorm_finalize::PORTS::EQUIVALENT_BIAS));
                auto &eps_tensor = tensors.at(batchnorm_finalize_node->get_tensor_at_port(Batchnorm_finalize::PORTS::EPSILON));
                auto &exp_avg_tensor = tensors.at(batchnorm_finalize_node->get_tensor_at_port(Batchnorm_finalize::PORTS::EXP_AVG));

                outgoing_nodes_for_tensors[sum_tensor->get_name()].push_back(batchnorm_finalize_node->get_name());
                outgoing_nodes_for_tensors[square_sum_tensor->get_name()].push_back(batchnorm_finalize_node->get_name());
                outgoing_nodes_for_tensors[mean_tensor->get_name()].push_back(batchnorm_finalize_node->get_name());
                outgoing_nodes_for_tensors[inv_var_tensor->get_name()].push_back(batchnorm_finalize_node->get_name());
                outgoing_nodes_for_tensors[prev_running_mean_tensor->get_name()].push_back(batchnorm_finalize_node->get_name());
                outgoing_nodes_for_tensors[prev_running_var_tensor->get_name()].push_back(batchnorm_finalize_node->get_name());
                outgoing_nodes_for_tensors[eps_tensor->get_name()].push_back(batchnorm_finalize_node->get_name());
                outgoing_nodes_for_tensors[exp_avg_tensor->get_name()].push_back(batchnorm_finalize_node->get_name());
                outgoing_nodes_for_tensors[scale_tensor->get_name()].push_back(batchnorm_finalize_node->get_name());
                outgoing_nodes_for_tensors[bias_tensor->get_name()].push_back(batchnorm_finalize_node->get_name());
                incoming_nodes_for_tensors[next_running_mean_tensor->get_name()].push_back(batchnorm_finalize_node->get_name());
                incoming_nodes_for_tensors[next_running_var_tensor->get_name()].push_back(batchnorm_finalize_node->get_name());
                incoming_nodes_for_tensors[eq_scale_tensor->get_name()].push_back(batchnorm_finalize_node->get_name());
                incoming_nodes_for_tensors[eq_bias_tensor->get_name()].push_back(batchnorm_finalize_node->get_name());

                incoming_tensors_for_nodes[batchnorm_finalize_node->get_name()].push_back(sum_tensor->get_name());
                incoming_tensors_for_nodes[batchnorm_finalize_node->get_name()].push_back(square_sum_tensor->get_name());
                incoming_tensors_for_nodes[batchnorm_finalize_node->get_name()].push_back(prev_running_mean_tensor->get_name());
                incoming_tensors_for_nodes[batchnorm_finalize_node->get_name()].push_back(prev_running_var_tensor->get_name());
                incoming_tensors_for_nodes[batchnorm_finalize_node->get_name()].push_back(eps_tensor->get_name());
                incoming_tensors_for_nodes[batchnorm_finalize_node->get_name()].push_back(exp_avg_tensor->get_name());
                incoming_tensors_for_nodes[batchnorm_finalize_node->get_name()].push_back(scale_tensor->get_name());
                incoming_tensors_for_nodes[batchnorm_finalize_node->get_name()].push_back(bias_tensor->get_name());
                incoming_tensors_for_nodes[batchnorm_finalize_node->get_name()].push_back(mean_tensor->get_name());
                incoming_tensors_for_nodes[batchnorm_finalize_node->get_name()].push_back(inv_var_tensor->get_name());
                outgoing_tensors_for_nodes[batchnorm_finalize_node->get_name()].push_back(next_running_mean_tensor->get_name());
                outgoing_tensors_for_nodes[batchnorm_finalize_node->get_name()].push_back(next_running_var_tensor->get_name());
                outgoing_tensors_for_nodes[batchnorm_finalize_node->get_name()].push_back(eq_scale_tensor->get_name());
                outgoing_tensors_for_nodes[batchnorm_finalize_node->get_name()].push_back(eq_bias_tensor->get_name());
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
            case Operation::Tag::Wgrad: {
                auto conv_node = std::static_pointer_cast<Wgrad>(node.second);
                auto &dy_tensor = tensors.at(conv_node->get_tensor_at_port(Wgrad::PORTS::DY));
                auto &x_tensor = tensors.at(conv_node->get_tensor_at_port(Wgrad::PORTS::X));
                auto &dw_tensor = tensors.at(conv_node->get_tensor_at_port(Wgrad::PORTS::DW));

                outgoing_nodes_for_tensors[x_tensor->get_name()].push_back(conv_node->get_name());
                outgoing_nodes_for_tensors[dy_tensor->get_name()].push_back(conv_node->get_name());
                incoming_nodes_for_tensors[dw_tensor->get_name()].push_back(conv_node->get_name());

                incoming_tensors_for_nodes[conv_node->get_name()].push_back(x_tensor->get_name());
                incoming_tensors_for_nodes[conv_node->get_name()].push_back(dy_tensor->get_name());
                outgoing_tensors_for_nodes[conv_node->get_name()].push_back(dw_tensor->get_name());
                break;
            }
        }
    }

    // https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
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

inline error_t Graph::run_graph_rules() const {
    int32_t major  = 0;
    int32_t minor  = 0;
    int32_t count  = 0;
    int32_t device = 0;

    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        return error_t::INVALID_CUDA_DEVICE;
    }
    if (count == 0 || cudaGetDevice(&device) != cudaSuccess) {
        return error_t::INVALID_CUDA_DEVICE;
    }
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) != cudaSuccess) {
        return error_t::INVALID_CUDA_DEVICE;
    }
    if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) != cudaSuccess) {
        return error_t::INVALID_CUDA_DEVICE;
    }
    auto device_version = (major * 100) + (minor * 10);

    auto cudnn_version = cudnnGetVersion();
    if (device_version < 700) {
        getLogger() << "Device version insufficient" << std::endl;
        return error_t::UNSUPPORTED_GRAPH_FORMAT;
    }

    bool is_supported = false;
    // No rules check below cudnn 8.9.0
    // Instead will be performed in the backend.
    if (8900 >= cudnn_version) { return error_t::OK;}

    // Section 3.3.1 https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#compile-single-op-engine
    if (nodes.size() == 1) {
        auto tag = nodes.begin()->second->get_tag();
        if (tag != Operation::Tag::Convolution &&
            tag != Operation::Tag::Batchnorm &&
            tag != Operation::Tag::Pointwise) {return error_t::UNSUPPORTED_GRAPH_FORMAT;}
    }
    // Section 3.3.3 https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#specialized-runtime-fusion-engines
    bool is_first_node = true;
    auto tag = nodes.find(sorted_nodes.front())->second->get_tag();
    switch (tag) {
        case Operation::Tag::Batchnorm: {
            auto supported_pattern = {Operation::Tag::Batchnorm, Operation::Tag::Pointwise, Operation::Tag::Pointwise, Operation::Tag::Pointwise};
            auto supported_pointwise_pattern = {PointwiseMode_t::ADD, PointwiseMode_t::RELU_FWD, PointwiseMode_t::CMP_GT};
            std::vector<Operation::Tag> actual_pattern = {Operation::Tag::Batchnorm};
            std::vector<PointwiseMode_t> actual_pointwise_pattern = {};
            (void) supported_pattern;
            (void) supported_pointwise_pattern;
            for (auto const &node_name : sorted_nodes) {
                if (true == is_first_node) {
                    is_first_node = false;
                    continue;
                }
                auto op_  = nodes.find(node_name)->second;
                auto tag_ = op_->get_tag();
                actual_pattern.push_back(tag_);
                if (tag_ == Operation::Tag::Pointwise) {
                    auto pointwise_op = std::static_pointer_cast<cudnn_frontend::graph::Pointwise>(op_);
                    actual_pointwise_pattern.push_back(pointwise_op->get_mode());
                }
            }
            is_supported = std::includes(supported_pattern.begin(), supported_pattern.end(), actual_pattern.begin(), actual_pattern.end());
            is_supported = std::includes(supported_pointwise_pattern.begin(), supported_pointwise_pattern.end(), actual_pointwise_pattern.begin(), actual_pointwise_pattern.end());
            getLogger() << "3.3.3.1. BnAddRelu supported" << std::endl;
            break;
        }
        default: {
            break;
        }
    }

    if (true == is_supported) {return error_t::OK;}
    // Section 3.3.4 https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#compile-specialized-engine
    // Section 3.3.2 https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#runtime-fusion-engine
    
    // Check if g1 can be applied
    if ((device_version < 800) && ((tag != Operation::Tag::Convolution) && (tag != Operation::Tag::Matmul))) {
        getLogger() << "Device version insufficient" << std::endl;
        return error_t::UNSUPPORTED_GRAPH_FORMAT;
    }

    std::set<Operation::Tag> g1_supported_pattern = {/*Operation::Tag::Concat, Operation::Tag::Signal */Operation::Tag::Pointwise};
    std::set<Operation::Tag> g2_supported_pattern = { /*Operation::Tag::ResampleFwd,
                                                        Operation::Tag::ResampleBwd,
                                                        Operation::Tag::Genstats*/
                                                        Operation::Tag::Reduction,
                                                        Operation::Tag::Pointwise};

    enum class Graph_parser_state {
        G1_PROCESS,
        G1_PROCESS_POINTWISE,
        G2_START,
        G2_PROCESS_REDUCTION_NODE_SEEN,
        G2_PROCESS_SIGNAL_NODE_SEEN,
    };

    Graph_parser_state state = Graph_parser_state::G1_PROCESS;
    is_supported = true;


    for (auto const &node_name : sorted_nodes) {
        auto op_  = nodes.find(node_name)->second;
        auto tag_ = op_->get_tag();

        switch (state) {
            case Graph_parser_state::G1_PROCESS:
            case Graph_parser_state::G1_PROCESS_POINTWISE: {
                if (tag_ == Operation::Tag::Convolution || tag_ == Operation::Tag::Matmul) {
                    state = Graph_parser_state::G2_START;
                // } else if ((tag_ == Operation::Tag::Resample_Fwd) || (tag_ == Operation::Tag::Resample_Bwd)) {
                //     actual_g2_pattern.push_back(tag_);
                //     state = Graph_parser_state::G2_START;
                } else {
                    // G1 nodes has to be one of concat / signal / pointwise
                    if (g1_supported_pattern.find(tag_) == g1_supported_pattern.end()) {is_supported = false;}

                    if (tag_ == Operation::Tag::Pointwise) {state = Graph_parser_state::G1_PROCESS_POINTWISE;}
                    // if G1 encounters pointwise, no other operation can be succeed
                    if ((state == Graph_parser_state::G1_PROCESS_POINTWISE)  && (tag_ != Operation::Tag::Pointwise)) {
                        is_supported = false;
                    }
                }
            } break;

            case Graph_parser_state::G2_START:
            case Graph_parser_state::G2_PROCESS_REDUCTION_NODE_SEEN:
            case Graph_parser_state::G2_PROCESS_SIGNAL_NODE_SEEN : {
                // g2 can only have nodes in g2_supported_patterns
                if (g2_supported_pattern.find(tag_) == g2_supported_pattern.end()) {is_supported = false;}

                // Only one of reduction or signal can be present and only as end node.
                if ((state == Graph_parser_state::G2_PROCESS_REDUCTION_NODE_SEEN)||
                    (state == Graph_parser_state::G2_PROCESS_SIGNAL_NODE_SEEN)) {is_supported = false;}

            } break;
        }
        if (is_supported == false) {break;}
    }

    // TODO
    if (is_supported == false) {return error_t::UNSUPPORTED_GRAPH_FORMAT;}
    return error_t::OK;
}

inline int64_t Graph::get_workspace_size() {
    return flat_node.get_workspace_size();
}

inline error_t Graph::validate(cudnnHandle_t handle) {
    (void) handle;
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
            case Operation::Tag::Dgrad: {
                getLogger() << "[cudnn_frontend] INFO: Adding the dgrad node named " << node.first << std::endl;
                auto dgrad_node = std::make_shared<DgradNode>(node.first, uid_offset);
                dgrad_node->props = std::static_pointer_cast<Dgrad>(node.second);
                dgrad_node->parent_node = &flat_node;
                flat_node.sub_nodes[node.first] = dgrad_node;
                uid_offset += 100;
                break;
            }
            case Operation::Tag::Genstats: {
                getLogger() << "[cudnn_frontend] INFO: Adding the genstats node named " << node.first << std::endl;
                auto genstats_node = std::make_shared<GenstatsNode>(node.first, uid_offset);
                genstats_node->props = std::static_pointer_cast<Genstats>(node.second);
                genstats_node->parent_node = &flat_node;
                flat_node.sub_nodes[node.first] = genstats_node;
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
            case Operation::Tag::Batchnorm_finalize: {
                getLogger() << "[cudnn_frontend] INFO: Adding the batch norm finalize node named " << node.first << std::endl;
                auto batchnorm_finalize_node = std::make_shared<BatchNormFinalizeNode>(node.first, uid_offset);
                batchnorm_finalize_node->props = std::static_pointer_cast<Batchnorm_finalize>(node.second);
                batchnorm_finalize_node->parent_node = &flat_node;
                flat_node.sub_nodes[node.first] = batchnorm_finalize_node;
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
            case Operation::Tag::Wgrad: {
                getLogger() << "[cudnn_frontend] INFO: Adding the wgrad node named " << node.first << std::endl;
                auto wgrad_node = std::make_shared<WgradNode>(node.first, uid_offset);
                wgrad_node->props = std::static_pointer_cast<Wgrad>(node.second);
                wgrad_node->parent_node = &flat_node;
                flat_node.sub_nodes[node.first] = wgrad_node;
                uid_offset += 100;
                break;
            }
        }
    }

    CHECK_CUDNN_FRONTEND_ERROR(infer_properties());
    CHECK_CUDNN_FRONTEND_ERROR(run_graph_rules());
    is_validated = true;
    return error_t::OK;
}

inline error_t Graph::build(cudnnHandle_t handle) {
    if (is_validated == false) {
        CHECK_CUDNN_FRONTEND_ERROR(validate(handle));
    }
    CHECK_CUDNN_FRONTEND_ERROR(flat_node.build(handle));
    return error_t::OK;
}

inline error_t Graph::execute(cudnnHandle_t handle, std::unordered_map<std::string, void *> var_pack) {
    CHECK_CUDNN_FRONTEND_ERROR(flat_node.execute(handle, var_pack));
    return error_t::OK;
}

} // namespace graph

} // namespace cudnn_frontend
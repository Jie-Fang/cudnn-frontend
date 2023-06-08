#pragma once

#include <unordered_map>

#include "graphs/cudnn_frontend_node_batchnorm.h"
#include "graphs/cudnn_frontend_node_bn_finalize.h"
#include "graphs/cudnn_frontend_node_conv_fprop.h"
#include "graphs/cudnn_frontend_node_conv_dgrad.h"
#include "graphs/cudnn_frontend_node_conv_wgrad.h"
#include "graphs/cudnn_frontend_node_dbn_weight.h"
#include "graphs/cudnn_frontend_node_genstats.h"
#include "graphs/cudnn_frontend_node_matmul.h"
#include "graphs/cudnn_frontend_node_pointwise.h"
#include "graphs/cudnn_frontend_node_reduction.h"
#include "graphs/cudnn_frontend_node_rng.h"
#include "graphs/cudnn_frontend_node_scaled_dot_product_attention.h"

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
    std::unordered_set<std::shared_ptr<Tensor>> tensors;
    std::vector<std::shared_ptr<Operation>> nodes;

    FlatNode flat_node{"flat_node", detail::Context{}};
    detail::Context& get_context();

    error_t run_graph_rules() const;

    std::shared_ptr<Tensor>
    output_tensor(std::string const &name) {
        auto tensor = std::make_shared<Tensor>(name);
        tensors.emplace(tensor);
        return tensor;
    }

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
    
    std::shared_ptr<Tensor> tensor(Tensor const& tensor);

    Batchnorm::Outputs batchnorm(Batchnorm::Inputs, Batchnorm const&);

    BN_finalize::Outputs bn_finalize(BN_finalize::Inputs, BN_finalize const&);

    std::shared_ptr<Tensor> conv_fprop(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, Conv_fprop const& conv);
    Conv_fprop::Outputs conv_fprop(Conv_fprop::Inputs, Conv_fprop const& conv);
    
    std::shared_ptr<Tensor> conv_dgrad(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, Conv_dgrad const&);
    Conv_dgrad::Outputs conv_dgrad(Conv_dgrad::Inputs, Conv_dgrad const&);

    std::shared_ptr<Tensor> conv_wgrad(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, Conv_wgrad const&);
    Conv_wgrad::Outputs conv_wgrad(Conv_wgrad::Inputs, Conv_wgrad const&);

    std::array<std::shared_ptr<Tensor>, 5> dbn_weight(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, DBN_weight const&);
    DBN_weight::Outputs dbn_weight(DBN_weight::Inputs, DBN_weight const& dbn_weight);

    std::array<std::shared_ptr<Tensor>, 2> genstats(std::shared_ptr<Tensor>, Genstats const&);
    Genstats::Outputs genstats(Genstats::Inputs, Genstats const&);

    Matmul::Outputs matmul(Matmul::Inputs, Matmul const&);
    std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, Matmul const&);
    
    std::shared_ptr<Tensor> pointwise(std::shared_ptr<Tensor>, Pointwise const& pointwise);
    std::shared_ptr<Tensor> pointwise(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, Pointwise const& pointwise);
    std::shared_ptr<Tensor> pointwise(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, Pointwise const& pointwise);
    Pointwise::Outputs pointwise(Pointwise::Inputs, Pointwise const& pointwise);
    
    Scaled_dot_product_attention::Outputs scaled_dot_product_attention(Scaled_dot_product_attention::Inputs const&, Scaled_dot_product_attention const&);
    
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
    error_t execute(cudnnHandle_t handle, std::unordered_map<std::shared_ptr<Tensor>, void *>);
    error_t execute(cudnnHandle_t handle, std::unordered_map<std::string, void *>);

    friend std::ostream& operator<<(std::ostream& os, const Graph& props);

    ~Graph() = default;
};

inline std::ostream& operator<<(std::ostream& os, const Graph& graph) {
    os << "{tensors: [\n";
    for(auto const& tensor: graph.tensors) {
        os << *(tensor) << ",";
    }
    os << "],"
    << "\nnodes: [\n";
    for(auto const& node: graph.nodes) {
        os << node << ",";
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

inline std::shared_ptr<Tensor> Graph::tensor(Tensor const& tensor) {
    auto tensor_ptr = std::make_shared<Tensor>(tensor);
    tensors.emplace(tensor_ptr);
    return tensor_ptr;
}

inline BN_finalize::Outputs Graph::bn_finalize(BN_finalize::Inputs inputs, BN_finalize const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<BN_finalize>(user_options);

    // Set outputs
    options->make_outputs([this](std::string const &name){return output_tensor(name);});

    // Set inputs
    options->inputs = inputs;

    nodes.emplace_back(options);

    return options->outputs;
}

inline Batchnorm::Outputs Graph::batchnorm(Batchnorm::Inputs inputs, Batchnorm const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Batchnorm>(user_options);

    // Set outputs
    options->make_outputs([this](std::string const &name){return output_tensor(name);});

    // Set inputs
    options->inputs = inputs;

    nodes.emplace_back(options);

    return options->outputs;
}

inline std::shared_ptr<Tensor> Graph::conv_fprop(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w, Conv_fprop const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Conv_fprop>(user_options);

    // Make required output tensors
    options->outputs.Y = output_tensor(options->get_name() + "_output");

    // Set inputs
    options->inputs.X = x;
    options->inputs.W = w;

    nodes.emplace_back(options);

    return options->outputs.Y;
}

inline Conv_fprop::Outputs Graph::conv_fprop(Conv_fprop::Inputs inputs, Conv_fprop const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Conv_fprop>(user_options);

    // Set outputs
    options->outputs.Y = output_tensor(options->get_name() + "_output");

    // Set inputs
    options->inputs = inputs;

    nodes.emplace_back(options);

    return options->outputs;
}

inline std::array<std::shared_ptr<Tensor>, 5> Graph::dbn_weight(std::shared_ptr<Tensor> dy, std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> mean, std::shared_ptr<Tensor> inv_variance, std::shared_ptr<Tensor> scale, DBN_weight const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<DBN_weight>(user_options);

    // Make required output tensors
    options->make_outputs([this](std::string const &name){return output_tensor(name);});

    // Set inputs
    options->inputs.DY = dy;
    options->inputs.X = x;
    options->inputs.SCALE = scale;
    options->inputs.MEAN = mean;
    options->inputs.INV_VARIANCE = inv_variance;

    nodes.emplace_back(options);

    return {options->outputs.DSCALE, options->outputs.DBIAS, options->outputs.EQ_SCALE_DY, options->outputs.EQ_SCALE_X, options->outputs.EQ_BIAS};
}

inline DBN_weight::Outputs Graph::dbn_weight(DBN_weight::Inputs inputs, DBN_weight const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<DBN_weight>(user_options);

    // Make required output tensors
    options->make_outputs([this](std::string const &name){return output_tensor(name);});

    // Set inputs
    options->inputs = inputs;

    nodes.emplace_back(options);

    return options->outputs;
}

inline std::shared_ptr<Tensor> Graph::conv_dgrad(std::shared_ptr<Tensor> dy, std::shared_ptr<Tensor> w, Conv_dgrad const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Conv_dgrad>(user_options);

    // Make required output tensors
    options->outputs.DX = output_tensor(options->get_name() + "_output");

    // Set inputs
    options->inputs.DY = dy;
    options->inputs.W = w;

    nodes.emplace_back(options);

    return options->outputs.DX;
}

inline Conv_dgrad::Outputs Graph::conv_dgrad(Conv_dgrad::Inputs inputs, Conv_dgrad const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Conv_dgrad>(user_options);

    // Make required output tensors
    options->outputs.DX = output_tensor(options->get_name() + "_output");

    // Set inputs
    options->inputs = inputs;

    nodes.emplace_back(options);

    return options->outputs;
}

inline std::array<std::shared_ptr<Tensor>, 2> Graph::genstats(std::shared_ptr<Tensor> x, Genstats const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Genstats>(user_options);

    // Set outputs
    options->outputs.SUM = output_tensor(options->get_name() + "_sum_output");
    options->outputs.SQ_SUM = output_tensor(options->get_name() + "_sq_sum_output");

    // Set inputs
    options->inputs.X = x;

    nodes.emplace_back(options);

    return {options->outputs.SUM, options->outputs.SQ_SUM};
}

inline Genstats::Outputs Graph::genstats(Genstats::Inputs inputs, Genstats const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Genstats>(user_options);

    // Make required output tensors
    options->outputs.SUM = output_tensor(options->get_name() + "_sum_output");
    options->outputs.SQ_SUM = output_tensor(options->get_name() + "_sq_sum_output");

    // Set inputs
    options->inputs = inputs;

    nodes.emplace_back(options);

    return options->outputs;
}

inline std::shared_ptr<Tensor> Graph::conv_wgrad(std::shared_ptr<Tensor> dy, std::shared_ptr<Tensor> x, Conv_wgrad const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Conv_wgrad>(user_options);

    // Make required output tensors
    options->outputs.DW = output_tensor(options->get_name() + "_output");

    // Set inputs
    options->inputs.X = x;
    options->inputs.DY = dy;

    nodes.emplace_back(options);

    return options->outputs.DW;
}

inline Conv_wgrad::Outputs Graph::conv_wgrad(Conv_wgrad::Inputs inputs, Conv_wgrad const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Conv_wgrad>(user_options);

    // Make required output tensors
    options->outputs.DW = output_tensor(options->get_name() + "_output");

    // Set inputs
    options->inputs = inputs;

    nodes.emplace_back(options);

    return options->outputs;
}

inline std::shared_ptr<Tensor> Graph::pointwise(std::shared_ptr<Tensor> a, Pointwise const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Pointwise>(user_options);

    // Make required output tensors
    options->outputs.OUT_0 = output_tensor(options->get_name() + "_output");

    // Set inputs
    options->inputs.IN_0 = a;

    nodes.emplace_back(options);

    return options->outputs.OUT_0;
}

inline std::shared_ptr<Tensor> Graph::pointwise(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, Pointwise const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Pointwise>(user_options);

    // Set outputs
    options->outputs.OUT_0 = output_tensor(options->get_name() + "_output");

    // Set inputs
    options->inputs.IN_0 = a;
    options->inputs.IN_1 = b;

    nodes.emplace_back(options);

    return options->outputs.OUT_0;
}

inline std::shared_ptr<Tensor> Graph::pointwise(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, std::shared_ptr<Tensor> c, Pointwise const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Pointwise>(user_options);

    options->outputs.OUT_0 = output_tensor(options->get_name() + "_output");

    // Set inputs
    options->inputs.IN_0 = a;
    options->inputs.IN_1 = b;
    options->inputs.IN_2 = c;

    nodes.emplace_back(options);

    return options->outputs.OUT_0;
}

inline Pointwise::Outputs Graph::pointwise(Pointwise::Inputs inputs, Pointwise const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Pointwise>(user_options);

    options->outputs.OUT_0 = output_tensor(options->get_name() + "_output");

    // Set inputs
    options->inputs = inputs;

    nodes.emplace_back(options);

    return options->outputs;
}

inline std::shared_ptr<Tensor> Graph::matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, Matmul const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Matmul>(user_options);

    options->outputs.C = output_tensor(options->get_name() + "_output");

    // Set inputs
    options->inputs.A = a;
    options->inputs.B = b;

    nodes.emplace_back(options);

    return options->outputs.C;
}

inline Matmul::Outputs Graph::matmul(Matmul::Inputs inputs, Matmul const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Matmul>(user_options);

    options->outputs.C = output_tensor(options->get_name() + "_output");

    // Set inputs
    options->inputs = inputs;

    nodes.emplace_back(options);

    return options->outputs;
}

inline Scaled_dot_product_attention::Outputs Graph::scaled_dot_product_attention(Scaled_dot_product_attention::Inputs const& inputs, Scaled_dot_product_attention const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Scaled_dot_product_attention>(user_options);
    
    // Make required output tensors
    options->outputs.O = output_tensor(options->get_name() + "_output");

    if(options->get_is_inference() == false) {
        options->outputs.S = output_tensor(options->get_name() + "_softmax_output");
    }

    // Set inputs
    options->inputs = inputs;

    nodes.emplace_back(options);

    return options->outputs;
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
    auto device_version = (major * 10) + (minor * 1);

    auto cudnn_version = cudnnGetVersion();
    RETURN_CUDNN_FRONTEND_ERROR_IF(device_version < 70, error_t::UNSUPPORTED_GRAPH_FORMAT);

    bool is_supported = false;

    // No rules for cudnn 8.9.0 and below.
    // Instead will be performed by backend.
    RETURN_CUDNN_FRONTEND_ERROR_IF(cudnn_version <= 8900, error_t::OK);

    auto const entrance_node_tag = nodes.front()->get_tag();

    if (nodes.size() == 1) {
        // Only contains checks for
        // Section 3.3.1 https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#compile-single-op-engine
        std::unordered_set<Operation::Tag> const supported_tags = {
                                                Operation::Tag::Conv_fprop
                                                , Operation::Tag::Conv_wgrad
                                                , Operation::Tag::Conv_dgrad
                                                , Operation::Tag::BN
                                                , Operation::Tag::Pointwise
                                            };
        RETURN_CUDNN_FRONTEND_ERROR_IF(supported_tags.find(entrance_node_tag) != supported_tags.end(), error_t::OK);
    }
    
    // Only contains checks for
    // Section 3.3.3 https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#specialized-runtime-fusion-engines
    switch (entrance_node_tag) {
        case Operation::Tag::BN: {
            // Only contains checks for Section 3.3.3.1
            auto bn_options = std::static_pointer_cast<cudnn_frontend::graph::Batchnorm>(nodes.front());

            // The pointwise nodes: Add, ReLU, and GT (greater than) are optional.
            if(std::any_of(std::next(nodes.begin(), 1), nodes.end(), [](auto const& node) {return node->get_tag() != Operation::Tag::Pointwise;})) {
                break;
            }

            auto pattern = {PointwiseMode_t::ADD, PointwiseMode_t::RELU_FWD, PointwiseMode_t::CMP_GT};
            std::vector<PointwiseMode_t> actual_pattern = {};
            for (auto itr = std::next(nodes.begin(), 1); itr != nodes.end(); itr++) {
                auto pointwise_options = std::static_pointer_cast<cudnn_frontend::graph::Pointwise>(*itr);
                actual_pattern.push_back(pointwise_options->get_mode().value());
            }
            if(!std::includes(pattern.begin(), pattern.end(), actual_pattern.begin(), actual_pattern.end())) {
                break;
            }

            // The attribute CUDNN_ATTR_OPERATION_NORM_FWD_MODE for the norm forward operation must be set to CUDNN_BATCH_NORM.
            // Hardcoded inside operation

            // The attribute CUDNN_ATTR_OPERATION_NORM_FWD_PHASE for the norm forward operation must be set to CUDNN_NORM_FWD_TRAINING.
            if(bn_options->get_forward_phase() != NormFwdPhase_t::TRAINING) {
                break;
            }

            // For FP16 and BF16 data types, the channel count C for the tensors must be a multiple of 8 while
            // for float data type the channel count must be a multiple of 4.
            auto const& X = bn_options->inputs.X;
            auto const X_data_type = X->get_data_type();
            auto const& X_dim = X->get_dim();
            if((X_data_type == DataType_t::FLOAT) && (X_dim[1]%4)) {
                break;
            }
            else if((X_data_type == DataType_t::HALF || X_data_type == DataType_t::BFLOAT16) && (X_dim[1]%8)) {
                break;
            }

            // These patterns are supported on devices with compute capability >= 8.0
            if(device_version < 80) {
                break;
            }
            
            return error_t::OK;
            break;
        }
        case Operation::Tag::Scaled_dot_product_attention: {
            // Only contains checks for
            // Section 3.3.3.3
            return error_t::OK;
        }
        default: {
            break;
        }
    }
    RETURN_CUDNN_FRONTEND_ERROR_IF(is_supported, error_t::OK);

    // Section 3.3.4 https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#compile-specialized-engine
    // Section 3.3.2 https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#runtime-fusion-engine
    
    // Check if g1 can be applied
    if ((device_version < 80) && ((entrance_node_tag != Operation::Tag::Conv_fprop) && (entrance_node_tag != Operation::Tag::Matmul))) {
        getLogger() << "Device version insufficient" << std::endl;
        return error_t::UNSUPPORTED_GRAPH_FORMAT;
    }

    std::set<Operation::Tag> g1_supported_pattern = {/*Operation::Tag::Concat, Operation::Tag::Signal */Operation::Tag::Pointwise};
    std::set<Operation::Tag> g2_supported_pattern = { /*Operation::Tag::ResampleFwd,
                                                        Operation::Tag::ResampleBwd*/
                                                        Operation::Tag::Genstats,
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


    for (auto const &node : nodes) {
        auto tag_ = node->get_tag();

        switch (state) {
            case Graph_parser_state::G1_PROCESS:
            case Graph_parser_state::G1_PROCESS_POINTWISE: {
                if (tag_ == Operation::Tag::Conv_fprop || tag_ == Operation::Tag::Matmul) {
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
    if (is_supported == false) {return error_t::OK;}
    return error_t::OK;
}

inline int64_t Graph::get_workspace_size() {
    return flat_node.get_workspace_size();
}

inline error_t Graph::validate(cudnnHandle_t handle) {
    (void) handle;

    for (auto &node : nodes) {
        switch (node->get_tag()) {
            case Operation::Tag::BN: {
                getLogger() << "[cudnn_frontend] INFO: Adding the batch norm node named " << node->get_name() << std::endl;
                auto batchnorm_node = std::make_shared<BatchNormNode>(node->get_name(), std::static_pointer_cast<Batchnorm>(node), get_context());
                flat_node.sub_nodes.push_back(batchnorm_node);
                break;
            }
            case Operation::Tag::BN_finalize: {
                getLogger() << "[cudnn_frontend] INFO: Adding the batch norm finalize node named " << node->get_name() << std::endl;
                auto bn_finalize_node = std::make_shared<BatchNormFinalizeNode>(node->get_name(), std::static_pointer_cast<BN_finalize>(node), get_context());
                flat_node.sub_nodes.push_back(bn_finalize_node);
                break;
            }
            case Operation::Tag::Conv_fprop: {
                getLogger() << "[cudnn_frontend] INFO: Adding the conv node named " << node->get_name() << std::endl;
                auto conv_node = std::make_shared<ConvolutionNode>(node->get_name(), std::static_pointer_cast<Conv_fprop>(node), get_context());
                flat_node.sub_nodes.push_back(conv_node);
                break;
            }
            case Operation::Tag::Conv_dgrad: {
                getLogger() << "[cudnn_frontend] INFO: Adding the dgrad node named " << node->get_name() << std::endl;
                auto dgrad_node = std::make_shared<DgradNode>(node->get_name(), std::static_pointer_cast<Conv_dgrad>(node), get_context());
                flat_node.sub_nodes.push_back(dgrad_node);
                break;
            }
            case Operation::Tag::Conv_wgrad: {
                getLogger() << "[cudnn_frontend] INFO: Adding the wgrad node named " << node->get_name() << std::endl;
                auto wgrad_node = std::make_shared<WgradNode>(node->get_name(), std::static_pointer_cast<Conv_wgrad>(node), get_context());
                flat_node.sub_nodes.push_back(wgrad_node);
                break;
            }
            case Operation::Tag::DBN_weight: {
                getLogger() << "[cudnn_frontend] INFO: Adding the batch norm finalize node named " << node->get_name() << std::endl;
                auto DBN_weight_node = std::make_shared<DBNWeightNode>(node->get_name(), std::static_pointer_cast<DBN_weight>(node), get_context());
                flat_node.sub_nodes.push_back(DBN_weight_node);
                break;
            }
            case Operation::Tag::Genstats: {
                getLogger() << "[cudnn_frontend] INFO: Adding the genstats node named " << node->get_name() << std::endl;
                auto genstats_node = std::make_shared<GenstatsNode>(node->get_name(), std::static_pointer_cast<Genstats>(node), get_context());
                flat_node.sub_nodes.push_back(genstats_node);
                break;
            }
            case Operation::Tag::Matmul: {
                getLogger() << "[cudnn_frontend] INFO: Adding the matmul node named " << node->get_name() << std::endl;
                auto matmul_node = std::make_shared<MatmulNode>(node->get_name(), std::static_pointer_cast<Matmul>(node), get_context());
                flat_node.sub_nodes.push_back(matmul_node);
                break;
            }
            case Operation::Tag::Pointwise: {
                getLogger() << "[cudnn_frontend] INFO: Adding the pointwise node named " << node->get_name() << std::endl;
                auto pointwise_node = std::make_shared<PointwiseNode>(node->get_name(), std::static_pointer_cast<Pointwise>(node), get_context());
                flat_node.sub_nodes.push_back(pointwise_node);
                break;
            }
            case Operation::Tag::Reduction: {
                getLogger() << "[cudnn_frontend] INFO: Adding the reduction node named " << node->get_name() << std::endl;
                auto reduction_node = std::make_shared<ReductionNode>(node->get_name(), std::static_pointer_cast<Reduction>(node), get_context());
                flat_node.sub_nodes.push_back(reduction_node);
                break;
            }
            case Operation::Tag::Rng: {
                getLogger() << "[cudnn_frontend] INFO: Adding the Rng node named " << node->get_name() << std::endl;
                auto rng_node = std::make_shared<RngNode>(node->get_name(), std::static_pointer_cast<Rng>(node), get_context());
                flat_node.sub_nodes.push_back(rng_node);
                break;
            }
            case Operation::Tag::Scaled_dot_product_attention: {
                getLogger() << "[cudnn_frontend] INFO: Adding the Scaled_dot_product_attention node named " << node->get_name() << std::endl;
                auto scaled_dot_product_attention_node = std::make_shared<ScaledDotProductAttentionNode>(node->get_name(), std::static_pointer_cast<Scaled_dot_product_attention>(node), get_context());
                flat_node.sub_nodes.push_back(scaled_dot_product_attention_node);
                break;
            }
            case Operation::Tag::Softmax:{
                break;
            }
        }
    }

    CHECK_CUDNN_FRONTEND_ERROR(flat_node.infer_properties());
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

inline error_t Graph::execute(cudnnHandle_t handle, std::unordered_map<std::shared_ptr<Tensor>, void *> var_pack) {
    CHECK_CUDNN_FRONTEND_ERROR(flat_node.execute(handle, var_pack));
    
    return error_t::OK;
}

} // namespace graph

} // namespace cudnn_frontend
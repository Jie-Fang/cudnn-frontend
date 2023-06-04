#pragma once

#include <unordered_map>

#include "graphs/cudnn_frontend_node_batchnorm.h"
#include "graphs/cudnn_frontend_node_dbn_weight.h"
#include "graphs/cudnn_frontend_node_bn_finalize.h"
#include "graphs/cudnn_frontend_node_convolution.h"
#include "graphs/cudnn_frontend_node_dgrad.h"
#include "graphs/cudnn_frontend_node_genstats.h"
#include "graphs/cudnn_frontend_node_matmul.h"
#include "graphs/cudnn_frontend_node_pointwise.h"
#include "graphs/cudnn_frontend_node_reduction.h"
#include "graphs/cudnn_frontend_node_scaled_dot_product_attention.h"
#include "graphs/cudnn_frontend_node_wgrad.h"

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
    std::vector<std::shared_ptr<Operation>> nodes;

    int64_t uid_offset = 1;

    FlatNode flat_node{"flat_node", 1};
    detail::Context& get_context();

    error_t run_graph_rules() const;

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

    std::shared_ptr<Tensor> conv(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, Convolution const& conv);
    Convolution::Outputs conv(Convolution::Inputs, Convolution const& conv);
    
    std::array<std::shared_ptr<Tensor>, 5> dbn_weight(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, DBN_weight const&);
    DBN_weight::Outputs dbn_weight(DBN_weight::Inputs, DBN_weight const& dbn_weight);

    std::shared_ptr<Tensor> dgrad(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, Dgrad const& dgrad);
    Dgrad::Outputs dgrad(Dgrad::Inputs, Dgrad const& dgrad);

    std::array<std::shared_ptr<Tensor>, 2> genstats(std::shared_ptr<Tensor>, Genstats const&);
    Genstats::Outputs genstats(Genstats::Inputs, Genstats const&);

    Matmul::Outputs matmul(Matmul::Inputs, Matmul const&);
    std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, Matmul const&);
    
    std::shared_ptr<Tensor> pointwise(std::shared_ptr<Tensor>, Pointwise const& pointwise);
    std::shared_ptr<Tensor> pointwise(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, Pointwise const& pointwise);
    std::shared_ptr<Tensor> pointwise(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, Pointwise const& pointwise);
    Pointwise::Outputs pointwise(Pointwise::Inputs, Pointwise const& pointwise);
    
    Scaled_dot_product_attention::Outputs scaled_dot_product_attention(Scaled_dot_product_attention::Inputs const&, Scaled_dot_product_attention const&);
    
    std::shared_ptr<Tensor> wgrad(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, Wgrad const& wgrad);
    Wgrad::Outputs wgrad(Wgrad::Inputs, Wgrad const& wgrad);

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
        os << *(tensor.second) << ",";
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
    tensors.emplace(tensor.get_name(), tensor_ptr);
    return tensor_ptr;
}

inline BN_finalize::Outputs Graph::bn_finalize(BN_finalize::Inputs inputs, BN_finalize const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<BN_finalize>(user_options);

    // Make required output tensors
    auto EQ_SCALE = std::make_shared<Tensor>(options->get_name() + "_EQ_SCALE_output");
    tensors.emplace(EQ_SCALE->get_name(), EQ_SCALE);
    auto EQ_BIAS = std::make_shared<Tensor>(options->get_name() + "_EQ_BIAS_output");
    tensors.emplace(EQ_BIAS->get_name(), EQ_BIAS);
    auto NEXT_RUNNING_MEAN = std::make_shared<Tensor>(options->get_name() + "_NEXT_RUNNING_MEAN_output");
    tensors.emplace(NEXT_RUNNING_MEAN->get_name(), NEXT_RUNNING_MEAN);
    auto NEXT_RUNNING_VAR = std::make_shared<Tensor>(options->get_name() + "_NEXT_RUNNING_VAR_output");
    tensors.emplace(NEXT_RUNNING_VAR->get_name(), NEXT_RUNNING_VAR);

    // Set outputs
    options->outputs.EQ_SCALE = EQ_SCALE;
    options->outputs.EQ_BIAS = EQ_BIAS;
    options->outputs.NEXT_RUNNING_MEAN = NEXT_RUNNING_MEAN;
    options->outputs.NEXT_RUNNING_VAR = NEXT_RUNNING_VAR;

    // Set inputs
    options->inputs = inputs;

    nodes.emplace_back(options);

    return options->outputs;
}

inline Batchnorm::Outputs Graph::batchnorm(Batchnorm::Inputs inputs, Batchnorm const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Batchnorm>(user_options);

    // Make required output tensors
    auto Y = std::make_shared<Tensor>(options->get_name() + "_Y_output");
    tensors.emplace(Y->get_name(), Y);
    auto MEAN = std::make_shared<Tensor>(options->get_name() + "_MEAN_output");
    tensors.emplace(MEAN->get_name(), MEAN);
    auto INV_VARIANCE = std::make_shared<Tensor>(options->get_name() + "_INV_VARIANCE_output");
    tensors.emplace(INV_VARIANCE->get_name(), INV_VARIANCE);
    auto NEXT_RUNNING_MEAN = std::make_shared<Tensor>(options->get_name() + "_NEXT_RUNNING_MEANoutput");
    tensors.emplace(NEXT_RUNNING_MEAN->get_name(), NEXT_RUNNING_MEAN);
    auto NEXT_RUNNING_VAR = std::make_shared<Tensor>(options->get_name() + "_NEXT_RUNNING_VAR_output");
    tensors.emplace(NEXT_RUNNING_VAR->get_name(), NEXT_RUNNING_VAR);

    // Set outputs
    options->outputs.Y = Y;
    options->outputs.MEAN = MEAN;
    options->outputs.INV_VARIANCE = INV_VARIANCE;
    options->outputs.NEXT_RUNNING_MEAN = NEXT_RUNNING_MEAN;
    options->outputs.NEXT_RUNNING_VAR = NEXT_RUNNING_VAR;

    // Set inputs
    options->inputs = inputs;

    nodes.emplace_back(options);

    return options->outputs;
}

inline std::shared_ptr<Tensor> Graph::conv(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w,Convolution const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Convolution>(user_options);

    // Make required output tensors
    auto Y = std::make_shared<Tensor>(options->get_name() + "_output");
    tensors.emplace(Y->get_name(), Y);
    options->outputs.Y = Y;

    // Set inputs
    options->inputs.X = x;
    options->inputs.W = w;

    nodes.emplace_back(options);

    return Y;
}

inline Convolution::Outputs Graph::conv(Convolution::Inputs inputs, Convolution const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Convolution>(user_options);

    // Make required output tensors
    auto Y = std::make_shared<Tensor>(options->get_name() + "_output");
    tensors.emplace(Y->get_name(), Y);

    // Set outputs
    options->outputs.Y = Y;

    // Set inputs
    options->inputs = inputs;

    nodes.emplace_back(options);

    return options->outputs;
}

inline std::array<std::shared_ptr<Tensor>, 5> Graph::dbn_weight(std::shared_ptr<Tensor> dy, std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> mean, std::shared_ptr<Tensor> inv_variance, std::shared_ptr<Tensor> scale, DBN_weight const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<DBN_weight>(user_options);

    // Make required output tensors
    auto dscale = std::make_shared<Tensor>(options->get_name() + "_dscale_output");
    tensors.emplace(dscale->get_name(), dscale);
    auto dbias = std::make_shared<Tensor>(options->get_name() + "_dbias_output");
    tensors.emplace(dbias->get_name(), dbias);
    auto eq_scale_dy = std::make_shared<Tensor>(options->get_name() + "_eq_scale_dy_output");
    tensors.emplace(eq_scale_dy->get_name(), eq_scale_dy);
    auto eq_scale_x = std::make_shared<Tensor>(options->get_name() + "_eq_scale_xoutput");
    tensors.emplace(eq_scale_x->get_name(), eq_scale_x);
    auto eq_bias = std::make_shared<Tensor>(options->get_name() + "_eq_bias_output");
    tensors.emplace(eq_bias->get_name(), eq_bias);

    // Set outputs
    options->outputs.DSCALE = dscale;
    options->outputs.DBIAS = dbias;
    options->outputs.EQ_SCALE_DY = eq_scale_dy;
    options->outputs.EQ_SCALE_X = eq_scale_x;
    options->outputs.EQ_BIAS = eq_bias;

    // Set inputs
    options->inputs.DY = dy;
    options->inputs.X = x;
    options->inputs.SCALE = scale;
    options->inputs.MEAN = mean;
    options->inputs.INV_VARIANCE = inv_variance;

    nodes.emplace_back(options);

    return {dscale, dbias, eq_scale_dy, eq_scale_x, eq_bias};
}

inline DBN_weight::Outputs Graph::dbn_weight(DBN_weight::Inputs inputs, DBN_weight const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<DBN_weight>(user_options);

    // Make required output tensors
    auto dscale = std::make_shared<Tensor>(options->get_name() + "_dscale_output");
    tensors.emplace(dscale->get_name(), dscale);
    auto dbias = std::make_shared<Tensor>(options->get_name() + "_dbias_output");
    tensors.emplace(dbias->get_name(), dbias);
    auto eq_scale_dy = std::make_shared<Tensor>(options->get_name() + "_eq_scale_dy_output");
    tensors.emplace(eq_scale_dy->get_name(), eq_scale_dy);
    auto eq_scale_x = std::make_shared<Tensor>(options->get_name() + "_eq_scale_xoutput");
    tensors.emplace(eq_scale_x->get_name(), eq_scale_x);
    auto eq_bias = std::make_shared<Tensor>(options->get_name() + "_eq_bias_output");
    tensors.emplace(eq_bias->get_name(), eq_bias);

    // Set outputs
    options->outputs.DSCALE = dscale;
    options->outputs.DBIAS = dbias;
    options->outputs.EQ_SCALE_DY = eq_scale_dy;
    options->outputs.EQ_SCALE_X = eq_scale_x;
    options->outputs.EQ_BIAS = eq_bias;

    // Set inputs
    options->inputs = inputs;

    nodes.emplace_back(options);

    return options->outputs;
}

inline std::shared_ptr<Tensor> Graph::dgrad(std::shared_ptr<Tensor> dy, std::shared_ptr<Tensor> w, Dgrad const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Dgrad>(user_options);

    // Make required output tensors
    auto DX = std::make_shared<Tensor>(options->get_name() + "_output");
    tensors.emplace(DX->get_name(), DX);
    options->outputs.DX = DX;

    // Set inputs
    options->inputs.DY = dy;
    options->inputs.W = w;

    nodes.emplace_back(options);

    return DX;
}

inline Dgrad::Outputs Graph::dgrad(Dgrad::Inputs inputs, Dgrad const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Dgrad>(user_options);

    // Make required output tensors
    auto DX = std::make_shared<Tensor>(options->get_name() + "_output");
    tensors.emplace(DX->get_name(), DX);

    // Set outputs
    options->outputs.DX = DX;

    // Set inputs
    options->inputs = inputs;

    nodes.emplace_back(options);

    return options->outputs;
}

inline std::array<std::shared_ptr<Tensor>, 2> Graph::genstats(std::shared_ptr<Tensor> x, Genstats const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Genstats>(user_options);

    // Make required output tensors
    auto sum = std::make_shared<Tensor>(options->get_name() + "_sum_output");
    tensors.emplace(sum->get_name(), sum);
    auto sq_sum = std::make_shared<Tensor>(options->get_name() + "_sq_sum_output");
    tensors.emplace(sq_sum->get_name(), sq_sum);

    // Set outputs
    options->outputs.SUM = sum;
    options->outputs.SQ_SUM = sq_sum;

    // Set inputs
    options->inputs.X = x;

    nodes.emplace_back(options);

    return {sum, sq_sum};
}

inline Genstats::Outputs Graph::genstats(Genstats::Inputs inputs, Genstats const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Genstats>(user_options);

    // Make required output tensors
    auto sum = std::make_shared<Tensor>(options->get_name() + "_sum_output");
    tensors.emplace(sum->get_name(), sum);
    auto sq_sum = std::make_shared<Tensor>(options->get_name() + "_sq_sum_output");
    tensors.emplace(sq_sum->get_name(), sq_sum);

    // Set outputs
    options->outputs.SUM = sum;
    options->outputs.SQ_SUM = sq_sum;

    // Set inputs
    options->inputs = inputs;

    nodes.emplace_back(options);

    return options->outputs;
}

inline std::shared_ptr<Tensor> Graph::wgrad(std::shared_ptr<Tensor> dy, std::shared_ptr<Tensor> x, Wgrad const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Wgrad>(user_options);

    // Make required output tensors
    auto DW = std::make_shared<Tensor>(options->get_name() + "_output");
    tensors.emplace(DW->get_name(), DW);
    options->outputs.DW = DW;

    // Set inputs
    options->inputs.X = x;
    options->inputs.DY = dy;

    nodes.emplace_back(options);

    return DW;
}

inline Wgrad::Outputs Graph::wgrad(Wgrad::Inputs inputs, Wgrad const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Wgrad>(user_options);

    // Make required output tensors
    auto DW = std::make_shared<Tensor>(options->get_name() + "_output");
    tensors.emplace(DW->get_name(), DW);

    // Set outputs
    options->outputs.DW = DW;

    // Set inputs
    options->inputs = inputs;

    nodes.emplace_back(options);

    return options->outputs;
}

inline std::shared_ptr<Tensor> Graph::pointwise(std::shared_ptr<Tensor> a, Pointwise const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Pointwise>(user_options);

    // Make required output tensors
    auto C = std::make_shared<Tensor>(options->get_name() + "_output");
    tensors.emplace(C->get_name(), C);

    // Set outputs
    options->outputs.OUT_0 = C;

    // Set inputs
    options->inputs.IN_0 = a;

    nodes.emplace_back(options);

    return C;
}

inline std::shared_ptr<Tensor> Graph::pointwise(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, Pointwise const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Pointwise>(user_options);

    // Make required output tensors
    auto C = std::make_shared<Tensor>(options->get_name() + "_output");
    tensors.emplace(C->get_name(), C);

    // Set outputs
    options->outputs.OUT_0 = C;

    // Set inputs
    options->inputs.IN_0 = a;
    options->inputs.IN_1 = b;

    nodes.emplace_back(options);

    return C;
}

inline std::shared_ptr<Tensor> Graph::pointwise(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, std::shared_ptr<Tensor> c, Pointwise const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Pointwise>(user_options);

    // Make required output tensors
    auto C = std::make_shared<Tensor>(options->get_name() + "_output");
    tensors.emplace(C->get_name(), C);

    // Set outputs
    options->outputs.OUT_0 = C;

    // Set inputs
    options->inputs.IN_0 = a;
    options->inputs.IN_1 = b;
    options->inputs.IN_2 = c;

    nodes.emplace_back(options);

    return C;
}

inline Pointwise::Outputs Graph::pointwise(Pointwise::Inputs inputs, Pointwise const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Pointwise>(user_options);

    // Make required output tensors
    auto C = std::make_shared<Tensor>(options->get_name() + "_output");
    tensors.emplace(C->get_name(), C);

    // Set outputs
    options->outputs.OUT_0 = C;

    // Set inputs
    options->inputs = inputs;

    nodes.emplace_back(options);

    return options->outputs;
}

inline std::shared_ptr<Tensor> Graph::matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, Matmul const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Matmul>(user_options);

    // Make required output tensors
    auto C = std::make_shared<Tensor>(options->get_name() + "_output");
    tensors.emplace(C->get_name(), C);

    // Set outputs
    options->outputs.C = C;

    // Set inputs
    options->inputs.A = a;
    options->inputs.B = b;

    nodes.emplace_back(options);

    return C;
}

inline Matmul::Outputs Graph::matmul(Matmul::Inputs inputs, Matmul const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Matmul>(user_options);

    // Make required output tensors
    auto C = std::make_shared<Tensor>(options->get_name() + "_output");
    tensors.emplace(C->get_name(), C);

    // Set outputs
    options->outputs.C = C;

    // Set inputs
    options->inputs = inputs;

    nodes.emplace_back(options);

    return options->outputs;
}

inline Scaled_dot_product_attention::Outputs Graph::scaled_dot_product_attention(Scaled_dot_product_attention::Inputs const& inputs, Scaled_dot_product_attention const& user_options) {

    // Copy over the options from the user
    auto options = std::make_shared<Scaled_dot_product_attention>(user_options);
    
    // Make required output tensors
    auto O = std::make_shared<Tensor>(options->get_name() + "_output");
    tensors.emplace(O->get_name(), O);
    options->outputs.O = O;

    if(options->get_is_inference() == false) {
        auto S = std::make_shared<Tensor>(options->get_name() + "_softmax_output");
        tensors.emplace(S->get_name(), S);
        options->outputs.S = S;
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
        auto tag = nodes[0]->get_tag();
        if (tag != Operation::Tag::Conv &&
            tag != Operation::Tag::BN &&
            tag != Operation::Tag::Pointwise) {return error_t::UNSUPPORTED_GRAPH_FORMAT;}
    }
    // Section 3.3.3 https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#specialized-runtime-fusion-engines
    bool is_first_node = true;
    auto tag = nodes[0]->get_tag();
    switch (tag) {
        case Operation::Tag::BN: {
            auto supported_pattern = {Operation::Tag::BN, Operation::Tag::Pointwise, Operation::Tag::Pointwise, Operation::Tag::Pointwise};
            auto supported_pointwise_pattern = {PointwiseMode_t::ADD, PointwiseMode_t::RELU_FWD, PointwiseMode_t::CMP_GT};
            std::vector<Operation::Tag> actual_pattern = {Operation::Tag::BN};
            std::vector<PointwiseMode_t> actual_pointwise_pattern = {};
            (void) supported_pattern;
            (void) supported_pointwise_pattern;
            for (auto const &node : nodes) {
                if (true == is_first_node) {
                    is_first_node = false;
                    continue;
                }
                auto tag_ = node->get_tag();
                actual_pattern.push_back(tag_);
                if (tag_ == Operation::Tag::Pointwise) {
                    auto pointwise_op = std::static_pointer_cast<cudnn_frontend::graph::Pointwise>(node);
                    actual_pointwise_pattern.push_back(pointwise_op->get_mode().value());
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
    if ((device_version < 800) && ((tag != Operation::Tag::Conv) && (tag != Operation::Tag::Matmul))) {
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


    for (auto const &node : nodes) {
        auto tag_ = node->get_tag();

        switch (state) {
            case Graph_parser_state::G1_PROCESS:
            case Graph_parser_state::G1_PROCESS_POINTWISE: {
                if (tag_ == Operation::Tag::Conv || tag_ == Operation::Tag::Matmul) {
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

    for (auto &node : nodes) {
        switch (node->get_tag()) {
            case Operation::Tag::BN: {
                getLogger() << "[cudnn_frontend] INFO: Adding the batch norm node named " << node->get_name() << std::endl;
                auto batchnorm_node = std::make_shared<BatchNormNode>(node->get_name(), std::static_pointer_cast<Batchnorm>(node), uid_offset);
                batchnorm_node->parent_node = &flat_node;
                flat_node.sub_nodes.push_back(batchnorm_node);
                uid_offset += 100;
                break;
            }
            case Operation::Tag::DBN_weight: {
                getLogger() << "[cudnn_frontend] INFO: Adding the batch norm finalize node named " << node->get_name() << std::endl;
                auto DBN_weight_node = std::make_shared<DBNWeightNode>(node->get_name(), std::static_pointer_cast<DBN_weight>(node), uid_offset);
                DBN_weight_node->parent_node = &flat_node;
                flat_node.sub_nodes.push_back(DBN_weight_node);
                uid_offset += 100;
                break;
            }
            case Operation::Tag::BN_finalize: {
                getLogger() << "[cudnn_frontend] INFO: Adding the batch norm finalize node named " << node->get_name() << std::endl;
                auto bn_finalize_node = std::make_shared<BatchNormFinalizeNode>(node->get_name(), std::static_pointer_cast<BN_finalize>(node), uid_offset);
                bn_finalize_node->parent_node = &flat_node;
                flat_node.sub_nodes.push_back(bn_finalize_node);
                uid_offset += 100;
                break;
            }
            case Operation::Tag::Conv: {
                getLogger() << "[cudnn_frontend] INFO: Adding the conv node named " << node->get_name() << std::endl;
                auto conv_node = std::make_shared<ConvolutionNode>(node->get_name(), std::static_pointer_cast<Convolution>(node), uid_offset);
                conv_node->parent_node = &flat_node;
                flat_node.sub_nodes.push_back(conv_node);
                uid_offset += 100;
                break;
            }
            case Operation::Tag::Dgrad: {
                getLogger() << "[cudnn_frontend] INFO: Adding the dgrad node named " << node->get_name() << std::endl;
                auto dgrad_node = std::make_shared<DgradNode>(node->get_name(), std::static_pointer_cast<Dgrad>(node), uid_offset);
                dgrad_node->parent_node = &flat_node;
                flat_node.sub_nodes.push_back(dgrad_node);
                uid_offset += 100;
                break;
            }
            case Operation::Tag::Genstats: {
                getLogger() << "[cudnn_frontend] INFO: Adding the genstats node named " << node->get_name() << std::endl;
                auto genstats_node = std::make_shared<GenstatsNode>(node->get_name(), std::static_pointer_cast<Genstats>(node), uid_offset);
                genstats_node->parent_node = &flat_node;
                flat_node.sub_nodes.push_back(genstats_node);
                uid_offset += 100;
                break;
            }
            case Operation::Tag::Matmul: {
                getLogger() << "[cudnn_frontend] INFO: Adding the matmul node named " << node->get_name() << std::endl;
                auto matmul_node = std::make_shared<MatMulNode>(node->get_name(), std::static_pointer_cast<Matmul>(node), uid_offset);
                matmul_node->parent_node = &flat_node;
                flat_node.sub_nodes.push_back(matmul_node);
                uid_offset += 100;
                break;
            }
            case Operation::Tag::Pointwise: {
                getLogger() << "[cudnn_frontend] INFO: Adding the pointwise node named " << node->get_name() << std::endl;
                auto pointwise_node = std::make_shared<PointwiseNode>(node->get_name(), std::static_pointer_cast<Pointwise>(node), uid_offset);
                pointwise_node->parent_node = &flat_node;
                flat_node.sub_nodes.push_back(pointwise_node);
                uid_offset += 100;
                break;
            }
            case Operation::Tag::Reduction: {
                getLogger() << "[cudnn_frontend] INFO: Adding the reduction node named " << node->get_name() << std::endl;
                auto reduction_node = std::make_shared<ReductionNode>(node->get_name(), std::static_pointer_cast<Reduction>(node), uid_offset);
                reduction_node->parent_node = &flat_node;
                flat_node.sub_nodes.push_back(reduction_node);
                uid_offset += 100;
                break;
            }
            case Operation::Tag::Scaled_dot_product_attention: {
                getLogger() << "[cudnn_frontend] INFO: Adding the Scaled_dot_product_attention node named " << node->get_name() << std::endl;
                auto scaled_dot_product_attention_node = std::make_shared<ScaledDotProductAttentionNode>(node->get_name(), std::static_pointer_cast<Scaled_dot_product_attention>(node), uid_offset);
                scaled_dot_product_attention_node->parent_node = &flat_node;
                flat_node.sub_nodes.push_back(scaled_dot_product_attention_node);
                uid_offset += 100;
                break;
            }
            case Operation::Tag::Softmax:{break;}
            case Operation::Tag::Wgrad: {
                getLogger() << "[cudnn_frontend] INFO: Adding the wgrad node named " << node->get_name() << std::endl;
                auto wgrad_node = std::make_shared<WgradNode>(node->get_name(), std::static_pointer_cast<Wgrad>(node), uid_offset);
                wgrad_node->parent_node = &flat_node;
                flat_node.sub_nodes.push_back(wgrad_node);
                uid_offset += 100;
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
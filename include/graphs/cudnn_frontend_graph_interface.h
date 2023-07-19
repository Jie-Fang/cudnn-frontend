#pragma once

#include <unordered_map>

#include "graphs/cudnn_frontend_node_batchnorm.h"
#include "graphs/cudnn_frontend_node_bn_finalize.h"
#include "graphs/cudnn_frontend_node_conv_fprop.h"
#include "graphs/cudnn_frontend_node_conv_dgrad.h"
#include "graphs/cudnn_frontend_node_conv_wgrad.h"
#include "graphs/cudnn_frontend_node_dbn.h"
#include "graphs/cudnn_frontend_node_dbn_weight.h"
#include "graphs/cudnn_frontend_node_genstats.h"
#include "graphs/cudnn_frontend_node_matmul.h"
#include "graphs/cudnn_frontend_node_pointwise.h"
#include "graphs/cudnn_frontend_node_reduction.h"
#include "graphs/cudnn_frontend_node_rng.h"
#include "graphs/cudnn_frontend_node_scaled_dot_product_attention.h"
#include "graphs/cudnn_frontend_node_scaled_dot_product_flash_attention.h"

#include "graphs/cudnn_frontend_graph_helpers.h"

namespace cudnn_frontend::graph {

class Plans {
    friend class Graph;
    Execution_plan_list list_of_engine_configs;

    public:

        Execution_plan_list &
        get_engine_configs() {
            return list_of_engine_configs;
        }

        Plans &filter_out_numeric_notes(std::vector<cudnnBackendNumericalNote_t> const &);
        Plans &filter_out_behavior_notes(std::vector<cudnnBackendBehaviorNote_t> const &);
        Plans &filter_out_workspace_greater_than(int64_t const workspace) {
            list_of_engine_configs.set_max_workspace_allowed(workspace);
            return *this;
        }

        error_t build_all_plans(cudnnHandle_t);

        inline error_t 
        check_support(cudnnHandle_t h){
            auto status = list_of_engine_configs.check_support(h);
            return status;
        }

        int64_t get_workspace_size();
        int64_t get_max_workspace_size();

        static error_t
        autotune_default_impl(Plans * plans, cudnnHandle_t handle, std::unordered_map<std::shared_ptr<Tensor_attributes>, void *> variants, void *workspace, void*) {

            auto &execution_plans = plans->get_engine_configs().get_execution_plans();

            // Create the variant pack for all the plans to use.
            std::vector<int64_t> uids;
            std::vector<void*>   ptrs;
            for (auto it : variants) {
                uids.push_back(it.first->get_uid());
                ptrs.push_back(it.second);
            }

            auto variantPack  = VariantPackBuilder()
                .setDataPointers(ptrs.size(), ptrs.data())
                .setUids(uids.size(), uids.data())
                .setWorkspacePointer(workspace)
                .build();

            std::vector<std::shared_ptr<ExecutionPlan>> time_sorted_plans;

            auto plan_cmp = [](std::shared_ptr<ExecutionPlan> a, std::shared_ptr<ExecutionPlan> b) {return a->getExecutionTime() < b->getExecutionTime();};
            std::set<std::shared_ptr<ExecutionPlan>, decltype(plan_cmp)> timed_execution_plans(plan_cmp);

            const int maxIterCount = 100;
            const float threshhold = 0.95f;
            uint64_t successful_plan_count = 0;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaDeviceSynchronize();

            cudaStream_t stream = nullptr;
            cudnnGetStream(handle, &stream);

            for (auto plan : plans->get_engine_configs().get_execution_plans()) {
                float time_ms       = 0.0f;
                float final_time_ms = 0.0f;
                float min_time_ms   = std::numeric_limits<float>::max();

                // Warm-up run
                auto warmup_status = cudnnBackendExecute(handle, plan->get_raw_desc(), variantPack.get_raw_desc());
                if (warmup_status != CUDNN_STATUS_SUCCESS) {
                    getLogger() << "[cudnn_frontend] Plan " << plan->getTag() << " failed with " << to_string(warmup_status) << std::endl;
                    continue;
                }
                successful_plan_count++;
                cudaDeviceSynchronize();

                for (int i = 0; i < maxIterCount; i++) {
                    cudaEventRecord(start, stream);

                    cudnnBackendExecute(handle, plan->get_raw_desc(), variantPack.get_raw_desc());

                    cudaEventRecord(stop, stream);
                    cudaEventSynchronize(stop);
                    cudaEventElapsedTime(&time_ms, start, stop);

                    final_time_ms = std::min(min_time_ms, time_ms);
                    if (time_ms / min_time_ms < threshhold) {
                        min_time_ms = final_time_ms;
                    } else {
                        break;
                    }
                }

                getLogger() << "[cudnn_frontend] Plan " << plan->getTag() << " took " << std::setw(10) << final_time_ms << std::endl;
                plan->setExecutionTime(final_time_ms);
                timed_execution_plans.insert(plan);

            }


            execution_plans.clear();
            for (auto sorted_plan : timed_execution_plans) {
                execution_plans.push_back(sorted_plan);
            }

            cudaEventDestroy(start);
            cudaEventDestroy(stop);

            getLogger() << "Autotuned " << successful_plan_count << " plans." << std::endl;
            return {error_code_t::OK, ""};
        }

        std::function<error_t(Plans *, cudnnHandle_t , std::unordered_map<std::shared_ptr<Tensor_attributes>, void *>, void *, void *)> autotune_impl = &Plans::autotune_default_impl;

        error_t autotune(cudnnHandle_t handle, std::unordered_map<std::shared_ptr<Tensor_attributes>, void *> variants, void *workspace, void *user_impl = nullptr) {
            auto error = autotune_impl(this, handle, variants, workspace, user_impl);
            return error;
        }
};

inline Plans& Plans::filter_out_behavior_notes(std::vector<cudnnBackendBehaviorNote_t> const &notes) {
    // TODO: The error returned is not propagate to user.
    // Should the return value be changed to error_code_t too?
    auto status = list_of_engine_configs.filter_out_behavior_notes(notes);
    if(status.is_bad()) {
        getLogger() << "[cudnn_frontend] ERROR: Filtering by behavioural notes failed." << std::endl; 
    }
    return *this;
}

inline Plans& Plans::filter_out_numeric_notes(std::vector<cudnnBackendNumericalNote_t> const &notes) {
    // TODO: The error returned is not propagate to user.
    // Should the return value be changed to error_code_t too?
    auto status = list_of_engine_configs.filter_out_numeric_notes(notes);
    if(status.is_bad()) {
        getLogger() << "[cudnn_frontend] ERROR: Filtering by numerical notes failed." << std::endl; 
    }
    return *this;
}

inline error_t Plans::build_all_plans(cudnnHandle_t h){
    auto status = list_of_engine_configs.build_all_plans(h);
    return status;
}


inline int64_t Plans::get_max_workspace_size(){
    return list_of_engine_configs.get_max_workspace_size();
}

inline int64_t Plans::get_workspace_size(){
    return list_of_engine_configs.get_workspace_size();
}

class Graph : public INode {
private:
    std::unordered_set<std::shared_ptr<Tensor_attributes>> tensors;

    std::shared_ptr<Tensor_attributes>
    output_tensor(std::string const &name) {
        auto tensor = std::make_shared<Tensor_attributes>();
        tensor->set_name(name);
        tensors.emplace(tensor);
        return tensor;
    }

public:
    Graph(): INode(detail::Context{}) {}

    Type getType() override {
        return Type::COMPOSITE;
    }

    Graph& set_intermediate_data_type(DataType_t type);
    Graph& set_io_data_type(DataType_t type);
    Graph& set_compute_data_type(DataType_t type);
    
    std::shared_ptr<Tensor_attributes> tensor(Tensor_attributes const& tensor);

    Batchnorm_attributes::Outputs batchnorm(Batchnorm_attributes::Inputs, Batchnorm_attributes);

    BN_finalize_attributes::Outputs bn_finalize(BN_finalize_attributes::Inputs, BN_finalize_attributes);

    std::shared_ptr<Tensor_attributes> conv_fprop(std::shared_ptr<Tensor_attributes>, std::shared_ptr<Tensor_attributes>, Conv_fprop_attributes);
    Conv_fprop_attributes::Outputs conv_fprop(Conv_fprop_attributes::Inputs, Conv_fprop_attributes);
    
    std::shared_ptr<Tensor_attributes> conv_dgrad(std::shared_ptr<Tensor_attributes>, std::shared_ptr<Tensor_attributes>, Conv_dgrad_attributes);
    Conv_dgrad_attributes::Outputs conv_dgrad(Conv_dgrad_attributes::Inputs, Conv_dgrad_attributes);

    std::shared_ptr<Tensor_attributes> conv_wgrad(std::shared_ptr<Tensor_attributes>, std::shared_ptr<Tensor_attributes>, Conv_wgrad_attributes);
    Conv_wgrad_attributes::Outputs conv_wgrad(Conv_wgrad_attributes::Inputs, Conv_wgrad_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 5> dbn_weight(std::shared_ptr<Tensor_attributes>, std::shared_ptr<Tensor_attributes>, std::shared_ptr<Tensor_attributes>, std::shared_ptr<Tensor_attributes>, std::shared_ptr<Tensor_attributes>, DBN_weight_attributes);
    DBN_weight_attributes::Outputs dbn_weight(DBN_weight_attributes::Inputs, DBN_weight_attributes);

    DBN_attributes::Outputs batchnorm_backward(DBN_attributes::Inputs, DBN_attributes);

    std::array<std::shared_ptr<Tensor_attributes>, 2> genstats(std::shared_ptr<Tensor_attributes>, Genstats_attributes);
    Genstats_attributes::Outputs genstats(Genstats_attributes::Inputs, Genstats_attributes);

    Matmul_attributes::Outputs matmul(Matmul_attributes::Inputs, Matmul_attributes);
    std::shared_ptr<Tensor_attributes> matmul(std::shared_ptr<Tensor_attributes>, std::shared_ptr<Tensor_attributes>, Matmul_attributes);
    
    std::shared_ptr<Tensor_attributes> pointwise(std::shared_ptr<Tensor_attributes>, Pointwise_attributes);
    std::shared_ptr<Tensor_attributes> pointwise(std::shared_ptr<Tensor_attributes>, std::shared_ptr<Tensor_attributes>, Pointwise_attributes);
    std::shared_ptr<Tensor_attributes> pointwise(std::shared_ptr<Tensor_attributes>, std::shared_ptr<Tensor_attributes>, std::shared_ptr<Tensor_attributes>, Pointwise_attributes);
    Pointwise_attributes::Outputs pointwise(Pointwise_attributes::Inputs, Pointwise_attributes);
    
    Scaled_dot_product_attention_attributes::Outputs scaled_dot_product_attention(std::shared_ptr<Tensor_attributes>, std::shared_ptr<Tensor_attributes>, std::shared_ptr<Tensor_attributes>, Scaled_dot_product_attention_attributes);
    Scaled_dot_product_flash_attention_attributes::Outputs scaled_dot_product_flash_attention(std::shared_ptr<Tensor_attributes>, std::shared_ptr<Tensor_attributes>, std::shared_ptr<Tensor_attributes>, Scaled_dot_product_flash_attention_attributes);

    Plans
    get_execution_plan_list(HeurMode_t mode);

    error_t set_execution_plans(Plans const & plan) {
        if (plan.list_of_engine_configs.get_candidate() == nullptr) {
            return {error_code_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED, "[cudnn_frontend] ERROR: No validate candidate for plan execution"};
        }
        execution_plans.emplace_back(plan.list_of_engine_configs.get_candidate());

        return {error_code_t::OK, ""};
    }
    
    error_t get_engine_configs(Execution_plan_list &plan_list) {
        getLogger() << "[cudnn_frontend] INFO: Extracting engine configs." << std::endl;

        if(engine_configs.size() == 0){
            return {error_code_t::HEURISTIC_QUERY_FAILED, "No valid engine configs for mode_a"};
        }
        plan_list.set_tag(engine_configs.begin()->first);
        plan_list.set_engine_configs(engine_configs.begin()->second);

        getLogger() << "[cudnn_frontend] INFO: Querying engine config properties for cfg_count " << engine_configs.begin()->second.size() << std::endl;
        CHECK_CUDNN_FRONTEND_ERROR(plan_list.query_properties());

        return {error_code_t::OK, ""};
    }

    error_t createOperationGraphs(cudnnHandle_t handle) override final {
        getLogger() << "Operation Graph has " << operations.size() << " operations." << std::endl;

        auto status = create_cudnn_operation_graphs(handle);
        if(status.is_bad()) {
            getLogger() << "[cudnn_frontend] ERROR: " << status.get_code() << " Failed to create execution plans for graph partitioning in FlatNode." << std::endl;
            return status;
        }

        return {error_code_t::OK, ""};
    }

};

inline Plans Graph::get_execution_plan_list(HeurMode_t mode) {
    Plans plan_list;
    // TODO: The error returned is not propagate to user.
    // Should the return value be changed to error_code_t too?
   
    auto status = query_heuristics(mode);
    if(status.is_bad()) {
        getLogger() << "[cudnn_frontend] ERROR: Failed to build in " << name << std::endl;
        return plan_list;
    }

    status = get_engine_configs(plan_list.list_of_engine_configs);
    if(status.is_bad()) {
        getLogger() << "[cudnn_frontend] ERROR: Querying engine configs failed." << std::endl; 
    }
    return plan_list;
}

inline Graph& Graph::set_intermediate_data_type(DataType_t const type) {
    context.set_intermediate_data_type(type);
    return *this;
}

inline Graph& Graph::set_io_data_type(DataType_t const type) {
    context.set_io_data_type(type);
    return *this;
}

inline Graph& Graph::set_compute_data_type(DataType_t const type) {
    context.set_compute_data_type(type);
    return *this;
}

inline std::shared_ptr<Tensor_attributes> Graph::tensor(Tensor_attributes const& tensor) {
    auto tensor_ptr = std::make_shared<Tensor_attributes>(tensor);
    tensors.emplace(tensor_ptr);
    return tensor_ptr;
}

inline BN_finalize_attributes::Outputs Graph::bn_finalize(BN_finalize_attributes::Inputs inputs, BN_finalize_attributes options) {
    // Set outputs
    options.make_outputs([this](std::string const &name){return output_tensor(name);});
    auto return_outputs = options.outputs;

    // Set inputs
    options.inputs = inputs;

    sub_nodes.emplace_back(std::make_unique<BatchNormFinalizeNode>(std::move(options), context));

    return return_outputs;
}

inline Batchnorm_attributes::Outputs Graph::batchnorm(Batchnorm_attributes::Inputs inputs, Batchnorm_attributes options) {
    
    // Set outputs
    options.make_outputs([this](std::string const &name){return output_tensor(name);});
    auto return_outputs = options.outputs;

    // Set inputs
    options.inputs.X = inputs.X;
    options.inputs.SCALE = inputs.SCALE;
    options.inputs.BIAS = inputs.BIAS;
    options.inputs.PREV_RUNNING_MEAN = inputs.PREV_RUNNING_MEAN;
    options.inputs.PREV_RUNNING_VAR = inputs.PREV_RUNNING_VAR;

    sub_nodes.emplace_back(std::make_unique<BatchNormNode>(std::move(options), context));

    return return_outputs;
}

inline DBN_attributes::Outputs Graph::batchnorm_backward(DBN_attributes::Inputs inputs, DBN_attributes options) {
    
    // Set outputs
    options.make_outputs([this](std::string const &name){return output_tensor(name);});
    auto return_outputs = options.outputs;

    // Set inputs
    options.inputs = inputs;

    sub_nodes.emplace_back(std::make_unique<DBNNode>(std::move(options), context));

    return return_outputs;
}

inline std::shared_ptr<Tensor_attributes> Graph::conv_fprop(std::shared_ptr<Tensor_attributes> x, std::shared_ptr<Tensor_attributes> w, Conv_fprop_attributes options) {

    // Make required output tensors 
    auto Y = output_tensor(options.get_name() + "_output");
    options.outputs.Y = Y;

    // Set inputs
    options.inputs.X = x;
    options.inputs.W = w;

    sub_nodes.emplace_back(std::make_unique<ConvolutionNode>(std::move(options), context));

    return Y;
}

inline Conv_fprop_attributes::Outputs Graph::conv_fprop(Conv_fprop_attributes::Inputs inputs, Conv_fprop_attributes options) {
    // Set outputs
    auto Y = options.outputs.Y = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs = inputs;

    sub_nodes.emplace_back(std::make_unique<ConvolutionNode>(std::move(options), context));

    return Conv_fprop_attributes::Outputs{Y};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 5> Graph::dbn_weight(std::shared_ptr<Tensor_attributes> dy, std::shared_ptr<Tensor_attributes> x, std::shared_ptr<Tensor_attributes> mean, std::shared_ptr<Tensor_attributes> inv_variance, std::shared_ptr<Tensor_attributes> scale, DBN_weight_attributes options) {
    // Make required output tensors
    options.make_outputs([this](std::string const &name){return output_tensor(name);});
    auto return_outputs = options.outputs;

    // Set inputs
    options.inputs.DY = dy;
    options.inputs.X = x;
    options.inputs.SCALE = scale;
    options.inputs.MEAN = mean;
    options.inputs.INV_VARIANCE = inv_variance;

    sub_nodes.emplace_back(std::make_unique<DBNWeightNode>(std::move(options), context));

    return {return_outputs.DSCALE, return_outputs.DBIAS, return_outputs.EQ_SCALE_DY, return_outputs.EQ_SCALE_X, return_outputs.EQ_BIAS};
}

inline DBN_weight_attributes::Outputs Graph::dbn_weight(DBN_weight_attributes::Inputs inputs, DBN_weight_attributes options) {
    // Make required output tensors
    options.make_outputs([this](std::string const &name){return output_tensor(name);});
    auto return_outputs = options.outputs;

    // Set inputs
    options.inputs = inputs;

    sub_nodes.emplace_back(std::make_unique<DBNWeightNode>(std::move(options), context));

    return return_outputs;
}

inline std::shared_ptr<Tensor_attributes> Graph::conv_dgrad(std::shared_ptr<Tensor_attributes> dy, std::shared_ptr<Tensor_attributes> w, Conv_dgrad_attributes options) {
    // Make required output tensors
    auto DX = options.outputs.DX = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs.DY = dy;
    options.inputs.W = w;

    sub_nodes.emplace_back(std::make_unique<DgradNode>(std::move(options), context));

    return DX;
}

inline Conv_dgrad_attributes::Outputs Graph::conv_dgrad(Conv_dgrad_attributes::Inputs inputs, Conv_dgrad_attributes options) {
    // Make required output tensors
    auto DX = options.outputs.DX = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs = inputs;

    sub_nodes.emplace_back(std::make_unique<DgradNode>(std::move(options), context));

    return Conv_dgrad_attributes::Outputs{DX};
}

inline std::array<std::shared_ptr<Tensor_attributes>, 2> Graph::genstats(std::shared_ptr<Tensor_attributes> x, Genstats_attributes options) {
    // Set outputs
    auto SUM = options.outputs.SUM = output_tensor(options.get_name() + "_sum_output");
    auto SQ_SUM = options.outputs.SQ_SUM = output_tensor(options.get_name() + "_sq_sum_output");

    // Set inputs
    options.inputs.X = x;

    sub_nodes.emplace_back(std::make_unique<GenstatsNode>(std::move(options), context));

    return {SUM, SQ_SUM};
}

inline Genstats_attributes::Outputs Graph::genstats(Genstats_attributes::Inputs inputs, Genstats_attributes options) {
    // Make required output tensors
    auto SUM = options.outputs.SUM = output_tensor(options.get_name() + "_sum_output");
    auto SQ_SUM = options.outputs.SQ_SUM = output_tensor(options.get_name() + "_sq_sum_output");

    // Set inputs
    options.inputs = inputs;

    sub_nodes.emplace_back(std::make_unique<GenstatsNode>(std::move(options), context));

    return Genstats_attributes::Outputs{SUM,SQ_SUM};
}

inline std::shared_ptr<Tensor_attributes> Graph::conv_wgrad(std::shared_ptr<Tensor_attributes> dy, std::shared_ptr<Tensor_attributes> x, Conv_wgrad_attributes options) {
    // Make required output tensors
    auto DW = options.outputs.DW = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs.X = x;
    options.inputs.DY = dy;

    sub_nodes.emplace_back(std::make_unique<WgradNode>(std::move(options), context));

    return DW;
}

inline Conv_wgrad_attributes::Outputs Graph::conv_wgrad(Conv_wgrad_attributes::Inputs inputs, Conv_wgrad_attributes options) {
    // Make required output tensors
    auto DW = options.outputs.DW = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs = inputs;

    sub_nodes.emplace_back(std::make_unique<WgradNode>(std::move(options), context));

    return Conv_wgrad_attributes::Outputs{DW};
}

inline std::shared_ptr<Tensor_attributes> Graph::pointwise(std::shared_ptr<Tensor_attributes> a, Pointwise_attributes options) {
    auto OUT_0 = options.outputs.OUT_0 = output_tensor(options.get_name() + "_output");
    
    // Set inputs
    options.inputs.IN_0 = a;

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(options), context));

    return OUT_0;
}

inline std::shared_ptr<Tensor_attributes> Graph::pointwise(std::shared_ptr<Tensor_attributes> a, std::shared_ptr<Tensor_attributes> b, Pointwise_attributes options) {
    auto OUT_0 = options.outputs.OUT_0 = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs.IN_0 = a;
    options.inputs.IN_1 = b;

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(options), context));

    return OUT_0;
}

inline std::shared_ptr<Tensor_attributes> Graph::pointwise(std::shared_ptr<Tensor_attributes> a, std::shared_ptr<Tensor_attributes> b, std::shared_ptr<Tensor_attributes> c, Pointwise_attributes options) {
    auto OUT_0 = options.outputs.OUT_0 = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs.IN_0 = a;
    options.inputs.IN_1 = b;
    options.inputs.IN_2 = c;

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(options), context));

    return OUT_0;
}

inline Pointwise_attributes::Outputs Graph::pointwise(Pointwise_attributes::Inputs inputs, Pointwise_attributes options) {
    auto OUT_0 = options.outputs.OUT_0 = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs = inputs;

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(std::move(options), context));

    return Pointwise_attributes::Outputs{OUT_0};
}

inline std::shared_ptr<Tensor_attributes> Graph::matmul(std::shared_ptr<Tensor_attributes> a, std::shared_ptr<Tensor_attributes> b, Matmul_attributes options) {
    auto C = options.outputs.C = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs.A = a;
    options.inputs.B = b;

    sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(options), context));

    return C;
}

inline Matmul_attributes::Outputs Graph::matmul(Matmul_attributes::Inputs inputs, Matmul_attributes options) {
    auto C = options.outputs.C = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs = inputs;

    sub_nodes.emplace_back(std::make_unique<MatmulNode>(std::move(options), context));

    return Matmul_attributes::Outputs{C};
}

inline Scaled_dot_product_attention_attributes::Outputs Graph::scaled_dot_product_attention(std::shared_ptr<Tensor_attributes> q, std::shared_ptr<Tensor_attributes> k, std::shared_ptr<Tensor_attributes> v, Scaled_dot_product_attention_attributes options) {    
    // Make required output tensors
    auto O = options.outputs.O = output_tensor(options.get_name() + "_output");
    auto S = options.outputs.S = output_tensor(options.get_name() + "_softmax_output");

    // Set inputs
    options.inputs.Q = q;
    options.inputs.K = k;
    options.inputs.V = v;

    sub_nodes.emplace_back(std::make_unique<ScaledDotProductAttentionNode>(std::move(options), context));

    return Scaled_dot_product_attention_attributes::Outputs{O, S};
}

inline Scaled_dot_product_flash_attention_attributes::Outputs Graph::scaled_dot_product_flash_attention(std::shared_ptr<Tensor_attributes> q, std::shared_ptr<Tensor_attributes> k, std::shared_ptr<Tensor_attributes> v, Scaled_dot_product_flash_attention_attributes options) {    
    // Make required output tensors
    auto O = options.outputs.O = output_tensor(options.get_name() + "::O");
    auto Stats = options.outputs.Stats = output_tensor(options.get_name() + "::Stats");

    // Set inputs
    options.inputs.Q = q;
    options.inputs.K = k;
    options.inputs.V = v;

    sub_nodes.emplace_back(std::make_unique<ScaledDotProductFlashAttentionNode>(std::move(options), context));

    return Scaled_dot_product_flash_attention_attributes::Outputs{O, Stats};
}

} // namespace cudnn_frontend::graph
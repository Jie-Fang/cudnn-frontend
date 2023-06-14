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
        Plans &build_plans(cudnnHandle_t);

        int64_t get_workspace_size();
        int64_t get_max_workspace_size();

        static error_t
        autotune_default_impl(Plans * plans, cudnnHandle_t handle, std::unordered_map<std::shared_ptr<Tensor>, void *> variants, void *workspace, void*) {

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
            return error_t::OK;
        }

        std::function<error_t(Plans *, cudnnHandle_t , std::unordered_map<std::shared_ptr<Tensor>, void *>, void *, void *)> autotune_impl = &Plans::autotune_default_impl;

        error_t autotune(cudnnHandle_t handle, std::unordered_map<std::shared_ptr<Tensor>, void *> variants, void *workspace, void *user_impl = nullptr) {
            auto error = autotune_impl(this, handle, variants, workspace, user_impl);
            return error;
        }
};

inline Plans& Plans::filter_out_behavior_notes(std::vector<cudnnBackendBehaviorNote_t> const &notes) {
    // TODO: The error returned is not propagate to user.
    // Should the return value be changed to error_t too?
    auto status = list_of_engine_configs.filter_out_behavior_notes(notes);
    if(status != error_t::OK) {
        getLogger() << "[cudnn_frontend] ERROR: Filtering by behavioural notes failed." << std::endl; 
    }
    return *this;
}

inline Plans& Plans::filter_out_numeric_notes(std::vector<cudnnBackendNumericalNote_t> const &notes) {
    // TODO: The error returned is not propagate to user.
    // Should the return value be changed to error_t too?
    auto status = list_of_engine_configs.filter_out_numeric_notes(notes);
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

class Graph : public INode {
private:
    std::unordered_set<std::shared_ptr<Tensor>> tensors;

    std::shared_ptr<Tensor>
    output_tensor(std::string const &name) {
        auto tensor = std::make_shared<Tensor>(name);
        tensors.emplace(tensor);
        return tensor;
    }

    std::vector<std::string> operation_names;
public:
    Graph(std::string const& name): INode(name, detail::Context{}) {}

    Type getType() override {
        return Type::COMPOSITE;
    }

    Graph& set_intermediate_data_type(DataType_t type);
    Graph& set_io_data_type(DataType_t type);
    Graph& set_compute_data_type(DataType_t type);
    
    std::shared_ptr<Tensor> tensor(Tensor const& tensor);

    Batchnorm::Outputs batchnorm(Batchnorm::Inputs, Batchnorm);

    BN_finalize::Outputs bn_finalize(BN_finalize::Inputs, BN_finalize);

    std::shared_ptr<Tensor> conv_fprop(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, Conv_fprop);
    Conv_fprop::Outputs conv_fprop(Conv_fprop::Inputs, Conv_fprop);
    
    std::shared_ptr<Tensor> conv_dgrad(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, Conv_dgrad);
    Conv_dgrad::Outputs conv_dgrad(Conv_dgrad::Inputs, Conv_dgrad);

    std::shared_ptr<Tensor> conv_wgrad(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, Conv_wgrad);
    Conv_wgrad::Outputs conv_wgrad(Conv_wgrad::Inputs, Conv_wgrad);

    std::array<std::shared_ptr<Tensor>, 5> dbn_weight(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, DBN_weight);
    DBN_weight::Outputs dbn_weight(DBN_weight::Inputs, DBN_weight);

    std::array<std::shared_ptr<Tensor>, 2> genstats(std::shared_ptr<Tensor>, Genstats);
    Genstats::Outputs genstats(Genstats::Inputs, Genstats);

    Matmul::Outputs matmul(Matmul::Inputs, Matmul);
    std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, Matmul);
    
    std::shared_ptr<Tensor> pointwise(std::shared_ptr<Tensor>, Pointwise);
    std::shared_ptr<Tensor> pointwise(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, Pointwise);
    std::shared_ptr<Tensor> pointwise(std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, Pointwise);
    Pointwise::Outputs pointwise(Pointwise::Inputs, Pointwise);
    
    Scaled_dot_product_attention::Outputs scaled_dot_product_attention(Scaled_dot_product_attention::Inputs, Scaled_dot_product_attention);
    Scaled_dot_product_flash_attention::Outputs scaled_dot_product_flash_attention(Scaled_dot_product_flash_attention::Inputs, Scaled_dot_product_flash_attention);

    error_t is_supported_node() override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Checking Graph Support..." << std::endl;
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

        auto const entrance_node_tag = sub_nodes.front()->getType();

        if (sub_nodes.size() == 1) {
            // Only contains checks for
            // Section 3.3.1 https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#compile-single-op-engine
            std::unordered_set<INode::Type> const supported_tags = {
                                                    INode::Type::CONVOLUTION
                                                    , INode::Type::WGRAD
                                                    , INode::Type::DGRAD
                                                    , INode::Type::BATCHNORM
                                                    , INode::Type::POINTWISE
                                                };
            RETURN_CUDNN_FRONTEND_ERROR_IF(supported_tags.find(entrance_node_tag) != supported_tags.end(), error_t::OK);
        }
        
        // Only contains checks for
        // Section 3.3.3 https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#specialized-runtime-fusion-engines
        switch (entrance_node_tag) {
            case INode::Type::BATCHNORM: {
                // Only contains checks for Section 3.3.3.1
                auto const& bn_options = static_cast<BatchNormNode*>(sub_nodes.front().get())->options;

                // The pointwise nodes: Add, ReLU, and GT (greater than) are optional.
                if(std::any_of(std::next(sub_nodes.begin(), 1), sub_nodes.end(), [](auto const& node) {return node->getType() != INode::Type::POINTWISE;})) {
                    break;
                }

                auto pattern = {PointwiseMode_t::ADD, PointwiseMode_t::RELU_FWD, PointwiseMode_t::CMP_GT};
                std::vector<PointwiseMode_t> actual_pattern = {};
                for (auto itr = std::next(sub_nodes.begin(), 1); itr != sub_nodes.end(); itr++) {
                    auto const& pointwise_options = static_cast<PointwiseNode*>(sub_nodes.front().get())->options;
                    actual_pattern.push_back(pointwise_options.get_mode().value());
                }
                if(!std::includes(pattern.begin(), pattern.end(), actual_pattern.begin(), actual_pattern.end())) {
                    break;
                }

                // The attribute CUDNN_ATTR_OPERATION_NORM_FWD_MODE for the norm forward operation must be set to CUDNN_BATCH_NORM.
                // Hardcoded inside operation

                // The attribute CUDNN_ATTR_OPERATION_NORM_FWD_PHASE for the norm forward operation must be set to CUDNN_NORM_FWD_TRAINING.
                if(bn_options.get_forward_phase() != NormFwdPhase_t::TRAINING) {
                    break;
                }

                // For FP16 and BF16 data types, the channel count C for the tensors must be a multiple of 8 while
                // for float data type the channel count must be a multiple of 4.
                auto const& X = bn_options.inputs.X;
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
            case INode::Type::SCALED_DOT_PRODUCT_ATTENTION: {
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
        if ((device_version < 80) && ((entrance_node_tag != INode::Type::CONVOLUTION) && (entrance_node_tag != INode::Type::MATMUL))) {
            getLogger() << "Device version insufficient" << std::endl;
            return error_t::UNSUPPORTED_GRAPH_FORMAT;
        }

        std::set<INode::Type> g1_supported_pattern = {/*Operation::Tag::Concat, Operation::Tag::Signal */INode::Type::POINTWISE};
        std::set<INode::Type> g2_supported_pattern = { /*Operation::Tag::ResampleFwd,
                                                            Operation::Tag::ResampleBwd*/
                                                            INode::Type::GENSTATS,
                                                            INode::Type::REDUCTION,
                                                            INode::Type::POINTWISE};

        enum class Graph_parser_state {
            G1_PROCESS,
            G1_PROCESS_POINTWISE,
            G2_START,
            G2_PROCESS_REDUCTION_NODE_SEEN,
            G2_PROCESS_SIGNAL_NODE_SEEN,
        };

        Graph_parser_state state = Graph_parser_state::G1_PROCESS;
        is_supported = true;


        for (auto const& node : sub_nodes) {
            auto tag_ = node->getType();

            switch (state) {
                case Graph_parser_state::G1_PROCESS:
                case Graph_parser_state::G1_PROCESS_POINTWISE: {
                    if (tag_ == INode::Type::CONVOLUTION || tag_ == INode::Type::MATMUL) {
                        state = Graph_parser_state::G2_START;
                    // } else if ((tag_ == Operation::Tag::Resample_Fwd) || (tag_ == Operation::Tag::Resample_Bwd)) {
                    //     actual_g2_pattern.push_back(tag_);
                    //     state = Graph_parser_state::G2_START;
                    } else {
                        // G1 nodes has to be one of concat / signal / pointwise
                        if (g1_supported_pattern.find(tag_) == g1_supported_pattern.end()) {is_supported = false;}

                        if (tag_ == INode::Type::POINTWISE) {state = Graph_parser_state::G1_PROCESS_POINTWISE;}
                        // if G1 encounters pointwise, no other operation can be succeed
                        if ((state == Graph_parser_state::G1_PROCESS_POINTWISE)  && (tag_ != INode::Type::POINTWISE)) {
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

    Plans
    get_execution_plan_list(HeurMode_t mode);

    error_t set_executor(Plans const & plan) {
        if (plan.list_of_engine_configs.get_candidate() == nullptr) {
            return error_t::GRAPH_EXECUTION_PLAN_CREATION_FAILED;
        }
        execution_plans.emplace_back(plan.list_of_engine_configs.get_candidate());

        return error_t::OK;
    }
    
    error_t get_engine_configs(HeurMode_t mode, Execution_plan_list &plan_list) {
        getLogger() << "[cudnn_frontend] INFO: Extracting engine configs." << std::endl;

        switch (mode) {
        case HeurMode_t::HEUR_MODE_A:
            if(mode_a_engine_configs.size() == 0){return error_t::HEURISTIC_QUERY_FAILED;}
            plan_list.set_tag(mode_a_engine_configs.begin()->first);
            plan_list.set_engine_configs(mode_a_engine_configs.begin()->second);
            break;
        case HeurMode_t::HEUR_MODE_B:
            if(mode_b_engine_configs.size() == 0){return error_t::HEURISTIC_QUERY_FAILED;}
            plan_list.set_tag(mode_b_engine_configs.begin()->first);
            plan_list.set_engine_configs(mode_b_engine_configs.begin()->second);
            break;
        case HeurMode_t::HEUR_MODE_FALLBACK:
            if(fallback_engine_configs.size() == 0){return error_t::HEURISTIC_QUERY_FAILED;}
            plan_list.set_tag(fallback_engine_configs.begin()->first);
            plan_list.set_engine_configs(fallback_engine_configs.begin()->second);
            break;
        }

        getLogger() << "[cudnn_frontend] INFO: Querying engine config properties." << std::endl;
        CHECK_CUDNN_FRONTEND_ERROR(plan_list.query_properties());

        return error_t::OK;
    }

    error_t createOperationGraphs(cudnnHandle_t handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partitioning FlatNode..." << std::endl;

        // Currently just make one large graph of operations from all sub nodes.
        for (auto const& node : sub_nodes) {
            getLogger() << "Getting the operation from " << name << std::endl;
            for (auto &operation : node->get_operations()) {
                operation_names.push_back(operation.first);
            }
        }

        getLogger() << "Operation Graph has " << operation_names.size() << " operations." << std::endl;

        auto status = create_cudnn_operation_graphs(handle, {operation_names});
        if(status != error_t::OK) {
            getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to create execution plans for graph partitioning in FlatNode." << std::endl;
            return status;
        }

        getLogger() << "[cudnn_frontend] INFO: Partitioned FlatNode." << std::endl;
        return error_t::OK;
    }

    error_t createExecutionPlans(cudnnHandle_t handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Creating Execution Plans..." << std::endl;

        (void)handle;
        // auto status = create_cudnn_execution_plan(handle);
        // if(status != error_t::OK) {
        //     getLogger() << "[cudnn_frontend] ERROR: " << status << " Failed to create execution plans for graph partitioning in FlatNode." << std::endl;
        //     return status;
        // }

        getLogger() << "[cudnn_frontend] INFO: Created Execution Plans." << std::endl;
        return error_t::OK;
    }
};

inline Plans Graph::get_execution_plan_list(HeurMode_t mode) {
    Plans plan_list;
    // TODO: The error returned is not propagate to user.
    // Should the return value be changed to error_t too?
    auto status = get_engine_configs(mode, plan_list.list_of_engine_configs);
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

inline std::shared_ptr<Tensor> Graph::tensor(Tensor const& tensor) {
    auto tensor_ptr = std::make_shared<Tensor>(tensor);
    tensors.emplace(tensor_ptr);
    return tensor_ptr;
}

inline BN_finalize::Outputs Graph::bn_finalize(BN_finalize::Inputs inputs, BN_finalize options) {
    // Set outputs
    options.make_outputs([this](std::string const &name){return output_tensor(name);});
    auto return_outputs = options.outputs;

    // Set inputs
    options.inputs = inputs;

    sub_nodes.emplace_back(std::make_unique<BatchNormFinalizeNode>(options.get_name(), std::move(options), get_context()));

    return return_outputs;
}

inline Batchnorm::Outputs Graph::batchnorm(Batchnorm::Inputs inputs, Batchnorm options) {
    
    // Set outputs
    options.make_outputs([this](std::string const &name){return output_tensor(name);});
    auto return_outputs = options.outputs;

    // Set inputs
    options.inputs = inputs;

    sub_nodes.emplace_back(std::make_unique<BatchNormNode>(options.get_name(), std::move(options), get_context()));

    return return_outputs;
}

inline std::shared_ptr<Tensor> Graph::conv_fprop(std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> w, Conv_fprop options) {

    // Make required output tensors 
    auto Y = output_tensor(options.get_name() + "_output");
    options.outputs.Y = Y;

    // Set inputs
    options.inputs.X = x;
    options.inputs.W = w;

    sub_nodes.emplace_back(std::make_unique<ConvolutionNode>(options.get_name(), std::move(options), get_context()));

    return Y;
}

inline Conv_fprop::Outputs Graph::conv_fprop(Conv_fprop::Inputs inputs, Conv_fprop options) {
    // Set outputs
    auto Y = options.outputs.Y = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs = inputs;

    sub_nodes.emplace_back(std::make_unique<ConvolutionNode>(options.get_name(), std::move(options), get_context()));

    return {.Y = Y};
}

inline std::array<std::shared_ptr<Tensor>, 5> Graph::dbn_weight(std::shared_ptr<Tensor> dy, std::shared_ptr<Tensor> x, std::shared_ptr<Tensor> mean, std::shared_ptr<Tensor> inv_variance, std::shared_ptr<Tensor> scale, DBN_weight options) {
    // Make required output tensors
    options.make_outputs([this](std::string const &name){return output_tensor(name);});
    auto return_outputs = options.outputs;

    // Set inputs
    options.inputs.DY = dy;
    options.inputs.X = x;
    options.inputs.SCALE = scale;
    options.inputs.MEAN = mean;
    options.inputs.INV_VARIANCE = inv_variance;

    sub_nodes.emplace_back(std::make_unique<DBNWeightNode>(options.get_name(), std::move(options), get_context()));

    return {return_outputs.DSCALE, return_outputs.DBIAS, return_outputs.EQ_SCALE_DY, return_outputs.EQ_SCALE_X, return_outputs.EQ_BIAS};
}

inline DBN_weight::Outputs Graph::dbn_weight(DBN_weight::Inputs inputs, DBN_weight options) {
    // Make required output tensors
    options.make_outputs([this](std::string const &name){return output_tensor(name);});
    auto return_outputs = options.outputs;

    // Set inputs
    options.inputs = inputs;

    sub_nodes.emplace_back(std::make_unique<DBNWeightNode>(options.get_name(), std::move(options), get_context()));

    return return_outputs;
}

inline std::shared_ptr<Tensor> Graph::conv_dgrad(std::shared_ptr<Tensor> dy, std::shared_ptr<Tensor> w, Conv_dgrad options) {
    // Make required output tensors
    auto DX = options.outputs.DX = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs.DY = dy;
    options.inputs.W = w;

    sub_nodes.emplace_back(std::make_unique<DgradNode>(options.get_name(), std::move(options), get_context()));

    return DX;
}

inline Conv_dgrad::Outputs Graph::conv_dgrad(Conv_dgrad::Inputs inputs, Conv_dgrad options) {
    // Make required output tensors
    auto DX = options.outputs.DX = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs = inputs;

    sub_nodes.emplace_back(std::make_unique<DgradNode>(options.get_name(), std::move(options), get_context()));

    return {.DX = DX};
}

inline std::array<std::shared_ptr<Tensor>, 2> Graph::genstats(std::shared_ptr<Tensor> x, Genstats options) {
    // Set outputs
    auto SUM = options.outputs.SUM = output_tensor(options.get_name() + "_sum_output");
    auto SQ_SUM = options.outputs.SQ_SUM = output_tensor(options.get_name() + "_sq_sum_output");

    // Set inputs
    options.inputs.X = x;

    sub_nodes.emplace_back(std::make_unique<GenstatsNode>(options.get_name(), std::move(options), get_context()));

    return {SUM, SQ_SUM};
}

inline Genstats::Outputs Graph::genstats(Genstats::Inputs inputs, Genstats options) {
    // Make required output tensors
    auto SUM = options.outputs.SUM = output_tensor(options.get_name() + "_sum_output");
    auto SQ_SUM = options.outputs.SQ_SUM = output_tensor(options.get_name() + "_sq_sum_output");

    // Set inputs
    options.inputs = inputs;

    sub_nodes.emplace_back(std::make_unique<GenstatsNode>(options.get_name(), std::move(options), get_context()));

    return {.SUM = SUM, .SQ_SUM = SQ_SUM};
}

inline std::shared_ptr<Tensor> Graph::conv_wgrad(std::shared_ptr<Tensor> dy, std::shared_ptr<Tensor> x, Conv_wgrad options) {
    // Make required output tensors
    auto DW = options.outputs.DW = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs.X = x;
    options.inputs.DY = dy;

    sub_nodes.emplace_back(std::make_unique<WgradNode>(options.get_name(), std::move(options), get_context()));

    return DW;
}

inline Conv_wgrad::Outputs Graph::conv_wgrad(Conv_wgrad::Inputs inputs, Conv_wgrad options) {
    // Make required output tensors
    auto DW = options.outputs.DW = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs = inputs;

    sub_nodes.emplace_back(std::make_unique<WgradNode>(options.get_name(), std::move(options), get_context()));

    return {.DW = DW};
}

inline std::shared_ptr<Tensor> Graph::pointwise(std::shared_ptr<Tensor> a, Pointwise options) {
    auto OUT_0 = options.outputs.OUT_0 = output_tensor(options.get_name() + "_output");
    
    // Set inputs
    options.inputs.IN_0 = a;

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(options.get_name(), std::move(options), get_context()));

    return OUT_0;
}

inline std::shared_ptr<Tensor> Graph::pointwise(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, Pointwise options) {
    auto OUT_0 = options.outputs.OUT_0 = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs.IN_0 = a;
    options.inputs.IN_1 = b;

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(options.get_name(), std::move(options), get_context()));

    return OUT_0;
}

inline std::shared_ptr<Tensor> Graph::pointwise(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, std::shared_ptr<Tensor> c, Pointwise options) {
    auto OUT_0 = options.outputs.OUT_0 = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs.IN_0 = a;
    options.inputs.IN_1 = b;
    options.inputs.IN_2 = c;

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(options.get_name(), std::move(options), get_context()));

    return OUT_0;
}

inline Pointwise::Outputs Graph::pointwise(Pointwise::Inputs inputs, Pointwise options) {
    auto OUT_0 = options.outputs.OUT_0 = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs = inputs;

    sub_nodes.emplace_back(std::make_unique<PointwiseNode>(options.get_name(), std::move(options), get_context()));

    return {.OUT_0 = OUT_0};
}

inline std::shared_ptr<Tensor> Graph::matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b, Matmul options) {
    auto C = options.outputs.C = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs.A = a;
    options.inputs.B = b;

    sub_nodes.emplace_back(std::make_unique<MatmulNode>(options.get_name(), std::move(options), get_context()));

    return C;
}

inline Matmul::Outputs Graph::matmul(Matmul::Inputs inputs, Matmul options) {
    auto C = options.outputs.C = output_tensor(options.get_name() + "_output");

    // Set inputs
    options.inputs = inputs;

    sub_nodes.emplace_back(std::make_unique<MatmulNode>(options.get_name(), std::move(options), get_context()));

    return {.C = C};
}

inline Scaled_dot_product_attention::Outputs Graph::scaled_dot_product_attention(Scaled_dot_product_attention::Inputs inputs, Scaled_dot_product_attention options) {    
    // Make required output tensors
    auto O = options.outputs.O = output_tensor(options.get_name() + "_output");
    auto S = options.outputs.S = output_tensor(options.get_name() + "_softmax_output");

    // Set inputs
    options.inputs = inputs;

    sub_nodes.emplace_back(std::make_unique<ScaledDotProductAttentionNode>(options.get_name(), std::move(options), get_context()));

    return {.S = S, .O = O};
}

inline Scaled_dot_product_flash_attention::Outputs Graph::scaled_dot_product_flash_attention(Scaled_dot_product_flash_attention::Inputs inputs, Scaled_dot_product_flash_attention options) {    
    // Make required output tensors
    auto O = options.outputs.O = output_tensor(options.get_name() + "_output");
    auto Stats = options.outputs.Stats = output_tensor(options.get_name() + "_softmax_output");

    // Set inputs
    options.inputs.Q = inputs.Q;
    options.inputs.K = inputs.K;
    options.inputs.V = inputs.V;

    sub_nodes.emplace_back(std::make_unique<ScaledDotProductFlashAttentionNode>(options.get_name(), std::move(options), get_context()));

    return {.O = O, .Stats = Stats};
}

} // namespace cudnn_frontend::graph
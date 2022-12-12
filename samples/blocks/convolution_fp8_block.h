#pragma once

#include <cudnn_frontend_Logging.h>

#include <graphs/cudnn_frontend_graph_helpers.h>
#include <graphs/cudnn_frontend_IBlock.h>
#include <graphs/cudnn_frontend_convolution_block.h>
#include <graphs/cudnn_frontend_pointwise_block.h>
#include <graphs/cudnn_frontend_reduction_block.h>

namespace cudnn_frontend {

class ConvolutionFP8Block : public IBlock {
private:

protected:

public:

    ConvolutionFP8Block(std::string const& name, int64_t const offset = 1)  : IBlock (name, offset) {
        auto conv_block = std::make_shared<ConvolutionBlock>("conv_block", offset);
        auto X_DQ_block = std::make_shared<PointwiseBlock>("X_DQ_block", offset + 100);
        auto W_DQ_block = std::make_shared<PointwiseBlock>("W_DQ_block", offset + 200);
        auto Y_Q_block = std::make_shared<PointwiseBlock>("Y_Q_block", offset + 300);
        auto amax_block = std::make_shared<ReductionBlock>("amax_block", offset + 400);

        sub_blocks.emplace("conv_block", conv_block);
        sub_blocks.emplace("X_DQ_block", X_DQ_block);
        sub_blocks.emplace("W_DQ_block", W_DQ_block);
        sub_blocks.emplace("Y_Q_block", Y_Q_block);
        sub_blocks.emplace("amax_block", amax_block);

        conv_block->parent_block = this;
        X_DQ_block->parent_block = this;
        W_DQ_block->parent_block = this;
        Y_Q_block->parent_block = this;
        amax_block->parent_block = this;
    }

    Type getType() override final {
        return Type::BLOCK;
    }

    int set_properties(std::string const& IBlock_name, convolution_node const& properties) {
        if(sub_blocks.count(IBlock_name) == 0) {
            return 1;
        }

        auto convolution_block_ptr = std::dynamic_pointer_cast<ConvolutionBlock>(sub_blocks.at(IBlock_name));
        if(convolution_block_ptr == nullptr) {
            return 1;
        }

        convolution_block_ptr->set_properties(IBlock_name, properties);
        return 0;
    }

    int set_properties(std::string const& IBlock_name, pointwise_node const& properties) {
        if(sub_blocks.count(IBlock_name) == 0) {
            return 1;
        }

        auto pointwise_block_ptr = std::dynamic_pointer_cast<PointwiseBlock>(sub_blocks.at(IBlock_name));
        if(pointwise_block_ptr == nullptr) {
            return 1;
        }

        pointwise_block_ptr->set_properties(IBlock_name, properties);
        return 0;
    }

    int set_properties(std::string const& IBlock_name, reduction_node const& properties) {
        if(sub_blocks.count(IBlock_name) == 0) {
            return 1;
        }

        auto reduction_block_ptr = std::dynamic_pointer_cast<ReductionBlock>(sub_blocks.at(IBlock_name));
        if(reduction_block_ptr == nullptr) {
            return 1;
        }

        reduction_block_ptr->set_properties(IBlock_name, properties);
        return 0;
    }

    int infer_properties() override final {        
        auto const& conv_block_ptr = std::dynamic_pointer_cast<ConvolutionBlock>(sub_blocks.at("conv_block"));
        auto conv_output_tensor = get_tensor_props(conv_block_ptr->props.port_to_name.at(convolution_node::PORTS::Y));
        conv_output_tensor->set_is_virtual(true);
        
        auto const& x_dq_block_ptr = std::dynamic_pointer_cast<PointwiseBlock>(sub_blocks.at("X_DQ_block"));
        auto x_dq_tensor = get_tensor_props(x_dq_block_ptr->props.port_to_name.at(pointwise_node::PORTS::Y));
        x_dq_tensor->set_is_virtual(true);
        
        auto const& w_dq_block_ptr = std::dynamic_pointer_cast<PointwiseBlock>(sub_blocks.at("W_DQ_block"));
        auto w_dq_tensor = get_tensor_props(w_dq_block_ptr->props.port_to_name.at(pointwise_node::PORTS::Y));
        w_dq_tensor->set_is_virtual(true);

        for(auto const& sub_block: sub_blocks) {
            sub_block.second->infer_properties();
        }
        return 0;
    }

    int validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating ConvolutionFP8Block..." << std::endl;

        for(auto const& sub_block: sub_blocks) {
            sub_block.second->validate();
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Validated ConvolutionFP8Block." << std::endl;
        return 0;
    }

    int partition(cudnnHandle_t& handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partioning ConvolutionFP8Block..." << std::endl;

        std::vector<Operation const*> operation_graph = {sub_blocks.at("conv_block")->operations.at("conv").get()
                                                        , sub_blocks.at("X_DQ_block")->operations.at("pointwise").get()
                                                        , sub_blocks.at("W_DQ_block")->operations.at("pointwise").get()
                                                        , sub_blocks.at("Y_Q_block")->operations.at("pointwise").get()
                                                        , sub_blocks.at("amax_block")->operations.at("reduction").get()};
        auto conv_pointwise_graph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(operation_graph.size(), operation_graph.data()).build();
        operation_graphs.push_back(std::make_shared<OperationGraph>(std::move(conv_pointwise_graph)));

        getLogger() << "[cudnn_frontend] INFO: " << "Partitioned ConvolutionFP8Block." << std::endl;
        return 0;
    }

    int build(cudnnHandle_t& handle) override final {

        infer_properties();
        validate();
        createTensors();
        createDescritpors();
        createOperations();
        partition(handle);
        createExecutionPlan(handle);

        return 0;
    }
    
    int execute(cudnnHandle_t& handle, std::unordered_map<std::string, void*> const& tensor_uid_to_pointer_map) override final {
        getLogger() << "[cudnn_frontend] INFO: ConvolutionFP8Block starting execution..." << std::endl;

        for(auto const& execution_plan: execution_plans) {
            getLogger() << "[cudnn_frontend] INFO: Executing " << execution_plan->getTag() << "..." << std::endl;
        
            std::vector<int64_t> uids;
            std::vector<void*> device_ptrs;

            uids.reserve(tensor_uid_to_pointer_map.size());
            device_ptrs.reserve(tensor_uid_to_pointer_map.size());

            for (auto const& p : tensor_uid_to_pointer_map) {
                uids.push_back(get_tensor_props(p.first)->get_uid());
                device_ptrs.push_back(p.second);
            }

            auto variant_pack = VariantPackBuilder()
                                .setDataPointers(device_ptrs.size(), device_ptrs.data())
                                .setUids(uids.size(), uids.data())
                                .build();

            auto status = cudnnBackendExecute(handle, execution_plan->get_raw_desc(), variant_pack.get_raw_desc());
            if (status != CUDNN_STATUS_SUCCESS) {
                return 1;
            }
            getLogger() << "[cudnn_frontend] INFO: Executed " << execution_plan->getTag() << "." << std::endl;
        }
        
        getLogger() << "[cudnn_frontend] INFO: ConvolutionFP8Block executed successfully." << std::endl;
        return 0;
    }
};

} // namespace cudnn_frontend
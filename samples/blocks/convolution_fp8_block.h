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

    ConvolutionFP8Block(int64_t const& offset = 1) {
        auto conv_block = std::make_shared<ConvolutionBlock>(offset);
        auto X_DQ_block = std::make_shared<PointwiseBlock>(offset + 100);
        auto W_DQ_block = std::make_shared<PointwiseBlock>(offset + 200);
        auto Y_Q_block = std::make_shared<PointwiseBlock>(offset + 300);
        auto amax_block = std::make_shared<ReductionBlock>(offset + 400);
        
        X_DQ_block->props.uids[pointwise_node::PORTS::X] = conv_block->props.uids[convolution_node::PORTS::Y];
        W_DQ_block->props.uids[pointwise_node::PORTS::X] = X_DQ_block->props.uids[pointwise_node::PORTS::Y];
        Y_Q_block->props.uids[pointwise_node::PORTS::X] = W_DQ_block->props.uids[pointwise_node::PORTS::Y];
        amax_block->props.uids[reduction_node::PORTS::X] = W_DQ_block->props.uids[pointwise_node::PORTS::Y];

        sub_blocks.emplace("conv_block", conv_block);
        sub_blocks.emplace("X_DQ_block", X_DQ_block);
        sub_blocks.emplace("W_DQ_block", W_DQ_block);
        sub_blocks.emplace("Y_Q_block", Y_Q_block);
        sub_blocks.emplace("amax_block", amax_block);
    }

    Type getType() override final {
        return Type::BLOCK;
    }

    int validate() override final {

        for(auto const& sub_block: sub_blocks) {
            sub_block.second->validate();
        }
        
        sub_blocks["conv_block"]->tensor_props.at(convolution_node::PORTS::Y).set_is_virtual(true);
        sub_blocks["conv_block"]->tensor_props.at(convolution_node::PORTS::Y).set_data_type(CUDNN_DATA_FLOAT);
        sub_blocks["X_DQ_block"]->tensor_props.at(convolution_node::PORTS::Y).set_is_virtual(true);
        sub_blocks["X_DQ_block"]->tensor_props.at(convolution_node::PORTS::Y).set_data_type(CUDNN_DATA_FLOAT);
        sub_blocks["W_DQ_block"]->tensor_props.at(convolution_node::PORTS::Y).set_is_virtual(true);
        sub_blocks["W_DQ_block"]->tensor_props.at(convolution_node::PORTS::Y).set_data_type(CUDNN_DATA_FLOAT);

        sub_blocks["X_DQ_block"]->tensor_props.at(pointwise_node::PORTS::X) = sub_blocks["conv_block"]->tensor_props.at(convolution_node::PORTS::Y);
        sub_blocks["W_DQ_block"]->tensor_props.at(pointwise_node::PORTS::X) = sub_blocks["X_DQ_block"]->tensor_props.at(pointwise_node::PORTS::Y);
        sub_blocks["Y_Q_block"]->tensor_props.at(pointwise_node::PORTS::X) = sub_blocks["W_DQ_block"]->tensor_props.at(pointwise_node::PORTS::Y);
        sub_blocks["amax_block"]->tensor_props.at(reduction_node::PORTS::X) = sub_blocks["W_DQ_block"]->tensor_props.at(pointwise_node::PORTS::Y);

        
        return 0;
    }

    int partition(cudnnHandle_t& handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partioning ConvolutionFP8Block..." << std::endl;

        std::vector<Operation const*> operation_graph = {sub_blocks["conv_block"]->operations["conv"].get()
                                                        , sub_blocks["X_DQ_block"]->operations["pointwise"].get()
                                                        , sub_blocks["W_DQ_block"]->operations["pointwise"].get()
                                                        , sub_blocks["Y_Q_block"]->operations["pointwise"].get()
                                                        , sub_blocks["amax_block"]->operations["reduction"].get()};
        auto conv_pointwise_graph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(operation_graph.size(), operation_graph.data()).build();
        operation_graphs.push_back(std::make_shared<OperationGraph>(std::move(conv_pointwise_graph)));

        getLogger() << "[cudnn_frontend] INFO: " << "Partitioned ConvolutionFP8Block." << std::endl;
        return 0;
    }

    int build(cudnnHandle_t& handle) override final {

        validate();
        createTensors();
        createDescritpors();
        createOperations();
        partition(handle);
        createExecutionPlan(handle);

        return 0;
    }
    
    int execute(cudnnHandle_t& handle, std::unordered_map<int64_t, void*> const& tensor_uid_to_pointer_map) override final {
        getLogger() << "[cudnn_frontend] INFO: ConvolutionFP8Block starting execution..." << std::endl;

        for(auto const& execution_plan: execution_plans) {
            getLogger() << "[cudnn_frontend] INFO: Executing " << execution_plan->getTag() << "..." << std::endl;
        
            std::vector<int64_t> uids;
            std::vector<void*> device_ptrs;

            uids.reserve(tensor_uid_to_pointer_map.size());
            device_ptrs.reserve(tensor_uid_to_pointer_map.size());

            for (auto const& p : tensor_uid_to_pointer_map) {
                uids.push_back(p.first);
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
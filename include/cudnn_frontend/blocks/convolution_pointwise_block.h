#pragma once

#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include <cudnn_frontend/blocks/helpers.h>
#include <cudnn_frontend/blocks/Iblock.h>
#include <cudnn_frontend/blocks/convolution_blocks.h>
#include <cudnn_frontend/blocks/pointwise_block.h>

namespace cudnn_frontend {

class ConvolutionPointwiseBlock : public IBlock {
private:

protected:

public:

    ConvolutionPointwiseBlock(int64_t const& offset = 1) {
        sub_blocks.emplace("conv_block", std::make_shared<ConvolutionBlock>(offset));
        sub_blocks.emplace("pointwise_block", std::make_shared<PointwiseBlock>(offset + 200));
    }

    Type getType() override final {
        return Type::BLOCK;
    }

    int validate() override final {

        for(auto const& sub_block: sub_blocks) {
            sub_block.second->validate();
        }

        sub_blocks["conv_block"]->tensor_props["Y"].is_virtual = true;
        sub_blocks["pointwise_block"]->tensor_props["X"] = sub_blocks["conv_block"]->tensor_props["Y"];

        return 0;
    }

    int partition(cudnnHandle_t& handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partioning ConvolutionPointwiseBlock..." << std::endl;

        std::vector<Operation const*> operation_graph = {sub_blocks["conv_block"]->operations["conv"].get(), sub_blocks["pointwise_block"]->operations["pointwise"].get()};
        auto conv_pointwise_graph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(operation_graph.size(), operation_graph.data()).build();
        operation_graphs.push_back(std::make_shared<OperationGraph>(std::move(conv_pointwise_graph)));

        getLogger() << "[cudnn_frontend] INFO: " << "Partitioned ConvolutionPointwiseBlock." << std::endl;
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
    
    int execute(cudnnHandle_t& handle) override final {
        (void)handle;
        return 0;
    } 
};

} // namespace cudnn_frontend
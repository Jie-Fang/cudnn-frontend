#pragma once

#include <cudnn_frontend_Heuristics.h>
#include <cudnn_frontend_Logging.h>

#include <graphs/cudnn_frontend_graph_helpers.h>
#include <graphs/cudnn_frontend_IBlock.h>
#include <graphs/cudnn_frontend_convolution_block.h>
#include <graphs/cudnn_frontend_pointwise_block.h>

namespace cudnn_frontend {

class ConvolutionPointwiseBlock : public IBlock {
private:

protected:

public:

    ConvolutionPointwiseBlock(std::string const& name, int64_t const offset = 1)  : IBlock (name, offset) {
        auto conv_block = std::make_shared<ConvolutionBlock>("conv_block", offset);
        auto pointwise_block = std::make_shared<PointwiseBlock>("pointwise_block", offset + 200);

        sub_blocks.emplace("conv_block", conv_block);
        sub_blocks.emplace("pointwise_block", pointwise_block);

        conv_block->parent_block = this;
        pointwise_block->parent_block = this;
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
    
    int infer_properties() override final {        
        auto const& conv_block_ptr = std::dynamic_pointer_cast<ConvolutionBlock>(sub_blocks.at("conv_block"));
        tensor_props.at(conv_block_ptr->props.port_to_name.at(convolution_node::PORTS::Y))->set_is_virtual(true);

        for(auto const& sub_block: sub_blocks) {
            sub_block.second->infer_properties();
        }
        return 0;
    }

    int validate() const override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Validating ConvolutionPointwiseBlock..." << std::endl;

        for(auto const& sub_block: sub_blocks) {
            sub_block.second->validate();
        }

        getLogger() << "[cudnn_frontend] INFO: " << "Validated ConvolutionPointwiseBlock." << std::endl;
        return 0;
    }

    int partition(cudnnHandle_t& handle) override final {
        getLogger() << "[cudnn_frontend] INFO: " << "Partioning ConvolutionPointwiseBlock..." << std::endl;

        std::vector<Operation const*> operation_graph = {sub_blocks.at("conv_block")->operations.at("conv").get(), sub_blocks.at("pointwise_block")->operations.at("pointwise").get()};
        auto conv_pointwise_graph = cudnn_frontend::OperationGraphBuilder().setHandle(handle).setOperationGraph(operation_graph.size(), operation_graph.data()).build();
        operation_graphs.push_back(std::make_shared<OperationGraph>(std::move(conv_pointwise_graph)));

        getLogger() << "[cudnn_frontend] INFO: " << "Partitioned ConvolutionPointwiseBlock." << std::endl;
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
        getLogger() << "[cudnn_frontend] INFO: ConvolutionPointwiseBlock starting execution..." << std::endl;

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
        
        getLogger() << "[cudnn_frontend] INFO: ConvolutionPointwiseBlock executed successfully." << std::endl;
        return 0;
    }
};

} // namespace cudnn_frontend
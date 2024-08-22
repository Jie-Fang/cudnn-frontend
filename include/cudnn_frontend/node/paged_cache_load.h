#pragma once

#include "../../cudnn_frontend_Heuristics.h"
#include "../../cudnn_frontend_Logging.h"

#include "../graph_helpers.h"
#include "../node_interface.h"

#include "pointwise.h"
#include "reduction.h"

namespace cudnn_frontend::graph {

class PagedCacheLoadNode : public NodeCRTP<PagedCacheLoadNode> {
   public:
    PagedCacheLoad_attributes attributes;

    PagedCacheLoadNode(PagedCacheLoad_attributes&& attributes_, detail::Context const& context)
        : NodeCRTP(context), attributes(std::move(attributes_)) {}

    Type
    getType() override final {
        return Type::PAGED_CACHE_LOAD;
    }

    error_t
    create_cudnn_operations(
        std::unordered_set<uid_t>& uids_involved_in_operations,
        std::vector<std::shared_ptr<cudnn_frontend::Operation>>& operations,
        managed_backend_descriptor_t& raw_operations,
        std::unordered_map<int64_t, std::shared_ptr<cudnn_frontend::Tensor>>& tensors) const override final {
            (void) uids_involved_in_operations;
            (void) operations;
            (void) raw_operations;
            (void) tensors;
            std::cout << "TODO(@mbreughe) create_cudnn_operations" << std::endl;
            return {error_code_t::OK, ""};
        }

    error_t
    pre_validate_node() const override final {
        CUDNN_FE_LOG_LABEL_ENDL("INFO: Validating PagedCacheLoadNode " << attributes.name << "...");

                    std::cout << "TODO(@mbreughe) pre_validate_node" << std::endl;


        return {error_code_t::OK, ""};
    }

    error_t
    infer_properties_node() override final {
        std::cout << "TODO(@mbreughe) infer_properties_node" << std::endl;


        return {error_code_t::OK, ""};
    }

#ifndef CUDNN_FRONTEND_SKIP_JSON_LIB
    virtual void
    serialize(json& j) const override final {
        j = attributes;
    }
#endif
};

inline void
INode::paged_cache_load(std::shared_ptr<Tensor_attributes> container,
               std::shared_ptr<Tensor_attributes> seqLen,
               std::shared_ptr<Tensor_attributes> pageTable,
               PagedCacheLoad_attributes attributes,
               std::shared_ptr<Tensor_attributes> yOut) {
    attributes.inputs[PagedCacheLoad_attributes::input_names::container] = container;
    attributes.inputs[PagedCacheLoad_attributes::input_names::seqLen]    = seqLen;
    attributes.inputs[PagedCacheLoad_attributes::input_names::pageTable] = pageTable;
    attributes.outputs[PagedCacheLoad_attributes::output_names::yOut]    = yOut;
    sub_nodes.emplace_back(std::make_unique<PagedCacheLoadNode>(std::move(attributes), context));
}
}  // namespace cudnn_frontend::graph
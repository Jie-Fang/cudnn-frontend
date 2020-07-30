#pragma once

#include <cudnn.h>

namespace cudnn_frontend {

static constexpr std::array<int64_t, 3> fallback_engine_conv_list  = {0, 1, 28};
static constexpr std::array<int64_t, 3> fallback_engine_dgrad_list = {0, 1, 25};
static constexpr std::array<int64_t, 3> fallback_engine_wgrad_list = {0, 1, 20};

class EngineFallbackList : public BackendDescriptor {
   private:
    auto
    get_fallback_list_size(cudnnBackendDescriptorType_t type) -> int64_t {
        switch (type) {
            case CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR:
                return fallback_engine_conv_list.size();
            case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR:
                return fallback_engine_dgrad_list.size();
            case CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR:
                return fallback_engine_wgrad_list.size();
            default:
                return 0;
        }
    }

   public:
    friend class EngineFallbackListBuilder;

    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_FALLBACK ENGINES :";
        return ss.str();
    }

    auto
    getFallbackList() -> std::vector<cudnnBackendDescriptor_t> & {
        return m_engine_configs;
    }

    ~EngineFallbackList() {
        for (auto i = 0; i < m_engine_configs.size(); i++) {
            if (m_engine_configs[i] != nullptr) {
                cudnnBackendDestroyDescriptor(m_engine_configs[i]);
                m_engine_configs[i] = nullptr;
            }
        }
    }
    EngineFallbackList(EngineFallbackList &&from)
        : BackendDescriptor(from.desc, from.get_status(), from.get_error()), mode(from.mode), opGraph(from.opGraph) {
        from.opGraph = nullptr;
        m_engine_configs.swap(from.m_engine_configs);
    }

   private:
    EngineFallbackList()                           = default;
    EngineFallbackList(EngineFallbackList const &) = delete;
    EngineFallbackList &
    operator=(EngineFallbackList const &) = delete;

    cudnnBackendDescriptor_t opGraph = nullptr;
    cudnnBackendDescriptorType_t mode;
    std::vector<cudnnBackendDescriptor_t> m_engine_configs;
};

///
/// EngineHeuristicsBuilder Class
/// Helper class used to build EngineHeuristics class
class EngineFallbackListBuilder {
   public:
    /** @defgroup EngineFallbackListBuilder
     *  Set individual property of EngineFallbackList class
     *  @{
     */
    //! Set operationGraph for the engine (opGraph is not destroyed)
    auto
    setOperationGraph(OperationGraph &opGraph_) -> EngineFallbackListBuilder & {
        m_fallback_list.opGraph = opGraph_.get_raw_desc();
        return *this;
    }
    auto
    setOperation(cudnnBackendDescriptorType_t mode) -> EngineFallbackListBuilder & {
        m_fallback_list.mode = mode;
        return *this;
    }
    /** @} */

    //! constructs the EngineFallbackList by calling the cudnn API
    //! Throws the appropriate error message
    EngineFallbackList &&
    build() {
        if (m_fallback_list.opGraph == nullptr) {
            set_error_and_throw_exception(&m_fallback_list,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: Check and Set the "
                                          "CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH field for heuristic");
            return std::move(m_fallback_list);
        };
        if (m_fallback_list.mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR) {
            for (auto i = 0; i < fallback_engine_conv_list.size(); i++) {
                auto engine = cudnn_frontend::EngineBuilder()
                                  .setGlobalEngineIdx(fallback_engine_conv_list[i])
                                  .setOperationGraph(m_fallback_list.opGraph)
                                  .build();
                auto engine_config = cudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
                m_fallback_list.m_engine_configs.emplace_back(engine_config.get_desc());
            }
        }
        if (m_fallback_list.mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
            for (auto i = 0; i < fallback_engine_dgrad_list.size(); i++) {
                auto engine = cudnn_frontend::EngineBuilder()
                                  .setGlobalEngineIdx(fallback_engine_dgrad_list[i])
                                  .setOperationGraph(m_fallback_list.opGraph)
                                  .build();
                auto engine_config = cudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
                m_fallback_list.m_engine_configs.emplace_back(engine_config.get_desc());
            }
        }
        if (m_fallback_list.mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) {
            for (auto i = 0; i < fallback_engine_wgrad_list.size(); i++) {
                auto engine = cudnn_frontend::EngineBuilder()
                                  .setGlobalEngineIdx(fallback_engine_wgrad_list[i])
                                  .setOperationGraph(m_fallback_list.opGraph)
                                  .build();
                auto engine_config = cudnn_frontend::EngineConfigBuilder().setEngine(engine).build();
                m_fallback_list.m_engine_configs.push_back(engine_config.get_desc());
            }
        }
        return std::move(m_fallback_list);
    }

    explicit EngineFallbackListBuilder()                         = default;
    ~EngineFallbackListBuilder()                                 = default;
    EngineFallbackListBuilder(EngineFallbackListBuilder &&)      = delete;
    EngineFallbackListBuilder(EngineFallbackListBuilder const &) = delete;
    EngineFallbackListBuilder &
    operator=(EngineFallbackListBuilder const &) = delete;

   private:
    EngineFallbackList m_fallback_list;
};
}
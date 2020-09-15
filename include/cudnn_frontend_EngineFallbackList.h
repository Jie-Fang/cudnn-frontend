/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */ 

#pragma once

#include <cudnn.h>

namespace cudnn_frontend {

static constexpr std::array<int64_t, 3> fallback_engine_conv_list  = {0, 1, 28};
static constexpr std::array<int64_t, 3> fallback_engine_dgrad_list = {0, 1, 25};
static constexpr std::array<int64_t, 3> fallback_engine_wgrad_list = {0, 1, 20};

class EngineFallbackList_v8 : public BackendDescriptor {
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
    friend class EngineFallbackListBuilder_v8;

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

    ~EngineFallbackList_v8() {
        for (auto i = 0; i < m_engine_configs.size(); i++) {
            if (m_engine_configs[i] != nullptr) {
                cudnnBackendDestroyDescriptor(m_engine_configs[i]);
                m_engine_configs[i] = nullptr;
            }
        }
    }
    EngineFallbackList_v8(EngineFallbackList_v8 &&from)
        : BackendDescriptor(from.desc, from.get_status(), from.get_error()), mode(from.mode), opGraph(from.opGraph) {
        from.opGraph = nullptr;
        m_engine_configs.swap(from.m_engine_configs);
    }

   private:
    EngineFallbackList_v8()                           = default;
    EngineFallbackList_v8(EngineFallbackList_v8 const &) = delete;
    EngineFallbackList_v8 &
    operator=(EngineFallbackList_v8 const &) = delete;

    cudnnBackendDescriptor_t opGraph = nullptr;
    cudnnBackendDescriptorType_t mode;
    std::vector<cudnnBackendDescriptor_t> m_engine_configs;
};

///
/// EngineFallBackListBuilder Class
/// Helper class used to build EngineFallBackList class
class EngineFallbackListBuilder_v8 {
   public:
    /** @defgroup EngineFallbackListBuilder_v8
     *  Set individual property of EngineFallbackList_v8 class
     *  @{
     */
    //! Set operationGraph for the engine (opGraph is not destroyed)
    auto
    setOperationGraph(OperationGraph_v8 &opGraph_) -> EngineFallbackListBuilder_v8 & {
        m_fallback_list.opGraph = opGraph_.get_raw_desc();
        return *this;
    }
    auto
    setOperation(cudnnBackendDescriptorType_t mode) -> EngineFallbackListBuilder_v8 & {
        m_fallback_list.mode = mode;
        return *this;
    }
    /** @} */

    //! constructs the EngineFallbackList_v8 by calling the cudnn API
    //! Throws the appropriate error message
    EngineFallbackList_v8 &&
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
                auto engine = cudnn_frontend::EngineBuilder_v8()
                                  .setGlobalEngineIdx(fallback_engine_conv_list[i])
                                  .setOperationGraph(m_fallback_list.opGraph)
                                  .build();
                auto engine_config = cudnn_frontend::EngineConfigBuilder_v8().setEngine(engine).build();
                m_fallback_list.m_engine_configs.emplace_back(engine_config.get_desc());
            }
        }
        if (m_fallback_list.mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR) {
            for (auto i = 0; i < fallback_engine_dgrad_list.size(); i++) {
                auto engine = cudnn_frontend::EngineBuilder_v8()
                                  .setGlobalEngineIdx(fallback_engine_dgrad_list[i])
                                  .setOperationGraph(m_fallback_list.opGraph)
                                  .build();
                auto engine_config = cudnn_frontend::EngineConfigBuilder_v8().setEngine(engine).build();
                m_fallback_list.m_engine_configs.emplace_back(engine_config.get_desc());
            }
        }
        if (m_fallback_list.mode == CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR) {
            for (auto i = 0; i < fallback_engine_wgrad_list.size(); i++) {
                auto engine = cudnn_frontend::EngineBuilder_v8()
                                  .setGlobalEngineIdx(fallback_engine_wgrad_list[i])
                                  .setOperationGraph(m_fallback_list.opGraph)
                                  .build();
                auto engine_config = cudnn_frontend::EngineConfigBuilder_v8().setEngine(engine).build();
                m_fallback_list.m_engine_configs.push_back(engine_config.get_desc());
            }
        }
        return std::move(m_fallback_list);
    }

    explicit EngineFallbackListBuilder_v8()                         = default;
    ~EngineFallbackListBuilder_v8()                                 = default;
    EngineFallbackListBuilder_v8(EngineFallbackListBuilder_v8 &&)      = delete;
    EngineFallbackListBuilder_v8(EngineFallbackListBuilder_v8 const &) = delete;
    EngineFallbackListBuilder_v8 &
    operator=(EngineFallbackListBuilder_v8 const &) = delete;

   private:
    EngineFallbackList_v8 m_fallback_list;
};
}

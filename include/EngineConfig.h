#pragma once

#include <array>
#include <memory>
#include <sstream>
#include <algorithm>
#include <functional>
#include <utility>

#include <cudnn.h>
#include <cudnn_backend.h>

#include "cudnn_backend_wrap_utils.h"
#include "Engine.h"

namespace cudnn_api_wrapper {
///
/// EngineConfig Class
/// This class tells the Configuration of the Engine in terms of the knob choices
/// Properties:
///    - num knobs
///    - Choice
///    - Engine
///
/// Use EngineConfigBuilder to build this class.
/// Describe returns a string describing the tensor class
///
class EngineConfig : public BackendDescriptor {
   public:
    friend class EngineConfigBuilder;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_ENGINECFG_DESCRIPTOR :";
        ss << " Number of knobs: " << numKnobs;
        return ss.str();
    }
    EngineConfig(EngineConfig &&from)
        : BackendDescriptor(from.desc, from.get_status(), from.get_error()), engine(from.engine), numKnobs(from.numKnobs) {
        from.engine = nullptr;
        bChoices = from.bChoices;
        from.bChoices.fill(nullptr);
    }
    ~EngineConfig() {
        if (desc != nullptr) {
            cudnnBackendDestroyDescriptor(desc);
        }
        for (uint64_t i = 0; i < bChoices.size(); i++) {
            if (bChoices[i] != nullptr) {
                cudnnBackendDestroyDescriptor(bChoices[i]);
            }
        }
        if (engine != nullptr) {
            cudnnBackendDestroyDescriptor(engine);
        }
    }

   private:
    EngineConfig() : BackendDescriptor(nullptr) {
        cudnnStatus_t status;
        for (uint64_t i = 0; i < bChoices.size(); i++) {
            status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR, &bChoices[i]);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(this, status, "Choice creation failed");
                break;
            }
        }
    }
    EngineConfig(EngineConfig const &) = delete;
    EngineConfig &
    operator=(EngineConfig const &) = delete;

    manager<cudnnBackendDescriptor_t> engine   = nullptr;
    int64_t numKnobs                  = 0;
    std::array<cudnnBackendDescriptor_t, CUDNN_KNOB_TYPE_COUNTS> bChoices = {}; //!< Opaque pointer to the backend knobs
};

///
/// EngineConfigBuilder Class
/// Helper class used to build EngineConfig class
class EngineConfigBuilder {
   public:
    /** @defgroup EngineConfigBuilder
     *  Set individual property of EngineConfig class
     *  @{
     */
    //! Set engine for the EngineConfig
    auto
    setEngine(Engine const &engine_) -> EngineConfigBuilder & {
        m_engine_config.engine = engine_.get_desc();
        auto &knobs = engine_.getKnobs();
        m_engine_config.numKnobs = knobs.size();
        for (auto i = 0; i < knobs.size(); i++) {
            cudnnStatus_t status;
            cudnnBackendKnobType_t type = knobs[i].getKnobType();
            int64_t value = knobs[i].getChoice();
            status = cudnnBackendSetAttribute(m_engine_config.bChoices[i], CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, &type);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(&m_engine_config, status, "Engine Config Set Descriptor Choice Type failed");
            }
            status = cudnnBackendSetAttribute(m_engine_config.bChoices[i], CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, CUDNN_TYPE_INT64, 1, &value);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(&m_engine_config, status, "Engine Config Set Descriptor Choice Value failed");
            }
            status = cudnnBackendFinalize(m_engine_config.bChoices[i]);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(&m_engine_config, status, "Choice Finalize failed");
            }
        }
        return *this;
    }
    /** @} */

    //! constructs the Engine Config by calling the cudnn API
    //! Throws the appropriate error message
    EngineConfig &&
    build() {
        if (m_engine_config.status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_engine_config, m_engine_config.status, "Engien config not initialized properly");
            return std::move(m_engine_config);
        }
        if (m_engine_config.engine == nullptr) {
            set_error_and_throw_exception(&m_engine_config, CUDNN_STATUS_BAD_PARAM,"Initialize the engine correctly");
            return std::move(m_engine_config);
        }
        // Create a descriptor. Memory allocation happens here.
        auto status = CUDNN_STATUS_SUCCESS;
        status      = cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &m_engine_config.desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_engine_config, status, "cudnn Create Descriptor failed");
            return std::move(m_engine_config);
        }

        status = cudnnBackendSetAttribute(m_engine_config.desc,
                                          CUDNN_ATTR_ENGINECFG_ENGINE,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          &m_engine_config.engine);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_engine_config, status, "cudnn Set Attribute Engine failed");
            return std::move(m_engine_config);
        }

        if (m_engine_config.numKnobs > 0) {
            status = cudnnBackendSetAttribute(m_engine_config.desc,
                                              CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              m_engine_config.numKnobs,
                                              m_engine_config.bChoices.data());
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(&m_engine_config, status, "cudnn Set Attribute Engine failed");
                return std::move(m_engine_config);
            }
        }

        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_engine_config.desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_engine_config, status, "Engine Config Finalize failed");
            return std::move(m_engine_config);
        }
        return std::move(m_engine_config);
    }

    explicit EngineConfigBuilder()                   = default;
    ~EngineConfigBuilder()                           = default;
    EngineConfigBuilder(EngineConfigBuilder &&)      = delete;
    EngineConfigBuilder(EngineConfigBuilder const &) = delete;
    EngineConfigBuilder &
    operator=(EngineConfigBuilder const &) = delete;

   private:
    EngineConfig m_engine_config;
};
}

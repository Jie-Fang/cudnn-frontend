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
        : BackendDescriptor(from.desc), engine(from.engine), numKnobs(from.numKnobs) {
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
            throw_if(status != CUDNN_STATUS_SUCCESS, "Choice creation failed");
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
            throw_if(status != CUDNN_STATUS_SUCCESS, "Choice Type set failed");
            status = cudnnBackendSetAttribute(m_engine_config.bChoices[i], CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, CUDNN_TYPE_INT64, 1, &value);
            throw_if(status != CUDNN_STATUS_SUCCESS, "Choice Value set failed");
            status = cudnnBackendFinalize(m_engine_config.bChoices[i]);
            throw_if(status != CUDNN_STATUS_SUCCESS, "Choice Finalize failed");
        }
        return *this;
    }
    /** @} */

    //! constructs the Engine Config by calling the cudnn API
    //! Throws the appropriate error message
    EngineConfig &&
    build() {
        throw_if([this]() { return m_engine_config.engine == nullptr; }, "Initialize the engine correctly");
        // Create a descriptor. Memory allocation happens here.
        auto status = CUDNN_STATUS_SUCCESS;
        status      = cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &m_engine_config.desc);
        throw_if([this, status]() { return (status != CUDNN_STATUS_SUCCESS); }, "cudnn Create Descriptor failed");

        status = cudnnBackendSetAttribute(m_engine_config.desc,
                                          CUDNN_ATTR_ENGINECFG_ENGINE,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          &m_engine_config.engine);
        throw_if([this, status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Engine is invalid");

        if (m_engine_config.numKnobs > 0) {
            status = cudnnBackendSetAttribute(m_engine_config.desc,
                                              CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                                              CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                              m_engine_config.numKnobs,
                                              m_engine_config.bChoices.data());
            throw_if([this, status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Choices are invalid");
        }

        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_engine_config.desc);
        throw_if([this, status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Engine Config Finalize failed");
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

#pragma once

#include <array>
#include <vector>
#include <memory>
#include <sstream>
#include <algorithm>
#include <functional>
#include <utility>

#include <cudnn.h>
#include <cudnn_backend.h>

#include "cudnn_frontend_utils.h"
#include "OperationGraph.h"

namespace cudnn_frontend {

///
/// Engine Class
/// This class tells the properties of the Engine on which performs the operation requested
/// Properties:
///    - Index
///    - OperationGraph
///
/// Use EngineBuilder to build this class.
/// Describe returns a string describing the tensor class
///
class Engine : public BackendDescriptor {
   private:
    Engine()               = default;
    Engine(Engine const &) = delete;
    Engine &
    operator=(Engine const &) = delete;

    /// Internal class which controls the different knobs for a given engine
    /// Has min-max and stride as the options.
    /// User has the option to set the required value as a choice.
    class Knob {
       public:
        Knob(cudnnBackendKnobType_t type_, int64_t max, int64_t min, int64_t stride_)
            : knobType(type_), maxValue(max), minValue(min), stride(stride_) {}

        std::string
        describe() const {
            std::stringstream ss;
            ss << "Knob:" << knobType;
            ss << " Min: " << minValue;
            ss << " Max: " << maxValue;
            ss << " Stride: " << stride;
            return ss.str();
        }

        void
        setChoice(uint64_t val_) {
            choice = val_;
        }

        const int64_t getChoice() const {
            return choice;
        }

        const cudnnBackendKnobType_t getKnobType() const {
            return knobType;
        }

        const int64_t getMinValue() const {
            return minValue;
        }

        const int64_t getMaxValue() const {
            return minValue;
        }

        const int64_t getStride() const {
            return minValue;
        }

       private:
        cudnnBackendKnobType_t knobType = CUDNN_KNOB_TYPE_COUNTS;
        int64_t maxValue = 0, minValue = 0, stride = 0; //!< min, max and stride of the knob value
        int64_t choice = 0; //!< Choice set by the user
    };

    manager<cudnnBackendDescriptor_t> opGraph = nullptr;
    int64_t idx                    = -1; //!< Global Index of the engine for the given operationGraph.
    int64_t numKnobs               =  0; //!< Count of the backend knobs in the engine
    std::array<cudnnBackendDescriptor_t, CUDNN_KNOB_TYPE_COUNTS> bKnobs   = {}; //!< Opaque pointer to the backend knobs
    std::vector<Knob> knobs;

    //! Called from the constructor builds the internal knobs vector
    void
    buildKnobs() {
        cudnnStatus_t status;
        for (auto i = 0; i < numKnobs; i++) {
            auto bKnob = bKnobs[i];
            cudnnBackendKnobType_t type;
            int64_t maxValue, minValue, stride, elemCount;
            status =
                cudnnBackendGetAttribute(bKnob, CUDNN_ATTR_KNOB_INFO_TYPE, CUDNN_TYPE_KNOB_TYPE, 1, &elemCount, &type);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(this, status, "CUDNN_BACKEND_ENGINE_DESCRIPTOR: CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR GetAttribute CUDNN_ATTR_KNOB_INFO_TYPE failed");
            }
            status = cudnnBackendGetAttribute(
                bKnob, CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, CUDNN_TYPE_INT64, 1, &elemCount, &maxValue);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(this, status, "CUDNN_BACKEND_ENGINE_DESCRIPTOR: CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR GetAttribute CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE Failed");
            }
            status = cudnnBackendGetAttribute(
                bKnob, CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, CUDNN_TYPE_INT64, 1, &elemCount, &minValue);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(this, status, "CUDNN_BACKEND_ENGINE_DESCRIPTOR: CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR GetAttribute CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE Failed");
            }
            status =
                cudnnBackendGetAttribute(bKnob, CUDNN_ATTR_KNOB_INFO_STRIDE, CUDNN_TYPE_INT64, 1, &elemCount, &stride);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(this, status, "CUDNN_BACKEND_ENGINE_DESCRIPTOR: CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR GetAttribute CUDNN_ATTR_KNOB_INFO_STRIDE Failed");
            }
            knobs.emplace_back(Knob(type, maxValue, minValue, stride));
        }
    }

   public:
    friend class EngineBuilder;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_ENGINE_DESCRIPTOR :";
        ss << " ID: " << idx;
        ss << " Has " << numKnobs << " knobs";
        return ss.str();
    }
    Engine(Engine &&from) : BackendDescriptor(from.desc, from.get_status(), from.get_error()), opGraph(from.opGraph), idx(from.idx) {
        cudnnStatus_t status;
        from.opGraph = nullptr;
        for (uint64_t i = 0; i < bKnobs.size(); i++) {
            status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR, &bKnobs[i]);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(this, status, "CUDNN_BACKEND_ENGINE_DESCRIPTOR: CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR cudnnCreate Failed");
            }
        }
        status = cudnnBackendGetAttribute(desc,
                                          CUDNN_ATTR_ENGINE_KNOB_INFO,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          CUDNN_KNOB_TYPE_COUNTS,
                                          &numKnobs,
                                          bKnobs.data());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(this, status, "CUDNN_BACKEND_ENGINE_DESCRIPTOR: GetAttribute CUDNN_ATTR_ENGINE_KNOB_INFO Query Failed");
        }
        buildKnobs();
    }
    ~Engine() {
        if (desc != nullptr) {
            cudnnBackendDestroyDescriptor(desc);
        }
        for (uint64_t i = 0; i < bKnobs.size(); i++) {
            if (bKnobs[i] != nullptr) {
                cudnnBackendDestroyDescriptor(bKnobs[i]);
            }
        }
        if (opGraph != nullptr) {
            cudnnBackendDestroyDescriptor(opGraph);
        }
    }

    //! Returns a vector of knobs to the user
    std::vector<Knob> const &
    getKnobs() const {
        return knobs;
    }
};

///
/// EngineBuilder Class
/// Helper class used to build Engine class
class EngineBuilder {
   public:
    /** @defgroup EngineBuilder
     *  Set individual property of Engine class
     *  @{
     */
    //! Set operationGraph for the engine
    auto
    setOperationGraph(OperationGraph const &opGraph_) -> EngineBuilder & {
        m_engine.opGraph = opGraph_.get_desc();
        return *this;
    }
    //! Set engine index for the engine
    auto
    setGlobalEngineIdx(int64_t idx_) -> EngineBuilder & {
        m_engine.idx = idx_;
        return *this;
    }
    /** @} */

    //! constructs the Engine by calling the cudnn API
    //! Throws the appropriate error message
    Engine &&
    build() {
        if (m_engine.idx < 0) {
            set_error_and_throw_exception(&m_engine, CUDNN_STATUS_BAD_PARAM, "CUDNN_BACKEND_ENGINE_DESCRIPTOR: Check and Set the CUDNN_ATTR_ENGINE_GLOBAL_INDEX to valid value");
            return std::move(m_engine);
        }
        if (m_engine.opGraph == nullptr) {
            set_error_and_throw_exception(&m_engine, CUDNN_STATUS_BAD_PARAM, "CUDNN_BACKEND_ENGINE_DESCRIPTOR: Check and Set CUDNN_ATTR_ENGINE_OPERATION_GRAPH to valid value");
            return std::move(m_engine);
        }

        // Create a descriptor. Memory allocation happens here.
        auto status = CUDNN_STATUS_SUCCESS;
        status      = cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &m_engine.desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_engine, status, "CUDNN_BACKEND_ENGINE_DESCRIPTOR: cudnnCreate Descriptor Failed");
            return std::move(m_engine);
        }

        status = cudnnBackendSetAttribute(
            m_engine.desc, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &m_engine.opGraph);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_engine, status, "CUDNN_BACKEND_ENGINE_DESCRIPTOR: SetAttribute CUDNN_ATTR_ENGINE_OPERATION_GRAPH Failed");
            return std::move(m_engine);
        }

        status =
            cudnnBackendSetAttribute(m_engine.desc, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &m_engine.idx);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_engine, status, "CUDNN_BACKEND_ENGINE_DESCRIPTOR: SetAttribute CUDNN_ATTR_ENGINE_GLOBAL_INDEX Failed");
            return std::move(m_engine);
        }

        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_engine.desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(&m_engine, status, "CUDNN_BACKEND_ENGINE_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_engine);
        }

        return std::move(m_engine);
    }

    explicit EngineBuilder()             = default;
    ~EngineBuilder()                     = default;
    EngineBuilder(EngineBuilder &&)      = delete;
    EngineBuilder(EngineBuilder const &) = delete;
    EngineBuilder &
    operator=(EngineBuilder const &) = delete;

   private:
    Engine m_engine;
};
}

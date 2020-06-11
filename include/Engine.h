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

#include "cudnn_backend_wrap_utils.h"
#include "OperationGraph.h"

namespace cudnn_api_wrapper {

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
            throw_if(status != CUDNN_STATUS_SUCCESS, "Knob Info query failed");
            status = cudnnBackendGetAttribute(
                bKnob, CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE, CUDNN_TYPE_INT64, 1, &elemCount, &maxValue);
            throw_if(status != CUDNN_STATUS_SUCCESS, "Knob Max Value query failed");
            status = cudnnBackendGetAttribute(
                bKnob, CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE, CUDNN_TYPE_INT64, 1, &elemCount, &minValue);
            throw_if(status != CUDNN_STATUS_SUCCESS, "Knob Min Value query failed");
            status =
                cudnnBackendGetAttribute(bKnob, CUDNN_ATTR_KNOB_INFO_STRIDE, CUDNN_TYPE_INT64, 1, &elemCount, &stride);
            throw_if(status != CUDNN_STATUS_SUCCESS, "Knob Stride Value query failed");
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
    Engine(Engine &&from) : BackendDescriptor(from.desc), opGraph(from.opGraph), idx(from.idx) {
        cudnnStatus_t status;
        from.opGraph = nullptr;
        for (uint64_t i = 0; i < bKnobs.size(); i++) {
            status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR, &bKnobs[i]);
            throw_if(status != CUDNN_STATUS_SUCCESS, "Knob creation failed");
        }
        status = cudnnBackendGetAttribute(desc,
                                          CUDNN_ATTR_ENGINE_KNOB_INFOS,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          CUDNN_KNOB_TYPE_COUNTS,
                                          &numKnobs,
                                          bKnobs.data());
        throw_if(status != CUDNN_STATUS_SUCCESS, "Knob count query failed");
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
        throw_if([this]() { return m_engine.idx < 0; }, "Set the engine idx to valid value");
        throw_if([this]() { return m_engine.opGraph == nullptr; }, "Set opset to valid value");

        // Create a descriptor. Memory allocation happens here.
        auto status = CUDNN_STATUS_SUCCESS;
        status      = cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &m_engine.desc);
        throw_if([this, status]() { return (status != CUDNN_STATUS_SUCCESS); }, "cudnn Create Descriptor failed");

        status = cudnnBackendSetAttribute(
            m_engine.desc, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &m_engine.opGraph);
        throw_if([this, status]() { return (status != CUDNN_STATUS_SUCCESS); }, "cudnn OperationGraph is invalid");

        status =
            cudnnBackendSetAttribute(m_engine.desc, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &m_engine.idx);
        throw_if([this, status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Engine Index is invalid");

        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_engine.desc);
        throw_if([this, status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Engine Finalize failed");

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

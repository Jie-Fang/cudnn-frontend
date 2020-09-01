#pragma once

#include <vector>

#include <cudnn.h>
#include <cudnn_backend.h>

#include "cudnn_frontend_OperationGraph.h"
#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {
///
/// Engine Heuristic Class
/// This class helps determine the engine from the operation graph
/// based on the heuristics
/// Properties:
///    - heuristic mode
///    - operation graph
///
/// Use EngineHeuristicsBuilder_v8 to build this class.
/// Describe returns a string describing the EngineHeuristics_v8 class
///
class EngineHeuristics_v8 : public BackendDescriptor {
   public:
    friend class EngineHeuristicsBuilder_v8;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR :";
        return ss.str();
    }

    EngineHeuristics_v8(EngineHeuristics_v8 &&from)
        : BackendDescriptor(from.desc, from.get_status(), from.get_error()), mode(from.mode), opGraph(from.opGraph) {
        from.opGraph = nullptr;
    }

    ~EngineHeuristics_v8() {
        if (desc != nullptr) {
            cudnnBackendDestroyDescriptor(desc);
        }
        for (auto i = 0u; i < m_heuristic_results.size(); ++i) {
            if (m_heuristic_results[i] != nullptr) {
                cudnnBackendDestroyDescriptor(m_heuristic_results[i]);
            }
        }
        m_heuristic_results.clear();
    }

    /** @defgroup EngineHeuristicsQuery
     *  Query individual property of EngineHeuristics_v8 class
     *  @{
     */
    //! Query the total count of the engines for the Operation Set
    auto
    getEngineConfig(int64_t count = 1) -> std::vector<cudnnBackendDescriptor_t> & {
        cudnnStatus_t status;
        for (auto i = 0u; i < count; ++i) {
            cudnnBackendDescriptor_t engConfig = nullptr;
            status = cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engConfig);
            if (status != CUDNN_STATUS_SUCCESS) {
                set_error_and_throw_exception(
                    this,
                    status,
                    "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: CUDNN_BACKEND_ENGINECFG_DESCRIPTOR cudnnCreate Failed");
                return m_heuristic_results;
            };
            m_heuristic_results.emplace_back(engConfig);
        }
        int64_t result = -1;
        status         = cudnnBackendGetAttribute(desc,
                                          CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          count,
                                          &result,
                                          m_heuristic_results.data());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                this, status, "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: GetAttribute CUDNN_ATTR_ENGINEHEUR_RESULTS Failed");
        };
        return m_heuristic_results;
    }

    //! Query the total count of the engine config for the Operation Set
    auto
    getEngineConfigCount(void) -> int64_t {
        cudnnStatus_t status;
        int64_t count = -1;
        status        = cudnnBackendGetAttribute(
            desc, CUDNN_ATTR_ENGINEHEUR_RESULTS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                this,
                status,
                "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: GetAttribute CUDNN_ATTR_ENGINEHEUR_RESULTS Count Failed");
        };
        return count;
    }
    /** @} */

   private:
    EngineHeuristics_v8()                         = default;
    EngineHeuristics_v8(EngineHeuristics_v8 const &) = delete;
    EngineHeuristics_v8 &
    operator=(EngineHeuristics_v8 const &) = delete;

    cudnnBackendHeurMode_t mode      = CUDNN_HEUR_MODE_INSTANT;
    cudnnBackendDescriptor_t opGraph = nullptr;
    std::vector<cudnnBackendDescriptor_t> m_heuristic_results;  //! storage of heuristic results
};

///
/// EngineHeuristicsBuilder_v8 Class
/// Helper class used to build EngineHeuristics_v8 class
class EngineHeuristicsBuilder_v8 {
   public:
    /** @defgroup EngineHeuristicsBuilder_v8
     *  Set individual property of EngineHeuristics_v8 class
     *  @{
     */
    //! Set operationGraph for the engine (opGraph is not destroyed)
    auto
    setOperationGraph(OperationGraph_v8 &opGraph_) -> EngineHeuristicsBuilder_v8 & {
        m_heuristics.opGraph = opGraph_.get_raw_desc();
        return *this;
    }
    //! Set cudnnHandle for the operations
    auto
    setHeurMode(cudnnBackendHeurMode_t mode_) -> EngineHeuristicsBuilder_v8 & {
        m_heuristics.mode = mode_;
        return *this;
    }
    /** @} */

    //! constructs the EngineHeuristics_v8 by calling the cudnn API
    //! Throws the appropriate error message
    EngineHeuristics_v8 &&
    build() {
        if (m_heuristics.opGraph == nullptr) {
            set_error_and_throw_exception(&m_heuristics,
                                          CUDNN_STATUS_BAD_PARAM,
                                          "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: Check and Set the "
                                          "CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH field for heuristic");
            return std::move(m_heuristics);
        };

        // Create a descriptor. Memory allocation happens here.
        auto status = CUDNN_STATUS_SUCCESS;
        status      = cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &m_heuristics.desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_heuristics, status, "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_heuristics);
        };

        status = cudnnBackendSetAttribute(m_heuristics.desc,
                                          CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          &m_heuristics.opGraph);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_heuristics,
                status,
                "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: SetAttribute  CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH Failed");
            return std::move(m_heuristics);
        };
        status = cudnnBackendSetAttribute(
            m_heuristics.desc, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &m_heuristics.mode);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_heuristics,
                status,
                "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: SetAttribute CUDNN_ATTR_ENGINEHEUR_MODE Failed");
            return std::move(m_heuristics);
        };

        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_heuristics.desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_heuristics, status, "CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR: cudnn Finalize failed");
            return std::move(m_heuristics);
        };

        return std::move(m_heuristics);
    }

    explicit EngineHeuristicsBuilder_v8()                       = default;
    ~EngineHeuristicsBuilder_v8()                               = default;
    EngineHeuristicsBuilder_v8(EngineHeuristicsBuilder_v8 &&)      = delete;
    EngineHeuristicsBuilder_v8(EngineHeuristicsBuilder_v8 const &) = delete;
    EngineHeuristicsBuilder_v8 &
    operator=(EngineHeuristicsBuilder_v8 const &) = delete;

   private:
    EngineHeuristics_v8 m_heuristics;
};
}

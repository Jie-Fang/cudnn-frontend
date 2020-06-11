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
/// ExecutionPlan Class
/// This class tells the Configuration of the Engine in terms of the knob choices
/// Properties:
///    - num knobs
///    - Choice
///    - Engine
///
/// Use ExecutionPlanBuilder to build this class.
/// Describe returns a string describing the tensor class
///
class ExecutionPlan : public BackendDescriptor {
   public:
    friend class ExecutionPlanBuilder;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR :";
        ss << " workspace_size : " << getWorkspaceSize();
        return ss.str();
    }
    ExecutionPlan(ExecutionPlan &&from)
        : BackendDescriptor(from.desc), handle(from.handle), engine_config(from.engine_config) {
        from.engine_config = nullptr;
    }
    ~ExecutionPlan() {
        if (desc != nullptr) {
            cudnnBackendDestroyDescriptor(desc);
        }
        if (engine_config != nullptr) {
            cudnnBackendDestroyDescriptor(engine_config);
        }
    }
    /** @defgroup ExecutionPlanQuery
     *  Query individual property of ExecutionPlan class
     *  @{
     */
    //! Query the workspace requirement for the given plan
    auto
    getWorkspaceSize(void) const -> const int64_t {
        uint64_t workSpaceSize = 0;
        auto status            = cudnnBackendGetAttribute(
            desc, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, NULL, &workSpaceSize);
        throw_if([this, status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Workspace Size query failed");
        throw_if([workSpaceSize]() { return (workSpaceSize < 0); }, "workSpaceSize invalid");
        return workSpaceSize;
    }

   private:
    ExecutionPlan()                      = default;
    ExecutionPlan(ExecutionPlan const &) = delete;
    ExecutionPlan &
    operator=(ExecutionPlan const &) = delete;

    manager<cudnnBackendDescriptor_t> engine_config = nullptr;
    cudnnHandle_t handle                   = nullptr;
};

///
/// ExecutionPlanBuilder Class
/// Helper class used to build ExecutionPlan class
class ExecutionPlanBuilder {
   public:
    /** @defgroup ExecutionPlanBuilder
     *  Set individual property of ExecutionPlan class
     *  @{
     */
    //! Set engine for the ExecutionPlan
    auto
    setHandle(cudnnHandle_t handle_) -> ExecutionPlanBuilder & {
        m_execution_plan.handle = handle_;
        return *this;
    }
    //! Set engine Config for the Plan
    auto
    setEngineConfig(EngineConfig const &engine_config_) -> ExecutionPlanBuilder & {
        m_execution_plan.engine_config = engine_config_.get_desc();
        return *this;
    }
    //! Set engine Config for the Plan
    auto
    setEngineConfig(cudnnBackendDescriptor_t & desc) -> ExecutionPlanBuilder & {
        m_execution_plan.engine_config = desc;
        desc = nullptr;
        return *this;
    }
    /** @} */

    //! constructs the Engine Config by calling the cudnn API
    //! Throws the appropriate error message
    ExecutionPlan &&
    build() {
        throw_if([this]() { return m_execution_plan.handle == nullptr; }, "Initialize the ExecutionPlan handle");
        throw_if([this]() { return m_execution_plan.engine_config == nullptr; }, "Initialize the EngineConfig ");
        // Create a descriptor. Memory allocation happens here.
        auto status = CUDNN_STATUS_SUCCESS;
        status      = cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &m_execution_plan.desc);
        throw_if([this, status]() { return (status != CUDNN_STATUS_SUCCESS); }, "cudnn Create Descriptor failed");

        status = cudnnBackendSetAttribute(m_execution_plan.desc,
                                          CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                          CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                          1,
                                          &m_execution_plan.engine_config);
        throw_if([this, status]() { return (status != CUDNN_STATUS_SUCCESS); }, "EngineConfig is invalid");
        status = cudnnBackendSetAttribute(
            m_execution_plan.desc, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &m_execution_plan.handle);
        throw_if([this, status]() { return (status != CUDNN_STATUS_SUCCESS); }, "Handle set is invalid");

        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_execution_plan.desc);
        throw_if([this, status]() { return (status != CUDNN_STATUS_SUCCESS); }, "ExecutionPlan Finalize failed");
        return std::move(m_execution_plan);
    }

    explicit ExecutionPlanBuilder()                    = default;
    ~ExecutionPlanBuilder()                            = default;
    ExecutionPlanBuilder(ExecutionPlanBuilder &&)      = delete;
    ExecutionPlanBuilder(ExecutionPlanBuilder const &) = delete;
    ExecutionPlanBuilder &
    operator=(ExecutionPlanBuilder const &) = delete;

   private:
    ExecutionPlan m_execution_plan;
};
}

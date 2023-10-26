#pragma once

#include <string>
#include <vector>

#include "../cudnn_frontend_EngineConfig.h"
#include "../cudnn_frontend_Logging.h"

namespace cudnn_frontend::graph {


/*
class Plans {
   public:

    static error_t
    autotune_default_impl(Plans* plans,
                          cudnnHandle_t handle,
                          std::unordered_map<std::shared_ptr<Tensor_attributes>, void*> variants,
                          void* workspace,
                          void*) {
        auto& execution_plans = plans->list_of_engine_configs.get_execution_plans();

        // Create the variant pack for all the plans to use.
        std::vector<int64_t> uids;
        std::vector<void*> ptrs;
        for (auto it : variants) {
            uids.push_back(it.first->get_uid());
            ptrs.push_back(it.second);
        }

        auto variantPack = VariantPackBuilder()
                               .setDataPointers(ptrs.size(), ptrs.data())
                               .setUids(uids.size(), uids.data())
                               .setWorkspacePointer(workspace)
                               .build();

        std::vector<std::shared_ptr<ExecutionPlan>> time_sorted_plans;

        auto plan_cmp = [](std::shared_ptr<ExecutionPlan> a, std::shared_ptr<ExecutionPlan> b) {
            return a->getExecutionTime() < b->getExecutionTime();
        };
        std::set<std::shared_ptr<ExecutionPlan>, decltype(plan_cmp)> timed_execution_plans(plan_cmp);

        const int maxIterCount         = 100;
        const float threshhold         = 0.95f;
        uint64_t successful_plan_count = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaDeviceSynchronize();

        cudaStream_t stream = nullptr;
        cudnnGetStream(handle, &stream);

        for (auto plan : plans->list_of_engine_configs.get_execution_plans()) {
            float time_ms       = 0.0f;
            float final_time_ms = 0.0f;
            float min_time_ms   = std::numeric_limits<float>::max();

            // Warm-up run
            auto warmup_status = cudnnBackendExecute(handle, plan->get_raw_desc(), variantPack.get_raw_desc());
            if (warmup_status != CUDNN_STATUS_SUCCESS) {
                getLogger() << "[cudnn_frontend] Plan " << plan->getTag() << " failed with " << to_string(warmup_status)
                            << std::endl;
                continue;
            }
            successful_plan_count++;
            cudaDeviceSynchronize();

            for (int i = 0; i < maxIterCount; i++) {
                cudaEventRecord(start, stream);

                cudnnBackendExecute(handle, plan->get_raw_desc(), variantPack.get_raw_desc());

                cudaEventRecord(stop, stream);
                cudaEventSynchronize(stop);
                cudaEventElapsedTime(&time_ms, start, stop);

                final_time_ms = std::min(min_time_ms, time_ms);
                if (time_ms / min_time_ms < threshhold) {
                    min_time_ms = final_time_ms;
                } else {
                    break;
                }
            }

            getLogger() << "[cudnn_frontend] Plan " << plan->getTag() << " took " << std::setw(10) << final_time_ms
                        << std::endl;
            plan->setExecutionTime(final_time_ms);
            timed_execution_plans.insert(plan);
        }

        execution_plans.clear();
        for (auto sorted_plan : timed_execution_plans) {
            execution_plans.push_back(sorted_plan);
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        getLogger() << "Autotuned " << successful_plan_count << " plans." << std::endl;
        return {error_code_t::OK, ""};
    }

    std::function<
        error_t(Plans*, cudnnHandle_t, std::unordered_map<std::shared_ptr<Tensor_attributes>, void*>, void*, void*)>
        autotune_impl = &Plans::autotune_default_impl;

    error_t
    autotune(cudnnHandle_t handle,
             std::unordered_map<std::shared_ptr<Tensor_attributes>, void*> variants,
             void* workspace,
             void* user_impl = nullptr) {
        auto error = autotune_impl(this, handle, variants, workspace, user_impl);
        return error;
    }
};

*/

}  // namespace cudnn_frontend::graph
/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <tuple>
#include <unordered_map>
#include <map>
#include <memory>
#include <mutex>

#include <cudnn_frontend_OperationGraph.h>
#include <cudnn_frontend_ExecutionPlan.h>

/// Execution Plan Caching:
/// Goal is to auto-tune once and then save the best auto-tuned result for a problem for later use.
/// For every unique Operation Graph (denoted by a string) we have a set of plans identified by a feature vector.
/// The feature vector could be Tensor dimension/data_type and so on.
/// Multiple operation Graph can share a feature vector type but may have different Execution Plan(s).

/// The v1 cache has the following format.
/// It is the reponsibility of the user to query the correct cache for the given device/operation graph combination.
/***
 *    device_id_0 Operation_Graph0 (conv_fprop)
 *    -------------------------------------------------------------------------------
 *    | Feature_vector_type0_val0   |  Plan A0   |
 *    | Feature_vector_type0_val1   |  Plan B0   |
 *    ===============================================================================
 * 
 *    device_id_0 Operation_Graph1 (dgrad)  
 *    -------------------------------------------------------------------------------
 *    | Feature_vector_type1_val0   |  Plan A1   |
 *    | Feature_vector_type1_val1   |  Plan B1   |
 *    ===============================================================================
 *    
 *    device_id_0 Operation_Graph2 (wgrad)  
 *    -------------------------------------------------------------------------------
 *    | Feature_vector_type2_val0   |  Plan B2   |  
 *    ===============================================================================
 * 
 *    device_id_1 Operation_Graph0 (conv_fprop)
 *    -------------------------------------------------------------------------------
 *    | Feature_vector_type0_val0   |  Plan A0   |
 *    | Feature_vector_type0_val1   |  Plan B0   |
 *    ===============================================================================
 * 
 *    device_id_1 Operation_Graph1 (dgrad)  
 *    -------------------------------------------------------------------------------
 *    | Feature_vector_type1_val0   |  Plan A1   |
 *    | Feature_vector_type1_val1   |  Plan B1   |
 *    ===============================================================================
 *     
 *    device_id_1 Operation_Graph2 (wgrad)  
 *    -------------------------------------------------------------------------------
 *    | Feature_vector_type2_val0   |  Plan B2   |  
 *    ===============================================================================
 */

namespace cudnn_frontend {




/// Plan Cache structure for the above table
class ExecutionPlanCache_v1 {

    struct compare {
        bool operator ()(const feature_vector_t & fv1, const feature_vector_t &fv2) const {
            for (uint32_t i = 0u; i < fv1.size(); i++) {
                if (fv1[i] < fv2[i]) {
                    return true;
                }
            }
        }
    };

    std::string name = "plan_cache_[unnamed]";

    /// String to map of feature_vector to execution plan
    /// For a given FeatureVector of type T according to the Operation Graph, we get the plan. 
    using FeatureVectorToPlanMap = std::map<cudnn_frontend::feature_vector_t, cudnn_frontend::ExecutionPlan, cudnn_frontend::ExecutionPlanCache_v1::compare>;
    FeatureVectorToPlanMap  cache;
    mutable std::mutex cache_mutex;
 public:

    void add_plan_to_cache(const cudnn_frontend::OperationGraph &op_graph,
                           const cudnn_frontend::ExecutionPlan &plan) {
        std::lock_guard<std::mutex> guard(cache_mutex);
        cache.insert(std::make_pair(op_graph.getFeatureVector(),plan));
        getLogger() << "[cudnn_frontend] Added to " << name << " " << op_graph.getTag() << std::endl;
    }

    ExecutionPlanCache_v1(const char * name_) {
        name = name_;
    }

    const std::string & get_name() const {
        return name;
    }

    // Plan is the output here.
    bool get_plan(const cudnn_frontend::OperationGraph &op_graph, 
                  const cudnn_frontend::ExecutionPlan *&plan) const {
        {
            std::lock_guard<std::mutex> guard(cache_mutex);
            auto it = cache.find(op_graph.getFeatureVector());

            if (it == cache.end()) {
                getLogger() << "[cudnn_frontend] Cached Plan Not Found in " << name << std::endl;
                return false;
            }
            plan = &(it->second);
        }
        getLogger() << "[cudnn_frontend] Cached Plan Found in " << name << std::endl;
        return true;
    }
};

using ExecutionPlanCache = ExecutionPlanCache_v1;

}
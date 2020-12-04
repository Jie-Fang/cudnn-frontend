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

#include <cudnn_frontend_EngineConfigGenerator.h>

auto
EngineConfigGenerator::cudnnGetPlan(cudnnHandle_t handle, cudnn_frontend::OperationGraph&& opGraph, Predicate pred)
    -> executionPlans {
    // Creating a set of execution plans that are supported.
    executionPlans plans;
    for (auto& engine_config : generate_engine_config(opGraph)) {
        try {
            plans.push_back(
                cudnn_frontend::ExecutionPlanBuilder().setHandle(handle).setEngineConfig(engine_config).build());
        } catch (cudnn_frontend::cudnnException e) {
            std::cout << "[INFO] Plan is not supported!" << std::endl;
        }
    }
    return filter(pred, plans);
}

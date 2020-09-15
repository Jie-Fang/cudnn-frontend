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

// If filter_fn returns true
// The engine config will be filtered out and will
// not be part of the to list.

static void
filter(std::vector<cudnnBackendDescriptor_t> &from,
       std::vector<cudnnBackendDescriptor_t> &to,
       std::function<bool(cudnnBackendDescriptor_t &)> filter_fn) {
    auto p = std::stable_partition(from.begin(), from.end(), filter_fn);
    // range insert with move
    to.insert(to.end(), std::make_move_iterator(p), std::make_move_iterator(from.end()));
    // erase the moved-from elements.
    from.erase(p, from.end());
}

static bool allowAll(cudnnBackendDescriptor_t & engine_config) {
    return false;
}

static bool
isNonDeterministic(cudnnBackendDescriptor_t &engine_config) {
    bool isNondeterministic         = false;
    cudnnBackendDescriptor_t engine = nullptr;
    cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine);
    int64_t engine_count = -1;
    auto status          = cudnnBackendGetAttribute(
        engine_config, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine_count, &engine);
    if (status == CUDNN_STATUS_SUCCESS) {
        cudnnBackendNumericalNote_t notes[CUDNN_NUMERICAL_NOTE_TYPE_COUNT];
        int64_t elem_count = 0;
        cudnnBackendGetAttribute(engine,
                                 CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                 CUDNN_TYPE_NUMERICAL_NOTE,
                                 CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
                                 &elem_count,
                                 notes);
        if (std::any_of(notes, notes + elem_count, [](cudnnBackendNumericalNote_t note) {
                return note == CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC;
            })) {
            isNondeterministic = true;
        }
    }
    cudnnBackendDestroyDescriptor(engine);
    return isNondeterministic;
}
}

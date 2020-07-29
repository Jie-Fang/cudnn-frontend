#pragma once

#include <cudnn.h>

namespace cudnn_frontend {

// If filter_fn returns true
// The engine config will be filtered out and will
// not be part of the to list.

void
filter(std::vector<cudnnBackendDescriptor_t> &from,
       std::vector<cudnnBackendDescriptor_t> &to,
       std::function<bool(cudnnBackendDescriptor_t &)> filter_fn) {
    auto p = std::stable_partition(from.begin(), from.end(), filter_fn);
    // range insert with move
    to.insert(to.end(), std::make_move_iterator(p), std::make_move_iterator(from.end()));
    // erase the moved-from elements.
    from.erase(p, from.end());
}

bool
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
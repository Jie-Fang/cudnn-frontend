#pragma once
#include <exception>

#include "cudnn_backend_base.h"

namespace cudnn_api_wrapper {

class cudnnException : public std::runtime_error {
   public:
    cudnnException(const char* message) throw() : std::runtime_error(message) {}
    virtual const char*
    what() const throw() {
        return std::runtime_error::what();
    }
};

static inline void
throw_if(std::function<bool()> expr, const char* message) {
    if (expr()) {
        throw cudnnException(message);
    }
}
static inline void
throw_if(bool expr, const char* message) {
    if (expr) {
        throw cudnnException(message);
    }
}

static inline void
set_error_and_throw_exception(BackendDescriptor *desc, cudnnStatus_t status, const char *message) {
    if (desc != nullptr) {
        desc->set_status(status);
        desc->set_error(message);
    }
#ifndef CUDNN_SUPPORTS_EXCEPTION 
    throw cudnnException(message);
#endif
}
}

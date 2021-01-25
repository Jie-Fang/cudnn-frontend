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

class OpaqueBackendPointer {
    cudnnBackendDescriptor_t m_desc = nullptr;  // Raw void pointer
    cudnnStatus_t status            = CUDNN_STATUS_SUCCESS;

   public:
    OpaqueBackendPointer(const OpaqueBackendPointer&) = delete;
    OpaqueBackendPointer&
    operator=(const OpaqueBackendPointer&)       = delete;
    OpaqueBackendPointer(OpaqueBackendPointer&&) = default;

    OpaqueBackendPointer(cudnnBackendDescriptorType_t type) { status = cudnnBackendCreateDescriptor(type, &m_desc); }
    ~OpaqueBackendPointer() { cudnnBackendDestroyDescriptor(m_desc); };
    cudnnBackendDescriptor_t const&
    get_backend_descriptor() const {
        return m_desc;
    }
    cudnnStatus_t
    get_status() const {
        return status;
    }
    bool
    is_good() const {
        return status == CUDNN_STATUS_SUCCESS;
    }
};

using ManagedOpaqueDescriptor = std::shared_ptr<OpaqueBackendPointer>;

static ManagedOpaqueDescriptor
make_shared_backend_pointer(cudnnBackendDescriptorType_t type) {
    return std::make_shared<OpaqueBackendPointer>(type);
}

///
/// BackendDescriptor class
/// Holds pointer to base BackendDescriptor class
class BackendDescriptor {
   public:
    //! Return a string describing the backend Descriptor
    virtual std::string
    describe() const = 0;

    //! Get a copy of the raw descriptor pointer. Ownership is reatined and
    //! gets deleted when out of scope
    cudnnBackendDescriptor_t
    get_raw_desc() const {
        return pointer->get_backend_descriptor();
    }

    //! Current status of the descriptor
    cudnnStatus_t
    get_status() const {
        return status;
    }

    //! Set status of the descriptor
    void
    set_status(cudnnStatus_t const status_) const {
        status = status_;
    }

    //! Set Diagonistic error message.
    void
    set_error(const char* message) const {
        err_msg = message;
    }

    //! Diagonistic error message if any
    const char*
    get_error() const {
        return err_msg.c_str();
    }

    ManagedOpaqueDescriptor
    get_desc() const {
        return pointer;
    }

    cudnnStatus_t
    initialize_managed_backend_pointer(cudnnBackendDescriptorType_t type) {
        pointer = make_shared_backend_pointer(type);
        return pointer->get_status();
    }

   protected:
    BackendDescriptor(ManagedOpaqueDescriptor pointer_, cudnnStatus_t status_, std::string err_msg_)
        : pointer(pointer_), status(status_), err_msg(err_msg_) {}
    BackendDescriptor() = default;

    ManagedOpaqueDescriptor pointer;

    mutable cudnnStatus_t status = CUDNN_STATUS_SUCCESS;  //!< Error code if any being set
    mutable std::string err_msg;                          //!< Error message if any being set
};
}

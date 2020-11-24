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

template <typename T>
using manager = T;

///
/// BackendDescriptor class
/// Holds pointer to base BackendDescriptor class
class BackendDescriptor {
   public:
    //! Return a string describing the backend Descriptor
    virtual std::string
    describe() const = 0;

    //! Get ownersip of the raw descriptor pointer
    cudnn_frontend::manager<cudnnBackendDescriptor_t>
    get_desc() const {
        cudnn_frontend::manager<cudnnBackendDescriptor_t> ptr = desc;
        desc                                                  = nullptr;
        return ptr;
    }

    //! Get a copy of the raw descriptor pointer. Ownership is reatined and
    //! gets deleted when out of scope
    cudnnBackendDescriptor_t
    get_raw_desc() const {
        return desc;
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

   protected:
    //! Constructor
    BackendDescriptor(cudnnBackendDescriptor_t& desc_, cudnnStatus_t status_, std::string err_msg_)
        : desc(desc_), status(status_), err_msg(err_msg_) {
        desc_ = nullptr;
    }
    BackendDescriptor(cudnnBackendDescriptor_t&& desc_) : desc(desc_) {}
    BackendDescriptor() = default;

    mutable cudnnBackendDescriptor_t desc = nullptr;

    mutable cudnnStatus_t status = CUDNN_STATUS_SUCCESS;  //!< Error code if any being set
    mutable std::string err_msg;                          //!< Error message if any being set
};
}

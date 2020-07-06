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
    set_status(cudnnStatus_t const status_) {
        status = status_;
    }

    //! Set Diagonistic error message.
    void
    set_error(const char* message) {
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

    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;  //!< Error code if any being set
    std::string err_msg;                          //!< Error message if any being set
};
}

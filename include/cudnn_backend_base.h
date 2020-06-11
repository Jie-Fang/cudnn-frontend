#pragma once

namespace cudnn_api_wrapper {

template <typename T> using manager = T;

///
/// BackendDescriptor class
/// Holds pointer to base BackendDescriptor class
class BackendDescriptor {
   public:
    //! Return a string describing the backend Descriptor
    virtual std::string
    describe() const = 0;

    //! Get ownersip of the raw descriptor pointer
    cudnn_api_wrapper::manager<cudnnBackendDescriptor_t>
    get_desc() const {
        cudnn_api_wrapper::manager<cudnnBackendDescriptor_t> ptr = desc;
        desc = nullptr;
        return ptr;
    }

    //! Get a copy of the raw descriptor pointer. OWnership is reatined and 
    //! get deleted when out of scope
    cudnnBackendDescriptor_t
    get_raw_desc() const {
        return desc;
    }
   protected:
    //! Constructor
    BackendDescriptor(cudnnBackendDescriptor_t& desc_)  : desc(desc_) { desc_ = nullptr; }
    BackendDescriptor(cudnnBackendDescriptor_t&& desc_) : desc(desc_) {}
    BackendDescriptor()                                                      = default;

    mutable cudnnBackendDescriptor_t desc = nullptr;
};
}

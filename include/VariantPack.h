#pragma once

#include <array>
#include <memory>
#include <sstream>
#include <algorithm>
#include <functional>
#include <utility>
#include <set>

#include <cudnn.h>
#include <cudnn_backend.h>

#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {

///
/// VariantPack Class
/// This class tells the Configuration of the Engine in terms of the knob choices
/// Properties:
///    - num knobs
///    - Choice
///    - Engine
///
/// Use VariantPackBuilder to build this class.
/// Describe returns a string describing the tensor class
///
class VariantPack : public BackendDescriptor {
   public:
    friend class VariantPackBuilder;
    std::string
    describe() const override {
        std::stringstream ss;
        ss << "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR :"
           << " has " << num_ptrs << " data pointers";
        return ss.str();
    }
    VariantPack(VariantPack &&from)
        : BackendDescriptor(from.desc, from.get_status(), from.get_error()),
          workspace(from.workspace),
          num_ptrs(from.num_ptrs) {
        std::copy(std::begin(from.data_pointers), std::end(from.data_pointers), data_pointers);
        std::copy(std::begin(from.uid), std::end(from.uid), uid);
    }
    ~VariantPack() {
        if (desc != nullptr) {
            ::cudnnBackendDestroyDescriptor(desc);
        }
    }

   private:
    VariantPack()                    = default;
    VariantPack(VariantPack const &) = delete;
    VariantPack &
    operator=(VariantPack const &) = delete;

    void *workspace         = nullptr;
    void *data_pointers[10] = {nullptr};
    int64_t uid[10]         = {-1};
    int64_t num_ptrs        = -1;
};

///
/// VariantPackBuilder Class
/// Helper class used to build VariantPack class
class VariantPackBuilder {
   public:
    /** @defgroup VariantPackBuilder
     *  Set individual property of VariantPack class
     *  @{
     */
    //! Set dataPointers for the VariantPack
    auto
    setDataPointers(int64_t num_ptr, void **ptrs) -> VariantPackBuilder & {
        std::copy(ptrs, ptrs + num_ptr, m_variant_pack.data_pointers);
        m_variant_pack.num_ptrs = num_ptr;
        return *this;
    }
    //! Set Uids for the VariantPack
    auto
    setUids(int64_t num_uids, int64_t *uid) -> VariantPackBuilder & {
        std::copy(uid, uid + num_uids, m_variant_pack.uid);
        return *this;
    }
    //! Initialize a set of pairs containing uid and data pointer.
    auto
    setDataPointers(std::set<std::pair<uint64_t, void *>> const &data_pointers) -> VariantPackBuilder & {
        auto i = 0;
        for (auto &data_pointer : data_pointers) {
            m_variant_pack.uid[i]           = data_pointer.first;
            m_variant_pack.data_pointers[i] = data_pointer.second;
            i++;
        }
        m_variant_pack.num_ptrs = data_pointers.size();
        return *this;
    }
    //! Set Workspace
    auto
    setWorkspacePointer(void *ws) -> VariantPackBuilder & {
        m_variant_pack.workspace = ws;
        return *this;
    }
    /** @} */

    //! constructs the Engine Config by calling the cudnn API
    //! Throws the appropriate error message
    VariantPack &&
    build() {
        // Create a descriptor. Memory allocation happens here.
        auto status = CUDNN_STATUS_SUCCESS;
        status      = cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &m_variant_pack.desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_variant_pack, status, "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_variant_pack);
        }

        status = cudnnBackendSetAttribute(m_variant_pack.desc,
                                          CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                          CUDNN_TYPE_VOID_PTR,
                                          m_variant_pack.num_ptrs,
                                          m_variant_pack.data_pointers);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_variant_pack,
                status,
                "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR: SetAttribute CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS Failed");
            return std::move(m_variant_pack);
        }

        status = cudnnBackendSetAttribute(m_variant_pack.desc,
                                          CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                          CUDNN_TYPE_INT64,
                                          m_variant_pack.num_ptrs,
                                          m_variant_pack.uid);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_variant_pack,
                status,
                "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR: SetAttribute CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS Failed");
            return std::move(m_variant_pack);
        }

        status = cudnnBackendSetAttribute(
            m_variant_pack.desc, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &m_variant_pack.workspace);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_variant_pack,
                status,
                "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR: SetAttribute CUDNN_ATTR_VARIANT_PACK_WORKSPACE Failed");
            return std::move(m_variant_pack);
        }

        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_variant_pack.desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_variant_pack, status, "CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_variant_pack);
        }
        return std::move(m_variant_pack);
    }

    explicit VariantPackBuilder()                  = default;
    ~VariantPackBuilder()                          = default;
    VariantPackBuilder(VariantPackBuilder &&)      = delete;
    VariantPackBuilder(VariantPackBuilder const &) = delete;
    VariantPackBuilder &
    operator=(VariantPackBuilder const &) = delete;

   private:
    VariantPack m_variant_pack;
};
}

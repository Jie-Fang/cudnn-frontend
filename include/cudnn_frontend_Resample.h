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

#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <memory>
#include <sstream>
#include <utility>

#include <cudnn.h>
#include <cudnn_backend.h>

#include "cudnn_frontend_utils.h"

namespace cudnn_frontend {

///
/// Resample Descriptor Class
/// This class tells the properties of the Resample operation
/// Properties:
///
/// Use ResampleDescBuilder_v8 to build this class.
/// Describe returns a string describing the Resample operation
///
class ResampleDesc_v8 : public BackendDescriptor {
   public:
    friend class ResampleDescBuilder_v8;
    std::string
    describe() const override {
        std::stringstream ss;
#if (CUDNN_VERSION >= 8500)
        char sep = ',';
         ss << "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR :"
           << " Datatype: " << to_string(compType) << " Mode: " << mode_strings[mode]
           << " Num Dimensions: " << nbSpatialDims 
           << " NanOpt " << std::to_string(nanOpt)
           << " PaddingMode " << padding_mode_strings[paddingMode]
           << " alpha " << alpha
           << " beta " << beta;
        ss << " WindowDimA [ ";
        for (auto i = 0; i < nbSpatialDims; i++) {
            ss << '(' << windowDimA[i].numerator << sep << windowDimA[i].denominator << ')' << sep;
        }
        ss << " ] prePaddingA [ ";
        for (auto i = 0; i < nbSpatialDims; i++) {
            ss << '(' << prePaddingA[i].numerator << sep << prePaddingA[i].denominator << ')' << sep;
        }
        ss << " ] postPaddingA [ ";
        for (auto i = 0; i < nbSpatialDims; i++) {
            ss << '(' << postPaddingA[i].numerator << sep << postPaddingA[i].denominator << ')' << sep;
        }
        ss << " ] strideA [ ";
        for (auto i = 0; i < nbSpatialDims; i++) {
            ss << '(' << strideA[i].numerator << sep << strideA[i].denominator << ')' << sep;
        }
        ss << " ]";
#endif
        return ss.str();
    }

    ResampleDesc_v8(ResampleDesc_v8 &&from) = default;
    ResampleDesc_v8 &
    operator=(ResampleDesc_v8 &&) = default;

    ~ResampleDesc_v8() = default;

    /** @defgroup ResampleDescBuilder_v8
     *  Get individual property of ResampleDesc_v8 class
     *  @{
     */
    
    cudnnDataType_t
    getCompType() const {
        return compType;
    }
    
    int64_t
    getNbSpatialDims() const {
        return nbSpatialDims;
    }

#if (CUDNN_VERSION >= 8500)

    cudnnResampleMode_t
    getMode() const {
        return mode;
    }

    cudnnNanPropagation_t
    getNanOpt() const {
        return nanOpt;
    }

    cudnnPaddingMode_t
    getPaddingMode() const {
        return paddingMode;
    }

    cudnnFraction_t const *
    getStrideA() const {
        return strideA;
    }

    cudnnFraction_t const *
    getPrePaddingA() const {
        return prePaddingA;
    }

    cudnnFraction_t const *
    getPostPaddingA() const {
        return postPaddingA;
    }

    cudnnFraction_t const *
    getWindowDimA() const {
        return windowDimA;
    }
#endif
    /** @} */

   private:

    ResampleDesc_v8()                    = default;
    ResampleDesc_v8(ResampleDesc_v8 const &) = delete;
    ResampleDesc_v8 &
    operator=(ResampleDesc_v8 const &) = delete;

    // default values for attributes 
    cudnnDataType_t compType = CUDNN_DATA_FLOAT;   
    cudnnNanPropagation_t nanOpt = CUDNN_NOT_PROPAGATE_NAN;
    
    int32_t nbSpatialDims = 0;
    double alpha = 1.0;
    double beta = 0.0;

#if (CUDNN_VERSION >= 8500)
    cudnnResampleMode_t mode = CUDNN_RESAMPLE_AVGPOOL;
    cudnnPaddingMode_t paddingMode = CUDNN_ZERO_PAD;
    // Shape attributes
    cudnnFraction_t windowDimA[CUDNN_DIM_MAX] = {{0,1},{0,1}};
    cudnnFraction_t prePaddingA[CUDNN_DIM_MAX] = {{0,1},{0,1}};
    cudnnFraction_t postPaddingA[CUDNN_DIM_MAX] = {{0,1},{0,1}};
    cudnnFraction_t strideA[CUDNN_DIM_MAX] = {{0,1},{0,1}};
#endif

    // string arrays for describing the enum operators
    std::string mode_strings[4] = {"CUDNN_RESAMPLE_NEAREST","CUDNN_RESAMPLE_BILINEAR","CUDNN_RESAMPLE_AVGPOOL","CUDNN_RESAMPLE_MAXPOOL"};
    std::string padding_mode_strings[3] = {"CUDNN_ZERO_PAD", "CUDNN_NEG_INF_PAD", "CUDNN_EDGE_VAL_PAD"};

    };

///
/// ResampleDescBuilder_v8 Class
/// Helper class used to build ResampleDesc_v8 class
class ResampleDescBuilder_v8 {
   public:
    /** @defgroup ResampleDescBuilder_v8
     *  Set individual property of ResampleDesc_v8 class
     *  @{
     */
    //! Set Datatype for the Resample Operation
    auto
    setMathPrecision(cudnnDataType_t data_type_) -> ResampleDescBuilder_v8 & {
        return setComputePrecision(data_type_);
    }

    auto
    setComputePrecision(cudnnDataType_t data_type_) ->  ResampleDescBuilder_v8 & {
        m_resampleDesc.compType = data_type_;
        return *this;
    }

#if (CUDNN_VERSION >= 8500)
    //! (Overloaded) Set post padding for the Resample Operation with cudnnFraction_t
    auto
    setPostPaddingA(cudnnFraction_t *arr) -> ResampleDescBuilder_v8 & {
        std::copy(arr, arr + m_resampleDesc.nbSpatialDims, m_resampleDesc.postPaddingA);
        return *this;
    }
 
    //! (Overloaded) Set post padding for the Resample Operation with int64_t
    auto
    setPostPaddingA(int64_t *arr) -> ResampleDescBuilder_v8 & {
        for (int i = 0; i < m_resampleDesc.nbSpatialDims; i++) {
            m_resampleDesc.postPaddingA[i].numerator = arr[i];
            m_resampleDesc.postPaddingA[i].denominator = 1;
        }
        return *this;
    }

    //! (Overloaded) Set pre padding for the Resample Operation with cudnnFraction_t
    auto
    setPrePaddingA(cudnnFraction_t *arr) -> ResampleDescBuilder_v8 & {
        std::copy(arr, arr + m_resampleDesc.nbSpatialDims, m_resampleDesc.prePaddingA);
        return *this;
    }
    
    //! (Overloaded) Set pre padding for the Resample Operation with int64_t
    auto
    setPrePaddingA(int64_t *arr) -> ResampleDescBuilder_v8 & {
        for (int i = 0; i < m_resampleDesc.nbSpatialDims; i++) {
            m_resampleDesc.prePaddingA[i].numerator = arr[i];
            m_resampleDesc.prePaddingA[i].denominator = 1;
        }
        return *this;
    }

    //! (Overloaded) Set stride for the Resample Operation with cudnnFraction_t
    auto
    setStrideA(cudnnFraction_t *arr) -> ResampleDescBuilder_v8 & {
        std::copy(arr, arr + m_resampleDesc.nbSpatialDims, m_resampleDesc.strideA);
        return *this;
    }
    
    //! (Overloaded) Set stride for the Resample Operation with int64_t
    auto
    setStrideA(int64_t *arr) -> ResampleDescBuilder_v8 & {
        for (int i = 0; i < m_resampleDesc.nbSpatialDims; i++) {
            m_resampleDesc.strideA[i].numerator = arr[i];
            m_resampleDesc.strideA[i].denominator = 1;
        }
        return *this;
    }

    //! (Overloaded) Set window dim for the Resample Operation with cudnnFraction_t
    auto
    setWindowDimA(cudnnFraction_t *arr) -> ResampleDescBuilder_v8 & {
        std::copy(arr, arr + m_resampleDesc.nbSpatialDims, m_resampleDesc.windowDimA);
        return *this;
    }

    //! (Overloaded) Set window dim for the Resample Operation with int64_t
    auto
    setWindowDimA(int64_t *arr) -> ResampleDescBuilder_v8 & {
        for (int i = 0; i < m_resampleDesc.nbSpatialDims; i++) {
            m_resampleDesc.windowDimA[i].numerator = arr[i];
            m_resampleDesc.windowDimA[i].denominator = 1;
        }
        return *this;
    }
    
     //! Set padding mode for the Resample Operation
    auto
    setPaddingMode(cudnnPaddingMode_t paddingMode_) -> ResampleDescBuilder_v8 & {
        m_resampleDesc.paddingMode = paddingMode_;
        return *this;
    }

    //! Set resample mode for the Resample Operation
    auto
    setResampleMode(cudnnResampleMode_t mode_) -> ResampleDescBuilder_v8 & {
        m_resampleDesc.mode = mode_;
        return *this;
    }

#endif

    //! Set beta value for the Resample Operation 
    auto
    setBeta(double beta_) -> ResampleDescBuilder_v8 & {
        m_resampleDesc.alpha = beta_;
        return *this;
    }
    
    //! Set nan propagation mode for the Resample Operation
    auto
    setNanPropagation(cudnnNanPropagation_t nanOpt_) -> ResampleDescBuilder_v8 & {
        m_resampleDesc.nanOpt = nanOpt_;
        return *this;
    }
    
    //! Set alpha value for the Resample Operation
    auto
    setAlpha(double alpha_) -> ResampleDescBuilder_v8 & {
        m_resampleDesc.alpha = alpha_;
        return *this;
    }

    //! Set number of spatial dims value for the Resample Operation
    auto
    setNbSpatialDim(int32_t nbSpatialDims_) -> ResampleDescBuilder_v8 & {
        m_resampleDesc.nbSpatialDims = nbSpatialDims_;
        return *this;
    }

    /** @} */

    //! constructs the ResampleDesc_v8 by calling the cudnn API
    //! Throws the appropriate error message
    ResampleDesc_v8 &&
    build() {
#if (CUDNN_VERSION >= 8500)
        // Sanity check if non-default fields have been set correctly.
        if (m_resampleDesc.nbSpatialDims < 0) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                CUDNN_STATUS_BAD_PARAM,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: Check and Set the nbSpatialDims field");
            return std::move(m_resampleDesc);
        };


        // Create a descriptor. Memory allocation happens here.
        auto status = m_resampleDesc.initialize_managed_backend_pointer(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc, status, "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: Bad descriptor created");
            return std::move(m_resampleDesc);
        }

        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc, status, "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: cudnnCreate Failed");
            return std::move(m_resampleDesc);
        }

        // Once Created lets set the descriptor parameters.
        status = cudnnBackendSetAttribute(m_resampleDesc.pointer->get_backend_descriptor(), 
                                          CUDNN_ATTR_RESAMPLE_MODE, 
                                          CUDNN_TYPE_RESAMPLE_MODE, 
                                          1,
                                          &(m_resampleDesc.mode));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_MODE Failed");
            return std::move(m_resampleDesc);
        }

        status = cudnnBackendSetAttribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_RESAMPLE_COMP_TYPE, 
                                          CUDNN_TYPE_DATA_TYPE,    
                                          1, 
                                          &(m_resampleDesc.compType));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_COMP_TYPE Failed");
            return std::move(m_resampleDesc);
        }

        status = cudnnBackendSetAttribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                            CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION,
                                            CUDNN_TYPE_NAN_PROPOGATION,
                                            1,
                                            &(m_resampleDesc.nanOpt));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_NAN_PROPAGATION Failed");
            return std::move(m_resampleDesc);
        }

        status = cudnnBackendSetAttribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                           CUDNN_ATTR_RESAMPLE_PADDING_MODE, 
                                           CUDNN_TYPE_PADDING_MODE, 
                                           1, 
                                           &(m_resampleDesc.paddingMode));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_PADDING_MODE Failed");
            return std::move(m_resampleDesc);
        }

        status = cudnnBackendSetAttribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                          CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS, 
                                          CUDNN_TYPE_INT64, 
                                          1, 
                                          &(m_resampleDesc.nbSpatialDims));
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS Failed");
            return std::move(m_resampleDesc);
        }

        status = cudnnBackendSetAttribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                            CUDNN_ATTR_RESAMPLE_WINDOW_DIMS,
                                            CUDNN_TYPE_FRACTION,
                                            m_resampleDesc.nbSpatialDims,
                                            m_resampleDesc.windowDimA);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_WINDOW_DIMS Failed");
            return std::move(m_resampleDesc);
        }

        status = cudnnBackendSetAttribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                            CUDNN_ATTR_RESAMPLE_PRE_PADDINGS,
                                            CUDNN_TYPE_FRACTION,
                                            m_resampleDesc.nbSpatialDims,
                                            m_resampleDesc.prePaddingA);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_PRE_PADDINGS Failed");
            return std::move(m_resampleDesc);
        }

        
        status = cudnnBackendSetAttribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                            CUDNN_ATTR_RESAMPLE_POST_PADDINGS,
                                            CUDNN_TYPE_FRACTION,
                                            m_resampleDesc.nbSpatialDims,
                                            m_resampleDesc.postPaddingA);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_POST_PADDINGS Failed");
            return std::move(m_resampleDesc);
        }

        
        status = cudnnBackendSetAttribute(m_resampleDesc.pointer->get_backend_descriptor(),
                                            CUDNN_ATTR_RESAMPLE_STRIDES,
                                            CUDNN_TYPE_FRACTION,
                                            m_resampleDesc.nbSpatialDims,
                                            m_resampleDesc.strideA);
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc,
                status,
                "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: SetAttribute CUDNN_ATTR_RESAMPLE_STRIDES Failed");
            return std::move(m_resampleDesc);
        }


        // Finalizing the descriptor
        status = cudnnBackendFinalize(m_resampleDesc.pointer->get_backend_descriptor());
        if (status != CUDNN_STATUS_SUCCESS) {
            set_error_and_throw_exception(
                &m_resampleDesc, status, "CUDNN_BACKEND_RESAMPLE_DESCRIPTOR: cudnnFinalize Failed");
            return std::move(m_resampleDesc);
        }
        getLogger() << "[cudnn_frontend] " << m_resampleDesc << std::endl;
        return std::move(m_resampleDesc);
#else 
    set_error_and_throw_exception(&m_resampleDesc,
                                    CUDNN_STATUS_NOT_SUPPORTED,
                                    "CUDNN_RESAMPLE_DESCRIPTOR: Not supported in this version");
    return std::move(m_resampleDesc);
#endif
    }

    explicit ResampleDescBuilder_v8()                  = default;
    ~ResampleDescBuilder_v8()                          = default;
    ResampleDescBuilder_v8(ResampleDescBuilder_v8 &&)      = delete;
    ResampleDescBuilder_v8(ResampleDescBuilder_v8 const &) = delete;
    ResampleDescBuilder_v8 &
    operator=(ResampleDescBuilder_v8 const &) = delete;

   private:
    ResampleDesc_v8 m_resampleDesc;
};
}

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:16:14 2017
@author: yanxu
"""

from __future__ import print_function
import re
import math
import argparse
import collections
import json
import traceback

from helpers.Flags import Flags
# import pdb

# gradually adding support to more functions
supported_function_list = [
    "cudnnGraphLibraryConfigInit",
    "cudnnBackendFinalizeGraphVisualize",
    "cudnnBackendFinalizeEngineCfgVisualize",
    "cudnnBackendExecuteGraphVisualize",
    "cudnnBackendExecuteInternal",
    "cudnnConvolutionForward", 
    "cudnnFindConvolutionForwardAlgorithm", "cudnnFindConvolutionForwardAlgorithmEx",
    "cudnnGetConvolutionForwardAlgorithm_v7", "cudnnGetConvolutionForwardAlgorithm",
    "cudnnConvolutionBackwardData",
    "cudnnFindConvolutionBackwardDataAlgorithm", "cudnnFindConvolutionBackwardDataAlgorithmEx",
    "cudnnGetConvolutionBackwardDataAlgorithm_v7", "cudnnGetConvolutionBackwardDataAlgorithm",
    "cudnnConvolutionBackwardFilter", 
    "cudnnFindConvolutionBackwardFilterAlgorithm","cudnnFindConvolutionBackwardFilterAlgorithmEx",
    "cudnnGetConvolutionBackwardFilterAlgorithm_v7", "cudnnGetConvolutionBackwardFilterAlgorithm",
    "cudnnConvolutionBackwardBias",
    "cudnnConvolutionBiasActivationForward",
    "cudnnActivationForward", "cudnnActivationBackward",
    "cudnnPoolingForward", "cudnnPoolingBackward",
    "cudnnSoftmaxForward", "cudnnSoftmaxBackward",
    "cudnnBatchNormalizationForwardInference",
    "cudnnBatchNormalizationForwardTraining",
    "cudnnBatchNormalizationBackward",
    "cudnnBatchNormalizationForwardTrainingEx",
    "cudnnBatchNormalizationBackwardEx",
    "cudnnNormalizationForwardTraining", "cudnnNormalizationBackward",
    "cudnnLRNCrossChannelForward", "cudnnAddTensor",
    "cudnnRNNForwardInference", "cudnnRNNForwardTraining",
    "cudnnRNNBackwardData", "cudnnRNNBackwardWeights",
    "cudnnRNNForwardInferenceEx", "cudnnRNNForwardTrainingEx",
    "cudnnRNNBackwardDataEx", "cudnnRNNBackwardWeightsEx",
    "cudnnRNNForward", "cudnnRNNBackwardData_v8", "cudnnRNNBackwardWeights_v8",
    "cudnnCTCLossInternal"]

supported_flag_list = [
    "conv", "getFwdAlgo", "findFwdAlgo",
    "dgrad", "getBwdDataAlgo", #"findBwdDataAlgo",
    "wgrad", "getBwdFilterAlgo", #"findBwdFilterAlgo",
    "bgrad",
    "convBiasAct",
    "poolf", "poolb", "activationf", "activationb",
    "softmaxf","softmaxb",
    "bnf", "bnfi","bnft","bnb",
    "lrnf","add",
    "RNNf","RNNb","rnnf","rnnb"]

unsupported_flag_list = []

cudnnVersionRegex = re.compile(r"CuDNN \(v([0-9]{4,5})(?: [0-9]+)?\) function", re.MULTILINE + re.DOTALL)

singleVarRegex = re.compile(r"[ ]*([a-zA-Z0-9_]+?): type=([a-zA-Z0-9_ ]+?); val=([a-zA-Z0-9\[\]\(\)\.\+\-,_ ]+?);")
singleJsonRegex = re.compile(r"[ ]*([a-zA-Z0-9_]+?): type=json; val=(.+?);")
singlePtrRegex = re.compile(r"[ ]*([a-zA-Z0-9_]+?): location=([devhost]+?); addr=([a-zA-Z0-9\[\]\(\)\.,_ ]+?);")
NULLPtrRegex = re.compile(r"[ ]*([a-zA-Z0-9_]+?): .+?[;:] .*?NULL_PTR;")
singleStructRegex = re.compile(r"[ ]*([a-zA-Z0-9_]+?): type=([a-zA-Z0-9_ ]+?):")

# Source of truth of knobs from cudnnBackendKnobType_t in include/cudnn_graph.h
# Source of truth of the mapping from getKnobFlag() in test/testEngine.h
backendKnobToTestFlag_convert = {"CUDNN_KNOB_TYPE_SPLIT_K":"knobSplitK=",
                                "CUDNN_KNOB_TYPE_SWIZZLE":"knobSwizzle=",
                                "CUDNN_KNOB_TYPE_TILE_SIZE":"knobTileOpt=",
                                "CUDNN_KNOB_TYPE_USE_TEX":"knobUseTex=",   
                                "CUDNN_KNOB_TYPE_EDGE":"knobEdge=",
                                "CUDNN_KNOB_TYPE_KBLOCK":"knobKBlock=",
                                "CUDNN_KNOB_TYPE_LDGA":"knobLdgA=",
                                "CUDNN_KNOB_TYPE_LDGB":"knobLdgB=",
                                "CUDNN_KNOB_TYPE_CHUNK_K":"knobChunkK=",
                                "CUDNN_KNOB_TYPE_SPLIT_H":"knobSplitH=",
                                "CUDNN_KNOB_TYPE_WINO_TILE":"knobWinoTile=",
                                "CUDNN_KNOB_TYPE_MULTIPLY":"knobMultiply=",
                                "CUDNN_KNOB_TYPE_SPLIT_K_BUF":"knobSplitKBuf=",
                                "CUDNN_KNOB_TYPE_TILEK":"knobTileK=",
                                "CUDNN_KNOB_TYPE_STAGES":"knobStages=",
                                "CUDNN_KNOB_TYPE_REDUCTION_MODE":"knobReductionMode=",
                                "CUDNN_KNOB_TYPE_CTA_SPLIT_K_MODE":"knobCTASplitKMode=",
                                "CUDNN_KNOB_TYPE_SPLIT_K_SLC":"knobSplitKSlices=",
                                "CUDNN_KNOB_TYPE_IDX_MODE":"knobIdxMode=",
                                "CUDNN_KNOB_TYPE_SLICED":"knobSliced=",
                                "CUDNN_KNOB_TYPE_SPLIT_RS":"knobSplitRS=",
                                "CUDNN_KNOB_TYPE_SINGLEBUFFER":"knobSingleBuffer=",
                                "CUDNN_KNOB_TYPE_LDGC":"knobLdgC=",
                                "CUDNN_KNOB_TYPE_SPECFILT":"knobSpecFilt=",
                                "CUDNN_KNOB_TYPE_KERNEL_CFG":"knobKernelCfg=",
                                "CUDNN_KNOB_TYPE_WORKSPACE":"knobWorkspaceOpt=",
                                "CUDNN_KNOB_TYPE_TILE_CGA":"knobTileCGA=",
                                "CUDNN_KNOB_TYPE_TILE_CGA_M":"knobTileCgaM=",
                                "CUDNN_KNOB_TYPE_TILE_CGA_N":"knobTileCgaN=",
                                "CUDNN_KNOB_TYPE_BLOCK_SIZE":"knobBlockOpt=",
                                "CUDNN_KNOB_TYPE_OCCUPANCY":"knobOccupancy=",
                                "CUDNN_KNOB_TYPE_ARRAY_SIZE_PER_THREAD":"knobNumPerThread=",
                                "CUDNN_KNOB_TYPE_NUM_C_PER_BLOCK":"knobNumPerCTA=",
                                "CUDNN_KNOB_TYPE_SPLIT_COLS":"knobSplitCols=",
                                "CUDNN_KNOB_TYPE_TILE_ROWS":"knobTileRows=",
                                "CUDNN_KNOB_TYPE_TILE_COLS":"knobTileCols=",
                                "CUDNN_KNOB_TYPE_LOAD_SIZE":"knobLoadSize=",
                                "CUDNN_KNOB_TYPE_CTA_COUNT":"knobCtaCount=",
                                "CUDNN_KNOB_TYPE_STREAM_K":"knobStreamK=",
                                "CUDNN_KNOB_TYPE_SPLIT_P_SLC":"knobSplitPSlices=",
                                "CUDNN_KNOB_TYPE_TILE_M":"knobTileM=",
                                "CUDNN_KNOB_TYPE_TILE_N":"knobTileN=",
                                "CUDNN_KNOB_TYPE_WARP_SPEC_CFG":"knobWarpSpecCfg=",
                                }

cudnnConvolutionBwdDataAlgo_convert = {
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 (0)" : "0",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 (1)" : "1",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT (2)" : "2",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING (3)" : "3",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD (4)" : "4",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED (5)" : "5",
    "CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT (6)" : "6"}

cudnnConvolutionFwdAlgo_convert = {
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM (0)" : "0",
    "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM (1)" : "1",
    "CUDNN_CONVOLUTION_FWD_ALGO_GEMM (2)" : "2",
    "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT (3)" : "3",
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT (4)" : "4",
    "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING (5)" : "5",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD (6)" : "6",
    "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED (7)" : "7",
    "CUDNN_CONVOLUTION_FWD_ALGO_COUNT (8)" : "8"}

cudnnConvolutionBwdFilterAlgo_convert = {
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 (0)" : "0",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 (1)" : "1",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT (2)" : "2",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 (3)" : "3",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD (4)" : "4",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED (5)" : "5",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING (6)" : "6",
    "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT (7)" : "7"}

cudnnCTCLossAlgo_convert = {
    "CUDNN_CTC_LOSS_ALGO_DETERMINISTIC (0)" : "0",
    "CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC (1)" : "1"}

cudnnCTCLossNormMode_convert = {
    "CUDNN_LOSS_NORMALIZATION_NONE (0)" : "0",
    "CUDNN_LOSS_NORMALIZATION_SOFTMAX (1)" : "1"}

cudnnCTCLossGradMode_convert = {
    "CUDNN_NOT_PROPAGATE_NAN (0)" : "0",
    "CUDNN_PROPAGATE_NAN (1)" : "1"}

cudnnActivationMode_convert = {
    "CUDNN_ACTIVATION_SIGMOID (0)"      : "0",
    "CUDNN_ACTIVATION_RELU (1)"         : "1",
    "CUDNN_ACTIVATION_TANH (2)"         : "2",
    "CUDNN_ACTIVATION_CLIPPED_RELU (3)" : "3",
    "CUDNN_ACTIVATION_ELU (4)"          : "4",
    "CUDNN_ACTIVATION_IDENTITY (5)"     : "5"}

cudnnNanPropagation_convert = {
    "CUDNN_NOT_PROPAGATE_NAN (0)" : "0",
    "CUDNN_PROPAGATE_NAN (1)"     : "1"}

cudnnSoftmaxAlgorithm_convert = {
    "CUDNN_SOFTMAX_FAST (0)" : "0",
    "CUDNN_SOFTMAX_ACCURATE (1)" : "1",
    "CUDNN_SOFTMAX_LOG (2)" : "2"}

cudnnSoftmaxMode_convert = {
    "CUDNN_SOFTMAX_MODE_INSTANCE (0)": "0",
    "CUDNN_SOFTMAX_MODE_CHANNEL (1)": "1"}

cudnnNormMode_convert = { "CUDNN_NORM_PER_ACTIVATION (0)" : "0",
                          "CUDNN_NORM_PER_CHANNEL (1)" : "1"}

cudnnBatchNormalizationMode_convert = { "CUDNN_BATCHNORM_PER_ACTIVATION (0)" : "0",
                                        "CUDNN_BATCHNORM_SPATIAL (1)" : "1",
                                        "CUDNN_BATCHNORM_SPATIAL_PERSISTENT (2)" : "2"}

cudnnPoolingMode_convert = {    "CUDNN_POOLING_MAX (0)": "0",
                                "CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING (1)": "1",
                                "CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING (2)": "2",
                                "CUDNN_POOLING_MAX_DETERMINISTIC (3)": "3"}

cudnnRNNModeConvert = {     "CUDNN_RNN_RELU (0)" : "0",
                            "CUDNN_RNN_TANH (1)" : "1",
                            "CUDNN_LSTM (2)" : "2",
                            "CUDNN_GRU (3)" : "3"}

# -rnnPaddingMode0: Packed Seq Major Old API (default),
# -rnnPaddingMode1: Packed Seq Major Extended API, i.e. "CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED (1)"
# -rnnPaddingMode2: Padded Seq Major Extended API, i.e. "CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED (0)"
# -rnnPaddingMode3: Padded Batch Major Extended API, i.e. "CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED (2)"
# Updated: "-rnnPaddingMode" option is replaced with "-rnnDataLayout" in https://p4sw-swarm.nvidia.com/changes/33009188
cudnnRNNDataLayoutConvert = {  "CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED (0)" : "0",
                                "CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED (1)" : "1",
                                "CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED (2)" : "2"}

cudnnDirectionModeConvert = {   "CUDNN_UNIDIRECTIONAL (0)" : "0",
                                "CUDNN_BIDIRECTIONAL (1)" : "1"}

cudnnRNNAlgoConvert = { "CUDNN_RNN_ALGO_STANDARD (0)" : "0",
                        "CUDNN_RNN_ALGO_PERSIST_STATIC (1)" : "1",
                        "CUDNN_RNN_ALGO_PERSIST_DYNAMIC (2)" : "2",
                        "CUDNN_RNN_ALGO_PERSIST_STATIC_SMALL_H (3)": "3"}

cudnnRNNBiasModeConvert = { "CUDNN_RNN_NO_BIAS (0)" : "0",
                            "CUDNN_RNN_SINGLE_INP_BIAS (1)" : "1",
                            "CUDNN_RNN_DOUBLE_BIAS (2)" : "2",
                            "CUDNN_RNN_SINGLE_REC_BIAS (3)" : "3"}

format_convert = {      "CUDNN_TENSOR_NCHW (0)": "0",
                        "CUDNN_TENSOR_NHWC (1)": "1",
                        "CUDNN_TENSOR_NCHW_VECT_C (2)" : "2",
                        "CUDNN_TENSOR_CHWN (3)" : "3"}

type_convert = {
                        "CUDNN_DATA_FLOAT (0)":    "s",
                        "CUDNN_DATA_DOUBLE (1)":   "d",
                        "CUDNN_DATA_HALF (2)":     "h",
                        "CUDNN_DATA_INT8 (3)":     "b",
                        "CUDNN_DATA_INT32 (4)":    "b",
                        "CUDNN_DATA_INT8x4 (5)":   "b",
                        "CUDNN_DATA_UINT8 (6)":    "u",
                        "CUDNN_DATA_UINT8x4 (7)":  "u",
                        "CUDNN_DATA_INT8x32 (8)":  "b",
                        "CUDNN_DATA_BFLOAT16 (9)": "g"}

mathType_convert = {
                        "CUDNN_DEFAULT_MATH (0)":    "0",
                        "CUDNN_TENSOR_OP_MATH (1)":   "1",
                        "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION (2)": "2",
                        "CUDNN_FMA_MATH (3)": "3",}

normAlgo_convert = {
    "CUDNN_NORM_ALGO_STANDARD (0)" : "0",
    "CUDNN_NORM_ALGO_PERSIST (1)" : "1"}

normOps_convert = {
    "CUDNN_NORM_OPS (0)" : "0",
    "CUDNN_NORM_OPS_ACTIVATION (1)" : "1",
    "CUDNN_NORM_OPS_NORM_ADD_ACTIVATION (2)" : "2"}

bnOps_convert = {
    "CUDNN_BATCHNORM_OPS_BN (0)" : "0,0",
    "CUDNN_BATCHNORM_OPS_BN_ACTIVATION (1)" : "0,1",
    "CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION (2)" : "1,1"}

# keep "seed" at the moment
cudnnTestKeyWord = ["&&&&", "@@@@", "####", "^^^^", "Gflops", "PERF"]
nonDeterministicKeyWord = ["Time", "time", "pid", "Process", "tid", "Thread", "Handle", "handle", "StreamId", "addr", "plan"] # "Gflops", "^^^^"


def stripNonDeterministic(fileContent, mode):
    # just delete all lines that show nondeterminstic things like time, pid, tid, pointer, seeds
    # mode 0 is print, mode 1 is return

    if mode == 1:
        cleanLog = ""

    for line in fileContent.split('\n'):
        # line = line.lower()
        line = line.replace("I! ","")
        line = line.replace("i! ","")

        if ("^^^^" in line) or ("Gflops" in line) or ("PERF" in line):
            # need to skip, not replace, because second run with "-T1" will have different number of these lines
            continue
        elif any(keyWord in line for keyWord in nonDeterministicKeyWord):
            line="(removed)"
        if mode == 0:
            print(line)
        elif mode == 1:
            cleanLog += (line + "\n")

    if mode == 0:
        return ""
    elif mode == 1:
        return cleanLog

def stripNonLog(fileContent, mode):
    # just delete all lines that show nondeterminstic things like time, pid, tid, pointer, seeds
    # mode 0 is print, mode 1 is return

    if mode == 1:
        cleanLog = ""

    for line in fileContent.split('\n'):
        # line = line.lower()
        line = line.replace("I! ","")
        line = line.replace("i! ","")

        if any(keyWord in line for keyWord in cudnnTestKeyWord):
            # need to skip, not replace, because second run with "-T1" will have different number of these lines
            continue
        elif line == "":
            continue
        elif any(keyWord in line for keyWord in nonDeterministicKeyWord):
            line="(removed)"
        if mode == 0:
            print(line)
        elif mode == 1:
            cleanLog += (line + "\n")

    if mode == 0:
        return ""
    elif mode == 1:
        return cleanLog

def stripLog(fileContent, mode):
    # just delete all lines that show nondeterminstic things like time, pid, tid, pointer, seeds
    # mode 0 is print, mode 1 is return

    if mode == 1:
        cleanLog = ""

    for line in fileContent.split('\n'):
        if any(keyWord in line for keyWord in ["I! ","i! "]):
            pass
        elif line == "":
            pass
        else:
            if mode == 0:
                print(line)
            elif mode == 1:
                cleanLog += (line + "\n")

    if mode == 1:
        return cleanLog


def getCudnnVersion(logSegment):
    version = re.findall(cudnnVersionRegex, logSegment)
    if not version:
        version = "alpha"
    else:
        version = version[0]
    return version

def getFunctionBlockRegexFromVersion(version):
    major = int(version[0])
    if major < 7:
        major = 8
        print("!!!! Warning version" + version + " is not supported. Default: assume v8 format.")
    if major > 9:
        major = 9
        print("!!!! Warning version" + version + " is not supported. Default: assume v9 format.")
    if major == 8:
        return re.compile(r"(CuDNN \(v{}\d{{3}}\) function \w+\(\) called:[a-zA-Z0-9+-=_:;.\(\)\[\]{{}}\"\r\n ]*?Time: [0-9Tdhmsincetar:\+\(\)\-\. ]+?[\r\n]+Process=[a-zA-Z0-9_=;\(\) ]*?\.)".format(major), re.MULTILINE + re.DOTALL)
    else: # freeze older version above for compatibality, and put latest below
        return re.compile(r"(CuDNN \(v{}\d{{4}}(?: [0-9]+)?\) function \w+\(\) called:[a-zA-Z0-9+-=_:;.\(\)\[\]{{}}\"\r\n ]*?Time: [0-9Tdhmsincetar:\+\(\)\-\. ]+?[\r\n]+Process=[a-zA-Z0-9_=;\(\) ]*?\.)".format(major), re.MULTILINE + re.DOTALL)

def getSingleFuncCallRegexFromVersion(version):
    major = int(version[0])
    if major < 7:
        major = 8
        print("!!!! Warning version" + version + " is not supported. Default: assume v8 format.")
    if major > 9:
        major = 9
        print("!!!! Warning version" + version + " is not supported. Default: assume v9 format.")
    if major == 8:
        return re.compile(r"CuDNN \(v{}\d{{3}}\) function (\w+)\(\) called:[ ]*[\r\n]+(.*?)Time: [0-9Tdhmsincetar:\+\(\)\-\. ]+?[\r\n]+Process=[a-zA-Z0-9_=;\(\) ]*?\.".format(major), re.MULTILINE + re.DOTALL)
    else: # freeze older version above for compatibality, and put latest below
        return re.compile(r"CuDNN \(v{}\d{{4}}(?: [0-9]+)?\) function (\w+)\(\) called:[ ]*[\r\n]+(.*?)Time: [0-9Tdhmsincetar:\+\(\)\-\. ]+?[\r\n]+Process=[a-zA-Z0-9_=;\(\) ]*?\.".format(major), re.MULTILINE + re.DOTALL)

def logToTest_compareExecutionPath( output1, output2):

    output1 = output1.replace("I! ","")
    output1 = output1.replace("i! ","")
    output2 = output2.replace("I! ","")
    output2 = output2.replace("i! ","")

    version = getCudnnVersion(output1)
    singleFuncCallRegex = getSingleFuncCallRegexFromVersion(version)

    funcCallList1 = re.findall(singleFuncCallRegex, output1)
    funcCallList2 = re.findall(singleFuncCallRegex, output2)

    computeCallList1 = []
    computeCallList2 = []

    for singleFuncCall in funcCallList1:
        funcName = singleFuncCall[0]
        if funcName in supported_function_list:
            computeCallList1.append((funcName,stripNonLog(singleFuncCall[1],1)))

    for singleFuncCall in funcCallList2:
        funcName = singleFuncCall[0]
        if funcName in supported_function_list:
            computeCallList2.append((funcName,stripNonLog(singleFuncCall[1],1)))

    if computeCallList1 == computeCallList2:
        return True
    else:
        print("===========\ntestLog1=\n" + output1 + "===========\n")
        print("===========\ntestLog2=\n" + output2 + "===========\n")
        return False

def post_processing(flags):
    if "A" in flags:
        valList = [float(i) for i in flags["A"][0].split(",")]
        areIntegers = [i.is_integer() for i in valList]
        if (all(areIntegers)):
            intArr = [int(i) for i in valList]
            flags["A"] = (",".join([str(i) for i in intArr]),)
    if "B" in flags:
        valList = [float(i) for i in flags["B"][0].split(",")]
        areIntegers = [i.is_integer() for i in valList]
        if (all(areIntegers)):
            intArr = [int(i) for i in valList]
            flags["B"] = (",".join([str(i) for i in intArr]),)
    return flags

def get_cudnn_flags(func_name, params, version):

    flags = Flags()

    if(func_name in  ["cudnnGraphLibraryConfigInit"]):
        flags["Dlib_config="] = (params["apiLog"],)
        return post_processing(flags)

    if(func_name in  ["cudnnBackendFinalizeGraphVisualize",
                      "cudnnBackendFinalizeEngineCfgVisualize",
                      "cudnnBackendExecuteGraphVisualize"]):
        flags["R"] = ("graphRunner",)

        graph_json = json.loads(params["apiLog"])
        if "engine" in graph_json.keys():
            engine_json = graph_json["engine"]
            if "engineId" in engine_json.keys():
                flags["backendEngine"] = (str(engine_json["engineId"]),)
            if "knobChoices" in engine_json and isinstance(engine_json["knobChoices"], dict):
                for k, v in engine_json["knobChoices"].items():
                    flags[backendKnobToTestFlag_convert[k]] = (str(v),)
            if "smVersion" in engine_json.keys():
                flags["minDevVer"] = (str(engine_json["smVersion"]),)

        flags["b"] = ("",)
        flags["gpuRef"] = ("",)

        # add a comment tag to indicate the source API call
        flags["jsonTag="] = (re.sub(r"^(cudnnBackend)?(.*?)(Visualize)?$", r"\2", func_name),)

        # remove extraneous information
        graph_json.pop("cudnnVersion", None)
        graph_json.pop("schemaVersion", None)
        graph_json.pop("engine", None)
        flags["jsonStr="] = ("'" + json.dumps(graph_json) + "'",)
        return post_processing(flags)

    if(func_name in  ["cudnnBackendExecuteInternal"]):
        if "operation" in params:
            if params["operation"] == "CONV_FORWARD":
                flags["R"] = ("conv",)
            elif params["operation"] == "CONV_BWD_DATA":
                flags["R"] = ("dgrad",)
            elif params["operation"] == "CONV_BWD_FILTER":
                flags["R"] = ("wgrad",)
            elif params["operation"] == "CONV_ADD_BIAS_ACT":
                flags["R"] = ("convBiasActBackend",)

        if "engine_id" in params:
            flags["backendEngine"] = (params["engine_id"],)

        if "target_sm_count" in params:
            if params["target_sm_count"] != "0":
                flags["targetSMCount"] = (params["target_sm_count"],)

        if "knobDesc" in params:
            if params["knobDesc"] != "NULL_PTR":
                for knob in params["knobDesc"]:
                    flags[backendKnobToTestFlag_convert[knob]] = (params["knobDesc"][knob],)

        inImageName  = "xDesc"
        inFilterName = "wDesc"
        outImageName = "yDesc"

        kc_multipler = 1
        if "8x32" in params[inImageName]["dataType"]:
            kc_multipler = 32
        if "8x4" in params[inImageName]["dataType"]:
            kc_multipler = 4
        if("Forward" in func_name):
            flags["R"] = ("conv",)

        if "dataType" in params[inImageName]:
            flags["Pin"] = (type_convert[params[inImageName]["dataType"]],)
            if "x32" in params[inImageName]["dataType"]:
                flags["nc32"] = ("",)
        if "dataType" in params[outImageName]:
            flags["Pout"] = (type_convert[params[outImageName]["dataType"]],)
        if "dataType" in params["convDesc"]:
            flags["Pcomp"] = (type_convert[params["convDesc"]["dataType"]],)

        if "nbDims" in params[inImageName]:
            nbDims = int(params[inImageName]["nbDims"])
            if nbDims == 5:
                flags["dim"] = ("3",)

        if "dimA" in params[inImageName]:
            params[inImageName]["dimA"][1] = str(int(params[inImageName]["dimA"][1]) * kc_multipler)
            flags["dimA"] = (",".join(params[inImageName]["dimA"][:nbDims]),)
        if "strideA" in params[inImageName]:
            params[inImageName]["strideA"][0] = str(int(params[inImageName]["strideA"][0]) * kc_multipler)
            flags["strideA"] = (",".join(params[inImageName]["strideA"][:nbDims]),)

        params[inFilterName]["dimA"][1] = str(int(params[inFilterName]["dimA"][1]) * kc_multipler)
        params[inFilterName]["dimA"][0] = str(int(params[inFilterName]["dimA"][0]) * kc_multipler)
        flags["filtA"] = (",".join(params[inFilterName]["dimA"][:nbDims]),)
        if "format" in params[inFilterName]:
            flags["filtFormat"] = (format_convert[params[inFilterName]["format"]],)

        if "mode" in params["convDesc"]:
            if(params["convDesc"]["mode"]=="CUDNN_CONVOLUTION (0)"):
                pass
            elif(params["convDesc"]["mode"]=="CUDNN_CROSS_CORRELATION (1)"):
                flags["x"] = ("",)
        # if "mathType" in params["convDesc"]:
        # no -Pmath when -backendEngine is present. see https://nvbugswb.nvidia.com/NvBugs5/SWBug.aspx?bugid=3755803&cmtNo=10
        #     flags["Pmath"] = (mathType_convert[params["convDesc"]["mathType"]],)
        if "padA" in params["convDesc"]:
            flags["padA"] = (",".join(params["convDesc"]["padA"][:(nbDims-2)]),)
        if "strideA" in params["convDesc"]:
            flags["convStrideA"] = (",".join(params["convDesc"]["strideA"][:(nbDims-2)]),)
        if "dilationA" in params["convDesc"]:
            flags["dilationA"] = (",".join(params["convDesc"]["dilationA"][:(nbDims-2)]),)
        if "groupCount" in params["convDesc"]:
            if params["convDesc"]["groupCount"] != "1":
                flags["groupCount"] = (params["convDesc"]["groupCount"],)

        if("alpha" in params):
            flags["A"] = (params["alpha"],)
        if("alpha1" in params and "alpha2" in params):
            flags["A"] = (",".join([params["alpha1"], params["alpha2"]]),)
        elif("alpha1" in params):
            flags["A"] = (params["alpha1"], )

        if("beta" in params):
            flags["B"] = (params["beta"],)

        if "strideA" in params[outImageName]:
            nbDims_yDesc = int(params[outImageName]["nbDims"])
            params[outImageName]["strideA"][0] = str(int(params[outImageName]["strideA"][0]) * kc_multipler)
            flags["strideOut"] = (",".join(params[outImageName]["strideA"][:nbDims_yDesc]),)
        flags["b"] = ("",)

        if params["operation"] == "CONV_ADD_BIAS_ACT":
            if params["activationDesc"]["mode"] == "CUDNN_ACTIVATION_SIGMOID (0)":
                flags["activFunc"] = ("0",)
            elif params["activationDesc"]["mode"] == "CUDNN_ACTIVATION_RELU (1)":
                flags["activFunc"] = ("1",)
            elif params["activationDesc"]["mode"] == "CUDNN_ACTIVATION_TANH (2)":
                flags["activFunc"] = ("2",)
            elif params["activationDesc"]["mode"] == "CUDNN_ACTIVATION_CLIPPED_RELU (3)":
                flags["activCoef"] = (params["activationDesc"]["coef"],)
                flags["activFunc"] = ("3",)
            elif params["activationDesc"]["mode"] == "CUDNN_ACTIVATION_ELU (4)":
                flags["activCoef"] = (params["activationDesc"]["coef"],)
                flags["activFunc"] = ("4",)
            elif params["activationDesc"]["mode"] == "CUDNN_ACTIVATION_IDENTITY (5)":
                flags["activFunc"] = ("5",)
            elif params["activationDesc"]["mode"] == "CUDNN_ACTIVATION_SWISH (6)":
                flags["activFunc"] = ("6",)

            if params["activationDesc"]["reluNanOpt"] == "CUDNN_PROPAGATE_NAN (1)":
                flags["propnan"] = ("",)

        return post_processing(flags)

    if(func_name in ["cudnnConvolutionForward"]):

        # Assume forward
        inImageName  = "xDesc"
        inFilterName = "wDesc"
        outImageName = "yDesc"

        kc_multipler = 1
        if "8x32" in params[inImageName]["dataType"]:
            kc_multipler = 32
        if "8x4" in params[inImageName]["dataType"]:
            kc_multipler = 4
        if("Forward" in func_name):
            flags["R"] = ("conv",)

        algo = cudnnConvolutionFwdAlgo_convert[params["algo"]]
        flags["algo"] = (algo,)

        if "dataType" in params[inImageName]:
            flags["Pin"] = (type_convert[params[inImageName]["dataType"]],)
            if "x32" in params[inImageName]["dataType"]:
                flags["nc32"] = ("",)
        if "dataType" in params[outImageName]:
            flags["Pout"] = (type_convert[params[outImageName]["dataType"]],)
        if "dataType" in params["convDesc"]:
            flags["Pcomp"] = (type_convert[params["convDesc"]["dataType"]],)

        if "nbDims" in params[inImageName]:
            nbDims = int(params[inImageName]["nbDims"])
            if nbDims == 5:
                flags["dim"] = ("3",)

        if "dimA" in params[inImageName]:
            params[inImageName]["dimA"][1] = str(int(params[inImageName]["dimA"][1]) * kc_multipler)
            flags["dimA"] = (",".join(params[inImageName]["dimA"][:nbDims]),)
        if "strideA" in params[inImageName]:
            flags["strideA"] = (",".join(params[inImageName]["strideA"][:nbDims]),)

        params[inFilterName]["dimA"][1] = str(int(params[inFilterName]["dimA"][1]) * kc_multipler)
        flags["filtA"] = (",".join(params[inFilterName]["dimA"][:nbDims]),)
        if "format" in params[inFilterName]:
            flags["filtFormat"] = (format_convert[params[inFilterName]["format"]],)

        if "mode" in params["convDesc"]:
            if(params["convDesc"]["mode"]=="CUDNN_CONVOLUTION (0)"):
                pass
            elif(params["convDesc"]["mode"]=="CUDNN_CROSS_CORRELATION (1)"):
                flags["x"] = ("",)
        if "mathType" in params["convDesc"]:
            flags["Pmath"] = (mathType_convert[params["convDesc"]["mathType"]],)
        if "padA" in params["convDesc"]:
            flags["padA"] = (",".join(params["convDesc"]["padA"][:(nbDims-2)]),)
        if "strideA" in params["convDesc"]:
            flags["convStrideA"] = (",".join(params["convDesc"]["strideA"][:(nbDims-2)]),)
        if "dilationA" in params["convDesc"]:
            flags["dilationA"] = (",".join(params["convDesc"]["dilationA"][:(nbDims-2)]),)
        if "groupCount" in params["convDesc"]:
            if params["convDesc"]["groupCount"] != "1":
                flags["groupCount"] = (params["convDesc"]["groupCount"],)

        if("alpha" in params):
            flags["A"] = (params["alpha"],)
        if("beta" in params):
            flags["B"] = (params["beta"],)

        if "strideA" in params[outImageName]:
            nbDims_yDesc = int(params[outImageName]["nbDims"])
            flags["strideOut"] = (",".join(params[outImageName]["strideA"][:nbDims_yDesc]),)
        flags["b"] = ("",)

        return post_processing(flags)

    if(func_name in ["cudnnConvolutionBackwardData", "cudnnConvolutionBackwardFilter"]):
        outDataTensor = "dyDesc"
        if(func_name == "cudnnConvolutionBackwardData"):
            flags["R"] = ("dgrad",)
            algo = cudnnConvolutionBwdDataAlgo_convert[params["algo"]]
            flags["algo"] = (algo,)
            inDataTensor = "dxDesc"
            filterTensor = "wDesc"

        elif(func_name == "cudnnConvolutionBackwardFilter"):
            flags["R"] = ("wgrad",)
            algo = cudnnConvolutionBwdFilterAlgo_convert[params["algo"]]
            flags["algo"] = (algo,)
            inDataTensor = "xDesc"
            filterTensor = "dwDesc"

        # extract Pin Pout Pcomp flags
        if "dataType" in params[inDataTensor]:
            flags["Pin"] = (type_convert[params[inDataTensor]["dataType"]],)
        if "dataType" in params[outDataTensor]:
            flags["Pout"] = (type_convert[params[outDataTensor]["dataType"]],)
        if "dataType" in params["convDesc"]:
            flags["Pcomp"] = (type_convert[params["convDesc"]["dataType"]],)
        if "mathType" in params["convDesc"]:
            flags["Pmath"] = (mathType_convert[params["convDesc"]["mathType"]],)

        if(params["convDesc"]["mode"]=="CUDNN_CONVOLUTION (0)"):
            pass
        elif(params["convDesc"]["mode"]=="CUDNN_CROSS_CORRELATION (1)"):
            flags["x"] = ("",)

        if params["convDesc"]["groupCount"] != "1":
            flags["groupCount"] = (params["convDesc"]["groupCount"],)

        nbDims = int(params[inDataTensor]["nbDims"])

        if nbDims == 5:
            flags["dim"] = ("3",)

        flags["dimA"] = (",".join(params[inDataTensor]["dimA"][:nbDims]),)
        if "strideA" in params[inDataTensor]:
            flags["strideA"] = (",".join(params[inDataTensor]["strideA"][:nbDims]),)

        flags["filtA"] = (",".join(params[filterTensor]["dimA"][:nbDims]),)
        if "format" in params[filterTensor]:
            flags["filtFormat"] = (format_convert[params[filterTensor]["format"]],)

        if "mathType" in params["convDesc"]:
            flags["Pmath"] = (mathType_convert[params["convDesc"]["mathType"]],)
        if "padA" in params["convDesc"]:
            flags["padA"] = (",".join(params["convDesc"]["padA"][:(nbDims-2)]),)
        if "strideA" in params["convDesc"]:
            flags["convStrideA"] = (",".join(params["convDesc"]["strideA"][:(nbDims-2)]),)
        if "dilationA" in params["convDesc"]:
            flags["dilationA"] = (",".join(params["convDesc"]["dilationA"][:(nbDims-2)]),)

        if("alpha" in params):
            flags["A"] = (params["alpha"],)
        if("beta" in params):
            flags["B"] = (params["beta"],)

        nbDims_yDesc = int(params[outDataTensor]["nbDims"])
        if "strideA" in params[outDataTensor]:
            flags["strideOut"] = (",".join(params[outDataTensor]["strideA"][:nbDims_yDesc]),)

        flags["b"] = ("",)

        return post_processing(flags)

    if(func_name in ["cudnnConvolutionBackwardBias"]):

        flags["R"] = ("bgrad",)

        if "dataType" in params["srcDesc"]:
            flags["P"] = (type_convert[params["srcDesc"]["dataType"]],)

        nbDims = int(params["srcDesc"]["nbDims"])
        if nbDims == 5:
            flags["dim"] = ("3",)
        if "dimA" in params["srcDesc"]:
            flags["dimA"] = (",".join(params["srcDesc"]["dimA"][:nbDims]),)
        if "strideA" in params["srcDesc"]:
            flags["strideA"] = (",".join(params["srcDesc"]["strideA"][:nbDims]),)

        if("alpha" in params):
            flags["A"] = (params["alpha"],)
        if("beta" in params):
            flags["B"] = (params["beta"],)

        flags["b"] = ("",)

        return post_processing(flags)

    if(func_name in ["cudnnConvolutionBiasActivationForward"]):

        inImageName  = "xDesc"
        inFilterName = "wDesc"
        outImageName = "yDesc"

        flags["R"] = ("convBiasAct",)

        algo = cudnnConvolutionFwdAlgo_convert[params["algo"]]
        flags["algo"] = (algo,)

        if "dataType" in params[inImageName]:
            flags["Pin"] = (type_convert[params[inImageName]["dataType"]],)
            if "x32" in params[inImageName]["dataType"]:
                flags["nc32"] = ("",)
        if "dataType" in params[outImageName]:
            flags["Pout"] = (type_convert[params[outImageName]["dataType"]],)
        if "dataType" in params["convDesc"]:
            flags["Pcomp"] = (type_convert[params["convDesc"]["dataType"]],)

        if "nbDims" in params[inImageName]:
            nbDims = int(params[inImageName]["nbDims"])
            if nbDims == 5:
                flags["dim"] = ("3",)
        if "dimA" in params[inImageName]:
            flags["dimA"] = (",".join(params[inImageName]["dimA"][:nbDims]),)
        if "strideA" in params[inImageName]:
            flags["strideA"] = (",".join(params[inImageName]["strideA"][:nbDims]),)


        flags["filtA"] = (",".join(params[inFilterName]["dimA"][:nbDims]),)
        if "format" in params[inFilterName]:
            flags["filtFormat"] = (format_convert[params[inFilterName]["format"]],)

        if "mode" in params["convDesc"]:
            if(params["convDesc"]["mode"]=="CUDNN_CONVOLUTION (0)"):
                pass
            elif(params["convDesc"]["mode"]=="CUDNN_CROSS_CORRELATION (1)"):
                flags["x"] = ("",)
        if "mathType" in params["convDesc"]:
            flags["Pmath"] = (mathType_convert[params["convDesc"]["mathType"]],)
        if "padA" in params["convDesc"]:
            flags["padA"] = (",".join(params["convDesc"]["padA"][:(nbDims-2)]),)
        if "strideA" in params["convDesc"]:
            flags["convStrideA"] = (",".join(params["convDesc"]["strideA"][:(nbDims-2)]),)
        if "dilationA" in params["convDesc"]:
            flags["dilationA"] = (",".join(params["convDesc"]["dilationA"][:(nbDims-2)]),)
        if "groupCount" in params["convDesc"]:
            if params["convDesc"]["groupCount"] != "1":
                flags["groupCount"] = (params["convDesc"]["groupCount"],)

        if "strideA" in params[outImageName]:
            nbDims_yDesc = int(params[outImageName]["nbDims"])
            flags["strideOut"] = (",".join(params[outImageName]["strideA"][:nbDims_yDesc]),)

        if params["activationDesc"]["mode"] == "CUDNN_ACTIVATION_SIGMOID (0)":
            flags["activFunc"] = ("0",)
        elif params["activationDesc"]["mode"] == "CUDNN_ACTIVATION_RELU (1)":
            flags["activFunc"] = ("1",)
        elif params["activationDesc"]["mode"] == "CUDNN_ACTIVATION_TANH (2)":
            flags["activFunc"] = ("2",)
        elif params["activationDesc"]["mode"] == "CUDNN_ACTIVATION_CLIPPED_RELU (3)":
            flags["activCoef"] = (params["activationDesc"]["coef"],)
            flags["activFunc"] = ("3",)
        elif params["activationDesc"]["mode"] == "CUDNN_ACTIVATION_ELU (4)":
            flags["activCoef"] = (params["activationDesc"]["coef"],)
            flags["activFunc"] = ("4",)
        elif params["activationDesc"]["mode"] == "CUDNN_ACTIVATION_IDENTITY (5)":
            flags["activFunc"] = ("5",)
        elif params["activationDesc"]["mode"] == "CUDNN_ACTIVATION_SWISH (6)":
            flags["activFunc"] = ("6",)

        if params["activationDesc"]["reluNanOpt"] == "CUDNN_PROPAGATE_NAN (1)":
            flags["propnan"] = ("",)

        if("alpha1" in params):
            flags["A"] = (params["alpha1"],)
        if("alpha2" in params):
            flags["B"] = (params["alpha2"],)

        if(params["zData"] == params["yData"]):
            flags["b_is_c "] = ("",)

        flags["b"] = ("",)

        return post_processing(flags)

    if(func_name in ["cudnnActivationForward","cudnnActivationBackward"]):

        if("Forward" in func_name):
            flags["R"] = ("activationf",)
        elif("Backward" in func_name):
            flags["R"] = ("activationb",)

        if "srcDesc" in params:
            inDesc = "srcDesc"
        elif "dyDesc" in params:
            inDesc = "dyDesc"

        flags["mode"] = (cudnnActivationMode_convert[params["activationDesc"]["mode"]],)
        flags["activCoef"] = (params["activationDesc"]["coef"],)
        if params["activationDesc"]["reluNanOpt"] == "CUDNN_PROPAGATE_NAN (1)":
            flags["propnan"] = ("",)

        if "dataType" in params[inDesc]:
            flags["P"] = (type_convert[params[inDesc]["dataType"]],)

        nbDims = int(params[inDesc]["nbDims"])
        if nbDims == 5:
            flags["dim"] = ("3",)
        flags["dimA"] = (",".join(params[inDesc]["dimA"][:nbDims]),)
        flags["strideA"] = (",".join(params[inDesc]["strideA"][:nbDims]),)

        if("alpha" in params):
            flags["A"] = (params["alpha"],)
        if("beta" in params):
            flags["B"] = (params["beta"],)

        flags["b"] = ("",)
        return post_processing(flags)

    if(func_name in ["cudnnPoolingForward", "cudnnPoolingBackward"]):

        if("Forward" in func_name):
            flags["R"] = ("poolf",)
            inputDesc = "srcDesc"
            outputDesc = "destDesc"
        elif("Backward" in func_name):
            # in backward only max pooling have valid srcDesc,
            # so we should use srcDiffDesc
            flags["R"] = ("poolb",)
            inputDesc = "destDiffDesc"
            outputDesc = "srcDiffDesc"

        flags["mode"] = (cudnnPoolingMode_convert[params["poolingDesc"]["mode"]],)

        if isinstance(params[inputDesc], dict):
            nbDims = int(params[inputDesc]["nbDims"])
        else:
            print("!!!! Warning " + inputDesc + " is NULL_PTR, expect to fail (why?)")
            return None

        if "dataType" in params[inputDesc]:
            flags["Pin"] = (type_convert[params[inputDesc]["dataType"]],)
            if "x32" in params[inputDesc]["dataType"]:
                flags["nc32"] = ("",)
        if "dataType" in params[outputDesc]:
            flags["Pout"] = (type_convert[params[outputDesc]["dataType"]],)

        if nbDims == 4:
            flags["win_h"] = (params["poolingDesc"]["windowDimA"][0],)
            flags["win_w"] = (params["poolingDesc"]["windowDimA"][1],)
            flags["pad_h"] = (params["poolingDesc"]["paddingA"][0],)
            flags["pad_w"] = (params["poolingDesc"]["paddingA"][1],)
        elif nbDims == 5:
            flags["dim"] = ("3",)
            flags["win_d"] = (params["poolingDesc"]["windowDimA"][0],)
            flags["win_h"] = (params["poolingDesc"]["windowDimA"][1],)
            flags["win_w"] = (params["poolingDesc"]["windowDimA"][2],)
            flags["pad_d"] = (params["poolingDesc"]["paddingA"][0],)
            flags["pad_h"] = (params["poolingDesc"]["paddingA"][1],)
            flags["pad_w"] = (params["poolingDesc"]["paddingA"][2],)

        flags["dimA"] = (",".join(params[inputDesc]["dimA"][:nbDims]),)
        flags["strideA"] = (",".join(params[inputDesc]["strideA"][:nbDims]),)
        flags["dimOut"] = (",".join(params[outputDesc]["dimA"][:nbDims]),)
        flags["strideOut"] = (",".join(params[outputDesc]["strideA"][:nbDims]),)
        # flags["padA"] = (",".join(params["poolingDesc"]["paddingA"][:(nbDims-2)]),)
        flags["convStrideA"] = (",".join(params["poolingDesc"]["strideA"][:(nbDims-2)]),)

        if params["poolingDesc"]["maxpoolingNanOpt"] == "CUDNN_PROPAGATE_NAN (1)":
            flags["propnan"] = ("",)

        if("alpha" in params):
            flags["A"] = (params["alpha"],)
        if("beta" in params):
            flags["B"] = (params["beta"],)

        flags["b"] = ("",)

        return post_processing(flags)

    if(func_name in ["cudnnBatchNormalizationForwardTraining","cudnnBatchNormalizationForwardInference","cudnnBatchNormalizationBackward"]):

        if("ForwardInference" in func_name):
            if version[0] == "8":
                flags["R"] = ("bnfi",)
            else:
                flags["R"] = ("bnf",)
        elif("ForwardTraining" in func_name):
            if version[0] == "8":
                flags["R"] = ("bnft",)
            else:
                flags["R"] = ("bnf",)
        elif("Backward" in func_name):
            flags["R"] = ("bnb",)

        if "mode" in params:
            flags["algo"] = (cudnnBatchNormalizationMode_convert[params["mode"]],)

        if "bottomDesc" in params:
            xDescName = "bottomDesc"
        elif "xDesc" in params:
            xDescName = "xDesc"

        if "bottomData" in params:
            xDataName = "bottomData"
        elif "xData" in params:
            xDataName = "xData"

        if "resultTopData" in params:
            yDataName = "resultTopData"
        elif "yData" in params:
            yDataName = "yData"

        if "topDiff" in params:
            dyDataName = "topDiff"
        elif "dyData" in params:
            dyDataName = "dyData"

        if "resultBottomDiff" in params:
            dxDataName = "topDiff"
        elif "dxData" in params:
            dxDataName = "dxData"

        if "dataType" in params[xDescName]:
            flags["P"] = (type_convert[params[xDescName]["dataType"]],)

        nbDims = int(params[xDescName]["nbDims"])
        if nbDims == 5:
            flags["dim"] = ("3",)

        if "dimA" in params[xDescName]:
            flags["dimA"] = (",".join(params[xDescName]["dimA"][:nbDims]),)
        if "strideA" in params[xDescName]:
            flags["strideA"] = (",".join(params[xDescName]["strideA"][:nbDims]),)

        if("Forward" in func_name):
            if(params[xDataName] == params[yDataName]):
                flags["inplace"] = ("",)
        elif("Backward" in func_name):
            if(params[dyDataName] == params[dxDataName]):
                flags["inplace"] = ("",)

        if "saveMean" in params and "saveInvVariance" in params:
            if params["saveMean"] == "NULL_PTR" or params["saveInvVariance"] == "NULL_PTR":
                flags["testSavedMeans"] = ("0",)
            else:
                flags["testSavedMeans"] = ("1",)
        elif "savedMean" in params and "savedInvVariance" in params:
            if params["savedMean"] == "NULL_PTR" or params["savedInvVariance"] == "NULL_PTR":
                flags["testSavedMeans"] = ("0",)
            else:
                flags["testSavedMeans"] = ("1",)

        if "activationDesc" in params and "mode" in params["activationDesc"]:
            flags["mode"] = (cudnnActivationMode_convert[params["activationDesc"]["mode"]],)

        if("Forward" in func_name):
            if("alpha" in params):
                flags["A"] = (params["alpha"],)
            if("beta" in params):
                flags["B"] = (params["beta"],)
        elif("Backward" in func_name):
            if("alphaDataDiff" in params and "alphaParamDiff" in params):
                flags["A"] = (params["alphaDataDiff"] + ',' + params["alphaParamDiff"],)
            if("betaDataDiff" in params and "betaParamDiff" in params):
                flags["B"] = (params["betaDataDiff"] + ',' + params["betaParamDiff"],)

        flags["b"] = ("",)
        return post_processing(flags)

    if(func_name in ["cudnnBatchNormalizationForwardTrainingEx","cudnnBatchNormalizationBackwardEx"]):

        if("ForwardTraining" in func_name):
            if version[0] == "8":
                flags["R"] = ("bnft",)
            else:
                flags["R"] = ("bnf",)
        elif("Backward" in func_name):
            flags["R"] = ("bnb",)

        if "mode" in params:
            flags["algo"] = (cudnnBatchNormalizationMode_convert[params["mode"]],)

        if "bnOps" in params:
            flags["bnTestEx"] = (bnOps_convert[params["bnOps"]],)

        if "bottomDesc" in params:
            xDescName = "bottomDesc"
        elif "xDesc" in params:
            xDescName = "xDesc"

        if "bottomData" in params:
            xDataName = "bottomData"
        elif "xData" in params:
            xDataName = "xData"

        if "resultTopData" in params:
            yDataName = "resultTopData"
        elif "yData" in params:
            yDataName = "yData"

        if "topDiff" in params:
            dyDataName = "topDiff"
        elif "dyData" in params:
            dyDataName = "dyData"

        if "resultBottomDiff" in params:
            dxDataName = "topDiff"
        elif "dxData" in params:
            dxDataName = "dxData"

        if "dataType" in params[xDescName]:
            flags["P"] = (type_convert[params[xDescName]["dataType"]],)

        nbDims = int(params[xDescName]["nbDims"])
        if nbDims == 5:
            flags["dim"] = ("3",)

        if "dimA" in params[xDescName]:
            flags["dimA"] = (",".join(params[xDescName]["dimA"][:nbDims]),)
        if "strideA" in params[xDescName]:
            flags["strideA"] = (",".join(params[xDescName]["strideA"][:nbDims]),)

        if("Forward" in func_name):
            if(params[xDataName] == params[yDataName]):
                flags["inplace"] = ("",)
        elif("Backward" in func_name):
            if(params[dyDataName] == params[dxDataName]):
                flags["inplace"] = ("",)

        if "saveMean" in params and "saveInvVariance" in params:
            if params["saveMean"] == "NULL_PTR" or params["saveInvVariance"] == "NULL_PTR":
                flags["testSavedMeans"] = ("0",)
            else:
                flags["testSavedMeans"] = ("1",)
        elif "savedMean" in params and "savedInvVariance" in params:
            if params["savedMean"] == "NULL_PTR" or params["savedInvVariance"] == "NULL_PTR":
                flags["testSavedMeans"] = ("0",)
            else:
                flags["testSavedMeans"] = ("1",)

        if "activationDesc" in params and "mode" in params["activationDesc"]:
            flags["mode"] = (cudnnActivationMode_convert[params["activationDesc"]["mode"]],)

        if("Forward" in func_name):
            if("alpha" in params):
                flags["A"] = (params["alpha"],)
            if("beta" in params):
                flags["B"] = (params["beta"],)
        elif("Backward" in func_name):
            if("alphaDataDiff" in params and "alphaParamDiff" in params):
                flags["A"] = (params["alphaDataDiff"] + ',' + params["alphaParamDiff"],)
            if("betaDataDiff" in params and "betaParamDiff" in params):
                flags["B"] = (params["betaDataDiff"] + ',' + params["betaParamDiff"],)

        flags["b"] = ("",)
        return post_processing(flags)

    if(func_name in ["cudnnNormalizationForwardTraining","cudnnNormalizationBackward"]):
        if("ForwardTraining" in func_name):
            flags["R"] = ("normft",)
        elif("Backward" in func_name):
            flags["R"] = ("normb",)
        
        if "mode" in params:
            flags["normMode"] = (cudnnNormMode_convert[params["mode"]],) 
        if "normOps" in params:
            flags["normOps"] = (normOps_convert[params["normOps"]],) 
        if "algo" in params:
            flags["algo"] = (normAlgo_convert[params["algo"]],) 
        if "bottomDesc" in params:
            xDescName = "bottomDesc"
        elif "xDesc" in params:
            xDescName = "xDesc"

        if "bottomData" in params:
            xDataName = "bottomData"
        elif "xData" in params:
            xDataName = "xData"

        if "resultTopData" in params:
            yDataName = "resultTopData"
        elif "yData" in params:
            yDataName = "yData"

        if "topDiff" in params:
            dyDataName = "topDiff"
        elif "dyData" in params:
            dyDataName = "dyData"

        if "resultBottomDiff" in params:
            dxDataName = "topDiff"
        elif "dxData" in params:
            dxDataName = "dxData"

        if "dataType" in params[xDescName]:
            flags["P"] = (type_convert[params[xDescName]["dataType"]],)

        nbDims = int(params[xDescName]["nbDims"])
        if nbDims == 5:
            flags["dim"] = ("3",)

        if "dimA" in params[xDescName]:
            flags["dimA"] = (",".join(params[xDescName]["dimA"][:nbDims]),)
        if "strideA" in params[xDescName]:
            flags["strideA"] = (",".join(params[xDescName]["strideA"][:nbDims]),)

        if("Forward" in func_name):
            if(params[xDataName] == params[yDataName]):
                flags["inplace"] = ("",)
        elif("Backward" in func_name):
            if(params[dyDataName] == params[dxDataName]):
                flags["inplace"] = ("",)

        if "resultSaveMeanData" in params and "resultSaveInvVarianceData" in params:
            if params["resultSaveMeanData"] == "NULL_PTR" or params["resultSaveInvVarianceData"] == "NULL_PTR":
                flags["testSavedMeans"] = ("0",)
            else:
                flags["testSavedMeans"] = ("1",)
        elif "saveMean" in params and "saveInvVariance" in params:
            if params["saveMean"] == "NULL_PTR" or params["saveInvVariance"] == "NULL_PTR":
                flags["testSavedMeans"] = ("0",)
            else:
                flags["testSavedMeans"] = ("1",)
        elif "savedMean" in params and "savedInvVariance" in params:
            if params["savedMean"] == "NULL_PTR" or params["savedInvVariance"] == "NULL_PTR":
                flags["testSavedMeans"] = ("0",)
            else:
                flags["testSavedMeans"] = ("1",)

        if "activationDesc" in params and "mode" in params["activationDesc"]:
            flags["mode"] = (cudnnActivationMode_convert[params["activationDesc"]["mode"]],)

        if("Forward" in func_name):
            if("alpha" in params):
                flags["A"] = (params["alpha"],)
            if("beta" in params):
                flags["B"] = (params["beta"],)
        elif("Backward" in func_name):
            if("alphaDataDiff" in params and "alphaParamDiff" in params):
                flags["A"] = (params["alphaDataDiff"] + ',' + params["alphaParamDiff"],)
            if("betaDataDiff" in params and "betaParamDiff" in params):
                flags["B"] = (params["betaDataDiff"] + ',' + params["betaParamDiff"],)

        if "groupCnt" in params:
            if params["groupCnt"] != "1":
                flags["groupCount"] = (params["groupCnt"],)

        flags["b"] = ("",)
        return post_processing(flags)

    if(func_name in ["cudnnSoftmaxForward","cudnnSoftmaxBackward"]):

        if("Forward" in func_name):
            flags["R"] = ("softmaxf",)
            inTensorName = "srcDesc"
        elif("Backward" in func_name):
            flags["R"] = ("softmaxb",)
            inTensorName = "srcDesc"

        nbDims = int(params[inTensorName]["nbDims"])
        if nbDims == 5:
            flags["dim"] = ("3",)

        if "dataType" in params["srcDesc"]:
            flags["P"] = (type_convert[params["srcDesc"]["dataType"]],)

        if "dimA" in params[inTensorName]:
            flags["dimA"] = (",".join(params[inTensorName]["dimA"][:nbDims]),)
        if "strideA" in params[inTensorName]:
            flags["strideA"] = (",".join(params[inTensorName]["strideA"][:nbDims]),)
        mode = cudnnSoftmaxMode_convert[params["mode"]]
        algo = cudnnSoftmaxAlgorithm_convert[params["algorithm"]]
        flags["mode"] = (str(int(algo)*2 + int(mode)),)

        if("alpha" in params):
            flags["A"] = (params["alpha"],)
        if("beta" in params):
            flags["B"] = (params["beta"],)

        flags["b"] = ("",)
        return post_processing(flags)

    if(func_name in ["cudnnLRNCrossChannelForward"]):
        flags["R"] = ("lrnf",)

        nbDims = int(params["xDesc"]["nbDims"])

        if nbDims == 4:
            flags["n"] = (params["xDesc"]["dimA"][0],)
            flags["c"] = (params["xDesc"]["dimA"][1],)
            flags["h"] = (params["xDesc"]["dimA"][2],)
            flags["w"] = (params["xDesc"]["dimA"][3],)
        else:
            flags["dimA"] = (",".join(params["xDesc"]["dimA"][:nbDims]),)

        if("alpha" in params):
            flags["A"] = (params["alpha"],)
        if("beta" in params):
            flags["B"] = (params["beta"],)

        flags["b"] = ("",)
        return post_processing(flags)

    if(func_name in ["cudnnAddTensor"]):
        flags["R"] = ("add",)

        nbDims = int(params["srcDestDesc"]["nbDims"])

        if nbDims == 4:
            flags["n"] = (params["srcDestDesc"]["dimA"][0],)
            flags["c"] = (params["srcDestDesc"]["dimA"][1],)
            flags["h"] = (params["srcDestDesc"]["dimA"][2],)
            flags["w"] = (params["srcDestDesc"]["dimA"][3],)
        else:
            flags["dimA"] = (",".join(params["srcDestDesc"]["dimA"][:nbDims]),)

        if "strideA" in params["srcDestDesc"]:
            flags["strideA"] = (",".join(params["srcDestDesc"]["strideA"][:nbDims]),)

        if "dataType" in params["srcDestDesc"]:
            flags["P"] = (type_convert[params["srcDestDesc"]["dataType"]],)

        mode = "?"
        # check tensor descriptors to know what kind of addition we are performing
        if (params["biasDesc"] == params["srcDestDesc"]):
            mode = "3" #CUDNN_ADD_FULL_TENSOR

        elif (params["biasDesc"] == params["srcDestDesc"] and (params["biasDesc"]["dimA"][0] == "1") and (params["biasDesc"]["dimA"][1] == params["srcDestDesc"]["dimA"][1])):
            # Add same value to all images
            mode = "1" #CUDNN_ADD_FEATURE_MAP

        elif (params["biasDesc"] == params["srcDestDesc"] and (params["biasDesc"]["dimA"][0] == "1") and (params["biasDesc"]["dimA"][1] == "1")):
            # Add pixels to all images, channels
            mode = "0" #CUDNN_ADD_IMAGE

        elif (params["srcDestDesc"]["dimA"][1] == params["biasDesc"]["dimA"][1]):
            nbElems = 1
            for n in range(nbDims):
                if n != 1:
                    nbElems *= int(params["biasDesc"]["dimA"][n])
            if nbElems == 1:
                # add channel to all images and pixels
                mode = "2" #CUDNN_ADD_SAME_C

        if mode == "?":
            print("!!!! Wrning: mode is not set properly")
        flags["mode"] = (mode,)

        if("alpha" in params):
            flags["A"] = (params["alpha"],)
        if("beta" in params):
            flags["B"] = (params["beta"],)

        flags["b"] = ("",)
        return post_processing(flags)


    if(func_name in ["cudnnFindConvolutionForwardAlgorithm", "cudnnFindConvolutionForwardAlgorithmEx",
                    "cudnnGetConvolutionForwardAlgorithm_v7", "cudnnGetConvolutionForwardAlgorithm"]):
        if(func_name == "cudnnFindConvolutionForwardAlgorithm" or func_name == "cudnnFindConvolutionForwardAlgorithmEx"):
            inImageName  = "srcDesc"
            inFilterName = "filterDesc"
            outImageName = "destDesc"
            flags["R"] = ("findFwdAlgo",)
        elif(func_name == "cudnnGetConvolutionForwardAlgorithm_v7" or func_name == "cudnnGetConvolutionForwardAlgorithm"):
            inImageName  = "srcDesc"
            inFilterName = "filterDesc"
            outImageName = "destDesc"
            flags["R"] = ("getFwdAlgo",)

        if "dataType" in params[inImageName]:
            flags["Pin"] = (type_convert[params[inImageName]["dataType"]],)
        if "dataType" in params[outImageName]:
            flags["Pout"] = (type_convert[params[outImageName]["dataType"]],)
        if "dataType" in params["convDesc"]:
            flags["Pcomp"] = (type_convert[params["convDesc"]["dataType"]],)

        if "nbDims" in params[inImageName]:
            nbDims = int(params[inImageName]["nbDims"])
            if nbDims == 5:
                flags["dim"] = ("3",)
        if "dimA" in params[inImageName]:
            flags["dimA"] = (",".join(params[inImageName]["dimA"][:nbDims]),)
        if "strideA" in params[inImageName]:
            flags["strideA"] = (",".join(params[inImageName]["strideA"][:nbDims]),)


        flags["filtA"] = (",".join(params[inFilterName]["dimA"][:nbDims]),)
        if "format" in params[inFilterName]:
            flags["filtFormat"] = (format_convert[params[inFilterName]["format"]],)

        if "mode" in params["convDesc"]:
            if(params["convDesc"]["mode"]=="CUDNN_CONVOLUTION (0)"):
                pass
            elif(params["convDesc"]["mode"]=="CUDNN_CROSS_CORRELATION (1)"):
                flags["x"] = ("",)
        if "mathType" in params["convDesc"]:
            flags["Pmath"] = (mathType_convert[params["convDesc"]["mathType"]],)
        if "padA" in params["convDesc"]:
            flags["padA"] = (",".join(params["convDesc"]["padA"][:(nbDims-2)]),)
        if "strideA" in params["convDesc"]:
            flags["convStrideA"] = (",".join(params["convDesc"]["strideA"][:(nbDims-2)]),)
        if "dilationA" in params["convDesc"]:
            flags["dilationA"] = (",".join(params["convDesc"]["dilationA"][:(nbDims-2)]),)
        if "groupCount" in params["convDesc"]:
            if params["convDesc"]["groupCount"] != "1":
                flags["groupCount"] = (params["convDesc"]["groupCount"],)

        if "requestedAlgoCount" in params:
            flags["mode"] = (params["requestedAlgoCount"],)

        if("alpha" in params):
            flags["A"] = (params["alpha"],)
        if("beta" in params):
            flags["B"] = (params["beta"],)

        if "strideA" in params[outImageName]:
            nbDims_yDesc = int(params[outImageName]["nbDims"])
            flags["strideOut"] = (",".join(params[outImageName]["strideA"][:nbDims_yDesc]),)
        flags["b"] = ("",)

        return post_processing(flags)

    if(func_name in ["cudnnFindConvolutionBackwardDataAlgorithm", "cudnnFindConvolutionBackwardDataAlgorithmEx",
                    "cudnnGetConvolutionBackwardDataAlgorithm_v7", "cudnnGetConvolutionBackwardDataAlgorithm",
                    "cudnnFindConvolutionBackwardFilterAlgorithm", "cudnnFindConvolutionBackwardFilterAlgorithmEx",
                    "cudnnGetConvolutionBackwardFilterAlgorithm_v7", "cudnnGetConvolutionBackwardFilterAlgorithm"]):

        if(func_name == "cudnnFindConvolutionBackwardDataAlgorithm" or func_name == "cudnnFindConvolutionBackwardDataAlgorithmEx"):
            outDataTensor = "diffDesc"
            inDataTensor = "gradDesc"
            filterTensor = "filterDesc"
            flags["R"] = ("findBwdDataAlgo",)
        elif(func_name == "cudnnGetConvolutionBackwardDataAlgorithm_v7" or func_name == "cudnnGetConvolutionBackwardDataAlgorithm"):
            outDataTensor = "diffDesc"
            inDataTensor = "gradDesc"
            filterTensor = "filterDesc"
            flags["R"] = ("getBwdDataAlgo",)
        elif(func_name == "cudnnFindConvolutionBackwardFilterAlgorithm" or func_name == "cudnnFindConvolutionBackwardFilterAlgorithmEx"):
            outDataTensor = "diffDesc"
            inDataTensor = "srcDesc"
            filterTensor = "gradDesc"
            flags["R"] = ("findBwdFilterAlgo",)
        elif(func_name == "cudnnGetConvolutionBackwardFilterAlgorithm_v7" or func_name == "cudnnGetConvolutionBackwardFilterAlgorithm"):
            outDataTensor = "diffDesc"
            inDataTensor = "srcDesc"
            filterTensor = "gradDesc"
            flags["R"] = ("getBwdFilterAlgo",)

        # extract Pin Pout Pcomp flags
        if "dataType" in params[inDataTensor]:
            flags["Pin"] = (type_convert[params[inDataTensor]["dataType"]],)
        if "dataType" in params[outDataTensor]:
            flags["Pout"] = (type_convert[params[outDataTensor]["dataType"]],)
        if "dataType" in params["convDesc"]:
            flags["Pcomp"] = (type_convert[params["convDesc"]["dataType"]],)
        if "mathType" in params["convDesc"]:
            flags["Pmath"] = (mathType_convert[params["convDesc"]["mathType"]],)

        if(params["convDesc"]["mode"]=="CUDNN_CONVOLUTION (0)"):
            pass
        elif(params["convDesc"]["mode"]=="CUDNN_CROSS_CORRELATION (1)"):
            flags["x"] = ("",)

        if params["convDesc"]["groupCount"] != "1":
            flags["groupCount"] = (params["convDesc"]["groupCount"],)

        nbDims = int(params[inDataTensor]["nbDims"])

        if nbDims == 5:
            flags["dim"] = ("3",)

        flags["dimA"] = (",".join(params[inDataTensor]["dimA"][:nbDims]),)
        if "strideA" in params[inDataTensor]:
            flags["strideA"] = (",".join(params[inDataTensor]["strideA"][:nbDims]),)

        flags["filtA"] = (",".join(params[filterTensor]["dimA"][:nbDims]),)
        if "format" in params[filterTensor]:
            flags["filtFormat"] = (format_convert[params[filterTensor]["format"]],)

        if "mathType" in params["convDesc"]:
            flags["Pmath"] = (mathType_convert[params["convDesc"]["mathType"]],)
        if "padA" in params["convDesc"]:
            flags["padA"] = (",".join(params["convDesc"]["padA"][:(nbDims-2)]),)
        if "strideA" in params["convDesc"]:
            flags["convStrideA"] = (",".join(params["convDesc"]["strideA"][:(nbDims-2)]),)
        if "dilationA" in params["convDesc"]:
            flags["dilationA"] = (",".join(params["convDesc"]["dilationA"][:(nbDims-2)]),)

        if "requestedAlgoCount" in params:
            flags["mode"] = (params["requestedAlgoCount"],)

        if("alpha" in params):
            flags["A"] = (params["alpha"],)
        if("beta" in params):
            flags["B"] = (params["beta"],)

        nbDims_yDesc = int(params[outDataTensor]["nbDims"])
        if "strideA" in params[outDataTensor]:
            flags["strideOut"] = (",".join(params[outDataTensor]["strideA"][:nbDims_yDesc]),)

        flags["b"] = ("",)

        return post_processing(flags)

    if(func_name in ["cudnnRNNForwardInference", "cudnnRNNForwardTraining", "cudnnRNNBackwardData", "cudnnRNNBackwardWeights"]):
        if (func_name == "cudnnRNNForwardInference"):
            flags["R"] = ("RNNf",)
        else:
            flags["R"] = ("RNNb",)

        if func_name == "cudnnRNNBackwardData":
            inImageName  = "dxDesc"
        else:
            inImageName  = "xDesc"

        # in v8 "mode" has changed to "cellMode"
        if "cellMode" in params["rnnDesc"]:
            flags["rnnCellMode"] = (cudnnRNNModeConvert[params["rnnDesc"]["cellMode"]],)
        elif "mode" in params["rnnDesc"]:
            flags["mode"] = (cudnnRNNModeConvert[params["rnnDesc"]["mode"]],)

        flags["rnnNumLayers"] = (params["rnnDesc"]["numLayers"],)
        flags["rnnInputSize"] = (params[inImageName]["dimA"][1],)
        flags["rnnHiddenSize"] = (params["rnnDesc"]["hiddenSize"],)
        flags["rnnbidirectional"] = (cudnnDirectionModeConvert[params["rnnDesc"]["bidirectional"]],)
        flags["rnnMiniBatch"] = (params[inImageName]["dimA"][0],)
        flags["rnnSeqLength"] = (params["seqLength"],)

        # in v8 "math_mode" has changed to "mathMode"
        if "mathMode" in params["rnnDesc"]:
            flags["Pmath"] = (mathType_convert[params["rnnDesc"]["mathMode"]],)
        elif "math_mode" in params["rnnDesc"]:
            flags["Pmath"] = (mathType_convert[params["rnnDesc"]["math_mode"]],)

        
        flags["algo"] = (cudnnRNNAlgoConvert[params["rnnDesc"]["algo"]],)

        if "dataType" in params[inImageName]:
            flags["Pin"] = (type_convert[params[inImageName]["dataType"]],)
            flags["Pout"] = (type_convert[params[inImageName]["dataType"]],)

        # in newer version it's called mathPrec
        if "mathType" in params["rnnDesc"]:
            flags["Pcomp"] = (type_convert[params["rnnDesc"]["mathType"]],)
        elif "mathPrec" in params["rnnDesc"]:
            flags["Pcomp"] = (type_convert[params["rnnDesc"]["mathPrec"]],)

        if "biasMode" in params["rnnDesc"]:
            flags["rnnBiasMode"] = (cudnnRNNBiasModeConvert[params["rnnDesc"]["biasMode"]],)

        flags["b"] = ("",)

        return post_processing(flags)
    if(func_name in ["cudnnRNNForwardInferenceEx", "cudnnRNNForwardTrainingEx", "cudnnRNNBackwardDataEx", "cudnnRNNBackwardWeightsEx", "cudnnRNNForward", "cudnnRNNBackwardData_v8", "cudnnRNNBackwardWeights_v8"]):
        if (func_name == "cudnnRNNForwardInferenceEx"):
            flags["R"] = ("RNNf",)
        elif (func_name in ["cudnnRNNForwardTrainingEx", "cudnnRNNBackwardDataEx", "cudnnRNNBackwardWeightsEx"]):
            # all training APIs are tested in backward tests
            flags["R"] = ("RNNb",)
        elif (func_name == "cudnnRNNForward"):
            if "fwdMode" in params:
                if params["fwdMode"] == "0":
                    # inference
                    flags["R"] = ("rnnf",)
                else:
                    # training
                    flags["R"] = ("rnnb",)          
        elif (func_name in ["cudnnRNNBackwardData_v8", "cudnnRNNBackwardWeights_v8"]):
            # all training APIs are tested in backward tests
            flags["R"] = ("rnnb",)

        inImageName  = "xDesc"

        # in v8 "mode" has changed to "cellMode"
        if "cellMode" in params["rnnDesc"]:
            flags["rnnCellMode"] = (cudnnRNNModeConvert[params["rnnDesc"]["cellMode"]],)
        elif "mode" in params["rnnDesc"]:
            flags["mode"] = (cudnnRNNModeConvert[params["rnnDesc"]["mode"]],)

        flags["rnnNumLayers"] = (params["rnnDesc"]["numLayers"],)
        flags["rnnInputSize"] = (params[inImageName]["dimA"][2],)
        flags["rnnHiddenSize"] = (params["rnnDesc"]["hiddenSize"],)
        flags["rnnbidirectional"] = (cudnnDirectionModeConvert[params["rnnDesc"]["bidirectional"]],)
        flags["rnnMiniBatch"] = (params[inImageName]["dimA"][1],)
        flags["rnnSeqLength"] = (params[inImageName]["dimA"][0],)

        # in v8 "math_mode" has changed to "mathMode"
        if "mathMode" in params["rnnDesc"]:
            flags["Pmath"] = (mathType_convert[params["rnnDesc"]["mathMode"]],)
        elif "math_mode" in params["rnnDesc"]:
            flags["Pmath"] = (mathType_convert[params["rnnDesc"]["math_mode"]],)

        flags["algo"] = (cudnnRNNAlgoConvert[params["rnnDesc"]["algo"]],)

        if "dataType" in params[inImageName]:
            flags["Pin"] = (type_convert[params[inImageName]["dataType"]],)
            flags["Pout"] = (type_convert[params[inImageName]["dataType"]],)
        
        if "layout" in params[inImageName]:
            flags["rnnDataLayout"] = (cudnnRNNDataLayoutConvert[params[inImageName]["layout"]],)
        # in newer version it's called mathPrec
        if "mathType" in params["rnnDesc"]:
            flags["Pcomp"] = (type_convert[params["rnnDesc"]["mathType"]],)
        elif "mathPrec" in params["rnnDesc"]:
            flags["Pcomp"] = (type_convert[params["rnnDesc"]["mathPrec"]],)

        if "biasMode" in params["rnnDesc"]:
            flags["rnnBiasMode"] = (cudnnRNNBiasModeConvert[params["rnnDesc"]["biasMode"]],)
        
        flags["b"] = ("",)

        return post_processing(flags)

    if(func_name in ["cudnnCTCLossInternal"]):

        flags["R"] = ("ctc",)

        if "dimA" in params["probsDesc"]:
            flags["cT"] = (params["probsDesc"]["dimA"][0],)
            flags["cN"] = (params["probsDesc"]["dimA"][1],)
            flags["cA"] = (params["probsDesc"]["dimA"][2],)

        # max value of labelLength array in minibatch (an approximation of -cL flag)
        if ("maxLabelLength" in params):
            flags["cL"] = (params["maxLabelLength"],)

        # NOTE: cannot recover -cR flag (max number of repeats in labelLength array)
        # given that repeats would have to be counted within labelLength array that exists on device

        algo = cudnnCTCLossAlgo_convert[params["algo"]]
        flags["algo"] = (algo,)

        if "compType" in params["ctcLossDesc"]:
            flags["P"] = (type_convert[params["ctcLossDesc"]["compType"]],)

        if ("normMode" in params["ctcLossDesc"]) and ("gradMode" in params["ctcLossDesc"]):
            normMode = cudnnCTCLossNormMode_convert[params["ctcLossDesc"]["normMode"]]
            gradMode = cudnnCTCLossGradMode_convert[params["ctcLossDesc"]["gradMode"]]
            if (normMode == "0") and (gradMode == "1"):
                flags["mode"] = ("0",)
            elif (normMode == "0") and (gradMode == "0"):
                flags["mode"] = ("1",)
            elif (normMode == "1") and (gradMode == "1"):
                flags["mode"] = ("2",)
            else:
                flags["mode"] = ("3",)

        flags["b"] = ("",)

        return post_processing(flags)

    return None



def parseVariable(paramList):
    if paramList.__len__() == 1:
        param = paramList[0]
        if "NULL_PTR" in param:
            name = re.findall(NULLPtrRegex, param)
            if name != []:
                return name[0], "NULL_PTR"
            else:
                raise ValueError("Cannot parse \"" + str(param) + "\" from here:")
                # return "unknown", 0
        elif "cudnnHandle_t" in param:
            # don't care handle value
            return "handle", 0
        elif "addr=" in param:
            var = re.findall(singlePtrRegex, param)
            if var != []:
                return var[0][0], var[0][2]
            else:
                raise ValueError("Cannot parse \"" + str(param) + "\" from here:")
                # return "unknown", 0
        elif "val=" in param:
            var = re.findall(singleJsonRegex, param)
            if var != []:
                name = var[0][0]
                val = var[0][1]
                return name, val
            
            var = re.findall(singleVarRegex, param)
            if var != []:
                name = var[0][0]
                val = var[0][2]
                if "[" in val and "]" in val:
                    array = val.replace('[','')
                    array = array.replace(']','')
                    array = array.replace(' ','')
                    array = array.split(',')
                    return name, array
                else:
                    return name, val
            else:
                raise ValueError("Cannot parse \"" + str(param) + "\" from here:")
                # return "unknown", 0
        else:
            raise ValueError("Cannot parse \"" + str(param) + "\" from here:")
            # return "unknown", 0

    else:
        totalLines = paramList.__len__()
        struct_dict = {}
        structSig = re.findall(singleStructRegex, paramList[0])

        if structSig != []:
            struct_name = structSig[0][0]
            indentLv = list()
            for lineNum in range(totalLines):
                line = paramList[lineNum]
                indentLv.insert( lineNum, (len(line) - len(line.lstrip()))/4.0)

            topIndentLv = indentLv[1]
            varLineStart = 1
            varLineEnd = 1
            for lineNum in range(1,totalLines):
                currIndentLv = indentLv[lineNum]

                if ( currIndentLv == topIndentLv ):
                    varLineStart = lineNum
                    varLineEnd = lineNum
                elif ( currIndentLv > topIndentLv ):
                    varLineEnd = lineNum

                if( lineNum == totalLines-1 or indentLv[lineNum+1] == topIndentLv ):
                    name, val = parseVariable(paramList[varLineStart : varLineEnd+1])
                    struct_dict[name] = val


            return struct_name, struct_dict
        else:
            return "unknownStructName", {"unknown":"unknown"}



def parseFuncCall(singleFuncCall):
    funcCall = {}
    funcCall["Function_name"] = singleFuncCall[0]
    if singleFuncCall[1] == '':
        funcCall["Function_params"] = {"unknown":"unknown"}
        return funcCall
    params = {}
    paramList = singleFuncCall[1].rstrip()
    paramList = paramList.split("\n")
    indentLv = list()
    for lineNum in range(len(paramList)):
        line = paramList[lineNum]
        indentLv.insert( lineNum, (len(line) - len(line.lstrip()))/4.0)

    totalLines = len(paramList)
    varLineStart = 0
    varLineEnd = 0
    topIndentLv = indentLv[0]

    varLineStart = 0
    varLineEnd = 0
    for lineNum in range(0,totalLines):
        currIndentLv = indentLv[lineNum]

        if ( currIndentLv == topIndentLv ):
            varLineStart = lineNum
            varLineEnd = lineNum
        elif ( currIndentLv > topIndentLv ):
            varLineEnd = lineNum

        if( lineNum == totalLines-1 or indentLv[lineNum+1] == topIndentLv ):
            name, val = parseVariable(paramList[varLineStart : varLineEnd+1])
            params[name] = val


    funcCall["Function_params"] = params
    return funcCall

def overrideSupportedFunctionList(src):
    global supported_function_list
    if src == '':
        return
    if '.' in src or not src.startswith('cudnn'):
        with open(src) as fin:
            funcNames = fin.readlines()
    else:
        funcNames = src.split(',')
    supported_function_list = list(filter(bool, map(str.strip, funcNames)))

def logToTest_checkFlagSupport(flags):

    if (flags["R"][0] not in supported_flag_list):
        return False
    elif "b" not in flags:
        # skip randomized ones
        return False
    elif "N" in flags:
        # skip commands that have multiple tests
        if flags["N"][0] != "1":
            return False
    else:
        for key in flags:
            if key in unsupported_flag_list:
                return False
        return True

def logToTest_generate(test_results, flags):
    # flags i.e. the correct answer is passed in just in case we need to follow flags like "-d0" or "-T1" so UID matches
    # but it is not used right now, since excecution path comparison is immune to difference in "-d0" or "-T1"

    cleanoutput = test_results.output.replace("I! ","")
    cleanoutput = cleanoutput.replace("i! ","")

    version = getCudnnVersion(cleanoutput)
    singleFuncCallRegex = getSingleFuncCallRegexFromVersion(version)
    funcCallList = re.findall(singleFuncCallRegex, cleanoutput)
    funcCallDictList = list()
    for singleFuncCall in funcCallList:
        funcName = singleFuncCall[0]
        if funcName in supported_function_list:
            funcCall = parseFuncCall(singleFuncCall)
            funcCallDictList.append(dict(funcCall))

    #always only look at the last supported function call
    if len(funcCallDictList) == 0:
        print("!!!! Warning, no supported function detected!")
        return "error"
    elif len(funcCallDictList) > 1:
        hasBackendExecuteInternal = False
        for n in range(len(funcCallDictList)):
            if funcCallDictList[n]["Function_name"] == "cudnnBackendExecuteInternal":
                hasBackendExecuteInternal = True
        
        if hasBackendExecuteInternal:
            # usually each legacy conv call would be followed by a cudnnBackendExecuteInternal, we should parse the conv call
            print("!!!! Warning, multiple supported function detected including cudnnBackendExecuteInternal, looking at the first one!")
            print([funcCallDictList[n]["Function_name"] for n in range(len(funcCallDictList))])
            singleFuncCall = funcCallDictList[0]
        else:
            # usually in BN backward or RNN backward it runs forward first, so we should parse the last call
            print("!!!! Warning, multiple supported function detected, looking at the last one!")
            print([funcCallDictList[n]["Function_name"] for n in range(len(funcCallDictList))])
            singleFuncCall = funcCallDictList[-1]
    else:
        singleFuncCall = funcCallDictList[0]

    gen_flags = get_cudnn_flags(singleFuncCall["Function_name"], singleFuncCall["Function_params"], version)

    if gen_flags is not None:
        return gen_flags
    else:
        print("!!!! Error, no flags output")
        return "error"

def flagsToLayer(flags):
    layerComponents = []
    testType = flags['R'][0]
    for key in flags:
        if ',' in flags[key][0]:
            layerComponents.append('{key}:"{value}"'.format(key=key, value=flags[key][0]))
        else:
            try:
                v = float(flags[key][0])
                layerComponents.append('{key}:{value:g}'.format(key=key, value=v))
            except ValueError:
                layerComponents.append('{key}:{value}'.format( key=key, value=flags[key][0]))

    runCmd = (" * ").join(layerComponents)
    return (testType, runCmd)

class Layer(object):
    def __init__(self, model_name, flags, repeats=1):
        self.model_name = model_name
        self.flags = flags
        self.test_name, self.layerCmd  = flagsToLayer(flags)
        self.repeats = repeats

    def toString(self, with_repeats=False, index_width=0):
        pre = "{{model}}_{{test}}_{{index:0{}d}}".format(index_width).format(model = self.model_name, test=self.test_name, index=self.index)
        if with_repeats:
            pre += '_R{:d}'.format(self.repeats)
        return '"{pre}" = {layer}'.format(pre=pre, layer=self.layerCmd)

    def __eq__(self, other):
        return self.layerCmd == other.layerCmd

def main():

    def make_help(s, has_choices=False):
        result = s + " [default: %(default)s]"
        if(has_choices):
            result += " [choices: %(choices)s]"
        return result

    arg_format = lambda prog: argparse.HelpFormatter(prog,max_help_position=100, width=160)
    parser = argparse.ArgumentParser(description='CuDNN Log File Utility', formatter_class=arg_format)
    parser._optionals.title = "Mutually Exclusive Options"

    mode_args = parser.add_mutually_exclusive_group(required=True)

    mode_args.add_argument('-stripNondeterministic', action='store_const', const=True, default=False,
        help=make_help("Option 1: Remove all nondeterminstic log content (e.g. time, pid, tid, addresses) for cleaner diff."))

    mode_args.add_argument('-stripLog', action='store_const', const=True, default=False,
        help=make_help("Option 2: Remove all API log from input file."))

    mode_args.add_argument('-genTestCmd', action='store_const', const=True, default=False,
        help=make_help("Option 3: Regenerate cudnnTest command from log."))

    mode_args.add_argument('-genRunCmd', action='store_const', const=True, default=False,
        help=make_help("Option 4: Regenerate cudnn_run.py command from log."))

    mode_args.add_argument('-filterLog', action='store_const', const=True, default=False,
        help=make_help("Option 5: Print filtered log; see -filterFunc option."))

    gen_args = parser.add_argument_group('General Options')

    gen_args.add_argument('-inputFile', '--input-file', '-i', metavar='"path"', dest='filePath', default=None , required=True,
        help=make_help("Required: Specify path to input log file"))

    gen_args.add_argument('-noRepeat', '--no-repeat', action='store_const', const=True, default=False,
        help=make_help('Avoid generating repeated tests in "-genTestCmd" and "-genRunCmd" mode'))

    gen_args.add_argument('-countRepeat', '---repeat', action='store_const', const=True, default=False,
        help=make_help('Avoid generating repeated tests in "-genTestCmd" and "-genRunCmd" mode'))

    gen_args.add_argument('-excludeFlags', '--exclude-flags', metavar='e.g."Pin,Pout,Pmath,Pcomp"', dest='excludedFlags', default=None , required=False,
        help=make_help('Remove certain flags from generated tests/layers, e.g. use "Pin,Pout,Pmath,Pcomp,formatIn,formatOut,filtFormat,strideA,strideOut,algo,P,A,B,x,b" for sweep layers'))

    gen_args.add_argument('-filterTests', '--filter-tests', metavar='e.g."conv,dgrad,wgrad"', dest='filterTests', default=None , required=False,
        help=make_help('Only generate tests/layers for certain ops'))

    gen_args.add_argument('-filterFunc', '--filter-func', metavar='a (newline separated) file of function names or a comma separated list', dest='filterFunc', default='' , required=False,
        help=make_help('Only generate tests/layers or print log for certain cudnn functions'))

    gen_args.add_argument('-modelName', '--model-name', metavar='e.g."myConvNet"', dest='modelName', default='AutoGenerated' , required=False,
        help=make_help('For naming the layers in "-genRunCmd" mode'))

    gen_args.add_argument('-noParsingError', action='store_const', const=True, default=False,
        help=make_help('Pass this to avoid printing parsing errors'))

    gen_args.add_argument('-targetVersion', metavar='"cudnn_version"', dest='cudnnVersion', default=None, required=False,
        help=make_help("Generate cudnnTests using cudnn version specified by the user"))

    args = parser.parse_args()

    # print("Reading file " + args.filePath)

    if args.filePath is not None:
        with open(args.filePath, "r") as in_file:
            fileContent = in_file.read()

    if args.stripNondeterministic == True:
        stripNonDeterministic(fileContent, mode=0)
        return

    elif args.stripLog == True:
        stripLog(fileContent, mode=0)
        return

    fileContent = fileContent.replace("I! ","")
    fileContent = fileContent.replace("i! ","")

    version = getCudnnVersion(fileContent)

    functionBlockRegex = getFunctionBlockRegexFromVersion(version)
    funcBlockList = re.findall(functionBlockRegex, fileContent)

    singleFuncCallRegex = getSingleFuncCallRegexFromVersion(version)

    allowRepeat = not args.no_repeat
    parsingError = list()

    if args.genTestCmd == True:
        loadModeFlag = Flags()
        listofcudnnTest = []
    elif args.genRunCmd == True:
        testCounts = collections.defaultdict(int)
        repeatCounts = collections.defaultdict(int)

        layer_list = []
    elif args.filterLog == True:
        pass

    overrideSupportedFunctionList(args.filterFunc) # effective in multiple modes; the default empty string bypasses this

    if args.filterTests is not None:
        allowedTestsList = args.filterTests.split(',')

    for singleFuncBlock in funcBlockList:
        
        funcCallContent = re.findall(singleFuncCallRegex,singleFuncBlock)[0]
        funcName = funcCallContent[0]
        if funcName in supported_function_list:
            if args.filterLog == True:
                print(singleFuncBlock, '\n')
                continue # filterLog mode prints logs only
            flags = Flags()
            try:
                # these are tricky steps, sometimes the log may be mixed up
                # we try not to error out and generate whatever we can
                singleFuncCallDict = dict(parseFuncCall(funcCallContent))
                
                if args.cudnnVersion is not None:
                    version = str(int(args.cudnnVersion[0])*1000)
            
                flags = get_cudnn_flags(singleFuncCallDict["Function_name"], singleFuncCallDict["Function_params"], version)
                	
            except Exception as e:
                parsingError.append('\n'.join(map(lambda line: "#" + line.rstrip(), (
                    "!!!! {}\n".format(repr(e)) + \
                    ''.join(traceback.format_tb(e.__traceback__)) + \
                    str(singleFuncBlock)
                ).splitlines())))

            # pdb.set_trace()
            if (args.filterTests is None) or (flags["R"][0] in allowedTestsList):
                if args.excludedFlags is not None:
                    excludedFlagsList = args.excludedFlags.split(',')
                    for singleExcludedFlag in excludedFlagsList:
                        if singleExcludedFlag in flags:
                            del flags[singleExcludedFlag]

                if flags.key_count() != 0:
                    # parse `-Dlib_config=...` flag and save it for later
                    if singleFuncCallDict["Function_name"] == "cudnnGraphLibraryConfigInit":
                        loadModeFlag = flags
                        continue

                    if args.genTestCmd == True:
                        testCmd = "cudnnTest " + str(flags) + " " + str(loadModeFlag)
                        if testCmd not in listofcudnnTest or allowRepeat:
                            listofcudnnTest.append(testCmd)
                        else:
                            pass
                    elif args.genRunCmd == True:
                        (testType, layerCmd) = flagsToLayer(flags)
                        repeatCounts[layerCmd] += 1

                        layer = Layer(args.modelName, flags)

                        if layer not in layer_list or allowRepeat:
                            layer.index = testCounts[testType]
                            testCounts[testType] += 1
                            layer_list.append(layer)
                        else:
                            layer = layer_list[layer_list.index(layer)]
                            layer.repeats += 1

                else:
                    parsingError.append('#!!!!' + singleFuncBlock.replace('\n','\n#'))


    if args.genTestCmd == True:
        print(("\n").join(listofcudnnTest))
    elif args.genRunCmd == True:
        print("\n".join(l.toString(True, 0 if len(layer_list) <= 1 else int(math.log10(len(layer_list)-1))+1) for l in layer_list))

    if parsingError and not args.noParsingError:
        errorSeperator = "#========================Parsing error========================"
        print("\n" + errorSeperator)
        print(("\n" + errorSeperator + "\n\n" + errorSeperator + "\n").join(parsingError))
        print(errorSeperator + "\n")

if __name__== "__main__":
    import sys
    main()


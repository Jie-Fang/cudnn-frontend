from utils import getFwdConvOutputDim, computeStrideNdTransposedPacked, reportCurrentTime, ImplementationError, create_nhwc_strides
import cudnn
reportCurrentTime("import_cudnn")
import json
import os
import sys
import argparse

from test_graph import test_graph

def read_json_test_dict(fname):
    if not os.path.exists(fname):
        raise FileNotFoundError(fname)

    with open(fname) as ifh:
        json_tests = json.load(ifh)
    reportCurrentTime("read_json_test_dict")
    return json_tests

# @raises ImplementationError
# TODO: keep track of which parameters were actually used (i.e., was as cli parameter specified that was never used? E.g., filtC)
def replace_single_param(json_test_def, abstract_params):
    SKIPABLE = ["dimOut", "k", "n", "c", "h", "w", "r", "s", "d"]
    FORMAT_ALL = "formatAll"
    IO_DATA_TYPE = "P"
    INT_LISTS = ["dimOut", "dimA", "filtA", "convStrideA", "dilationA", "padA", "dimB", "Reduce_dimY", "Scale_dim", "CmpB_dim"]
    catch_all = {"formatIn": FORMAT_ALL, "filtFormat": FORMAT_ALL, "formatOut": FORMAT_ALL, "Pin": IO_DATA_TYPE, "Pout": IO_DATA_TYPE}
    layout_params = ["formatIn", "filtFormat", "formatOut", FORMAT_ALL]

    if isinstance(json_test_def, str) and "<" in json_test_def and ">" in json_test_def:
        abstract_param = json_test_def.strip("<>")
        concrete_param = None

        # Most common case: replace a parameter with what we found on the command line
        if abstract_param in abstract_params:
            concrete_param = abstract_params[abstract_param]
        # Some parameters have a catch-all instead (e.g., formatIn is specified by formatAll)
        elif abstract_param in catch_all:
            concrete_param = abstract_params[catch_all[abstract_param]]
        # Some parameters can be skipped in the first pass of parameter replacement (e.g., dimOut as it will be derived later)
        elif abstract_param in SKIPABLE:
            return json_test_def
        else:
            raise ImplementationError("CLI parameter {} not provided".format(abstract_param))
            
        # Now that we have found the concrete parameter, we may need to do some post processing
        if isinstance(concrete_param, str) and abstract_param in INT_LISTS:
            # Some of the kv args in test command put [] around int lists
            return list(eval(concrete_param.strip("[]")))

        else:
            return concrete_param

    # If this wasn't an abstract parameter, just return it
    return json_test_def

# @raises ImplementationError
def replace_abstract_test_params(json_test_def, abstract_params):
    if isinstance(json_test_def, dict):
        for key in json_test_def.keys():
            if not isinstance(json_test_def[key], dict) and not isinstance(json_test_def[key], list):
                json_test_def[key] = replace_single_param(json_test_def[key], abstract_params)
            else:
                replace_abstract_test_params(json_test_def[key], abstract_params)
    elif isinstance(json_test_def, list):
        for index, item in enumerate(json_test_def):
            if not isinstance(item, dict) and not isinstance(item, list):
                json_test_def[index] = replace_single_param(item, abstract_params)
            else:
                replace_abstract_test_params(item, abstract_params)
      
    return json_test_def

def run_test_from_legacy_args(parent_args, unparsed_graphRunner_args):
    print("Running json graph")

    kTEST_NAME = "jsonTestName"
    kDATA_TYPES = ['s', 'h', 'float', 'half', "g", "b", "int8", "int32", "bool", "bf16", "i"]
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Step 0: A quick sanitizing of the remaining arguments to be further parsed by run_test_from_legacy_args
    sanitized_graphRunner_args = []
    for item in unparsed_graphRunner_args:
        if not "=" in item:
            sanitized_graphRunner_args.append(item)
        else:
            stripped = item.strip('=')
            if "==" in stripped:
                sanitized_graphRunner_args.extend(item.split("=="))
            elif "=" in stripped:
                sanitized_graphRunner_args.extend(item.split("="))
            else:
                sanitized_graphRunner_args.append(stripped)
    
    # General args
    l_parser = argparse.ArgumentParser("legacy_graph_runner", allow_abbrev=False)
    l_parser.add_argument("-" + kTEST_NAME, required=True)
    l_parser.add_argument("-jsonPath", dest="json_fname", action='store', default=os.path.join(SCRIPT_DIR, "json_graph_defs",  "fusionGraphTests.json"))
    l_parser.add_argument("-T", dest="timing_loop", action="store", type=int, default=0, help="positive value: the number of times to run the kernel, no refcheck; 0: run kernel once, run refcheck")
    l_parser.add_argument("-p", dest="pseudo_random_generator", action="store", choices=["r"], help="Only normal distribution (\"r\") supported")
    # These are somewhat complicated. We expect something like -kv=layout NCHH. In addition, multiple -kv invocations can happen. This means 2 things:
    # 1) we need to set nargs to 2, so argparse knows it expects 2 values (in case above this is layout and NCHW). 
    # 2), we need to specify action as append and build a list of kv values
    l_parser.add_argument("-kv", action='append', nargs=2, dest='key_values', default=[], help="kv values to be specified as [-kv=<key> <value>]+ e.g., -kv=layout NCHW.")
    # TODO(@mbreughe): Add proper support for this 
    l_parser.add_argument("-minDevVer", default=0, type=int)
    l_parser.add_argument("-groupCount", default=1, type=int)
    l_parser.add_argument("-atol", default=0.1, type=float)
    l_parser.add_argument("-rtol", default=0.1, type=float)
    # Convolution related params (TODO(@mbreughe): add grouping)
    l_parser.add_argument("-x", "-convMode", dest='convMode', action='store_const', const='CUDNN_CROSS_CORRELATION', default='CUDNN_CONVOLUTION')
    l_parser.add_argument("-dim", type=int, default=2)
    l_parser.add_argument("-pad_d", action='store', default=0)
    l_parser.add_argument("-pad_h", action='store', default=0)
    l_parser.add_argument("-pad_w", action='store', default=0)
    l_parser.add_argument("-u", action='store', default=1)
    l_parser.add_argument("-v", action='store', default=1)
    l_parser.add_argument("-dimA", default=None) # Alterantively we can use nargs='+' and specify as -dimA 1 2 3 4 (without comma's)
    l_parser.add_argument("-filtA", default=None)
    l_parser.add_argument("-dimOut", default=None) # Don't specify a default as there is special logic for it in replace_implicit_params
    l_parser.add_argument("-convStrideA")   # DO NOT SPECIFY A DEFAULT. It will get taken care of in the second parsing pass
    l_parser.add_argument("-dilationA")     # DO NOT SPECIFY A DEFAULT. It will get taken care of in the second parsing pass
    l_parser.add_argument("-padA")          # DO NOT SPECIFY A DEFAULT. It will get taken care of in the second parsing pass
    l_parser.add_argument("-A", type=int, choices=[1], help="Dummy argument to support default convolution with alpha=1")
    l_parser.add_argument("-B", type=int, choices=[0], help="Dummy argument to support default convolution with beta=0")
    l_parser.add_argument("-n", help="Deprecated. This overrides the N dimension of a convolution.")
    # GEMM related params
    l_parser.add_argument("-gemm_B", type=int, action="store")
    l_parser.add_argument("-gemm_M", type=int, action="store")
    l_parser.add_argument("-gemm_N", type=int, action="store")
    l_parser.add_argument("-gemm_K", type=int, action="store")
    l_parser.add_argument("-gemm_layout_A", action="store")
    l_parser.add_argument("-gemm_layout_B", action="store")
    l_parser.add_argument("-gemm_layout_C", action="store")
    # Type related
    l_parser.add_argument("-Pin", choices=kDATA_TYPES)
    l_parser.add_argument("-Pout", choices=kDATA_TYPES)
    l_parser.add_argument("-Pcomp", choices=kDATA_TYPES)
    l_parser.add_argument("-P", help="Data type for both I/O in case Pin or Pout is not specified", choices=kDATA_TYPES)
    l_parser.add_argument("-Tin", choices=kDATA_TYPES)
    l_parser.add_argument("-Tout", choices=kDATA_TYPES)
    l_parser.add_argument("-Tcomp", choices=kDATA_TYPES)
    l_parser.add_argument("-Tstore", choices=kDATA_TYPES)
    l_parser.add_argument("-outType", choices=kDATA_TYPES)
    l_parser.add_argument("-reduceY_Type", choices=kDATA_TYPES)
    # Format related
    format_options = l_parser.add_argument_group("format_options")
    format_options.add_argument("-formatIn", type=int, choices=[0,1])
    format_options.add_argument("-formatOut", type=int, choices=[0,1])
    format_options.add_argument("-filtFormat", type=int, choices=[0,1])
    # TODO(@mbreughe): make this mutually exclusive with the rest (Cannot use python's add_mutual_exclusive_group here. I tried.)
    format_options.add_argument("-formatAll", type=int, choices=[0,1])

    #Other
    l_parser.add_argument("-backendEngine", dest='backendEngine', action='store', default=-1, required=False)
    l_parser.add_argument("-Dforce_jit_dbg", action='store', default=None, type=int)

    # Ignored arguments
    # TODO(@mbreughe): Extracted ignored_keys from ignored_args
    ignored_args = l_parser.add_argument_group('ignored_args')
    ignored_args.add_argument("-d", action='store', default=None)
    ignored_args.add_argument("-b", action='store_true', default=None)
    ignored_args.add_argument("-S", action='store_true', default=None)
    ignored_args.add_argument("-gpuRef", action='store_true', default=None)
    ignored_args.add_argument("-engineCfgSweep", action='store', default=None, required=False)
    ignored_args.add_argument("-knobSplitKSlices", action='store', default=None, required=False)
    ignored_args.add_argument("-knobKernelCfg", action='store', default=None, required=False)
    ignored_args.add_argument("-serialization", action='store', default=None, required=False)
    ignored_args.add_argument("-pref", action='store', default=None, required=False)
    ignored_args.add_argument("-Pmath", action='store', default=None, required=False)

    ignored_keys = [argument.dest for argument in ignored_args._group_actions]

    # Step 1: First parsing and cleanup pass
    legacy_args = l_parser.parse_args(sanitized_graphRunner_args)
    
    abstract_params = vars(legacy_args)

    # Special treatment for kv parameters:
    for kv in legacy_args.key_values:
        assert len(kv) == 2
        k = kv[0].strip(":")
        v = kv[1].strip(":")
        abstract_params[k] = v

    # Remove the unparsed key_values
    del abstract_params['key_values']

    # Step 2: Second parsing pass -- Some params' default value is dependent on other specifications
    spatial_dims = abstract_params["dim"]
    if abstract_params['padA'] is None:
        padA = [0] * spatial_dims
        if spatial_dims == 3:
            padA[0] = int(abstract_params["pad_d"])
            padA[1] = int(abstract_params["pad_h"])
            padA[2] = int(abstract_params["pad_w"])
        elif spatial_dims == 2:
            padA[0] = int(abstract_params["pad_h"])
            padA[1] = int(abstract_params["pad_w"])
        else:
            raise ValueError()
        abstract_params['padA'] = padA
    
    if abstract_params['convStrideA'] is None:
        convStrideA = [1] * spatial_dims
        if spatial_dims == 3:
            convStrideA[0] = int(abstract_params["u"])
            convStrideA[1] = int(abstract_params["v"])
            convStrideA[2] = 1
        elif spatial_dims == 2:
            convStrideA[0] = int(abstract_params["u"])
            convStrideA[1] = int(abstract_params["v"])
        else:
            raise ValueError()
        abstract_params['convStrideA'] = convStrideA
    
    # dilation default is array of ones
    # stride default is 1, which is derived from cpp harness: first it's 0, but then getConvNdDesc and json_util override it with 1
    for param_name in ["dilationA"]:
        if abstract_params[param_name] is None:
            abstract_params[param_name] = [1] * spatial_dims

    # Deprecated: specifying batch dim through -n
    if abstract_params["n"] is not None:
        if abstract_params["dimA"] is not None:
            disassembled = abstract_params["dimA"].split(",")
            disassembled[0] = abstract_params["n"]
            abstract_params["dimA"] = ",".join(disassembled)
        if abstract_params["dimOut"] is not None:
            disassembled = abstract_params["dimOut"].split(",")
            disassembled[0] = abstract_params["n"]
            abstract_params["dimOut"] = ",".join(disassembled)

    # Deprecated: default filter dims.
    default_filt_dims = [96] + [3] * (spatial_dims+1)
    if abstract_params["filtA"] is None:
        abstract_params["filtA"] = default_filt_dims

    # Deprecated: -r, -s, -t could override filtA in cudnnTest. We removed this from the CLI in pycudnnTest.

    # Deprecated: infer convolution's dimOut parameter
    # TODO(@mbreughe): cudnnTest has been treating convolution tests specially. This means that there are tests
    # that use convolution-style dimension inference, even though the test does not have a convolution operator.
    # To avoid confusions with this, we should redefine these tests (e.g., ScaleBiasReductionCol2D_abstract, ScaleBiasReductionCol3D_abstract)
    # In addition, the code below should only be present in replace_implicit_params, when we know that a convolution operator is used.
    if abstract_params.get("dimA") and abstract_params.get("filtA") and (not abstract_params.get("dimOut")):
        dimA = list(eval(abstract_params["dimA"]))

        def get_integer_list(param):
            if isinstance(param, str):
                param = list(eval(param))
            return param
        
        filtA = get_integer_list(abstract_params["filtA"])
        padA = get_integer_list(abstract_params["padA"])
        convStrideA = get_integer_list(abstract_params["convStrideA"])
        dilationA = get_integer_list(abstract_params["dilationA"])
        inferred_params = get_conv_dim_placeholders(dimA, filtA, padA, convStrideA, dilationA)
        abstract_params["dimOut"] = inferred_params ["dimOut"]
        abstract_params["n"] = inferred_params["n"]
        abstract_params["c"] = inferred_params["c"]
        abstract_params["h"] = inferred_params["h"]
        abstract_params["w"] = inferred_params["w"]
        abstract_params["k"] = inferred_params["k"]
        abstract_params["r"] = inferred_params["r"]
        abstract_params["s"] = inferred_params["s"]
        abstract_params["d"] = inferred_params["d"]

    # Ignored arguments:
    for key in ignored_keys:
        if key in abstract_params and abstract_params[key] is not None:
            print ("Note: Argument -{} ignored.".format(key))
            del abstract_params[key]

    # Remove any parameters that were not specified or didn't have a default:
    for key in list(abstract_params.keys()):
        if abstract_params[key] is None:
            del abstract_params[key]

    # Step 3: Third parsing -- attempt to convert anything that is numeric for future convenience
    for key in abstract_params:
        val = abstract_params[key]
        if isinstance(val, str) and val.isnumeric():
            try:
                val = int(val)
            except:
                val = float(val)
            abstract_params[key] = val

    json_test_name = legacy_args.jsonTestName

    reportCurrentTime("arg_parse_2")
    json_tests = read_json_test_dict(legacy_args.json_fname)
    
    assert json_test_name in json_tests
    abstract_test_dict = json_tests[json_test_name]
    try:
        concrete_test_dict = replace_abstract_test_params(abstract_test_dict, abstract_params)
        reportCurrentTime("replace_abstract_test_params")
        # TODO(@mbreughe): Make this safer
        # Overwrite the environment variable that controls forced JIT-ing
        force_jit_env_before = os.environ.get("CUDNN_FORCE_JIT_DBG", None)
        if "Dforce_jit_dbg" in abstract_params:
            os.environ["CUDNN_FORCE_JIT_DBG"] = str(legacy_args.Dforce_jit_dbg)

        # Step 4: Replace implicit parameters and create the test graph
        testGraph = setup_test_graph_from_json(concrete_test_dict, legacy_args.backendEngine)

        if legacy_args.timing_loop == 0:
            run_test_from_json_definition(testGraph, concrete_test_dict)
        else:
            testGraph.build_cudnn_graph()
            testGraph.cudnn_execute(int(legacy_args.timing_loop))

        # Now recover the environment variable
        if force_jit_env_before is not None:
            os.environ["CUDNN_FORCE_JIT_DBG"] = force_jit_env_before
        elif "CUDNN_FORCE_JIT_DBG" in os.environ:
            del os.environ["CUDNN_FORCE_JIT_DBG"]
    except ImplementationError as e:
        print("MB Unsupported: ", e.reason)
        raise e

# @brief: A utility class to help parse graphRunner json graph definitions
# @details: This is a utility class to extract infomration from a json graph definition (graphRunner json format)
# In general its functions will be working with 1 or 2 different dictionaries:
#   1. self.jnode the json graph definition itself, containing properties of the operation (e.g., padding, dilation, etc.), but also I/O tensor names
#   2. self.operation_mapping which contains information on how to map json graph definiton keywords onto pycudnn
class Legacy_operation:
    mapping = None
    OPERATION = "operation"
    PW_OPERATION = "pointwise"
    PW_MODE = "pointwiseMode"
    OP_MAPPING = "operation"
    NAME = "name"
    OUTPUT = "output"
    INPUT = "input"
    PROPS = "properties"
    VAL_MAP = "value_mapping"
    MAPPING_FILE = os.path.join(os.path.dirname(__file__), "json_graph_fe_mapping.json")
    
    def setup_operation_mapping():
        fname = Legacy_operation.MAPPING_FILE
        assert os.path.exists(fname)
        with open(fname) as ifh:
            Legacy_operation.mapping = json.load(ifh)
    
    def __init__(self, node):
        if Legacy_operation.mapping is None:
            Legacy_operation.setup_operation_mapping()
        
        operation = node[Legacy_operation.OPERATION]
        self.legacy_operation = operation

        try:
            # Pointwise operations are double nested since they have an operation mode
            if operation == Legacy_operation.PW_OPERATION:
                pw_mode = node[Legacy_operation.PW_MODE]
                self.operation_mapping = Legacy_operation.mapping[operation][pw_mode]
            else:
                self.operation_mapping = Legacy_operation.mapping[operation]
        except KeyError as orig_e:
            print(orig_e.__str__())
            e = ImplementationError("Operation:[{}]".format(str(orig_e)))
            raise e

        # keep track of the json node
        self.jnode = node

    def get_legacy_operation_type(self):
        return self.legacy_operation

    def get_name(self):
        return self.jnode[Legacy_operation.NAME]

    def get_pycudnn_operation_name(self):
        return self.operation_mapping[Legacy_operation.OP_MAPPING]

    #@brief: Return the tensor names of the outputs
    def get_output_tensor_names(self):
        outputs = []
        # The mapping contains how they are called in graphRunner test defs (e.g., "Y" for conv and pointwise)
        for output_key in self.operation_mapping[Legacy_operation.OUTPUT]:
            outputs.append(self.jnode[output_key])

        return outputs
    
    def translate_to_pycudnn_value(self, leg_prop, leg_value):
        if Legacy_operation.VAL_MAP in self.operation_mapping and leg_prop in self.operation_mapping[Legacy_operation.VAL_MAP]:
            # TODO(@mbreughe): can we get rid of the eval?
            return eval(self.operation_mapping[Legacy_operation.VAL_MAP][leg_prop][leg_value])
        else:
            return Legacy_value.translate_to_pycudnn_value(leg_prop, leg_value)
    
    def get_operation_properties(self):
        property_map = {}
        for leg_prop, pycudnn_prop in self.operation_mapping[Legacy_operation.PROPS].items():
            leg_value = self.jnode[leg_prop]
            property_map[pycudnn_prop] = self.translate_to_pycudnn_value(leg_prop, leg_value)

        return property_map
    
    def get_input_name_mapping(self):
        input_map = {}
        for leg_input, pycudnn_input in self.operation_mapping[Legacy_operation.INPUT].items():
            input_map[pycudnn_input] = self.jnode[leg_input]

        return input_map

    # @example io_tensor is a legacy name, e.g., "W", or "X"
    def get_io_tensor_name(self, io_tensor):
        if io_tensor in self.jnode:
            return self.jnode[io_tensor]
        return None
    
# @note: we could opt to store the mapping in a file like we do for operation
# however, the mapping is much fewer, so we decide to avoid the overhead for now
class Legacy_tensor:
    mapping = {"dataType": "data_type", "dim": "dim", "name": "name", "stride": "stride"}
    # TODO(@mbreughe): implement the following fill related properties
    mapping.update({"fill": "fill", "min": "min", "max": "max", "value": "value", "mean": "mean", "std_dev": "std_dev"})

    def __init__(self, jtensor):
        self.jtensor = jtensor

    def get_data_type(self):
        return Legacy_value.translate_to_pycudnn_value("dataType", self.jtensor["dataType"])
    
    def get_dim(self):
        return Legacy_value.translate_to_pycudnn_value("dim", self.jtensor["dim"])
    
    def get_stride(self):
        dim = self.jtensor["dim"]

        # Just return test provided strides if already present
        if self.jtensor.get("stride", None):
            return self.jtensor["stride"]

        layout = self.jtensor["layout"]
        nbDims = len(dim)

        if layout == "ROW_MAJOR":
            axes_order = list(range(nbDims))
            strides = computeStrideNdTransposedPacked(nbDims, dim, axes_order)
        elif layout == "COL_MAJOR":
            axes_order = list(range(nbDims))
            swap_val = axes_order[-1]
            axes_order[-1] = axes_order[-2]
            axes_order[-2] = swap_val
            strides = computeStrideNdTransposedPacked(nbDims, dim, axes_order)
        elif layout == "NCHW" or str(layout) == '0':
            axes_order = list(range(nbDims))
            strides = computeStrideNdTransposedPacked(nbDims, dim, axes_order)
        elif layout == "NHWC" or str(layout) == '1':
            #TODO (@mbreughe): get this to work with the formula above
            strides = create_nhwc_strides(dim)

        return strides

    def get_tensor_properties(self):
        pycudnn_props = {}
        for key, value in self.jtensor.items():
            # special treatment for layout
            if key == "layout":
                pycudnn_props["stride"] = self.get_stride()
                layout = self.jtensor["layout"]
                # TODO(@mbreughe): fix this in test_graph.py
                # It shouldn't be necessary to specify this through both strides and layout
                if layout == "NCHW" or str(layout) == '0':
                    pycudnn_props[key] = "NCHW"
                elif layout == "NHWC" or str(layout) == '1':
                    pycudnn_props[key] = "NHWC"
            
            elif key in Legacy_tensor.mapping:
                    new_key = Legacy_tensor.mapping[key]
                
                    pycudnn_props[new_key] = Legacy_value.translate_to_pycudnn_value(key, value)

            else:
                raise ImplementationError("Unsupported tensor property \"{}\"".format(key))

        return pycudnn_props

# This is to be used as static class only
class Legacy_value:
    mapping = {"dataType": {"s": cudnn.data_type.FLOAT, "h": cudnn.data_type.HALF, 
                        "g": cudnn.data_type.BFLOAT16, "b": cudnn.data_type.INT8,
                        "float": cudnn.data_type.FLOAT, "half": cudnn.data_type.HALF
                        , "bool": cudnn.data_type.BOOLEAN, "int8": cudnn.data_type.INT8, "int32": cudnn.data_type.INT32
                        , "bf16": cudnn.data_type.BFLOAT16, "i": cudnn.data_type.INT32}}
    indirection = {"mathPrec": "dataType"}

    @staticmethod
    def translate_to_pycudnn_value(legacy_key_name, legacy_value):
        if legacy_key_name in Legacy_value.mapping:
            return Legacy_value.mapping[legacy_key_name][legacy_value]
        elif legacy_key_name in Legacy_value.indirection:
            return Legacy_value.translate_to_pycudnn_value(Legacy_value.indirection[legacy_key_name], legacy_value)
        
        return legacy_value
    
def get_conv_dim_placeholders(X_tensor_dim, filter_tensor_dim, padA, stdA, dilA):
    spatial_dims = len(X_tensor_dim) - 2

    dimOut = [None] * (2+spatial_dims)
    
    dimOut[0] = X_tensor_dim[0]
    dimOut[1] = filter_tensor_dim[0]
    
    for d in range(0,spatial_dims):
        dimOut[d+2] = getFwdConvOutputDim(X_tensor_dim[d+2], padA[d], filter_tensor_dim[d+2],
                                    stdA[d], dilA[d])
    
    input_nbDims = len(X_tensor_dim)
    filter_nbDims = len(filter_tensor_dim)

    inferred_params = dict()
    inferred_params["dimOut"] = dimOut
    inferred_params["n"] = X_tensor_dim[0]
    inferred_params["c"] = X_tensor_dim[1]
    inferred_params["h"] = X_tensor_dim[input_nbDims - 2]
    inferred_params["w"] = X_tensor_dim[input_nbDims - 1]
    inferred_params["k"] = filter_tensor_dim[0]
    inferred_params["r"] = filter_tensor_dim[filter_nbDims - 2]
    inferred_params["s"] = filter_tensor_dim[filter_nbDims - 1]
    # Assign NCDHW 5d tensor D abstract param if present
    inferred_params["d"] = dimOut[2] if len(dimOut) == 5 else None
    return inferred_params

# @brief: Replace derived parameters
# @details: Some parameters in the test definition are not explicitly specified and depend on several other specifications of the test
# e.g., dimOut: when the operation is a convolution fprop, it depends on the dimensions of the input tensor and output tensor
def replace_implicit_params(legacy_ops, jtensor_dict):
    # We could consider doing something smart where we skip this step if there are no
    # implicit parameters left to replace (e.g., from the first pass to replace_abstract_test_params we keep a hint)
    # This could save us time (but is this optimization really worth the complexity?)
    
    implicit_params = {}

    # Replace dimOut
    conv_ops = ["dgrad", "wgrad", "conv"]

    legacy_operations = [op.get_legacy_operation_type() for op in legacy_ops]

    # Is any of the operations a convolution operation?
    if set(legacy_operations).intersection(set(conv_ops)):
        for legacy_op in legacy_ops:
            if legacy_op.get_legacy_operation_type() in conv_ops:
                break
        
        filter_tensor_name = legacy_op.get_io_tensor_name("W")
        if filter_tensor_name is None:
            filter_tensor_name = legacy_op.get_io_tensor_name("dW")

        X_tensor_name = legacy_op.get_io_tensor_name("X")
        if X_tensor_name is None:
            X_tensor_name = legacy_op.get_io_tensor_name("dX")
        
        for jtensor in jtensor_dict:
            if jtensor["name"] == X_tensor_name:
                X_tensor_dim = jtensor["dim"]
            elif jtensor["name"] == filter_tensor_name:
                filter_tensor_dim = jtensor["dim"]

        padA = legacy_op.jnode["pad"]
        stdA = legacy_op.jnode["stride"]
        dilA = legacy_op.jnode["dilation"]
        
        inferred_params = get_conv_dim_placeholders(X_tensor_dim, filter_tensor_dim, padA, stdA, dilA)
        implicit_params.update(inferred_params)

    # Skip any property inferencing for matmuls
    elif set(legacy_operations).intersection(set(['matmul'])):
        pass
    # Pointwise operations
    elif set(legacy_operations).intersection(set(["pointwise", "reduction"])):
        X_tensor_name = legacy_ops[0].get_io_tensor_name("X")
        if X_tensor_name is None:
            X_tensor_name = legacy_ops[0].get_io_tensor_name("dX")
        
        for jtensor in jtensor_dict:
            if jtensor["name"] == X_tensor_name:
                X_tensor_dim = jtensor["dim"]

        dim = X_tensor_dim
        implicit_params["n"] = dim[0]
        implicit_params["c"] = dim[1]
        implicit_params["h"] = dim[-2]
        implicit_params["w"] = dim[-1]
        # Assign NCDHW 5d tensor D abstract param if present
        implicit_params["d"] = dim[-3] if len(dim) == 5 else None
        
    jtensor_dict = replace_abstract_test_params(jtensor_dict, implicit_params)
    return jtensor_dict

def setup_test_graph_from_json(json_dict, backendEngine=-1):
    testGraph = test_graph()
    testGraph.set_backend_engine(backendEngine)

    jtensor_dict = json_dict["tensors"]
    jnode_list = json_dict["nodes"]

    legacy_ops = [Legacy_operation(jnode) for jnode in jnode_list]

    # Replace any remaining abstract parameters from the tensor dictionary (e.g., dimOut)
    # After this call we can use legacy_ops and jtensor_dict to construct the pycudnn graph and tensors
    jtensor_dict = replace_implicit_params(legacy_ops, jtensor_dict)

    TGTensors = {}
    # Dictionary of operation names to (test_graph operation, LegacyOperation) tuples
    Operations = {}

    # Create all pycudnn operation nodes (without input) and their associated output tensors
    for legacy_op in legacy_ops:
        name = legacy_op.get_name()

        # Create node without input and add to the graph
        test_graph_op = create_test_graph_node(legacy_op)
        testGraph.nodes.append(test_graph_op)

        # Bind testGraph outputs with tensor names
        output_names = legacy_op.get_output_tensor_names()
        for output_name, output_tensor in zip(output_names, test_graph_op.output):
            TGTensors[output_name] = output_tensor
        
        Operations[name] = (test_graph_op, legacy_op)

    # Propagate any properties from the json test graph's output tensors
    # At this point TGTensors only contains output tensors. 
    # Since output tensors are created in cudnn as a result of adding an operation, 
    # we did not propagate any specifications yet
    for jtensor in jtensor_dict:
        tg_tensor = TGTensors.get(jtensor["name"], None)
        legacy_tensor = Legacy_tensor(jtensor)
        if tg_tensor is None:
            continue

        tg_tensor.set_data_type(legacy_tensor.get_data_type())
        tg_tensor.set_dim(legacy_tensor.get_dim())
        tg_tensor.set_stride(legacy_tensor.get_stride())
    
    # Identify all tensors in jtensor_dict that are not output tensors
    input_tensors = [tensor for tensor in jtensor_dict if not tensor["name"] in TGTensors]

    # Create a TestTensor for every input tensor
    for jtensor in input_tensors:
        legacy_tensor = Legacy_tensor(jtensor)
        
        t = testGraph.tensor(**legacy_tensor.get_tensor_properties())
        TGTensors[jtensor["name"]] = t
    
    # Finalize the connections by adding inputs and properties to the operation nodes
    for name, (test_graph_op, legacy_op) in Operations.items():
        test_graph_op.set_kwargs(create_kwargs(legacy_op, TGTensors))

    return testGraph

# @brief: main function to run json graphs with reference check
# @TODO(mbreughe): rename this function
def run_test_from_json_definition(testGraph, json_dict):
    # TODO(@mbreughe)
    # we can do a front-end test by using the json dimensions of the virtual tensors

    graph = testGraph.build_cudnn_graph()
    # Read in rtol/atol from json
    atol = 1e-2
    rtol = 1e-2
    if "tolerances" in json_dict:
        atol = float(json_dict["tolerances"]["abs"])
        rtol = float(json_dict["tolerances"]["rel"])

    reportCurrentTime("test_setup")      
    testGraph.cudnn_execute_and_compare_to_reference(atol=atol, rtol=rtol)

# @brief: Create a pycudnn node from the legacy_op
def create_test_graph_node(legacy_op):
    name = legacy_op.get_name()
    op_name = legacy_op.get_pycudnn_operation_name()
    op_ptr = getattr(cudnn.pygraph, op_name)
    return test_graph.create_operation(op_ptr, name)

# @brief: Combine properties from the legacy op, and input tensors to create the pycudnn kwargs
def create_kwargs(legacy_op, TGTensors):
    kwargs = {}
    kwargs.update(legacy_op.get_operation_properties())
    input_name_mapping = legacy_op.get_input_name_mapping()

    for pycudnn_input_name, tensor_name in input_name_mapping.items():
        kwargs[pycudnn_input_name] = TGTensors[tensor_name]
    
    return kwargs
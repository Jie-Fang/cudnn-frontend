import json
import os
import cudnn
import pytest
import sys

from test_graph import test_graph, operation
from utils import getFwdConvDilatedFilterDim, getFwdConvPaddedImageDim, getFwdConvOutputDim

class ImplementationError(Exception):
    def __init__(self, reason):
        self.reason = reason

def read_json_test_dict(fname):
    if not os.path.exists(fname):
        raise FileNotFoundError(fname)

    with open(fname) as ifh:
        json_tests = json.load(ifh)
    return json_tests

# @raises ImplementationError
# TODO: keep track of which parameters were actually used (i.e., was as cli parameter specified that was never used? E.g., filtC)
def replace_single_param(json_test_def, abstract_params):
    SKIPABLE = ["dimOut", "k", "n", "c", "h", "w"]
    FORMAT_ALL = "formatAll"
    INT_LISTS = ["dimOut", "dimA", "filtA", "convStrideA", "dilationA", "padA"]
    catch_all = {"formatIn": FORMAT_ALL, "filtFormat": FORMAT_ALL, "formatOut": FORMAT_ALL}
    layout_params = ["formatIn", "filtFormat", "formatOut", FORMAT_ALL]
    defaults = {"groupCount": 1}
    if isinstance(json_test_def, str) and "<" in json_test_def and ">" in json_test_def:
        abstract_param = json_test_def.strip("<>")
        concrete_param = None
                
        # Most common case: replace a parameter with what we found on the command line
        if abstract_param in abstract_params:
            concrete_param = abstract_params[abstract_param]
        # Some parameters have a catch-all instead (e.g., formatIn is specified by formatAll)
        elif abstract_param in catch_all:
            concrete_param = abstract_params[catch_all[abstract_param]]
        # Some parameters have default values if unspecified
        elif abstract_param in defaults:
            concrete_param = defaults[abstract_param]
        # Some parameters are booleans that are set on the command line, or have default value otherwise
        elif abstract_param == "convMode":
            if 'x' in abstract_params:
                concrete_param = "CUDNN_CROSS_CORRELATION"
            else:
                concrete_param = "CUDNN_CONVOLUTION"
        # Some parameters can be skipped in the first pass of parameter replacement (e.g., dimOut as it will be derived later)
        elif abstract_param in SKIPABLE:
            return json_test_def
        else:
            raise ImplementationError("CLI parameter {} not provided".format(abstract_param))
        
        # Now that we have found the concrete parameter, we may need to do some post processing
        if isinstance(concrete_param, str) and abstract_param in INT_LISTS:
            return list(eval(concrete_param))

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

def run_test_from_legacy_args(json_fname, args):
    print("Running from legacy json graph definition")
    kTEST_NAME = "jsonTestName"

    # Parse the cudnnTest arguments
    abstract_params = dict()
    for kv in args.split(" -"): # note the space: this is to ensure we split something like -atol5e-3 correctly
        kv_pair = kv.split(":")
        assert len(kv_pair) == 2
        k = kv_pair[0].strip("-").strip("=")
        v = kv_pair[1].strip("-").strip("=")
        abstract_params[k] = v

    json_test_name = abstract_params[kTEST_NAME]

    json_tests = read_json_test_dict(json_fname)
    
    assert json_test_name in json_tests
    abstract_test_dict = json_tests[json_test_name]
    try:
        concrete_test_dict = replace_abstract_test_params(abstract_test_dict, abstract_params)
        run_test_from_json_definition(concrete_test_dict)
    except ImplementationError as e:
        print("MB Unsupported: ", e.reason)
        sys.exit(1)
# A helper function to read json dictionaries
# @note: scope tells us that the dictionary is being loaded only once
@pytest.fixture(scope="module")
def json_dict(request):
    fname = request.param
    return read_json_test_dict(fname)

# Main entry point for json defined graphs
# @param json_dict: implicit call to a fixture using the json file name provided on the command line
# @param test_name: the specific test to be ran
def test_json_graph(json_dict, test_name):
    assert test_name in json_dict
    run_test_from_json_definition(json_dict[test_name])

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
    mapping = {"dataType": "data_type", "dim": "dim"}

    def __init__(self, jtensor):
        self.jtensor = jtensor

    def get_data_type(self):
        return Legacy_value.translate_to_pycudnn_value("dataType", self.jtensor["dataType"])
    
    def get_dim(self):
        return Legacy_value.translate_to_pycudnn_value("dim", self.jtensor["dim"])
    
    def get_tensor_properties(self):
        pycudnn_props = {}
        for key, value in self.jtensor.items():
            new_key = key
            if key in Legacy_tensor.mapping:
                new_key = Legacy_tensor.mapping[key]
            pycudnn_props[new_key] = Legacy_value.translate_to_pycudnn_value(key, value)

        return pycudnn_props

# This is to be used as static class only
class Legacy_value:
    mapping = {"dataType": {"s": cudnn.data_type.FLOAT, "h": cudnn.data_type.HALF,
               "float": cudnn.data_type.FLOAT, "half": cudnn.data_type.HALF}}
    indirection = {"mathPrec": "dataType"}

    @staticmethod
    def translate_to_pycudnn_value(legacy_key_name, legacy_value):
        if legacy_key_name in Legacy_value.mapping:
            return Legacy_value.mapping[legacy_key_name][legacy_value]
        elif legacy_key_name in Legacy_value.indirection:
            return Legacy_value.translate_to_pycudnn_value(Legacy_value.indirection[legacy_key_name], legacy_value)
        elif legacy_key_name == "layout":
            # TODO(@mbreughe): calculate strides instead. but if you do, also change test_graph code
            if legacy_value in ["NHWC", "NCHW"]:
                return legacy_value
            elif int(legacy_value) == 0:
                return "NCHW"
            elif int(legacy_value) == 1:
                return "NHWC"
            else:
                raise ValueError("Unknown value {} for {}".format(legacy_value, legacy_key_name))

        return legacy_value
    
def get_conv_dim_placeholders(legacy_op, jtensor_dict):
    filter_tensor_name = legacy_op.get_io_tensor_name("W")
    if filter_tensor_name is None:
        filter_tensor_name = legacy_op.get_io_tensor_name("dW")

    X_tensor_name = legacy_op.get_io_tensor_name("X")
    if X_tensor_name is None:
        X_tensor_name = legacy_op.get_io_tensor_name("dX")

    # Note that this is limited to 4 dimensions. It is here to mimic what we had for legacy reasons
    dimOut = [None] * 4
    for jtensor in jtensor_dict:
        if jtensor["name"] == X_tensor_name:
            X_tensor_dim = jtensor["dim"]
        elif jtensor["name"] == filter_tensor_name:
            filter_tensor_dim = jtensor["dim"]
    
    dimOut[0] = X_tensor_dim[0]
    dimOut[1] = filter_tensor_dim[0]
    
    padA = legacy_op.jnode["pad"]
    stdA = legacy_op.jnode["stride"]
    dilA = legacy_op.jnode["dilation"]
    for d in range(0,2):
        dimOut[d+2] = getFwdConvOutputDim(X_tensor_dim[d+2], padA[d], filter_tensor_dim[d+2],
                                    stdA[d], dilA[d])
    
    input_nbDims = len(X_tensor_dim)
    filter_nbDims = len(filter_tensor_dim)

    N = X_tensor_dim[0]
    C = X_tensor_dim[1]
    H = X_tensor_dim[input_nbDims - 2]
    W = X_tensor_dim[input_nbDims - 1]
    K = filter_tensor_dim[0]
    R = filter_tensor_dim[filter_nbDims - 2]
    S = filter_tensor_dim[filter_nbDims - 1]

    return [dimOut, N, C, H, W, K, R, S]

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
        dimOut, N, C, H, W, K, R, S = get_conv_dim_placeholders(legacy_op, jtensor_dict)

        implicit_params["dimOut"] = dimOut
        implicit_params["n"] = N
        implicit_params["c"] = C
        implicit_params["h"] = H
        implicit_params["w"] = W
        implicit_params["k"] = K
        implicit_params["r"] = R
        implicit_params["s"] = S
        
    jtensor_dict = replace_abstract_test_params(jtensor_dict, implicit_params)
    return jtensor_dict

# @brief: main function to run json graphs
def run_test_from_json_definition(json_dict):
    testGraph = test_graph()

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

    # TODO(@mbreughe)
    # we can do a front-end test by using the json dimensions of the virtual tensors

    graph = testGraph.build_cudnn_graph()

    # Read in rtol/atol from json
    atol = 1e-2
    rtol = 1e-2
    if "tolerances" in json_dict:
        atol = float(json_dict["tolerances"]["abs"])
        rtol = float(json_dict["tolerances"]["rel"])
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
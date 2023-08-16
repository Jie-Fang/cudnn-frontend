import json
import os
import cudnn
import pytest
import sys

from test_graph import test_graph, operation

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
    FORMAT_ALL = "formatALL"
    INT_LISTS = ["dimOut", "dimA", "filtA", "convStrideA", "dilationA", "padA"]
    catch_all = {"formatIn": FORMAT_ALL, "filtFormat": FORMAT_ALL, "formatOut": FORMAT_ALL}
    defaults = {"groupCount": 1}
    if isinstance(json_test_def, str) and "<" in json_test_def and ">" in json_test_def:
        abstract_param = json_test_def.strip("<>")
        concrete_param = None
        print("Replacing", abstract_param)

        # Some parameters can be skipped in the first pass of parameter replacement (e.g., dimOut as it will be derived later)
        if abstract_param in SKIPABLE:
            return json_test_def
        # Most common case: replace a parameter with what we found on the command line
        elif abstract_param in abstract_params:
            concrete_param = abstract_params[abstract_param]
        # Some parameters have a catch-all instead (e.g., formatIn is specified by formatAll)
        elif abstract_param in catch_all:
            concrete_param = replace_single_param(abstract_params, catch_all[abstract_param])
        # Some parameters have default values if unspecified
        elif abstract_param in defaults:
            concrete_param = defaults[abstract_param]
        # Some parameters are booleans that are set on the command line, or have default value otherwise
        elif abstract_param == "convMode":
            if 'x' in abstract_params:
                concrete_param = "CUDNN_CROSS_CORRELATION"
            else:
                concrete_param = "CUDNN_CONVOLUTION"
        else:
            raise ImplementationError("CLI parameter {} not provided".format(abstract_param))
        
        # Now that we have found the concrete parameter, we may need to do some post processing
        if abstract_param in INT_LISTS:
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
        for item in json_test_def:
            if not isinstance(item, dict) and not isinstance(item, list):
                item = replace_single_param(item, abstract_params)
            else:
                replace_abstract_test_params(item, abstract_params)
            
    return json_test_def

def run_test_from_legacy_args(json_fname, args):
    print("RUnning from legacy")
    kTEST_NAME = "jsonTestName"

    # Parse the cudnnTest arguments
    abstract_params = dict()
    for kv in args.split(" -"): # note the space: this is to ensure we split something like -atol5e-3 correctly
        kv_pair = kv.split(":")
        assert len(kv_pair) == 2
        k = kv_pair[0].strip("-").strip("=")
        v = kv_pair[1].strip("-").strip("=")
        abstract_params[k] = v
    print (abstract_params)

    json_test_name = abstract_params[kTEST_NAME]

    json_tests = read_json_test_dict(json_fname)
    
    assert json_test_name in json_tests
    abstract_test_dict = json_tests[json_test_name]
    try:
        print (abstract_test_dict)
        concrete_test_dict = replace_abstract_test_params(abstract_test_dict, abstract_params)
        print (abstract_test_dict)
        run_test_from_json_definition(concrete_test_dict)
    except ImplementationError as e:
        print("MB Unsupported: ", e.reason)
        sys.exit(0)

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
    
    def get_operation_properties(self):
        property_map = {}
        for leg_prop, pycudnn_prop in self.operation_mapping[Legacy_operation.PROPS].items():
            property_map[pycudnn_prop] = self.jnode[leg_prop]

        return property_map
    
    def get_input_name_mapping(self):
        input_map = {}
        for leg_input, pycudnn_input in self.operation_mapping[Legacy_operation.INPUT].items():
            input_map[pycudnn_input] = self.jnode[leg_input]

        return input_map

# @brief: main function to run json graphs
def run_test_from_json_definition(json_dict):
    testGraph = test_graph()

    jtensor_dict = json_dict["tensors"]
    jnode_list = json_dict["nodes"]

    TGTensors = {}
    # Dictionary of operation names to (test_graph operation, LegacyOperation) tuples
    Operations = {}
    # Create all pycudnn operation nodes (without input) and their associated output tensors
    for node in jnode_list:
        legacy_op = Legacy_operation(node)
        name = legacy_op.get_name()

        # Create node without input and add to the graph
        test_graph_op = create_test_graph_node(legacy_op)
        testGraph.nodes.append(test_graph_op)

        # Bind testGraph outputs with tensor names
        output_names = legacy_op.get_output_tensor_names()
        for output_name, output_tensor in zip(output_names, test_graph_op.output):
            TGTensors[output_name] = output_tensor
        
        Operations[name] = (test_graph_op, legacy_op)

    # At this point TGTensors only contains output tensors. 
    # Identify all tensors in jtensor_dict that are not output tensors
    input_tensors = [tensor for tensor in jtensor_dict if not tensor["name"] in TGTensors]

    # Create a TestTensor for every input tensor
    for tensor in input_tensors:
        # TODO(@mbreughe): are properties correctly transferred? 
        t = testGraph.tensor(**tensor)
        TGTensors[tensor["name"]] = t

    # Finalize the connections by adding inputs and properties to the operation nodes
    for name, (test_graph_op, legacy_op) in Operations.items():
        test_graph_op.set_kwargs(create_kwargs(legacy_op, TGTensors))

    # TODO(@mbreughe)
    # we can do a front-end test by using the json dimensions of the virtual tensors

    graph = testGraph.build_cudnn_graph()

    print(graph)

    # TODO(@mbreughe): read in rtol/atol from json
    testGraph.cudnn_execute_and_compare_to_reference(atol=1e-2,rtol=1e-2)

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
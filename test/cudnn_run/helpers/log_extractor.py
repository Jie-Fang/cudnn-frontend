"""For cudnnTest: This file defined the parser to extract data from the test stdout
Parsers are regex to process one line at a time
"""

import re           # For regular expressions
import collections  # For namedtuple

# Define patterns and tuples for all parsed outputs A LogExtractor consist of a pattern for matching,
# and a namedtuple type for matched group.
# Only type with single field is extractable presently.
class LogExtractor(collections.namedtuple('_NT_LogExtractor', ('pattern', 'nt_type', 'types'))):
    '''Extractor information from output log'''
    name2extractor_map = collections.OrderedDict()

    to_verbose_map = {}
    to_compact_map = {}
    to_type_map = {}

    @classmethod
    def addExtractor(cls, name, pattern, fields, types, re_flags=0):
        '''Add a new extractor (or replaces old one) by {name} based on {pattern},
        with namedtuple(typename, fields) output type, where typename is the string name prepended by _NT_

        {pattern} can be a format string for regular expression.  The following fields are supported:
            FLOAT_GROUP
            INT_GROUP
        '''
        typename = '_NT_{}'.format('_'.join(name.capitalize().split())) # auto-generate namedtuple type name, no space allowed.
        pattern = pattern.format( # substitute common matching group
                FLOAT_GROUP = r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)',
                INT_GROUP   = r'([+-]?\d+)',
                UINT_GROUP  = r'(\d+)',
                UID_GROUP   = r'([\da-z]{32})')
        inst = cls (
                pattern = re.compile(pattern, re_flags),
                nt_type = collections.namedtuple(typename, fields),
                types   = types)
        cls.name2extractor_map[name] = inst # NOTE: updating key does not change the order

        types_split = types.split(' ')

        if len(inst.nt_type._fields) != len(types_split):
            raise TypeError("Fields & Types size does not match: %s & %s" % (inst.nt_type._fields, types_split))

        if len(inst.nt_type._fields) == 1:
            verbose_name = "%s.%s" % (name, inst.nt_type._fields[0])

            cls.to_verbose_map[name] = verbose_name
            cls.to_verbose_map[verbose_name] = verbose_name

            cls.to_compact_map[name] = name
            cls.to_compact_map[verbose_name] = name

            cls.to_type_map[name] = types_split[0]
            cls.to_type_map[verbose_name] = types_split[0]
        else:
            for field, type_name in zip(inst.nt_type._fields, types_split):
                verbose_name = "%s.%s" % (name, field)

                cls.to_verbose_map[verbose_name] = verbose_name
                cls.to_compact_map[verbose_name] = verbose_name

                cls.to_type_map[verbose_name] = type_name

    @classmethod
    def getExtractorList(cls):
        '''Return a list of string of valid --extract arguments'''
        return [name if len(extractor.nt_type._fields) == 1  # --extract name for single field extractor
                else '{}.{}'.format(name, field)                     # --extract name.field for multiple field extractor
                for name, extractor in cls.name2extractor_map.items()
                for field in extractor.nt_type._fields
                ]

    @classmethod
    def getExtractorListVerbose(cls):
        return ['{}.{}'.format(name, field)                     # --extract name.field for multiple field extractor
                for name, extractor in cls.name2extractor_map.iteritems()
                for field in extractor.nt_type._fields]

    @classmethod
    def getVerboseName(cls, name):
        return cls.to_verbose_map[name]

    @classmethod
    def getCompactName(cls, name):
        return cls.to_compact_map[name]

    @classmethod
    def getType(cls, name):
        return cls.to_type_map[name]

    def extractToTuple(self, output):
        '''Search {output} for pattern, if found return namedtuple instance for it'''
        match = self.pattern.search(output)
        if match:
            num_fields = len(self.nt_type._fields)
            matched_groups = match.groups()
            if len(matched_groups) != num_fields:
                raise Exception("Unable to fit groups {:s} into Tuple \"{:s}\" with fields {:s}"
                        .format(matched_groups, self.nt_type.__name__, self.nt_type._fields))
            return self.nt_type(*(matched_groups[:num_fields]))
        return None

    def make_default_tuple(self, defval=None):
        '''Return an instance of self.nt_type with {defval} for all fields'''
        return self.nt_type(*((defval,) * len(self.nt_type._fields)))

#*******************************************************************************
#* Initialize all extractors
#*******************************************************************************
def reals_str(count):
    return " ".join(["REAL"] * count)

def ints_str(count):
    return " ".join(["INTEGER"] * count)

LogExtractor.addExtractor( # Processing line and flags
        name     = 'processing',
        pattern  = r'Processing line (.*): (.*)',
        fields   = "line flags",
        types    = "INTEGER TEXT")

LogExtractor.addExtractor( # GPU spec cuDNN
        name     = 'gpu',
        pattern  = r'\^\^\^\^\s+CSV\s+(\d+),(\d+),(.+),(\d+)',
        fields   = "sm cap clock mem",
        types    = ints_str(4))

# Test Flags (only works on output parsed from cudnn_run.py log)
LogExtractor.addExtractor(
        name     = 'test_flags',
        pattern  = r'Test Flags: (.*)',
        fields   = "test_flags",
        types    = "TEXT")

# Layer Name (only works on output parsed from cudnn_run.py log)
LogExtractor.addExtractor(
        name     = 'layer_name',
        pattern  = r'Layer Name: (.*)',
        fields   = "layer_name",
        types    = "TEXT")

# Unique Flags (only works on output parsed from cudnn_run.py log)
LogExtractor.addExtractor(
        name     = 'unique_flags',
        pattern  = r'Unique Flags: (.*)',
        fields   = "unique_flags",
        types    = "TEXT")

LogExtractor.addExtractor( # Input descriptor
        name     = 'srcDesc',
        pattern  = r'(?:ImageTensor \(input\)|DGradTensor \(output\)|DGradTensor \(input\))'
        r'.*?sizes=\[n=(\d+),c=(\d+),(\d+),(\d+)(?:,(\d+))?\] '
        r'.*?strides=\[(\d+),(\d+),(\d+),(\d+)(?:,(\d+))?\] vect=(\d) packed=(\d) type=(HALF|FLOAT|DOUBLE)',
        fields   = "dimA_0 dimA_1 dimA_2 dimA_3 dimA_4 strideA_0 strideA_1 strideA_2 strideA_3 strideA_4 vect packed dataType",
        types    = ints_str(12) + " TEXT")

LogExtractor.addExtractor( # filter descriptor
        name     = 'filterDesc',
        pattern  = r'(?:FilterTensor \(input\)|WGradTensor \(output\)).*?sizes=\[k=(\d+),c=(\d+),(\d+),(\d+)(?:,(\d+))?\].*?format=(\d+)',
        fields   = "dimA_0 dimA_1 dimA_2 dimA_3 dimA_4 format",
        types    = ints_str(6))

LogExtractor.addExtractor( # output descriptor
        name     = 'destDesc',
        pattern  = r'(?:RespTensor \(output\)|DGradTensor \(output\)|DiffTensor \(input\)).*?sizes=\[n=(\d+),c=(\d+),(\d+),(\d+)(?:,(\d+))?\] '
        '.*?strides=\[(\d+),(\d+),(\d+),(\d+)(?:,(\d+))?\] vect=(\d) packed=(\d) type=(HALF|FLOAT|DOUBLE)',
        fields   = "dimA_0 dimA_1 dimA_2 dimA_3 dimA_4 strideA_0 strideA_1 strideA_2 strideA_3 strideA_4 vect packed dataType",
        types    = ints_str(12) + " TEXT")

LogExtractor.addExtractor( # Convolution desciptor
        name     = 'convDesc',
        pattern  = r'args: Conv : pad=\[(\d+),(\d+)(?:,(\d+))?\] ?strides=\[(\d+),(\d+)(?:,(\d+))?\]'
        r' dilation=\[(\d+),(\d+)(?:,(\d+))?\]'
        r' mode=(CONV|CORR) math=(DEFAULT|TENSOR_OP)'
        r' groupCount=(\d+) dataType=(HALF|FLOAT|DOUBLE)',
        fields   = "padA_0 padA_1 padA_2 strideA_0 strideA_1 strideA_2 dilationA_0 dilationA_1 dilationA_2 mode math groupCount dataType",
        types    = ints_str(9) + " TEXT TEXT INTEGER TEXT")

LogExtractor.addExtractor( # algorithm
        name     = 'algo',
        pattern  = r'Algo (?:according|passed by) (preference|user): (\d+)',
        fields   = "chosen_by choice",
        types    = "TEXT INTEGER")

# ^^^^ Launch time: elapsed = 9.05991e-06 sec
LogExtractor.addExtractor(  # CPU launch time in seconds
    name     = "cpu_time",
    pattern  = r"\^\^\^\^ Launch time: elapsed = {FLOAT_GROUP} sec",
    fields   = "cpu_time_seconds",
    types    = "REAL")

# ^^^^ primary grid info: CudaGraphNodeKernel(sm80_xmma_wgrad_implicit_gemm_indexed_f16f16_f16f32_f32_nhwckrsc_nhwc_tilesize32x256x32_stage3_warpsize1x4x1_g1_tensor16x8x16_execute_kernel_cudnn_train<<<(7,1,39),(128,1,1),0,55296>>>())
LogExtractor.addExtractor( # CPU launch time in seconds
    name     = "primary_kernel",
    pattern  = r"^\^\^\^\^ primary grid info: CudaGraphNodeKernel\((.*?)\(\)\)$",
    fields   = "primary_kernel",
    types    = "TEXT")

# ^^^^ Workspace size = 51597488 Bytes (49 MB)
# ^^^^ Workspace size: 1985680 (1 MB)
LogExtractor.addExtractor(  # Total number of elements input and output tensor
    name     = "workspace_size_mb",
    pattern  = r"\^\^\^\^ Workspace [sS]ize\s*:\s*\d+\s+\(\s+{UINT_GROUP}\s*MB\)",
    fields   = "workspace_size_mb",
    types    = "INTEGER")

LogExtractor.addExtractor( # Total number of elements input and output tensor
        name     = 'workspace_size',
        pattern  = r'Workspace Size: {UINT_GROUP} \(.*?MB\)',
        fields   = 'workspace_size',
        types    = "INTEGER")

LogExtractor.addExtractor( # Heurgen query
        name     = 'query',
        pattern  = r'BackendQuery=(.*)',
        fields   = "flags_str",
        types    = "TEXT")

LogExtractor.addExtractor(  # Heuristics engine config print
    name     = "heur_list",
    pattern  = r"^\^\^\^\^ Heuristics List\[(\d+)\]: -backendEngine(\d+), -knobStr([\d,-]*)$",
    fields   = "idx engine knob",
    types    = "INTEGER INTEGER TEXT")

LogExtractor.addExtractor( # Heuristics engine config print
    name     = "heur_list_with_nnTM",
    pattern  = r"^\^\^\^\^ Heuristics List\[(\d+)\]: -backendEngine(\d+), -knobStr([\d,-]*) -predicted_time: (.*)$",
    fields   = "idx engine knob nnTM_predicted_time",
    types    = "INTEGER INTEGER TEXT TEXT")

LogExtractor.addExtractor( # Heuristics engine config print
    name     = "heur_engine_config",
    pattern  = r"\^\^\^\^ Heuristics EngineConfig: -backendEngine(\d+), -knobStr(.*)",
    fields   = "heur_engine heur_knob",
    types    = "INTEGER TEXT")

LogExtractor.addExtractor( # Heuristics engine config print
    name     = "predicted_time",
    pattern  = r"\^\^\^\^ Cost model predicted time = (.*) sec, (.*) sec",
    fields   = "nnTM_time KTM_time",
    types    = "TEXT TEXT")

LogExtractor.addExtractor( # graphRunner predicted time
    name     = "graphRunner_predicted_time_msec",
    pattern  = r"\^\^\^\^ Engine config predicted time = (.*) msec\.",
    fields   = "graphRunner_predicted_time_msec",
    types    = "TEXT")

LogExtractor.addExtractor( # cuda graph print
    name     = "graph",
    pattern  = r"\^\^\^\^[\w\s]+grid info: CudaGraphInfo\((.*)\)",
    fields   = "graph",
    types    = "TEXT")

LogExtractor.addExtractor( # heuristic gen tuner
        name     = 'tune',
        pattern  = r'Heurgen Information: \((\d+)\),\((\d+)\),\((\d+)\),\((.*?)\),\((.*?)\)',
        fields   = "gpu_id opset_id engine_id opset_params knob_choices",
        types    = "INTEGER INTEGER INTEGER TEXT TEXT")

LogExtractor.addExtractor( # median CUDA elapsed time in milliseconds
        name     = 'median_time',
        pattern  = r'CUDA elapsed median = {FLOAT_GROUP} msec',
        fields   = "median_time_msec",
        types    = "REAL")

LogExtractor.addExtractor( # SOL measured GFlOPS
        name     = "sol_gflops",
        pattern  = r"measured Gflops = {FLOAT_GROUP} ",
        fields   = "sol_gflops",
        types    = "REAL")

LogExtractor.addExtractor( # SOL measured memory bandwidth
        name     = 'mem_per_sec',
        pattern  = r'measured GiB/s = {FLOAT_GROUP}',
        fields   = 'sol_mem_per_sec',
        types    = "REAL")

# Extractors are added in order:
LogExtractor.addExtractor( # elapsed time in seconds
        name     = 'time',
        pattern  = r'\^\^\^\^ CUDA : elapsed = {FLOAT_GROUP} sec',
        fields   = "seconds",
        types    = "REAL")

LogExtractor.addExtractor( # elapsed time in seconds
    name     = "heuristics_overhead_time",
    pattern  = r"\^\^\^\^ Heuristic query overhead = {FLOAT_GROUP} sec, Tendor and graph creation overhead = {FLOAT_GROUP} sec, Plan build overheads = \[{FLOAT_GROUP} sec, {FLOAT_GROUP} sec, {FLOAT_GROUP} sec, (\d+)\]",
    fields   = "query_time, tensor_graph_creation_overhead, build_plan_time_top1, build_plan_time_top10, build_plan_time_all, build_plan_valid_engine_config_count",
    types    = "REAL REAL REAL REAL REAL INTEGER")

LogExtractor.addExtractor( # maximum absolute error seen
        name     = 'max_abs_err',
        pattern  = r'^^^^ max_rel_seen={FLOAT_GROUP} rel_threshold={FLOAT_GROUP} max_abs_seen={FLOAT_GROUP}, dead_region={FLOAT_GROUP}',
        fields   = "RelErr RelThreshold AbsErr DeadRegion",
        types    = reals_str(4))

LogExtractor.addExtractor( # passed-result
        name     = 'passed',
        pattern  = r'(PASSED)',
        fields   = "result",
        types    = "TEXT")

LogExtractor.addExtractor( # cuDNN error message
        name     = 'err_msg',
        pattern  = r'First error msg      : (.*)',
        fields   = "err_msg",
        types    = "TEXT")

LogExtractor.addExtractor( # cuDNN test status
        name     = 'status',
        pattern  = r'\$\$\$\$ Test on line (.*) returned status (.*)',
        fields   = "line status",
        types    = "INTEGER TEXT")

LogExtractor.addExtractor( # cuDNN test status
        name     = 'stable_times',
        pattern  = r'Most Stable Times: \[(.*?)\]',
        fields   = "times",
        types    = "TEXT")

LogExtractor.addExtractor( # cuDNN test status
        name     = 'stable_percent',
        pattern  = r'Most Stable Percent: {FLOAT_GROUP}',
        fields   = "percent",
        types    = "REAL")

LogExtractor.addExtractor( # cuDNN test status
        name     = 'conditional_stable_percent',
        pattern  = r'WARNING: Only {FLOAT_GROUP}\% stable',
        fields   = "percent",
        types    = "REAL")

LogExtractor.addExtractor( # cuDNN test status
        name     = 'execution_times',
        pattern  = r'All Execution Times: \[(.*?)\]',
        fields   = "times",
        types    = "TEXT")

LogExtractor.addExtractor( # cuDNN test status
        name     = 'launch_times',
        pattern  = r'All Launch Times: \[(.*?)\]',
        fields   = "times",
        types    = "TEXT")

LogExtractor.addExtractor(  # graph compilation time
    name     = "executionPlan_making_time",
    pattern  = r"\^\^\^\^ ExecutionPlan making time = {FLOAT_GROUP} msec.",
    fields   = "executionPlan_making_time_msec",
    types    = "REAL")

LogExtractor.addExtractor(  # elapsed time in seconds
    name     = "graph_median_time",
    pattern  = r"\^\^\^\^ Graph median CUDA time = {FLOAT_GROUP} msec",
    fields   = "graph_median_time_msec",
    types    = "REAL")

LogExtractor.addExtractor(  # elapsed time in seconds
    name     = "graphRunner_compilation_time",
    pattern  = r"\!\!\!\! VarPack setup CPU time 5 = {FLOAT_GROUP} usec\.",
    fields   = "graphRunner_compilation_time_usec",
    types    = "REAL")

LogExtractor.addExtractor(  # cuDNN test status
    name     = "graphRunner_engine_config",
    pattern  = r"\#\#\#\# Running config: {{\"backendEngine\":\"(.*)\", \"knob\":\"(.*)\"}}",
    fields   = "engine_id knob_choices",
    types    = "INTEGER TEXT")

LogExtractor.addExtractor(  # cuDNN test status
    name     = "graphRunner_engine_config_test_flag_str",
    pattern  = r"\#\#\#\# To reproduce, append \".*(\-backendEngine.*)\" at the end\.",
    fields   = "graphRunner_engine_config_test_flag_str",
    types    = "TEXT")

LogExtractor.addExtractor( # cuDNN test status
        name     = 'runtime_error',
        pattern  = r'(runtime_error)',
        fields   = "runtime_error",
        types    = "TEXT")

LogExtractor.addExtractor( # cuDNN test status
        name     = 'error',
        pattern  = r'ERROR: (.*?)$',
        fields   = "error",
        types    = "TEXT")

LogExtractor.addExtractor( # cuDNN test status
        name     = 'time_stats',
        pattern  = r'Time Stats: {FLOAT_GROUP} \| {FLOAT_GROUP} \| {FLOAT_GROUP} \| {FLOAT_GROUP}',
        fields   = "total total_execution total_launch sample_count",
        types    = "REAL REAL REAL REAL")

LogExtractor.addExtractor( # cuDNN test status
    name     = "CASK_shader",
    pattern  = r"Choice\(engine_id=(\d+),choices=\[(.*) Shader name: (.*)",
    fields   = "engine_id knob_full_str shader_name",
    types    = "INTEGER TEXT TEXT")

#*******************************************************************************
#* Unsectioned Stuff
#*******************************************************************************
LogExtractor.addExtractor(  # CUDNN test UID
    name     = "test_UID",
    pattern  = r"@@@@ Test UID             : {UID_GROUP}",
    fields   = "test_UID",
    types    = "TEXT")


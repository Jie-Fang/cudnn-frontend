import re 
from helpers.utility import get_default_src

DEFAULT_BACKEND_H           = get_default_src() + "/../include/cudnn_graph.h"
DEFAULT_BACKENDAPIUTILS_CPP = get_default_src() + "/../test/backendApiUtils.cpp"

class KnobMap:
    def __init__(self, file_name=None):
        if file_name == None:
            file_name = DEFAULT_BACKEND_H
            
        with open(file_name, 'r') as in_file:
            content = in_file.read()

        enum_pat = re.compile("typedef\s+enum\s+{(.*?)}\s+cudnnBackendKnobType_t\s*;", re.DOTALL | re.MULTILINE)

        enum_matches = enum_pat.findall(content)

        if len(enum_matches) != 1:
            raise Exception("[KnobMap] Incorrect number of pattern matches: %d" % len(enum_matches))

        enum_content = enum_matches[0]

        key_val_pat = re.compile('(CUDNN_[A-Z_0-9]+)\s*?=\s*?([\-0-9]+)')

        key_vals = key_val_pat.findall(enum_content)

        if len(key_vals) == 0:
            raise Exception("Found no keys...")

        self.name_to_idx_map = {}
        self.idx_to_name_map = {}

        for key_val in key_vals:
            self.name_to_idx_map[key_val[0]] = key_val[1]
            self.idx_to_name_map[key_val[1]] = key_val[0]

    def get_knob_name(self, knob_idx):
        knob_idx = str(knob_idx)

        if knob_idx not in self.idx_to_name_map:
            raise Exception("[KnobMap] Unable to find knob idx: %s" % knob_idx)

        return self.idx_to_name_map[knob_idx]

    def get_knob_idx(self, knob_name):
        if knob_name not in self.name_to_idx_map:
            raise Exception("[KnobMap] Unable to find knob name: %s" % knob_name)

        return self.name_to_idx_map[knob_name]

class KnobFlagMap:
    def __init__(self, file_name=None):
        if file_name == None:
            file_name = DEFAULT_BACKENDAPIUTILS_CPP
            
        with open(file_name, 'r') as in_file:
            content = in_file.read()

        enum_pat = re.compile('getCudnnTestOpts\(cudnnBackendKnobType_t\s+\w+\)\s+{(.*?)}\s+return "";\n}', re.DOTALL | re.MULTILINE)

        enum_matches = enum_pat.findall(content)

        if len(enum_matches) != 1:
            raise Exception("[KnobFlagMap] Incorrect number of pattern matches: %d" % len(enum_matches))

        enum_content = enum_matches[0]
        
        key_val_pat = re.compile('case (\w+):.*?return "(.*?)";', re.DOTALL | re.MULTILINE)

        key_vals = key_val_pat.findall(enum_content)

        if len(key_vals) == 0:
            raise Exception("Found no keys...")

        self.enum_name_to_flag_name = {}
        self.flag_name_to_enum_name = {}

        for key_val in key_vals:
            self.enum_name_to_flag_name[key_val[0]] = key_val[1]
            self.flag_name_to_enum_name[key_val[1]] = key_val[0]

    def get_flag_name(self, enum_name):
        enum_name = str(enum_name)

        if enum_name not in self.enum_name_to_flag_name:
            raise Exception("[KnobFlagMap] Unable to find enum name: %s" % enum_name)

        return self.enum_name_to_flag_name[enum_name]

    def get_knob_name(self, flag_name):
        if flag_name not in self.flag_name_to_enum_name_map:
            raise Exception("Unable to find flag name: %s" % flag_name)

        return self.flag_name_to_enum_name_map[flag_name]

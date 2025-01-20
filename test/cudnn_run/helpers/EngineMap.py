import re 
from helpers.utility import get_default_src

DEFAULT_BACKEND_ENGINES_H = get_default_src() + "/cnn_infer/include/backend_engines.h"

class EngineMap:
    def __init__(self, file_name=None):
        if file_name == None:
            file_name = DEFAULT_BACKEND_ENGINES_H
            
        with open(file_name, 'r') as in_file:
            content = in_file.read()

        enum_pat = re.compile("typedef\s+enum\s+{(.*?)}\s+cudnnBackendEngineName_t\s*;", re.DOTALL | re.MULTILINE)

        enum_matches = enum_pat.findall(content)

        if len(enum_matches) != 1:
            raise Exception("[EngineMap] Incorrect number of pattern matches: %d" % len(enum_matches))

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

    def get_engine_name(self, engine_idx):
        engine_idx = str(engine_idx)

        if engine_idx not in self.idx_to_name_map:
            raise Exception("[EngineMap] Unable to find engine idx: %s" % engine_idx)

        return self.idx_to_name_map[engine_idx]

    def get_engine_idx(self, engine_name):
        if engine_name not in self.name_to_idx_map:
            raise Exception("[EngineMap] Unable to find engine name: %s" % engine_name)

        return self.name_to_idx_map[engine_name]

    def get_all_engine_indices(self):
        return self.idx_to_name_map.keys()
        
    def get_all_engine_names(self):
        return self.name_to_idx_map.keys()

from helpers.EngineMap import EngineMap
from helpers.KnobMap import KnobMap, KnobFlagMap
from helpers.utility import flags_from_descs_str
from collections import namedtuple
import re

class HeuristicsParser:
    def __init__(self):
        self.engine_map    = EngineMap()
        self.knob_map      = KnobMap()
        self.knob_flag_map = KnobFlagMap()
    
        self.ENGINE_CONFIG   = namedtuple("ENGINE_CONFIG", "engine_idx knobs_str knobs_dict")
        self.CASE            = namedtuple("CASE", "case_idx test_flags opset_params eng_cfgs")
        self.case_pat   = re.compile('.*?(\d+).*?{"(.*?){(.*?)} -> \[(.*?)\]"}.*')
        self.engcfg_pat = re.compile('eng(\d+)\+knobs\((.*?)\)')
        
        self.CASE_LIST = namedtuple("CASE_LIST", "case_list_idx case_list")
        self.case_list_pat   = re.compile('  /\* (\d+) \*/ (.*?)$')
        
        self.KNOB_CHOICE_DATA = namedtuple("KNOB_CHOICE_DATA", "knob_str knob_choices knob_dict knob_flag_str knob_idx")
        self.kcd_pat = re.compile("\s*([-0-9,]+)\s*//\s*(\d+)$")
        
        self.CFG = namedtuple("CFG", "engine_idx kcd_idx kcd knob_count")
        self.cfg_pat = re.compile("engcfg_t\((\d+),(NULL|\d+),(\d+)\),")
        
        self.NODE = namedtuple("NODE", "node_idx cond_idx cond_value lower_node_idx higher_node_idx lower_leaf_idx higher_leaf_idx")
        self.node_pat = re.compile("  /\* (\d+) \*/ {(\d+)\s*,\s*(\d+)\s*,\s*(LEAF\s*\|)?\s*(\d+)\s*,\s*(LEAF\s*\|)?\s*(\d+)},")
        
        self.LEAF_DATA = namedtuple("LEAF_DATA", "engcfg_idxs")
        self.ld_pat = re.compile("\s*([0-9,]+)\s*")
        
        self.LEAF = namedtuple("LEAF", "leaf_idx leaf_count")
        self.leaf_pat = re.compile("\s*\{(\d+).*?(\d+)\}.*?")
        
        self.LEAF_CASE = namedtuple("LEAF_CASE", "case_idx")
        self.leaf_case_pat = re.compile(".*?(\d+).*?")

        self.DECISION_TREE = namedtuple('DECISION_TREE', 'cases case_list kcd cfg nodes ld leafs leaf_cases')

    #*******************************************************************************
    #* Parsing of "cases"
    #*******************************************************************************
    def process_case(self, line):
        case_idx, test_flags_str_label_form, opset_params_str, eng_cfgs_str = self.case_pat.match(line).groups()
        
        test_flags = flags_from_descs_str(test_flags_str_label_form)
        
        opset_params = [int(s) for s in opset_params_str.split(',')]
        
        eng_cfgs = []
        for eng_idx, knob_str in self.engcfg_pat.findall(eng_cfgs_str):
            knob_dict = {}
            
            knob_flags_str = ""
            if knob_str != "":
                knob_choices = knob_str.split(',')
                
                for knob_choice_idx in range(0, len(knob_choices), 2):
                    knob_name = self.knob_map.get_knob_name(knob_choices[knob_choice_idx])
                    
                    knob_value = knob_choices[knob_choice_idx+1]
                    
                    knob_flag_name = self.knob_flag_map.get_flag_name(knob_name)
                    
                    knob_flags_str += "%s=%s " % (knob_flag_name, knob_value)
                    
                    knob_dict[knob_name] = knob_value
            
                knob_flags_str = knob_flags_str[:-1]
                
            eng_cfgs.append(self.ENGINE_CONFIG(int(eng_idx), knob_flags_str, knob_dict))
            
        return self.CASE(int(case_idx), test_flags, opset_params, eng_cfgs)
        
    
    #*******************************************************************************
    #* Parsing of "case_lists"
    #*******************************************************************************
    def process_case_list(self, line):
        case_list_idx, case_list_str = self.case_list_pat.match(line).groups()
        
        case_lists = [int(s) for s in case_list_str.split(',') if s != ''][:-1]
        
        return self.CASE_LIST(int(case_list_idx), case_lists)
            
    #*******************************************************************************
    #* Parsing of "kcd"
    #*******************************************************************************
    def process_knob_choice_data(self, line):
        knob_str, knob_idx = self.kcd_pat.match(line).groups()
        
        knob_choices = [int(s) for s in knob_str.split(',') if s != '']
       
        knob_flags_str = ""
        knob_dict = {}
        
        for knob_choice_idx in range(0, len(knob_choices), 2):
            knob_name = self.knob_map.get_knob_name(knob_choices[knob_choice_idx])
            
            knob_value = knob_choices[knob_choice_idx+1]
            
            knob_flag_name = self.knob_flag_map.get_flag_name(knob_name)
            
            knob_flags_str += "%s=%s " % (knob_flag_name, knob_value)
            
            knob_dict[knob_name] = knob_value
    
        knob_flags_str = knob_flags_str[:-1]
        
        return self.KNOB_CHOICE_DATA(knob_str, knob_choices, knob_dict, knob_flags_str, int(knob_idx))
        
    #*******************************************************************************
    #* Parsing of "cfg"
    #*******************************************************************************
    def process_cfg(self, line):
        line = line.replace(' ', '')
        line = line.replace('kcd+', '')
        engine_idx, kcd_idx, knob_count = self.cfg_pat.match(line).groups()
        
        kcd_idx = int(kcd_idx) if kcd_idx != "NULL" else None
        
        return self.CFG(int(engine_idx), kcd_idx, None, int(knob_count))
        
    #*******************************************************************************
    #* Parsing of "nodes"
    #*******************************************************************************
    def process_node(self, line):
        node_match = self.node_pat.match(line)
        
        if node_match == None:
            return None
            
        node_idx, cond_idx, cond_value, left_leaf, left_idx, right_leaf, right_idx = node_match.groups()
        
        lower_node_idx = None
        higher_node_idx = None
        lower_leaf_idx = None
        higher_leaf_idx = None
        
        if left_leaf == None:
            lower_node_idx = int(left_idx)
        else:
            lower_leaf_idx = int(left_idx)
            
        if right_leaf == None:
            higher_node_idx = int(right_idx)
        else:    
            higher_leaf_idx = int(right_idx)
        
        return self.NODE(int(node_idx), int(cond_idx), int(cond_value), lower_node_idx, higher_node_idx, lower_leaf_idx, higher_leaf_idx)
    
    #*******************************************************************************
    #* Parsing of "ld" (leaf data)
    #*******************************************************************************
    def process_ld(self, line):
        line = line.replace(' ', '')
        line = line.replace('cfg+', '')
    
        ld_str = self.ld_pat.match(line).groups()[0]
        engcfg_idxs = [int(s) for s in ld_str.split(',') if s != '']
    
        return self.LEAF_DATA(engcfg_idxs)
    
    #*******************************************************************************
    #* Parsing of "leafs"
    #*******************************************************************************
    def process_leaf(self, line):
        leaf_idx, leaf_count = self.leaf_pat.match(line).groups()
    
        return self.LEAF(int(leaf_idx), int(leaf_count))
    
    #*******************************************************************************
    #* Parsing of "leaf_cases"
    #*******************************************************************************
    def process_leaf_case(self, line):
        case_idx = self.leaf_case_pat.match(line).groups()[0]
    
        return self.LEAF_CASE(int(case_idx))
    
    #*******************************************************************************
    #* Parse and process heuristic file
    #*******************************************************************************
    
    def parse_file(self, filename):
        STAGES = ["cases", "case_lists", "kcd", "cfg", "nodes", "ld", "leafs", "leaf_cases"]
        process_stages = [
                          self.process_case,
                          self.process_case_list,
                          self.process_knob_choice_data,
                          self.process_cfg,
                          self.process_node,
                          self.process_ld,
                          self.process_leaf,
                          self.process_leaf_case
                         ]
        STAGE_PATS = [re.compile("static\s+const.*%s.*" % stage) for stage in STAGES]
        STAGE_TUPS = {}
        for stage in STAGES:
            STAGE_TUPS[stage] = []
    
        empty_line_pat = re.compile("\s*$")
        end_stage_pat = re.compile("\s*};\s*")
        
        cur_stage_idx = -1
        cur_stage_state = None
       
        with open(filename) as f:
            for line in f:
                if empty_line_pat.match(line):
                    continue
                    
                if (cur_stage_idx + 1 < len(STAGES)) and STAGE_PATS[cur_stage_idx + 1].match(line):
                    cur_stage_idx += 1
                    cur_stage_state = "OPEN"
                    
                elif (cur_stage_state == "OPEN") and (end_stage_pat.match(line)):
                    cur_stage_state = "CLOSED"
                    
                elif cur_stage_state == "OPEN":
                    if cur_stage_idx >= len(process_stages):
                        print("Unknown stage: ")
                        print((STAGES[cur_stage_idx], cur_stage_state, line))
    
                    info = process_stages[cur_stage_idx](line)
    
                    if info:
                        STAGE_TUPS[STAGES[cur_stage_idx]].append(info)
    
        cases = STAGE_TUPS["cases"]
        case_list = STAGE_TUPS["case_lists"]
        kcd = STAGE_TUPS["kcd"]
        cfg = STAGE_TUPS["cfg"]
        nodes = STAGE_TUPS["nodes"]
        ld = STAGE_TUPS["ld"]
        leafs = STAGE_TUPS["leafs"]
        leaf_cases = STAGE_TUPS["leaf_cases"]
        
        data = self.DECISION_TREE(cases, case_list, kcd, cfg, nodes, ld, leafs, leaf_cases)
    
        # associate knob data with each engine config
        kcd_idx_map = {}
        for i, knob in enumerate(kcd):
            kcd_idx_map[knob.knob_idx] = i
        for i, c in enumerate(cfg):
            if c.kcd_idx:
                cfg[i] = self.CFG(c.engine_idx, c.kcd_idx, kcd[kcd_idx_map[c.kcd_idx]], c.knob_count)
        del(kcd_idx_map)
    
        # leaf data sanity checks
        if len(ld) != len(leafs):
            print("ld (%d) and leaf cases (%d) not equal in length. Exiting." % (len(ld), len(leafs)))
            return
        if len(ld) != len(leaf_cases):
            print("ld (%d) and leaf cases (%d) not equal in length. Exiting." % (len(ld), len(leaf_cases)))
            return
        if len(leaf_cases) != len(case_list):
            print("Leaf cases (%d) and case_list (%d) not equal in length. Exiting." % (len(leaf_cases), len(case_list)))
            return

        for i, node in enumerate(nodes):
            if i != node.node_idx:
                print(nodes[i-1])
                print(i, node.node_idx)
                print(nodes[i+1])
                break
    
        return data

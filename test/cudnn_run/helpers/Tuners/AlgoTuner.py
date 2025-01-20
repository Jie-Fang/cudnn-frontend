# AlgoTuner is responsible for heuristics based on algo chosen
# 
from DecisionTreeGen.DecisionTreeGenerator import DecisionTreeGenerator
from collections import OrderedDict, defaultdict
from helpers.cudnn_interface import OutputParser
from HeurgenTuner  import HeurgenTuner
import re
import operator    
import itertools
from collections import deque
from copy import deepcopy

# Removes backdoor from string of flags
def scrub_algo(flags):
    resized_flags = [flag[:5] for flag in flags]

    algo_idx = resized_flags.index('-algo')

    if algo_idx == -1:
        raise Exception("[ALGO TUNING] Unable to find -algo in %s" % str(flags))

    return flags[:algo_idx] + flags[algo_idx+1:]

def get_default(val, default):
    if(val == '' or val == None):
        return default

    return val

def get_param_vals(run):
    case = {}

    case["n"] = int(run.parsed.srcDesc.dimA_0)
    case["c"] = int(run.parsed.srcDesc.dimA_1)
    case["h"] = int(run.parsed.srcDesc.dimA_2)
    case["w"] = int(run.parsed.srcDesc.dimA_3)
    case["d"] = int(get_default(run.parsed.srcDesc.dimA_4, '1'))

    case["k"] = int(run.parsed.filterDesc.dimA_0)
    case["r"] = int(run.parsed.filterDesc.dimA_2)
    case["s"] = int(run.parsed.filterDesc.dimA_3)
    case["t"] = int(get_default(run.parsed.filterDesc.dimA_4, '1'))

    case["u"] = int(run.parsed.convDesc.strideA_0)
    case["v"] = int(run.parsed.convDesc.strideA_1)
    case["x"] = int(get_default(run.parsed.convDesc.strideA_2, '1'))

    case["pad_h"] = int(run.parsed.convDesc.padA_0)
    case["pad_w"] = int(run.parsed.convDesc.padA_1)
    case["pad_d"] = int(get_default(run.parsed.convDesc.padA_2, '0'))

    case["dil_h"] = int(run.parsed.convDesc.dilationA_0)
    case["dil_w"] = int(run.parsed.convDesc.dilationA_1)
    case["dil_d"] = int(get_default(run.parsed.convDesc.dilationA_2, '1'))
    
    return case

def is_tensor_NHWC_c_packed(tensor):
    if (int(tensor.strideA_1) != 1): 
        return 0
    if (int(tensor.strideA_3) < int(tensor.dimA_1) * int(tensor.strideA_1)): 
        return 0
    if (int(tensor.strideA_2) < int(tensor.dimA_3) * int(tensor.strideA_3)):        
        return 0
    if (int(tensor.strideA_0) < int(tensor.dimA_2) * int(tensor.strideA_2)):
        return 0
    return 1

def get_int_type(type_str):
    types = ["FLOAT", "DOUBLE", "HALF", "INT8", "INT32", "INT8x4", "UINT8", "UINT8x4", "UINT8x32"]

    if type_str not in types:
        raise Exception("Type not found: %s" % type_str)
        
    return types.index(type_str)
    
def get_int_type2(type_str):
    types = ["s", "d", "h"]

    if type_str not in types:
        raise Exception("Type not found: %s" % type_str)
        
    return types.index(type_str)
    
def get_int_mode(mode_str):
    modes = ["CONV", "CORR"]

    if mode_str not in modes:
        raise Exception("Mode not found: %s" % type_str)
        
    return modes.index(mode_str)

def get_flag(flag_str, flag_name, fail_on_missing=True, missing_result=None):
    flag_name = "-" + flag_name
    if flag_name not in flag_str:
        if fail_on_missing:
            raise Exception("[AlgoTuner] Unable to find %s in %s" % (flag_name, flag_str))
        else:
            return missing_result
            
    start_index = flag_str.find(flag_name) + len(flag_name)
    end_index   = flag_str.find(' ', start_index)

    return flag_str[start_index:end_index]
    
def get_chooser_vals(output, flags):
    choosers_dict = OrderedDict()
    output_parser = OutputParser(output)

    dimA = get_flag(flags, 'dimA').split(',')
    
    if len(dimA) == 4:
        dimA.append('1')
        
    filtA = get_flag(flags, 'filtA').split(',')
       
    if len(filtA) == 4:
        filtA.append('1')
        
    n, c, h, w, d = [int(dim) for dim in dimA]
    
    
    k, _, r, s, t = [int(dim) for dim in filtA]
    
    if '-n' in flags:
        n = int(get_flag(flags, 'n'))
    
    # extract all input chooser values from output_parser
    choosers_dict['n'] = n #
    choosers_dict['c'] = c #int(output_parser["srcDesc"].dimA_1)
    choosers_dict['h'] = h #int(output_parser["srcDesc"].dimA_2)
    choosers_dict['w'] = w #int(output_parser["srcDesc"].dimA_3)
    choosers_dict['d'] = d #int(get_default(output_parser["srcDesc"].dimA_4, '1'))
        
    choosers_dict['k'] = k #int(output_parser["filterDesc"].dimA_0)
    choosers_dict['r'] = r #int(output_parser["filterDesc"].dimA_2)
    choosers_dict['s'] = s #int(output_parser["filterDesc"].dimA_3)
    choosers_dict["t"] = t #int(get_default(output_parser["filterDesc"].dimA_4, '1'))
    
    choosers_dict["u"] = int(output_parser["convDesc"].strideA_0)
    choosers_dict["v"] = int(output_parser["convDesc"].strideA_1)
    choosers_dict["x"] = int(get_default(output_parser["convDesc"].strideA_2, '1'))

    choosers_dict["pad_h"] = int(output_parser["convDesc"].padA_0)
    choosers_dict["pad_w"] = int(output_parser["convDesc"].padA_1)
    choosers_dict["pad_d"] = int(get_default(output_parser["convDesc"].padA_2, '0'))

    choosers_dict["dil_h"] = int(output_parser["convDesc"].dilationA_0)
    choosers_dict["dil_w"] = int(output_parser["convDesc"].dilationA_1)
    choosers_dict["dil_d"] = int(get_default(output_parser["convDesc"].dilationA_2, '1'))

    choosers_dict["in_format"] = int(get_flag(flags, 'formatIn', False, '0'))
    choosers_dict["filter_format"] = int(get_flag(flags, 'filtFormat', False, '0'))
    choosers_dict["out_format"] = int(get_flag(flags, 'formatOut', False, '0'))
    
    choosers_dict['in_type'] = get_int_type2(get_flag(flags, 'Pin'))
    choosers_dict['comp_type'] = get_int_type2(get_flag(flags, 'Pcomp'))
    choosers_dict['out_type'] = get_int_type2(get_flag(flags, 'Pout'))
    
    choosers_dict['groupCount'] = int(output_parser["convGroup"].groupCount)
    
    choosers_dict['conv_mode'] = get_int_mode(output_parser["convMode"].mode)

    return choosers_dict
    
def map_choice(choice):
    return (str(int(choice) / 10), str(int(choice) % 2))

def permutations(possibles):
    to_explore = deque()
    
    to_explore.append([])
    
    for explore_rank in range(len(possibles)):
        
        new_explore = deque()
        
        while len(to_explore) > 0:
            cur_explore = to_explore.pop()
            
            for poss_rank in possibles[explore_rank]:
                
                if poss_rank not in cur_explore:
                    new_explore.append( cur_explore + [poss_rank] )
        
        to_explore = new_explore
    
    return [tuple(val) for val in to_explore]
        
def prune_cases(cases):
    equivalences = set()
    
    choice_counts = {}
    
    for case_idx in range(len(cases)):
        
        case = cases[case_idx]
        if case['choice'] in choice_counts:
            choice_counts[case['choice']] += 1
        else:
            choice_counts[case['choice']] = 1
            
    cases = deepcopy(cases)
            
    for case_idx, case in enumerate(cases):
        possibles = []
        
        INF_VAL = 1e-7
        
        best_choice = {}
        for rank_a, choice_a in enumerate(case['choice']):
            mapped_a = map_choice(choice_a)
            
            time_a = case['choice_times'][mapped_a]
            
            if time_a == "inf":
                continue
                
            # Any mathMode can replace '1'
            if '1' not in best_choice:
                best_choice['1'] = time_a
                
            # Ignore mathMode '1' for later setting of '0'
            if mapped_a[1] == '1':
                continue
                
            # Set '0' if it hasn't been set; we know our current case is mathMode '0'
            if '0' not in best_choice:
                best_choice['0'] = time_a
                
            
        for rank_a, choice_a in enumerate(case['choice']):
            
            mapped_a = map_choice(choice_a)
            
            time_a = case['choice_times'][mapped_a]

            eff_time_a = INF_VAL if time_a == "inf" else time_a

            # Allow up to 2.5% per rank
            allowed_diff = 1.0 + ((rank_a+1) * (.01 * 2.5))

            best_time = best_choice[mapped_a[1]]
            
            if best_time != None:
                # Get diff, 2.00 for 2x if best_time is twice as fast
                best_rel = eff_time_a / best_time
                
                # Get half of perf loss; 2.00 becomes (2.0 - 1.0) * 0.50 = 0.50
                best_buffer = (best_rel - 1.0) * 0.50
                
                # Allow max of either 2.5% per rank or 50% of current perf loss
                allowed_diff = max(allowed_diff, 1.0 + best_buffer)

            rank_a_possibles = []
            
            for rank_b, choice_b in enumerate(case['choice']):
                
                # We know it is faster if rank is same or smaller
                if rank_b <= rank_a:
                    rank_a_possibles.append(choice_b)
                    continue
                    
                time_b = case['choice_times'][map_choice(choice_b)]
                    
                eff_time_b = INF_VAL if time_b == "inf" else time_b
                
                if (eff_time_b / eff_time_a) < (allowed_diff):
                    rank_a_possibles.append(choice_b)
                    
            possibles.append(rank_a_possibles)
        
        
        best_permutation_count = choice_counts[case['choice']]
        best_permutation_value = case['choice']

        for permutation_idx, permutation_value in enumerate(permutations(possibles)):
            tup_permutation = tuple(permutation_value)
                
            if tup_permutation not in choice_counts:
                continue
                
            permutation_count = choice_counts[tup_permutation]
            
            if permutation_count > best_permutation_count:
                best_permutation_count = permutation_count
                best_permutation_value = permutation_value
        
        if best_permutation_value != case['choice']:
            print(("Switching Case", case["flags"], case['choice'], "->", best_permutation_value, "(", choice_counts[case['choice']], "->", best_permutation_count, ")", case['choice_times']))
            cases[case_idx]['choice'] = best_permutation_value
            choice_counts[case['choice']] -= 1
            choice_counts[best_permutation_value] += 1
        else:
            print(("Not Case", case["flags"], case['choice'], "->", best_permutation_value, "(", choice_counts[case['choice']], "->", best_permutation_count, ")", case['choice_times']))
           
    return cases

class AlgoTuner:
    def __init__(self, name, choices):
        # Initialize user-provided values
        self.name      = name
        self.choices   = choices

        # Initialize class values
        self.cases      = []
        self.tree       = None
        self.prev_valid = False
        # full algo choosers
        self.params     = ["n", "c", "h", "w", "d", "k", "r", "s", "t", "u", "v", "x", "pad_h", "pad_w", "pad_d", "dil_h", "dil_w", "dil_d", "in_format", "filter_format", "out_format", "in_type", "comp_type", "out_type", "groupCount", "conv_mode"]
        self.choosers   = self.params

        self.skipped = 0

    def add_case(self, case_flags, case_outputs):
        
        algo_pat = re.compile(r'algo:([0-9]+)')
        math_pat = re.compile(r'Pmath:([0-9]+)')

        # Initialize choice_times dict for each choice to float('inf')
        choice_times = dict( (choice, float('inf')) for choice in self.choices )
        
        chooser_vals = None
        num_out_with_no_time = 0
            
        # For each output get time and fill self.choice_times
        for output in case_outputs:
                
            output_parser = OutputParser(output)
            
            if output_parser["time"] == None:
                num_out_with_no_time = num_out_with_no_time + 1
                continue

            algo = algo_pat.search(output).group(1)
            math = math_pat.search(output).group(1)
            choice_times[(algo, math)] = float(output_parser["time"].seconds)

            run_chooser_vals = get_chooser_vals(output, case_flags)

            if chooser_vals == None:                
                chooser_vals = run_chooser_vals
                
            elif chooser_vals != run_chooser_vals:                
                raise Exception("[AlgoTuner] Conflict for tuners: chooser_vals %s and run_chooser_vals %s" % (str(chooser_vals), str(run_chooser_vals)))
        
        sorted_choices = [choice_time[0] for choice_time in sorted(choice_times.items(), key=operator.itemgetter(1))]

        # create a decimal bit-mask of sorted_choices
        sorted_choices = [str(int(a)*10+int(m)) for (a,m) in sorted_choices]

        if (chooser_vals == None) and (num_out_with_no_time == len(case_outputs)):            
            print('case %s skipped. No output has valid time. Total output cases: %s. chooser_val is None' % (case_flags, str(len(case_outputs))))            
            self.skipped = self.skipped + 1
            return
        elif (chooser_vals == None):
            raise Exception("chooser_vals None and output has times %s" % str(case_outputs))

        case_dict = {}

        case_dict['choosers'] = chooser_vals

        case_dict['choice']   = tuple(sorted_choices)

        case_dict['index'] = len(self.cases)

        case_dict['flags'] = case_flags
        
        case_dict['choice_times'] = choice_times

        self.cases.append(case_dict)
        

    def generate_heuristic(self):
        if(self.name == None):
            raise Exception("[AlgoTuner] Calling heuristic generation on heuristic with no name")
            
        # Generate classifier
        tree_gen = DecisionTreeGenerator(prune_cases(self.cases), [])
        
        self.tree = tree_gen.gen_tree()

        self.prev_valid = True

    def classify(self, choosers):
        # If heuristic is no longer valid, update it
        if( not self.prev_valid ):
            self.generate_heuristic()

        # Return classification value
        return self.tree.get_class(choosers) 

    def output_heuristic(self, path):
        # Orphan tuners shouldn't run heuristics
        if(self.name != None):
            # If previous heuristic is not valid, remake it
            if(not self.prev_valid):
                self.generate_heuristic()

            HEADER_TAIL = "/////////////////////////////////////////////////////////////////\n"
            dont_edit = "// DO NOT EDIT THIS FILE!\n"
            auto_gen = "// It was automatically generated by heurcheck (heurgen.py)\n"
                       
            function_signature = "const int (*%s(%s))[%d]" % (self.name, ', '.join(["int %s" % chooser for chooser in self.choosers]), len(self.choices))
                        
            with open("%s/%s.cpp" % (path, self.name), 'w') as out_file:
                out_file.write(HEADER_TAIL)
                out_file.write(dont_edit)
                out_file.write(auto_gen)
                out_file.write(HEADER_TAIL)
                out_file.write('\n')
                
                out_file.write('\n')
                out_file.write('\n')
                out_file.write("#include \"%s.h\"\n\n" % self.name)
                out_file.write('\n')
                out_file.write(self.tree.get_str(self.name))
                
            with open("%s/%s.h" % (path, self.name), 'w') as out_file:
                out_file.write(HEADER_TAIL)
                out_file.write(dont_edit)
                out_file.write(auto_gen)
                out_file.write(HEADER_TAIL)
                out_file.write('\n')
                
                out_file.write("#ifndef %s_H\n" % self.name.upper())
                out_file.write("#define %s_H\n" % self.name.upper())
                out_file.write("\n")
                out_file.write("#include \"../cudnn.h\"\n")
                out_file.write("\n")
                out_file.write(function_signature + ";\n")
                out_file.write("\n");
                out_file.write("#endif\n");


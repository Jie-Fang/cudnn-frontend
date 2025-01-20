# AlgoTuner is responsible for heuristics based on algo chosen
# 
from DecisionTreeGen.DecisionTreeGenerator import DecisionTreeGenerator
from collections import OrderedDict, defaultdict
from helpers.cudnn_interface import OutputParser
from helpers.utility import flags_from_descs_str

import operator
from copy import deepcopy

def prune_cases(cases):
    equivalences = set()
    
    choice_counts = {}
    
    for case_idx in range(len(cases)):
        
        case = cases[case_idx]
        
        cur_choice = case['choice']
        if cur_choice in choice_counts:
            choice_counts[cur_choice] += 1
        else:
            choice_counts[cur_choice] = 1
            
    cases = deepcopy(cases)
            
    for case_idx, case in enumerate(cases):
        possibles = []
        
        INF_VAL = 99e99
                
        cur_choice = case['choice']
        
        cur_time   = case['choice_times'][cur_choice]
        
        allowed_diff = 1.05

        best_choice = cur_choice
        best_time   = cur_time
        
        for poss_choice in case['choice_times']:
            
            poss_time = case['choice_times'][poss_choice]
                
            if poss_time == "inf":
                continue
            
            if poss_choice not in choice_counts:
                continue
            
            if choice_counts[poss_choice] <= choice_counts[best_choice]:
                continue
            
            if (poss_time / best_time) < (allowed_diff):
                best_choice = poss_choice
                best_time   = poss_time
                
        if best_choice != cur_choice:
            print(("Switching Case (Sub-Algo)", case["flags"], cur_choice, "->", best_choice, "(", choice_counts[cur_choice], "->", choice_counts[best_choice], ")", case['choice_times']))
            cases[case_idx]['choice'] = best_choice
            choice_counts[case['choice']] = choice_counts[case['choice']] - 1
            choice_counts[best_choice] = choice_counts[best_choice] + 1
        else:
            print(("Not Case (Sub-Algo)", case["flags"], cur_choice, "->", best_choice, "(", choice_counts[cur_choice], "->", choice_counts[best_choice], ")", case['choice_times']))
           
    return cases
    
def get_flag(flag_str, flag_name):
    flag_name = "-" + flag_name
    if flag_name not in flag_str:
        raise Exception("[AlgoTuner] Unable to find %s in %s" % (flag_name, flag_str))
        
    start_index = flag_str.find(flag_name) + len(flag_name)
    end_index   = flag_str.find(' ', start_index)

    return flag_str[start_index:end_index]
    
class HeurgenTuner:
    def __init__(self, name, choices, choosers, bin_name):
        # Initialize user-provided values
        self.name      = name
        self.choices   = [choice.strip() for choice in choices.split(',') if choice.strip() != '']
        self.choosers  = [chooser.strip() for chooser in choosers.split(',') if chooser.strip() != '']

        # Initialize class values
        self.cases      = []
        self.tree       = None
        self.prev_valid = False
        self.bin_name   = bin_name
        
    def add_case(self, case_flags, case_outputs):
        # Initialize choice_times dict for each choice to float('inf')
        choice_times = dict( (choice, float('inf')) for choice in self.choices )
        
        chooser_vals = None
        
        excluded_debugs = [] #[str(i) for i in [401, 402, 403, 404, 405, 406, 407, 409, 410, 411, 412, 413, 414, 415, 417, 418, 419, 420, 421, 422, 423, 425, 426, 427, 428, 429, 430, 431, 433, 434, 435, 436, 437, 438, 439, 441, 442, 443, 444, 445, 446, 447, 449, 450, 451, 452, 453, 454, 455, 457, 458, 459, 460, 461, 462, 463, 465, 466, 467, 468, 469, 470, 471, 473, 474, 475, 476, 477, 478, 479, 481, 482, 483, 484, 485, 486, 487]]

        excluded_best = float('inf')
        excluded_best_choice = None
        layer_name = None
        
        # For each output get time and fill self.choice_times
        for output in case_outputs:

            output_parser = OutputParser(output)
            
            if output_parser["time"] == None:
                continue

            if output_parser["tune"] == None:
                continue
                
            choice = flags_from_descs_str(output_parser["test_flags"].test_flags)[0]["Dheurgen_dbg="][0]
            time = float(output_parser["time"].seconds)
                
            if ("-Rconv" in case_flags or "-Rdgrad" in case_flags) and choice in excluded_debugs:
                if time < excluded_best:
                    excluded_best = time
                    excluded_best_choice = choice

                if layer_name == None:
                    layer_name = output_parser["layer_name"].layer_name
                    
                continue
            
            choice_times[choice] = time
            
            run_chooser_vals = [tuner.strip() for tuner in output_parser["tune"].tuners.split(",")]            

            if chooser_vals == None:                
                chooser_vals = run_chooser_vals
                
            elif chooser_vals != run_chooser_vals:
                for output in case_outputs:
                    print(output)
                raise Exception("[HeurgenTuner] Conflict for tuners: %s and %s for case flags %s" % (str(chooser_vals), str(output_parser["tune"].tuners), str(case_flags)))
        
        sorted_choices = [choice_time[0] for choice_time in sorted(choice_times.items(), key=operator.itemgetter(1))]

        best_choice = sorted_choices[0]
        
        excluded_perf_ratio = choice_times[best_choice] / excluded_best
        
        print(choice_times)
        
        if (excluded_perf_ratio > 1.10):
            print("[LOST PERF] Layer \"%s\" lost %4.2f%% perf due to selecting %s (%fms) instead of %s (%fms); flags \"%s\"" % (layer_name, 100*excluded_perf_ratio, best_choice, choice_times[best_choice], excluded_best_choice, excluded_best, case_flags))
            
        if chooser_vals == None:
            return 
            
        if len(chooser_vals) != len(self.choosers):
            raise Exception("[HeurgenTuner] Conflict with size of chooser values and names: %s and %s" % (str(chooser_vals), str(self.choosers)))
                        
        case_dict = {}
        
        case_dict['choosers'] = OrderedDict( (self.choosers[idx], int(chooser_vals[idx])) for idx in range(len(self.choosers)) )

        # cudnn / cublas pruning
        if 'cublasTest' in case_outputs[0]:
            mod_value_index = 0
            cublas_sorted_choices = []
            # prune cublas cases to only 32 unique (algo, splitK) pairs
            is_visited = {}
            for choice in sorted_choices:
                if mod_value_index == 32:
                    break
                algo = int(choice) % 1000
                splitK = ((int(choice)/1000) % 100000) > 1

                unique_id = (algo, splitK)

                if not unique_id in is_visited:
                    cublas_sorted_choices.append(choice)
                    mod_value_index += 1
                    is_visited[unique_id] = mod_value_index

            case_dict['choice']   = tuple(cublas_sorted_choices)
        
        else:
            case_dict['choice']   = tuple(sorted_choices)[0]
    
        case_dict['index']    = len(self.cases)
        
        case_dict['flags']    = case_flags
        
        case_dict['choice_times'] = choice_times
        
        self.cases.append(case_dict)

    def generate_heuristic(self):
        if(self.name == None):
            raise Exception("[HeurgenTuner] Calling heuristic generation on heuristic with no name")
            
        if len(self.cases) == 0:
            raise Exception("[HeurgenTuner] No cases to train heuristic %s" % self.name)
            
        # Generate classifier
        if "hmma_elig" in self.choosers and "data_type" in self.choosers and "comp_type" in self.choosers:
            tree_gen = DecisionTreeGenerator(self.cases, ["hmma_elig", "data_type", "comp_type"])
        else:
            tree_gen = DecisionTreeGenerator(prune_cases(self.cases), [])
        
        self.tree = tree_gen.gen_tree()
        
        self.prev_valid = True

    def classify(self, choosers):
        # If heuristic is no longer valid, update it
        if( not self.prev_valid ):
            self.generate_heuristic()

        # Return classification value
        return self.tree.get_class(choosers) 

    def get_tree(self):
        # If heuristic is no longer valid, update it
        if( not self.prev_valid ):
            self.generate_heuristic()
            
        return deepcopy(self.tree)
        
    def output_heuristic(self, path):
        # Orphan tuners shouldn't run heuristics
        if(self.name != None):
            # If previous heuristic is not valid, remake it
            if(not self.prev_valid):
                self.generate_heuristic()
                
            HEADER_TAIL = "/////////////////////////////////////////////////////////////////\n"
            dont_edit = "// DO NOT EDIT THIS FILE!\n"
            auto_gen = "// It was automatically generated by heurcheck (heurgen.py)\n"
            
            function_signature = "int %s(%s)" % (self.name, ', '.join(["int %s" % chooser for chooser in self.choosers]))
            
            with open("%s/%s.cpp" % (path, self.name), 'w') as out_file:
                out_file.write(HEADER_TAIL)
                out_file.write(dont_edit)
                out_file.write(auto_gen)
                out_file.write(HEADER_TAIL)
                out_file.write('\n')
                out_file.write('\n')
                out_file.write("#include \"%s.h\"\n\n" % self.name)
                out_file.write('\n')
                out_file.write(self.tree.get_str(self.name))
                
            with open("%s/%s.h" % (path, self.name), 'w') as out_file:
                out_file.write("#ifndef %s_H\n" % self.name.upper())
                out_file.write("#define %s_H\n" % self.name.upper())
                out_file.write("\n")
                out_file.write("#include \"../cudnn.h\"\n")
                out_file.write("\n")
                out_file.write(function_signature + ";\n")
                out_file.write("\n");
                out_file.write("#endif\n");

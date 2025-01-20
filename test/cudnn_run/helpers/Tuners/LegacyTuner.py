# AlgoTuner is responsible for heuristics based on algo chosen
# 
from helpers.Tuners.DecisionTreeGen.DecisionTreeGenerator import DecisionTreeGenerator
from collections import OrderedDict, defaultdict, namedtuple
from helpers.cudnn_interface import OutputParser
import re
import operator    
import itertools
import collections
from copy import deepcopy


class Possibles:
    
    def __init__(self, name_map, min_ranks, max_ranks, exact_order=None):
        self.min_ranks = tuple(min_ranks)
        self.max_ranks = tuple(max_ranks)
        self.name_map = name_map 
        self.name_unmap = {}
        for name in self.name_map:
            self.name_unmap[self.name_map[name]] = name
            
        self.names = list(sorted(self.name_map.keys()))

        self.exact_order = exact_order
        
        condensed = self.get_condensed()

    @classmethod
    def fromtimes(cls, times):
        exact_order = sorted(times, key=times.get)
 
        name_map    = {}
        for key in sorted(exact_order):
            name_map[key] = len(name_map)

        min_ranks = [None] * len(exact_order)
        max_ranks = [None] * len(exact_order)
        
        best_choice = exact_order[0]
        best_time   = times[best_choice]
        
        cur_rank      = 0
        
        while cur_rank < len(exact_order):
            cur_rank_best = exact_order[cur_rank]
            cur_best_time = times[cur_rank_best]
            
            allowed_diff = 1.0 + ((cur_rank+1) * (.01 * 2.5))

            # Get diff, 2.00 for 2x if best_time is twice as fast
            best_rel = cur_best_time / best_time
            
            # Get half of perf loss; 2.00 becomes (2.0 - 1.0) * 0.50 = 0.50
            best_buffer = (best_rel - 1.0) * 0.50
            
            # Allow max of either 2.5% per rank or 50% of current perf loss
            allowed_diff = max(allowed_diff, 1.0 + best_buffer)
            
            rank_choice_count = 1
            
            while cur_rank+rank_choice_count < len(exact_order):
                cur_order_idx = cur_rank + rank_choice_count
                
                cur_choice = exact_order[cur_order_idx]
                
                cur_time = times[cur_choice]
                
                if (cur_time / cur_best_time) >= (allowed_diff):
                    break
                    
                rank_choice_count += 1
                    
            for cur_order_idx in range(cur_rank, cur_rank+rank_choice_count):
                min_ranks[name_map[exact_order[cur_order_idx]]] = cur_rank
                max_ranks[name_map[exact_order[cur_order_idx]]] = cur_rank+rank_choice_count-1
                
            cur_rank += rank_choice_count
        
        return cls(name_map, min_ranks, max_ranks, exact_order)
        
    def get_min_order(self):
        
        result = []
        
        for condensed_possibles in self.get_condensed():
            for poss in condensed_possibles:
                result += [poss]

        return tuple(result)
        
    def get_condensed(self):
        result = []
        
        for rank in range(len(self.min_ranks)):
            cases_in_rank = [self.name_unmap[idx] for idx, min_rank in enumerate(self.min_ranks) if min_rank == rank]
            
            if len(cases_in_rank) > 0:
                result.append(tuple(cases_in_rank))
        
        return tuple(result)

    def get_str(self, include_exact_order=False):
        condensed = self.get_condensed()
        
        result = ""
        
        for cur_possibles in condensed:

            if len(cur_possibles) == 1:
                result += "%s " % str(cur_possibles[0])
                
            else:
                result += "(%s) " % ' '.join([str(cur_possible) for cur_possible in cur_possibles])
               
        exact_order_str = " [None]" 
        
        if self.exact_order != None:
            exact_order_str = " [%s]" % (' '.join(self.exact_order))
        
        result = result[:-1]
        
        if include_exact_order:
            result += exact_order_str

        return result
        
    def get_intervals(self):
        return None
        
    def __str__(self):
        return self.get_str()
        
    def get_possibles(self):
        condensed = self.get_condensed()
        
        result = []
        
        cur_poss = []
        
        for rank in range(len(self.min_ranks)):
            if rank < len(condensed):
                cur_poss += list(condensed[rank])

            result += [cur_poss[:]]

        return result

class PermTreeNode:
    def __init__(self, parent, name):
        self.count = {}
        self.children  = {}
        self.parent    = parent
        self.name      = name
        
    def exists(self, node_name):
        return node_name in self.children
        
    def get(self, node_name, create=False):
        if node_name not in self.children:
            if create:
                self.children[node_name] = PermTreeNode(self, node_name)
            else:
                return None
                
        return self.children[node_name]

    def get_count(self, depth):
        if depth not in self.count:
            return 0
        return self.count[depth]
        
    def update_parents(self, depth):
        cur_parent = self.parent
        
        while (cur_parent != None) and (cur_parent.get_count(depth) < self.get_count(depth)):
            cur_parent.count[depth] = self.count[depth]
        
            cur_parent = cur_parent.parent
            
    def get_depth(self):
        cur_parent = self.parent
        
        depth = 0
        
        while cur_parent != None:
            depth += 1
            cur_parent = cur_parent.parent
            
        return depth
        
    def __str__(self):
        tab_space = '|   ' * self.get_depth()
        
        result = ""
        
        for child in sorted(self.children):
            child_node = self.children[child]
            
            counts = ','.join(['%d:%d' % (c, i) for c,i in child_node.count.items()])
            result += "%s%s [%s] (%s)\n" % (tab_space, str(child), counts, child_node.get_order_str())

            result += str(child_node)

        return result 
        
    def get_order(self):
        order = [self.name]
        
        cur_parent = self.parent
        
        while (cur_parent != None) and (cur_parent.name != None):
            order += [cur_parent.name]
            cur_parent = cur_parent.parent
        
        return tuple(reversed(order))
        
    def get_order_str(self):
        return ' '.join([str(i.engine_id) for i in self.get_order()])

class PermTree:
    def __init__(self):
        self.tree = PermTreeNode(None, None)
        
    def add_order(self, order, increment=True):
        iter = self.tree

        for cur in order:
            iter = iter.get(cur, create=True)
                
        if increment:
            depth = iter.get_depth()
            iter.count[depth] = iter.get_count(depth) + 1
            
            iter.update_parents(depth)
            
    def __str__(self):
        return str(self.tree)
        
    def get_best(self, possibles):
        class_len = len(possibles)

        to_explore = collections.deque()
        
        to_explore.append(self.tree)
        
        possible_nodes = []
        
        best_node   = None
        
        while len(to_explore) > 0:
            cur_node = to_explore.popleft()
            
            depth = cur_node.get_depth()
            
            next_node = None
                
            for possible in possibles[depth:]:
                next_node = None
                
                for poss in possible:
                    if poss in cur_node.children:
                        child_node = cur_node.children[poss]
                        
                        if (best_node != None) and (child_node != None) and (child_node.get_count(class_len) <= best_node.get_count(class_len)):
                            continue
                            
                        if next_node == None:
                            next_node = child_node
                        else:
                            to_explore.append(child_node)
                            
                cur_node = next_node
                
                if cur_node == None:
                    break
                    
            if best_node == None:
                best_node = next_node
                
            if (next_node != None) and (next_node.get_count(class_len) > best_node.get_count(class_len)):
                best_node = next_node
                
        return best_node
                    
    def increment_possibles(self, possibles):
        class_len = len(possibles)

        to_explore = collections.deque()
        
        to_explore.append(self.tree)
        
        possible_nodes = []
        
        while len(to_explore) > 0:
            cur_node = to_explore.popleft()
            
            depth = cur_node.get_depth()
            
            next_node = None
            
            for possible in possibles[depth:]:
                next_node = None
                
                for poss in possible:
                    if poss in cur_node.children:
                        child_node = cur_node.children[poss]
                            
                        if next_node == None:
                            next_node = child_node
                        else:
                            to_explore.append(child_node)
                            
                cur_node = next_node
                
                if cur_node == None:
                    break
                    
            if next_node:
                next_node.count[class_len] = next_node.get_count(class_len) + 1

                next_node.update_parents(class_len)
                
        return None
        
def prune_cases(cases):
    # Disable pruning due to issues with Pmath0 (it is overly aggressive for these cases)
    return cases
    
    cases = deepcopy(cases)

    tot_cases = len(cases)
    
    case_possibles = [Possibles.fromtimes(case['choice_times']) for case in cases]
    
    best_exact_order_tree = PermTree()
    
    for case_poss in case_possibles:
        best_exact_order_tree.add_order(case_poss.exact_order)

    best_order_tree = PermTree()
    
    for case_poss in case_possibles:
        best_node = best_exact_order_tree.get_best(case_poss.get_possibles())
        
        if best_node:
            best_order_tree.add_order(best_node.get_order(), False)
        else:
            best_order_tree.add_order(case_poss.get_min_order(), False)
            
    for case_poss in case_possibles:
        best_order_tree.increment_possibles(case_poss.get_possibles())
            
    for case, case_poss in zip(cases, case_possibles):
        best_choice = best_order_tree.get_best(case_poss.get_possibles()).get_order()
        
        if best_choice != case['choice']:
            print("Switching %s:\n    %s ->\n    %s" % (case["flags"], str(case['choice']), str(best_choice)))
            case['choice'] = best_choice
        else:
            print("Not Switching %s:\n    %s" % (case["flags"], str(case['choice'])))
            
    return cases

TIMED_ENGINE_CFG = namedtuple("ChoiceTime", "engine_cfg time")

class LegacyTuner:
    def __init__(self, name):
        # Initialize user-provided values
        self.name      = name

        # Initialize class values
        self.cases      = []
        self.tree       = None
        self.prev_valid = False

    def add_opset_cases(self, cases):
        # Initialize choice_times dict for each choice to float('inf')
        per_engine_best = {}
        
        for case in cases:
            case_engine_id = case.choice.engine_id
            
            if (case_engine_id not in per_engine_best) or (case.time < per_engine_best[case_engine_id].time):
                per_engine_best[case.choice.engine_id] = case
                
        choice_times = {}
        
        for engine_id in per_engine_best:
            case = per_engine_best[engine_id]
            
            choice_times[case.choice] = case.time

        sorted_choices = [choice_time[0] for choice_time in sorted(choice_times.items(), key=operator.itemgetter(1))]
            
        case_dict = {}

        params_tup                = tuple([int(s) for s in cases[0].opset.params.split(',')])

        # To support legacy logs that have 26 feature vector
        if (len(params_tup) == 26) :
            case_dict['choosers']     = params_tup + (0,)
        else :
            case_dict['choosers']     = params_tup

        case_dict['choice']       = tuple(sorted_choices)

        case_dict['index']        = len(self.cases)

        case_dict['flags']        = cases[0].test_flags
        
        case_dict['choice_times'] = choice_times

        self.cases.append(case_dict)
        
    def generate_heuristic(self):
        if(self.name == None):
            raise Exception("[AlgoTuner] Calling heuristic generation on heuristic with no name")
        
        # "15" is dilation depth; a proxy for 2d vs 3d
        #  - 2d cases always have dilation depth = 0
        #  - all current known 3d cases always have dilation depth = 1
        # "21" is 'x' data type (should match 'w' & 'y'). 
        # "26" is compute type 
        pref_choosers = ["15", "21", "26"]
            
        # Generate classifier
        tree_gen = DecisionTreeGenerator(prune_cases(self.cases), pref_choosers)
        
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
                       
            function_signature = "const int *%s(int *input, int *output)"
                        
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
                
                out_file.write("#ifndef HEUR_%s_H\n" % self.name.upper())
                out_file.write("#define HEUR_%s_H\n" % self.name.upper())
                out_file.write("\n")
                out_file.write("#include \"cuprintf.h\"\n")
                out_file.write("#include \"cudnn_cnn.h\"\n")
                out_file.write("#include \"heur_types.h\"\n")
                
                out_file.write("\n")
                out_file.write("const leaf_t %s(int32_t *input, int32_t input_size);\n" % self.name)
                out_file.write("\n")
                out_file.write("#endif\n")


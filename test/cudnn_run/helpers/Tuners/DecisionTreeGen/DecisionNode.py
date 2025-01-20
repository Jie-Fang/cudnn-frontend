# A DecisionNode can either be a decision node or a value node
# - Decision Node: Switches to left/right
# -     If case[d_param] <= value, go left; else go right
# - Value Node: Just contains a value (it is a leaf node)
# -     d_param should be NULL; this node should only be used for value

import math
import operator
import collections

class DecisionNode:
    def __init__(self, d_param, value, cases_count, cases_offset, cases_ref, node_index):
        # Assume no children to begin with
        self.left = None
        self.right = None

        # Save decision parameter to self
        self.d_param = d_param

        # Save value to self
        self.value = value

        # Save number of cases (total number of cases contained within node)
        self.cases_count = cases_count
        
        # Save case offset (the index of the first case within node)
        self.cases_offset = cases_offset
        
        # Save reference to list of cases
        self.cases_ref = cases_ref
        
        # Save node_index 
        self.node_index = node_index

    # If this is a decision node, returns next node; returns None otherwise
    def get_next(self, case):
        if(self.is_decision()):
            if(case[self.d_param] <= self.value):
                return self.left
            else:
                return self.right
        
        raise ValueError

    # Returns true if this node is a decision node
    def is_decision(self):
        if(self.d_param == None):
            return False
            
        return True

def sort_uniq(l):
    return sorted(set(l))
    
def calc_entropy(stats):
    return -1 * sum([(math.log(stat, 2) * stat) for stat in stats if stat != 0])
    
def gen_node_sub(chooser_names, raw_cases, cases_begin, cases_end, pref_chooser_list):
    cases = raw_cases[cases_begin:cases_end]
    
    # Get all parameter options
    chooser_options = [ sort_uniq([case["choosers"][idx] for case in cases]) for idx in range(len(cases[0]["choosers"])) ]
    
    # Get all class options
    class_options_list = sort_uniq([val["choice"] for val in cases])

    # Set up dictionary mapping class option to index in class_options_list
    class_options_dict = {class_options_list[idx]: idx for idx in range(len(class_options_list))}

    # If there is only class, return a Value node of that class
    if(len(class_options_list) == 1):
        return ((None, cases[0]["choice"]), [])

    # If we don't have any parameter options...
    if(max( len(option) for option in chooser_options ) == 1):
        
        # Get all class counts
        class_counts = [ len([case for case in cases if case["choice"] == class_opt]) for class_opt in class_options_list ]
        
        # Get class with largest count
        max_class_index, max_class_count = max(enumerate(class_counts), key=operator.itemgetter(1))

        # Get name of best class
        best_class      = class_options_list[max_class_index]

        # Get percentage of this best class
        best_class_perc = float(100*max_class_count) / len(cases)

        # Print message of conflict to user and heuristic analyzer script
        print("Conflict between following cases: %s" % (cases))
        print("Resolving with class %s with %4.2f%% accuracy" % (best_class, best_class_perc))
        print("Start printing flags with conflict")
        for case in cases:
            print(case['flags'])
        print("End printing flags with conflict")

        # Assume value node with most likely class
        return ((None, best_class), [])
        
    # Initiliaze best to None (we haven't seen any cases)
    best = None

    count = 0
    
    pref_chooser_idx = None
    
    for pref_chooser in pref_chooser_list:
        tmp_chooser_idx = chooser_names.index(pref_chooser)
        
        if len(chooser_options[tmp_chooser_idx]) > 1:
            pref_chooser_idx = tmp_chooser_idx
            break

    # Loop through each parameter
    for chooser_index in range(len(chooser_options)):
        if pref_chooser_idx != None and pref_chooser_idx != chooser_index:
            continue

        # Get param name
        chooser = chooser_index

        # Get param options
        options = chooser_options[chooser_index]

        # Loop through each option (value); exclude last because we split from current index to next index
        for option_index in range(len(options) - 1):

            # Split between current and next option
            split = (options[option_index] + options[option_index+1]) / 2
            
            group_a = []
            group_b = []
            
            class_counts_a = [0] * len(class_options_list)
            class_counts_b = [0] * len(class_options_list)

            for case in cases:
                if case["choosers"][chooser] <= split:
                    group_a.append(case)
                    class_counts_a[class_options_dict[case["choice"]]] += 1
                    
                else:
                    group_b.append(case)
                    class_counts_b[class_options_dict[case["choice"]]] += 1
                    
            # Get all cases in each side of the split
            count_a = len(group_a)
            count_b = len(group_b)
            
            # Normalize to percentages
            percent_a = [ float(val) / count_a for val in class_counts_a ]
            percent_b = [ float(val) / count_b for val in class_counts_b ]

            # Get weights of each group            
            weight_a = float(count_a) / len(cases)
            weight_b = float(count_b) / len(cases)

            # Calculate entropy from percentages
            entropy = weight_a * calc_entropy(percent_a) + weight_b * calc_entropy(percent_b)

            # Set this to best (if it is better or our first)
            if(best == None or best["entropy"] > entropy):
                best = {"entropy": entropy, "chooser": chooser, "split": split, "group_a": group_a, "group_b": group_b}

    return ((best["chooser"], best["split"]), [best["group_a"], best["group_b"]])
    
def gen_node(chooser_names, cases, pref_chooser_list):
    # Check that all given preferred choosers are available
    # - For example, fail if 'm' is a preferred chooser but the available choosers are only ['n', 'k']
    for pref_chooser in pref_chooser_list:
        if pref_chooser not in chooser_names:
            raise Exception("[DT Generation] Pref chooser '%s' not found in chooser_names '%s'" % (pref_chooser, chooser_names))

    # Define a NumberRange to be used by indicating a range of cases
    NumberRange = collections.namedtuple("NumberRange", "begin end")
    
    # Define some variables representing hard-coded direction
    # - This direction is used in each iteration of DT generation
    # - This direction indicates if parent.left or parent.right should be set to current result
    # - Note that the enumerations MUST match the naming used by node for "left" and "right"
    LEFT  = "left"
    RIGHT = "right"
    
    # Named tuple representing parent node & direction
    # - "node" gives a pointer to the parent node
    # - "direction" represents LEFT or RIGHT; see above definition of LEFT/RIGHT for more clarification
    ParentNode = collections.namedtuple("ParentNode", "node direction")
    
    # Define a DT Generation Iteration
    # - "parent" indicates the ParentNode information (what is our current node's parent and direction)
    # - "case_range" indicates the range of cases handled by this iteration
    GenIteration = collections.namedtuple("GenIteration", "parent range")
    
    # Create first iteration
    # - For clarity on all GenIteration values, see above definition
    # - This iteration has no Parent information as the root has not been created
    #  - This is a special case, the current Node of this iteration will be considered the root
    iter_stack = collections.deque()
    iter_stack.append(GenIteration(None, NumberRange(0, len(cases))))
    
    # Initialize root to None to begin with
    root = None
    
    # Initialize node count to 0
    node_count = 0
    
    # Loop until all iterations have been enumerated (when we have no more remaining iterations)
    while len(iter_stack) > 0:
        # Obtain current parent & case range from iter_stack
        cur_parent, cur_range  = iter_stack.pop()

        # Obtain current node by using sub-call to gen_node_sub
        node_info, add_to_stack = gen_node_sub(chooser_names, cases, cur_range.begin, cur_range.end, pref_chooser_list)

        # Create node
        cur_node = DecisionNode(node_info[0], node_info[1], cur_range.end-cur_range.begin, cur_range.begin, cases, node_count)
        
        # Increment node count (because we just created one)
        node_count += 1
                
        # If current parent is None, it means this iteration is computing the Root node
        if cur_parent == None:
            # To set root, we should make sure it hasn't already been set
            if root != None:
                raise Exception("[DT Generation] Root already set but attempting to set again...")
                
            # Set root node
            root = cur_node
            
        # Otherwise, current parent is a valid node and we should set its left/right to current node
        else:
            # Assuming direction is left, equivalent to the following code: cur_parent.left = cur_node
            setattr(cur_parent.node, cur_parent.direction, cur_node)
            
        # Add new iterations if there are any
        if len(add_to_stack) == 2:
            # Update cases to new order (LEFT on left, RIGHT on right)
            cases[cur_range.begin:cur_range.end] = add_to_stack[0] + add_to_stack[1]

            # Define begin/middle/end for our new Iterations
            # - The LEFT iteration will be from begin -> middle
            # - The RIGHT iteration will be from middle -> end
            cur_begin = cur_range.begin
            cur_middle = cur_range.begin + len(add_to_stack[0])
            cur_end = cur_range.end
            
            # Append LEFT & RIGHT iteration to stack
            iter_stack.append( GenIteration(ParentNode(cur_node, LEFT),  NumberRange(cur_begin, cur_middle)) )
            iter_stack.append( GenIteration(ParentNode(cur_node, RIGHT), NumberRange(cur_middle, cur_end)) )
                        
    # Return computed root
    return (root, node_count)

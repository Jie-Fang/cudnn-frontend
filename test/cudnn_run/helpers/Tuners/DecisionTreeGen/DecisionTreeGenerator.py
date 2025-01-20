# A DecisionTreeGenerator is able to generate a DecisionTree by using given cases by the user

# There are two primary purposes:
#   1. add_case to add cases to be generated in decision tree
#   2. gen_tree to generate a DecisionTree that classifies each given case

import helpers.Tuners.DecisionTreeGen.DecisionTree as DecisionTree
import helpers.Tuners.DecisionTreeGen.DecisionNode as DecisionNode
import copy        

def check_valid(a):
    # Must have "choice" parameter
    if(not("choice" in a)):
        raise KeyError
        
    # "choice" parameter must be string
    if(not(isinstance(a["choice"], str) or isinstance(a["choice"], tuple))):
        raise TypeError
        
    # Loop through all keys but "choice"
    for key in a["choosers"]:
        if(not isinstance(key, int)):
            raise TypeError
            
    # Must have at least 1 chooser
    if(len(a["choosers"]) < 1):
        raise KeyError
            
def check_same(a, b):
    # Get keys in set form
    a_keys = set(a)
    b_keys = set(b)
    
    # Check matching keys
    if(a_keys != b_keys):
        raise KeyError
    
    # Check same type
    if( type(a["choice"]) != type(b["choice"]) ):
        raise TypeError
        
class DecisionTreeGenerator:
    def __init__(self, cases=[], pref_choosers=[]):
        # Initialize cases to given (or empty list)
        self.cases = []
        
        # Initialize choices to empty (defines all observed choices)
        self.all_choices_dict = {}
       
        # Add each of the given user cases
        for case in cases:
            self.add_case(case)
            
        self.pref_choosers = [pref_chooser for pref_chooser in pref_choosers]

    def add_case(self, case):
        # Case must be valid
        check_valid(case)
            
        # If we have an existing case
        if(len(self.cases) > 0):
            # Ensure this new case matches existing precedent
            check_same(case, self.cases[0])
        
        # Add case to cases
        self.cases.append(copy.deepcopy(case))
        
        for choice in case['choice']:
            self.all_choices_dict[choice] = None
            
        return True
        
    def get_params(self):
        # Early exit if we don't have any cases
        if(len(self.cases) == 0):
            raise ValueError
      
        # Get params for first case (used for reference)
        params = [str(i) for i in range(len(self.cases[0]["choosers"]))]
            
        return params
        
    def get_multi_count(self):
        # Early exit if we don't have any cases
        if(len(self.cases) == 0):
            raise ValueError
           
        # If a list, return list count
        if(isinstance(self.cases[0]["choice"], tuple)):
            return len(self.cases[0]["choice"])
        # If not a list, return 1 size
        else:
            return 1
            
        # Something wrong with system if this happens
        raise SystemError
        
    def gen_tree(self):
        # Get params for cases
        param_names = self.get_params()

        # Determine if there are multiple values
        multi_count = self.get_multi_count()

        # Generate tree 
        root, node_count = DecisionNode.gen_node(param_names, self.cases, self.pref_choosers)
        
        # Create Tree class from generated tree
        tree = DecisionTree.DecisionTree(param_names, multi_count, root, node_count, self.all_choices_dict.keys())

        # Clean up nodes
        tree.prune_nodes()

        return tree

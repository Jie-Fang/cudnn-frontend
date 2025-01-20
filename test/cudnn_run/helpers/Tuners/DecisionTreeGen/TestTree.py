from DecisionTreeGenerator import DecisionTreeGenerator

cases = [{"a": 30, "b": 2, "class": "Apple"}, {"a": 30, "b": 2, "class": "Apple"}, {"a": 30, "b": 2, "class": "Dog"}, {"a": 3, "b": 4, "class": "False"}, {"a": 20, "b": 15, "class": "True"}, {"a": 50, "b": 10, "class": "False"}]

cases_dict = []

for case in cases:
    case_dict = {}
    
    case_dict['choosers'] = dict( [(chooser, case[chooser]) for chooser in case if chooser != "class"] )

    case_dict['choice']   = case["class"]

    case_dict['index']    = len(cases_dict)

    case_dict['flags']    = " ".join(["-%s%d" % (chooser, case[chooser]) for chooser in case if chooser != "class"] )

    cases_dict.append(case_dict)
   
# Create Generator from cases in dictionary form
TreeGen = DecisionTreeGenerator(cases_dict)

# Generate tree
import time

begin = time.time()
tree = TreeGen.gen_tree()
end = time.time()

print("New Time: %f" % (end - begin))

begin = time.time()
tree_str = tree.get_str("int")
end = time.time()

print("New Time: %f" % (end - begin))

print(tree_str)

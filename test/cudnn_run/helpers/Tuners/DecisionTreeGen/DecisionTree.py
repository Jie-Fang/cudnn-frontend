# A DecisionTree is a Tree constructed with DecisionNodes
#   All branch nodes are decision nodes
#   All leaf nodes are value nodes
# There are two primary purposes:
#   1. Classify a given case to leaf node value
#   2. Print out DecisionTree using C syntax
import helpers.Tuners.DecisionTreeGen.DecisionNode
from collections import deque
import collections

def merge_node(tree):
    
    return False
    
    # Start exploration at root
    to_explore = deque()
    to_explore.append((tree.root, None, None))

    any_merged = False
    # While there are nodes to explore
    while(len(to_explore) > 0):       
        node, parent, direction = to_explore.popleft()

        # For decision nodes
        if(node.is_decision()):
            
            if node.left.is_decision() or node.right.is_decision():
                # Explore left node (will be done after printing if condition)
                to_explore.append((node.left, node, "left"))
                
                # Explore right node (will be done after printing if condition)
                to_explore.append((node.right, node, "right"))
                
            elif parent != None:
                if node.left.value == node.right.value:
                
                    self, d_param, value, cases_count, cases_offset, cases_ref, node_index
                    
                    setattr(parent, direction, DecisionNode.DecisionNode(None, node.left.value, node.left.cases_count + node.right.cases_count))
                    
                    del node.left
                    del node.right
                    del node
        
                    any_merged = True
                    
    return any_merged
    
class DecisionTree:
    def __init__(self, d_params, multi_count, root, node_count, all_choices):
        # Set root
        self.root = root
        
        # Save decision parameters to self
        self.d_params = d_params
        
        # Save multi_count choice to self
        #   Determines if this tree's leaf nodes have multiple values
        self.multi_count = multi_count
        
        # Save node count
        self.node_count = node_count
        
        # Save all choices (sorted)
        self.all_choices = sorted(all_choices)

    # If this is a decision node, returns next node
    def get_class(self, case):
        # Early exit if tree is empty
        if(self.root == None):
            raise ValueError
        
        # Start iteration at root
        cur_node = self.root

        # While in decision node
        while(cur_node.is_decision()):
            # Decide on next node
            cur_node = cur_node.get_next(case)
        
        # Return value once at leaf node
        return cur_node.value
        
    def prune_nodes(self):
        while(merge_node(self)):
            pass

    def get_str(self, function_name, type_name="int"):
        if type_name != "int":
            raise Exception("type_name must be int")
            
        # Start exploration at root
        to_explore = deque()
        
        to_explore.append(self.root)

        # Initial Inputs
        INPUT_PARAMS = self.d_params
        RETURN_COUNT = self.multi_count
        INPUT_COUNT = len(INPUT_PARAMS)
        FUNCTION_NAME = function_name

        # Assume empty string to begin with
        result = ""
        
        result += "#include <stdio.h>\n\n"

        result += "#include \"cudnn_cnn.h\" // For CUPRINTF \n"
        result += "static const int32_t LEAF = 1 << 31;\n"
        result += "#ifdef HEUR_TREE_DBG\n"
        result += "struct case_t{\n"
        result += "    const char *flags;\n"
        result += "};\n"
        
        result += "\n"
        
        result += "static const case_t cases[] = {\n"
        
        for case_idx, case in enumerate(self.root.cases_ref):
            
            case_flags  = case['flags']
            
            choosers = case['choosers']
            
            chooser_str = ", ".join(["%s" % chooser for chooser in choosers])
            
            choice_str = ", ".join(["eng%d+knobs(%s)" % (choice.engine_id, ",".join(choice.knob_choices)) for choice in case['choice']])
            
            case_str = "\"" + case['flags'].replace("\"", "\\\"") + " {" + chooser_str + "} -> [" + choice_str + "]\""

            result += "  /* %d */ {%s},\n" % (case_idx, case_str)
            
        result += "};\n"
        
        # Define a Leaf to be a choice & index to case_list
        Leaf = collections.namedtuple("Leaf", "choice case_list_index")
        
        SubNode = collections.namedtuple("SubNode", "index is_leaf")
        
        # Define a MemNode to be the info necessary for a Node in memory (stored in file) 
        MemNode = collections.namedtuple("MemNode", "key threshold left right")
        
        leafs      = []
        case_lists = []
        nodes      = []
        
        case_lists_count = 0
        
        # Map from node_index to decision_node_index
        map_node_index = {}
        
        if not self.root.is_decision():
            sub_node = SubNode(len(leafs), True)
            leafs.append(Leaf(self.root.value, 0))
            
            case_lists.append([i for i in range(self.root.cases_offset, self.root.cases_offset + self.root.cases_count)] + [-1])
            
            case_lists_count += len(case_lists[-1])
            
            nodes.append(MemNode(0, 0, sub_node, sub_node))
        
        else:
            # Traverse all nodes and insert into list
            while(len(to_explore) > 0):       
                # Get node
                node = to_explore.popleft()

                child_nodes = [None, None]
                
                if node.is_decision():
                    for child_index, child_node in enumerate([node.left, node.right]):
                        # Only explore left if decision node
                        if child_node.is_decision():
                            child_nodes[child_index] = SubNode(child_node.node_index, False)
                            
                            to_explore.append(child_node)

                        else:
                            child_nodes[child_index] = SubNode(len(leafs), True)
                            
                            leafs.append(Leaf(child_node.value, case_lists_count))
                            
                            case_lists.append([i for i in range(child_node.cases_offset, child_node.cases_offset + child_node.cases_count)] + [-1])
                            
                            case_lists_count += len(case_lists[-1])
                  
                map_node_index[node.node_index] = len(nodes)
                
                nodes.append(MemNode(node.d_param, node.value, child_nodes[0], child_nodes[1]))

        result += "static const int32_t case_lists[] = {\n"
        case_lists_count_cur = 0
        for cur_case_list in case_lists:
            result += "  /* %d */ %s,\n" % (case_lists_count_cur, ", ".join([str(i) for i in cur_case_list]))
            case_lists_count_cur += len(cur_case_list)
        result += "};\n"

        result += "#define CASE_LIST(list) , list\n"
        result += "#else\n"
        result += "#define CASE_LIST(...)\n"
        result += "#endif // HEUR_TREE_DBG\n"

        result += "\n"
        
        result += "\n"
        
        KnobChoiceMetaData = collections.namedtuple("KnobChoiceMetaData", "id offset")
        
        choice_meta_dict = {}
        
        choice_offset = 0
        
        for choice_idx, choice in enumerate(self.all_choices):
            choice_meta_dict[choice] = KnobChoiceMetaData(choice_idx, choice_offset)
            
            choice_offset += len(choice.knob_choices)
        
        result += "// kc = Knob Choice Data (shortened to reduce file size)\n"
        
        result += "static const int32_t kcd[] = {\n"
            
        for choice in self.all_choices:
            if len(choice.knob_choices) > 0:
                meta_data = choice_meta_dict[choice]
                result += "  " + ",".join([str(i) for i in choice.knob_choices]) + ", // %d\n" % meta_data.offset
        
        result += "};\n"
        
        result += "static const engcfg_t cfg[] = {\n";
        
        for choice in self.all_choices:
            meta_data = choice_meta_dict[choice]
            
            choice_count = len(choice.knob_choices)
            
            if choice_count > 0:
                result += "  engcfg_t(%d, kcd + %d, %d), // c_off %d\n" % (choice.engine_id, meta_data.offset, choice_count, meta_data.id)
            else:
                result += "  engcfg_t(%d, NULL, 0), // c_off %d\n" % (choice.engine_id, meta_data.id)
                
        result += "};\n"

        result += " // Array of four elements {key, threshold, left, right}\n"
        result += "static const int32_t nodes[][4] = {\n"
        result += "  /* keys: %s */\n" % ", ".join(self.d_params)
        for node_index, node in enumerate(nodes):
            left_node_str = ("LEAF | %d" % node.left.index) if node.left.is_leaf else ("%d" % map_node_index[node.left.index])
            right_node_str = ("LEAF | %d" % node.right.index) if node.right.is_leaf else ("%d" % map_node_index[node.right.index])

            result += "  /* %d */ {%d, %d, %s, %s},\n" % (node_index, node.key, node.threshold, left_node_str, right_node_str)
        result += "};\n"

        result += "// cfgd = Config Data\n"
        
        LeafMetaData = collections.namedtuple("LeafMetaData", "offset")
        
        leaf_meta_dict = {}
        
        leaf_offset = 0
        for leaf_idx, leaf in enumerate(leafs):
            leaf_meta_dict[leaf_idx] = LeafMetaData(leaf_offset)
            
            leaf_offset += len(leaf.choice)
            
        result += "// Leaf data\n"
        result += "static const engcfg_t* ld[] = {\n"
        for leaf_index, leaf in enumerate(leafs):
            choice_str = ", ".join(["cfg + %d" % choice_meta_dict[choice].id for choice in leaf.choice])
                
            result += "  %s,\n" % choice_str
            
        result += "};\n"
        
        cur_leaf_offset = 0
        
        result += "//2-D array showing start and size of leaf_data\n"        
        result += "static const int32_t leafs[][2] = {\n"        
        for leaf_index, leaf in enumerate(leafs):
            result += "  {%d, %d},\n" % (leaf_meta_dict[leaf_index].offset, len(leaf.choice))
            
        result += "};\n"

        result += "#ifdef HEUR_TREE_DBG\n"
        result += "static const char* key_strings[]={%s};\n" % (','.join(["\"%s\"" % s for s in INPUT_PARAMS]))
        
        result += "static const int32_t *leaf_cases[] = {\n"        
        
        for leaf_index, leaf in enumerate(leafs):
            result += "  case_lists + %d,\n" % leaf.case_list_index
            
        result += "};\n"
        
        result += "#endif\n"

        result += "\n"
        
        result += "const leaf_t %s(int32_t *input, int32_t input_size) {\n" % function_name
        
        result += "#ifdef HEUR_TREE_DBG\n"
        result += "    CUPRINTF(\"---------START OF HEURISTIC INFO-----------\\n\");\n"
        result += "    CUPRINTF(\"inputs[%d] = [\", input_size);\n"
        result += "    for(int32_t input_idx = 0; input_idx < input_size; ++input_idx) {\n"
        result += "        CUPRINTF(\"%d,\", input[input_idx]);\n"
        result += "    }\n"
        result += "    CUPRINTF(\"]\\n\\n\");\n"
        result += "#endif\n\n"

        result += "    // Input size must match the size trained with\n"
        result += "    if(input_size != FEATURE_VECTOR_SIZE) {\n"
        result += "#ifdef HEUR_TREE_DBG\n"
        result += "        CUPRINTF(\"Heuristic input size does not match: %d != %d\\n---------END OF HEURISTIC INFO-----------\\n\", input_size, FEATURE_VECTOR_SIZE);\n"
        result += "#endif\n"
        result += "        leaf_t err(nullptr, 0);\n"
        result += "        return err;\n"
        result += "    }\n\n"
    
        result += "#ifdef HEUR_TREE_DBG\n"
        result += "    CUPRINTF(\"Heuristic Traversal Path:\\n\");\n"
        result += "#endif\n"

        result += "    int32_t next = 0;\n"
        result += "    while((next & LEAF) == 0){\n"
        result += "        const int32_t *node = nodes[next];\n"
        result += "        if(input[node[0]] <= node[1]) {\n"
        result += "#ifdef HEUR_TREE_DBG\n"
        result += "            CUPRINTF(\" %s <= %d;\\n\", key_strings[node[0]], node[1]);\n"
        result += "#endif\n"
        result += "            next = node[2];\n"
        result += "        } else {\n"
        result += "#ifdef HEUR_TREE_DBG\n"
        result += "            CUPRINTF(\" %s > %d;\\n\", key_strings[node[0]], node[1]);\n"
        result += "#endif\n"
        result += "            next = node[3];\n"
        result += "        }\n"
        result += "    }\n"

        result += "    int32_t leaf_idx = (next & ~(LEAF));\n\n"
        
        result += "    const int32_t *leaf = leafs[leaf_idx];\n"
        
        result += "    \n";
        
        result += "#ifdef HEUR_TREE_DBG\n"
        
        result += "    CUPRINTF(\"\\nEngine/Knob Choices:\\n\");\n"
        result += "    for(int cfg_idx = 0; cfg_idx < leaf[1]; ++cfg_idx) \n"
        result += "    {\n"
        result += "        const engcfg_t *cfg = ld[leaf[0] + cfg_idx];\n"
        result += "        CUPRINTF(\"    Engine[%d], \", (int)cfg->engine_id);\n"
        result += "        CUPRINTF(\"Knobs(\");\n"
        result += "        \n"
        result += "        for(int knob_idx = 0; knob_idx < cfg->knob_count; ++knob_idx) {\n"
        result += "                CUPRINTF(\"%d,\", (int)*(cfg->start + knob_idx));\n"
        result += "        }\n"
        result += "        CUPRINTF(\")\\n\");\n"
        result += "    }\n"
        result += "    CUPRINTF(\"\\nTrained From Cases:\\n\");\n"
        result += "    for(const int32_t *cur_case_idx = leaf_cases[leaf_idx]; *cur_case_idx != -1; cur_case_idx++) {\n"
        result += "        CUPRINTF(\"[%d]: %s\\n\", (int)*cur_case_idx, cases[*cur_case_idx].flags);\n"
        result += "    }\n"
        result += "    CUPRINTF(\"---------END OF HEURISTIC INFO-----------\\n\");\n"
        
        result += "#endif\n"
        result += "    leaf_t ret(ld + leaf[0], leaf[1]);\n"

        result += "    return ret;\n"
        
        result += "}\n\n"

        return result
        
    def get_c_str(self, type_name=None):
        # Start exploration at root
        to_explore = deque()
        
        to_explore.append((self.root, 1))

        # Assume empty string to begin with
        result = ""

        # -- Write out function code --
        # While there are nodes to explore
        while(len(to_explore) > 0):            
            # Get node and tab level
            
            node, tab = to_explore.popleft()

            # Construct tab string (tab number of tab characters)
            tab_str = ("    " * tab)
            
            # Just print out any strings we get
            if(isinstance(node, str)):
                result += "%s%s\n" % (tab_str, node)

            # If not a string, it should be a node
            else:
                # Construct insertion array (to be inserted at beginning of exploration)
                to_insert = deque()
                
                # For decision nodes
                if(node.is_decision()):
                    # Insert left hand if condition
                    to_insert.append(("if(%s <= %s){" % (str(node.d_param), str(node.value)), tab))

                    # Explore left node (will be done after printing if condition)
                    to_insert.append((node.left, tab+1))

                    # Insert closing bracket
                    to_insert.append(("}", tab))

                    # Insert right hand if condition
                    to_insert.append(("if(%s > %s){" % (str(node.d_param), str(node.value)), tab))
                    
                    # Explore right node (will be done after printing if condition)
                    to_insert.append((node.right, tab+1))

                    # Insert closing bracket
                    to_insert.append(("}", tab))

                else:
                    # If multi-count node
                    if(self.multi_count > 1):
                        # Show number of elements this class has in train set
                        to_insert.append(("// %d elements" % node.cases_count, tab))
                        to_insert.append(("    // Case Indices: [%s]" % ",".join(["0"]), tab))
                        # to_insert.append(("    // Case Indices: [%s]" % ",".join([str(case["index"]) for case in node.cases]), tab))
                        
                        if type_name == None:
                            raise Exception("type_name not provided when generating Decision Node. look for \"get_str\" called without any values")
                        
                        result_header = "static const %s r[%d] = {" % (type_name, len(node.value))
                        
                        indent_per_elem = ' ' * len(result_header)
                        all_vals = (",\n%s" % indent_per_elem).join(node.value)
                        
                        result_lines = result_header + all_vals + "};"
                        
                        for result_line in result_lines.split('\n'):
                            to_insert.append((result_line, tab))
                            
                        # Early exit (we have filled up result array)
                        to_insert.append(("return &r;", tab))
                                
                    else:
                        to_insert.append(("// %d elements" % node.cases_count, tab))
                        
                        to_insert.append(("// %s" % ",".join([str(s) for s in node.cases_ref[node.cases_offset:node.cases_offset + node.cases_count]]), tab))
                        # to_insert.append(("    // Case Indices: [%s]" % ",".join([str(case["index"]) for case in node.cases]), tab))
                        
                        # On a single-value node, simply return result
                        to_insert.append(("return %s;" % node.value, tab))

                # Insert into exploration (at beginning)
                while len(to_insert) > 0:
                    to_explore.appendleft(to_insert.pop())

        return result

from collections import namedtuple
from re import compile, search
from sys import exc_info
import random

from .utility import is_ignored_line, flags_from_descs_str, cross_flags, get_flags_list_intersection, OrderedDefaultDict, flags_match_a_in_b_lists
from .utility_py3 import *
from .Flags import Flags

layer_pat = compile('\s*"(.*)"\s*=\s*(.*?)\s*$')

Layer = namedtuple("Layer", "base_name test_name test_diff_flags flags")


def gen_flags_from_file(layer_file_name, include_flags_list, include_layer_name, exclude_flags_list,
                        exclude_layer_name, global_flags_list, label_db, rand_sample_per_layer):
    '''Generator for Flag objects for each parsed in {layer_file_name}, with labels from {label_db}
    - flags from {global_flags_list} are appended to each flag object
    - only layer whose name is in {include_layer_name} are generated
    - on Flags that contains {include_flags_list} and not any of non-empty entry of {exclude_flags_list} are generated.
    '''

    rand_instance = random.Random()

    with open(layer_file_name, "r") as layer_file:
        for (line_index, line) in enumerate(layer_file):
            # Ignore commented or empty lines
            if is_ignored_line(line):
                continue

            try:
                match_line = layer_pat.match(line)

                if match_line == None:
                    raise Exception("Invalid layer description given")

                (layer_name, layer_descs) = match_line.groups()

                if include_layer_name:
                    # if `include_layer_name` set, accept this layer only if the layer name match
                    if isinstance(include_layer_name, basestring):
                        include_layer_name = [include_layer_name]
                    if isinstance(include_layer_name, list) and include_layer_name:
                        for layer_name_regex in include_layer_name:
                            if search(layer_name_regex, layer_name) is not None:
                                break   # regex matched, accept this layer
                        else:
                            continue  # none of the regex matched, skip and check next layer
                    else:
                        continue  # skip, check next layer

                if exclude_layer_name:
                    # if `exclude_layer_name` set, accept this layer when the layer name not match
                    if isinstance(exclude_layer_name, basestring):
                        exclude_layer_name = [exclude_layer_name]
                    if isinstance(exclude_layer_name, list) and exclude_layer_name:
                        if any(search(regex, layer_name) for regex in exclude_layer_name):
                            continue  # regex matched, skip this layer
                        # at this point: if not skipped by any exclude regex, this layer is accepted

                layer_descs_flags_list = flags_from_descs_str(
                    layer_descs, label_db)

                if not layer_descs_flags_list:
                    continue

                layer_descs_flags_list = cross_flags(layer_descs_flags_list,
                                                     global_flags_list)

                if not layer_descs_flags_list:
                    continue

                filtered_layer_descs_flags_list = [
                    sub_flag  #
                    for layer_descs_flags in layer_descs_flags_list
                    for sub_flag in layer_descs_flags.get_sub_flags()
                    if flags_match_a_in_b_lists(include_flags_list, [sub_flag])
                    and (all(len(x) == 0 for x in exclude_flags_list)
                         or not flags_match_a_in_b_lists(
                             exclude_flags_list, [sub_flag]))
                ]

                if len(filtered_layer_descs_flags_list) == 0:
                    continue

                unique_flags = get_flags_list_intersection(
                    filtered_layer_descs_flags_list)

                if rand_sample_per_layer > -1:
                    # Ensure that each layer gets its own flags (order of layers should not matter)
                    rand_instance.seed(layer_name)

                    filtered_layer_descs_flags_list = rand_instance.sample(
                        filtered_layer_descs_flags_list, rand_sample_per_layer)

                for layer_descs_flags in filtered_layer_descs_flags_list:
                    yield (layer_name, unique_flags, layer_descs_flags)

            except Exception as e:
                # Store traceback info (to find where real error spawned)
                t, v, tb = exc_info()

                # Re-raise exception with line info
                exception = Exception(
                    "[LAYER PARSING] %s (at %s:%s)" % (str(e), layer_file_name,
                                                       line_index + 1))
                reraise(t, exception, tb)
    return


def get_layers_count_from_file(layer_file_name, include_flags_list, include_layer_name,
                               exclude_flags_list, exclude_layer_name, global_flags_list, label_db):
    if split_flag_keys == None:
        split_flag_keys = []

    gen_layers = gen_flags_from_file(
        layer_file_name, include_flags_list, include_layer_name, exclude_flags_list,
        exclude_layer_name, global_flags_list, split_flag_keys, label_db)

    result_count = 0

    for (layer_name, unique_flags, flags) in gen_layers:
        result_count += flags.get_sub_flags_count()

    return result_count


class DuplicateFlagDetector:
    def __init__(self):
        self.flag_names = {}
        self.flag_values = {}
        self.all_flags = set()

    def get_flag_name_idx(self, flag_name):
        if not flag_name in self.flag_names:
            self.flag_names[flag_name] = len(self.flag_names)

        return self.flag_names[flag_name]

    def get_flag_value_idx(self, flag_value):
        if not flag_value in self.flag_values:
            self.flag_values[flag_value] = len(self.flag_values)

        return self.flag_values[flag_value]

    #returns false if the flags are already present.
    def add_flags(self, flags):
        flag_indices_dict = {}

        for flag in flags:
            flag_name_idx = self.get_flag_name_idx(flag)
            flag_value_idx = self.get_flag_value_idx(flags[flag])

            flag_indices_dict[flag_name_idx] = flag_value_idx

        flag_indices_list = []
        for flag_name_idx in sorted(flag_indices_dict):
            flag_indices_list.append(flag_name_idx)
            flag_indices_list.append(flag_indices_dict[flag_name_idx])

        flag_indices_tuple = tuple(flag_indices_list)

        if flag_indices_tuple in self.all_flags:
            return False

        self.all_flags.add(flag_indices_tuple)

        return True


def gen_layers_from_file(layer_file_name, include_flags_list, include_layer_name,
                         exclude_flags_list, exclude_layer_name, global_flags_list, label_db,
                         dup_detector, rand_sample_per_layer):
    gen_layers = gen_flags_from_file(
        layer_file_name, include_flags_list, include_layer_name, exclude_flags_list,
        exclude_layer_name, global_flags_list, label_db, rand_sample_per_layer)

    for (layer_name, unique_flags, flags) in gen_layers:
        for sub_flag in flags.get_sub_flags():
            test_diff_flags = sub_flag.get_flags_for_keys(unique_flags)

            test_name = layer_name + test_diff_flags.get_str(
                prefix='_', delimiter='')

            # Skip duplicate flags
            if (dup_detector and not dup_detector.add_flags(sub_flag)):
                print("[DuplicateFlag][layer: " + layer_name + "] " + " [Flags]" + str(flags) + " [SubFlags]" + str(sub_flag))
                continue

            yield Layer(layer_name, test_name, test_diff_flags, sub_flag)

    return


if __name__ == "__main__":
    test_str = '''"layer_name1" = n: 1

                  "layer_name2" = c: 2 * d: 4 * label_import'''

    label_db = {"label_import": [Flags()]}

    print(get_layers_from_str(test_str, label_db, "test_layer_name"))

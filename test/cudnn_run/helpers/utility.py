import contextlib
import os.path
import time

from csv import reader
from re import compile

from .Flags import Flags
from .utility_py3 import *

# Regex name match (capture a range of ASCII values)
re_name_match = "[\x21-\x7E]+"

empty_pat = compile('\s*$')
comment_pat = compile('\s*//.*$')


def is_empty_line(line):
    return empty_pat.match(line) != None


def is_comment_line(line):
    return comment_pat.match(line) != None


def is_ignored_line(line):
    if is_empty_line(line) or is_comment_line(line):
        return True

    return False


def split_and_strip(string, split_by, maxsplit=-1):
    return [val.strip() for val in string.split(split_by, maxsplit)]


def strip_exclude_quotes(string):
    result = ""

    in_quotes = False

    for char_idx, char in enumerate(string):
        if char == '"' and (char_idx == 0 or string[char_idx - 1] != '\\'):
            in_quotes = not in_quotes

        if not char.isspace() or in_quotes:
            result += char

    return result


def split_comma(string):
    if string == None:
        return []

    stripped_string = strip_exclude_quotes(string)

    stripped_split = next(reader(StringIO(stripped_string)), [''])

    result = [val for val in stripped_split]

    return result


def split_space(string):
    if string == None:
        return []

    unstripped_result = next(
        reader(StringIO(str(string)), delimiter=' '), [''])

    result = [val.strip() for val in unstripped_result if val.strip() != ""]

    return result


def get_shell_list(flags):
    return split_space(str(flags))


def cross_flags(flags_bases, flags_overrides):
    result = []

    for flags_base in flags_bases:
        for flags_override in flags_overrides:
            result.append(flags_base + flags_override)

    return result


def flags_from_descs_str(string, label_db=None):
    if string == None:
        return [Flags()]

    # Case is assumed to be descriptors (and error if not)
    split_by_bar = split_and_strip(string, '*')

    flags = [Flags()]

    for val in split_by_bar:
        if ':' in val:
            (desc_key, desc_val) = split_and_strip(val, ':', 1)

            for flags_idx in range(len(flags)):
                if desc_key == 'jsonStr=':# treat json as a whole unit, do not split with comma, instead strip single quotes
                    flags[flags_idx][desc_key] = tuple([desc_val[2:-2]])
                elif desc_key == 'minDevVer' and ( desc_key in flags[flags_idx].key_order ):  #
                    flags[flags_idx][desc_key] = tuple( [str( max( int(flags[flags_idx][desc_key][0]) , int(desc_val) )  )] )
                else:
                    flags[flags_idx][desc_key] = tuple(split_comma(desc_val))

        elif is_empty_line(val):
            # Ignore empty areas between bars
            pass

        else:
            cross_label_name = val

            if not (cross_label_name in label_db):
                raise Exception("Label \"%s\" not found" % cross_label_name)

            flags = cross_flags(flags, label_db[cross_label_name])

    return flags


def flags_match_a_in_b(flags_a, flags_b):
    if flags_a.next_multi_flag():
        raise Exception("Error matching multi-flag a:" + repr(flags_a))

    if flags_b.next_multi_flag():
        raise Exception("Error matching multi-flag b:" + repr(flags_b))

    if flags_a.key_count() == 0:
        return True

    for flag_key in flags_a:
        if not (flag_key in flags_b):
            return False

        if flags_a[flag_key] != flags_b[flag_key]:
            return False

    return True


def flags_match_a_in_b_lists(flags_list_a, flags_list_b):
    for flags_a in flags_list_a:
        for flags_b in flags_list_b:
            for sub_flags_a in flags_a.get_sub_flags():
                for sub_flags_b in flags_b.get_sub_flags():
                    if flags_match_a_in_b(sub_flags_a, sub_flags_b):
                        return True
    return False


def get_flags_list_intersection(flags_list):

    if flags_list == None:
        raise Exception("None is not a valid flags list")

    if len(flags_list) == 0:
        raise Exception("Empty flags list given")

    base_flags = dict((key, flags_list[0][key]) for key in flags_list[0])

    intersecting_flags = set(
        [flag for flag in base_flags if len(base_flags[flag]) > 1])

    for flags in flags_list[1:]:
        for flag in base_flags:
            if flag not in flags or base_flags[flag] != flags[flag] or len(
                    flags[flag]) > 1:
                intersecting_flags.add(flag)

        for flag in flags:
            if flag not in base_flags or flags[flag] != base_flags[
                    flag] or len(base_flags[flag]) > 1:
                intersecting_flags.add(flag)

    return intersecting_flags


def modify_layer(layer, flags):
    new_test_name = "_".join(
        [layer.test_name,
         flags.get_str(prefix='_', delimiter='')])

    new_test_diff_flags = layer.test_diff_flags.get_copy()
    for flag in flags:
        new_test_diff_flags[flag] = flags[flag]

    new_flags = layer.flags.get_copy()
    for flag in flags:
        new_flags[flag] = flags[flag]

    return layer._replace(
        test_name=new_test_name,
        test_diff_flags=new_test_diff_flags,
        flags=new_flags)


def modify_layer_backendQuery(layer):
    backendQuery_flags = Flags()

    backendQuery_flags["backendQuery"] = ('', )

    return modify_layer(layer, backendQuery_flags)


def get_modified_layers(layer, super_flags):
    result = []

    for flags in super_flags.get_sub_flags():
        result.append(modify_layer(layer, flags))

    return result


from collections import OrderedDict, defaultdict


class OrderedDefaultDict(OrderedDict):
    def __init__(self, default_factory=None, *args, **kwargs):
        super(OrderedDefaultDict, self).__init__(*args, **kwargs)
        self.default_factory = default_factory


def format_time(seconds_float):
    ms_left = seconds_float * 1000

    day_left = int(int(ms_left) / (24 * 60 * 60 * 1000))
    ms_left -= day_left * (24 * 60 * 60 * 1000)

    hour_left = int(int(ms_left) / (60 * 60 * 1000))
    ms_left -= hour_left * (60 * 60 * 1000)

    min_left = int(int(ms_left) / (60 * 1000))
    ms_left -= min_left * (60 * 1000)

    sec_left = int(int(ms_left) / 1000)
    ms_left -= sec_left * (1000)

    result = ""
    if (day_left > 0):
        result += "%d days " % day_left

    if (hour_left > 0):
        result += "%d hours " % hour_left

    if (min_left > 0):
        result += "%d minutes " % min_left

    if (sec_left > 0):
        result += "%d seconds " % sec_left

    if (ms_left > 0):
        result += "%d milliseconds " % ms_left

    return result[:-1]


def set_default_dir(filename, dir_path, force=False):
    '''path join {dir_path}/{filename} if {filename} is simple basename, unless {force} is True'''
    if force or os.path.basename(filename) == filename:
        filename = os.path.join(dir_path, filename)
    return filename


def flag_to_dict(flag_str):
    '''Convert a flag string to dictionary
    {flag_str} is a '*' separated list of key-value pairs separated by ':', e.g.

    R:conv * formatIn:1 * formatOut:1 * dimA:"1,32,1,1"

    returns

        {
            'R'         : 'conv',
            'formatIn'  : '1',
            'formatOut' : '1',
            'dimA'      : ['1', '32', '1', '1']
        }

    - Does not perform type conversion to float/int, etc.
    - Does not current support direct product of flag values, e.g. R:conv,dgrad
    '''
    out = {}
    for s in flag_str.split('*'):
        s = s.strip().split(':', 1)
        if any(s):
            key, val = s
            if val.startswith('"') and val.endswith('"'):
                out[key] = val.strip('"').split(',')  # split quoted list
            else:
                out[key] = val
    return out


@contextlib.contextmanager
def stopwatch(name):
    '''stopwatch to time a block of python code

        with stopwatch('title string'):
            {python code suite}

    will print

        "title string: 0.001s"

    after running {python code suite}
    '''
    t0 = time.time()
    try:
        yield t0
    finally:
        print("{}: {:g}s".format(name, time.time() - t0))


def hashable_elem(a):
    try:
        hash(a)
    except TypeError:
        return False

    return True


def hashable_list(l):
    for a in l:
        if not hashable_elem(a):
            return False

    return True


def get_unique(l):
    if hashable_list(l):
        result = {}

        for a in l:
            result[a] = None

        return list(result.keys())

    result = []

    for a in l:
        if a not in result:
            result.append(a)

    return result


# Get src path assuming scripts/helpers is stored in default path (cudnn/scripts/helpers)
def get_default_src():
    helpers_directory = os.path.dirname(__file__)

    default_src = os.path.realpath(helpers_directory + "/../../src")

    return default_src


# eof

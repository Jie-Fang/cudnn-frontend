# This module defines the class "Flags"
# See bottom __main__() function for explanations of how to use the class.

from collections import deque
import copy

try:
    from .utility_py3 import basestring
except ImportError:
    pass


def quote_if_comma(string):
    if "," in string:
        return '"' + string + '"'
    return string


class Flags:
    def __init__(self, flags=None, key_order=None):
        if flags == None:
            self.flags_dict = {}
            self.hard_copy_flags = self.do_nothing
        else:
            self.flags_dict = flags

        if key_order == None:
            self.key_order = []
            self.hard_copy_keys = self.do_nothing
        else:
            self.key_order = key_order

    def do_nothing(self):
        pass

    # Hard copy flags should only occur if the previous flags were set by reference
    # Otherwise, calling this function should do nothing
    def hard_copy_flags(self):
        new_flags = {}

        new_flags.update(self.flags_dict)

        self.flags_dict = new_flags

        self.hard_copy_flags = self.do_nothing

    # Hard copy keys should only occur if the previous keys were set by reference
    # Otherwise, calling this function should do nothing
    def hard_copy_keys(self):
        self.key_order = self.key_order[:]

        self.hard_copy_keys = self.do_nothing

    def __setitem__(self, key, value):
        if not isinstance(key, basestring):
            raise Exception("Following key is not a string:", key)

        if not isinstance(value, tuple):
            raise Exception("Following value is not a tuple:", value)

        if len(value) == 0:
            raise Exception("Empty list is not allowed for value:", value)

        for tup_elem in value:
            if not isinstance(tup_elem, basestring):
                raise Exception("Given value contains non-string element:",
                                tup_elem)

        self.hard_copy_flags()
        self.flags_dict[key] = value

        if not (key in self.key_order):
            self.hard_copy_keys()
            self.key_order.append(key)

    def __delitem__(self, key):
        if not isinstance(key, basestring):
            raise Exception("Following key is not a string:", key)

        if key in self.flags_dict:
            self.hard_copy_keys()
            self.key_order.remove(key)

            self.hard_copy_flags()
            del self.flags_dict[key]

    def __getitem__(self, key):
        if not isinstance(key, basestring):
            raise Exception("Following key is not a string:", key)

        if key in self.flags_dict:
            return self.flags_dict[key]

        raise Exception("Unable to find key {:s} in {:s}".format(
            key, str(self.flags_dict)))

    def __iter__(self):
        return iter(self.key_order)

    def __contains__(self, key):
        return key in self.flags_dict

    def __len__(self):
        return len(self.flags_dict)

    def next_multi_flag(self):
        for key in self.key_order:
            if len(self.flags_dict[key]) > 1:
                return key

        return None

    def get_multi_keys(self):
        multi_keys = []

        for key in self.key_order:
            if len(self.flags_dict[key]) > 1:
                multi_keys.append(key)

        return multi_keys

    def key_count(self):
        return len(self.key_order)

    def is_multiple(self):
        if self.next_multi_flag():
            return True

        return False

    def get_sub_flags_count(self):
        result = 1

        for key in self.key_order:
            result *= len(self[key])

        return result

    def get_sub_flags(self):
        result_flags = []

        to_parse = deque()

        to_parse.append(self)

        while len(to_parse) > 0:
            cur_elem = to_parse.popleft()

            next_multi = cur_elem.next_multi_flag()

            if next_multi:
                base_copy = {}
                base_copy.update(cur_elem.flags_dict)
                del base_copy[next_multi]

                for flag_val in reversed(cur_elem.flags_dict[next_multi]):
                    mod_copy = {}
                    mod_copy.update(base_copy)
                    mod_copy[next_multi] = (flag_val, )

                    to_parse.appendleft(Flags(mod_copy, self.key_order))

            else:
                yield cur_elem

        return

    def get_flags_for_keys(self, keys):
        result = Flags()

        for key in self.key_order:
            if key in keys:
                result[key] = self[key]

        return result

    def get_multi_flags(self):
        result = []

        for key in self.key_order:
            if len(self.flags_dict[key]) > 1:
                result.append(key)

        return result

    def get_str(self,
                prefix='-',
                delimiter=' ',
                seperator='',
                quotify_comma=False):
        if self.next_multi_flag():
            raise Exception(
                "Cannot obtain string of key {:s} in flags: {:s}".format(
                    self.next_multi_flag(), str(self.flags_dict)))

        if quotify_comma:
            return delimiter.join([
                prefix + key + seperator + quote_if_comma(
                    self.flags_dict[key][0]) for key in self.key_order
            ])

        return delimiter.join([
            prefix + key + seperator + self.flags_dict[key][0]
            for key in self.key_order
        ])

    def get_descs_str(self):
        return self.get_str(
            prefix='', delimiter=' * ', seperator=':', quotify_comma=True)

    def get_copy(self):
        return Flags(copy.deepcopy(self.flags_dict), self.key_order[:])

    def __str__(self):
        return self.get_str()

    def __repr__(self):
        return "Flags({}, {})".format(self.flags_dict, self.key_order)

    def __add__(self, other):
        result = self.get_copy()

        for other_flag in other:
            if (other_flag in result.key_order) and (other_flag == 'minDevVer'):
                result[other_flag] = tuple( [str(max( int(result[other_flag][0]) , int(other[other_flag][0]) ))] )
            else:
                result[other_flag] = other[other_flag]

        return result

    def copy_without(self, flag_name):
        flags_dict = {flag: self[flag] for flag in self if flag != flag_name}

        key_order = [key for key in self.key_order if key != flag_name]

        return Flags(flags_dict, key_order)

    def __sub__(self, other):
        flags_dict = {flag: self[flag] for flag in self if flag not in other}

        key_order = [key for key in self.key_order if key not in other]

        return Flags(flags_dict, key_order)

    def __hash__(self):
        return hash(tuple(sorted(self.flags_dict.items())))

    def __eq__(self, other):
        return self is other or \
            (isinstance(other, Flags) and
             self.flags_dict == other.flags_dict and
             self.key_order == other.key_order)


def match_str(a, b):
    if a != b:
        print("\t[ERROR] Mismatch of {:s} and {:s}".format(a, b))
        return False

    print("\t[SUCCESS] Correct match of {:s}".format(a))
    return True


if __name__ == "__main__":

    mismatch_count = 0

    ##########################################################
    # SINGLE TEST
    ##########################################################
    print("=== SINGLE TEST===")

    flags = Flags()

    flags["R"] = ("conv", )
    flags["R"] = ("dgrad", )

    flags["n"] = ("128", )
    flags["n"] = ("64", )

    flags["c"] = ("128", )

    golden_str = "-Rdgrad -n64 -c128"

    if not match_str(str(flags), golden_str):
        mismatch_count += 1

    ##########################################################
    # MULTI TEST
    ##########################################################
    print("\n\n=== MULTI TEST===")

    flags = Flags()
    flags = Flags()

    flags["R"] = ("conv", "dgrad")
    flags["n"] = tuple([str(i) for i in [1, 2]])

    golden_strings = ["-Rconv -n1", "-Rconv -n2", "-Rdgrad -n1", "-Rdgrad -n2"]

    for (sub_flag, golden_str) in zip(flags.get_sub_flags(), golden_strings):
        if not match_str(str(sub_flag), golden_str):
            mismatch_count += 1

    ##########################################################
    # ERROR CASE: SET INT KEY
    ##########################################################
    print("\n\n=== SET INT KEY ===")

    flags = Flags()

    try:
        flags[0] = "blank"
        mismatch_count += 1
        print("[ERROR] Integer key should not be settable")
    except:
        print("[SUCCESS] Integer key is not settable")

    ##########################################################
    # ERROR CASE: SET EMPTY TUPLE
    ##########################################################
    print("\n\n=== SET EMPTY TUPLE ===")

    flags = Flags()

    try:
        flags["blank"] = ()
        mismatch_count += 1
        print("[ERROR] Empty tuple value should not be settable")
    except:
        print("[SUCCESS] Empty tuple value is not settable")

    ##########################################################
    # ERROR CASE: SET INT VALUE
    ##########################################################
    print("\n\n=== SET INT VALUE ===")

    flags = Flags()

    try:
        flags["blank"] = 0
        mismatch_count += 1
        print("[ERROR] Integer value should not be settable")
    except:
        print("[SUCCESS] Integer value is not settable")

    ##########################################################
    # ERROR CASE: SET VALUE TO TUPLE OF INTS
    ##########################################################
    print("\n\n=== SET VALUE TO TUPLE OF INTS ===")

    flags = Flags()

    try:
        flags["blank"] = (0, 1)
        mismatch_count += 1
        print("[ERROR] Tuple of ints value should not be settable")
    except:
        print("[SUCCESS] Tuple of ints value is not settable")

    ##########################################################
    # ERROR CASE: GET INT KEY
    ##########################################################
    print("\n\n=== GET INT KEY ===")

    flags = Flags()

    try:
        flags[0] = ("blank", )
        test_get = flags[0]
        mismatch_count += 1
        print("[ERROR] Int key should not be gettable")
    except:
        print("[SUCCESS] Int key is not gettable")

    print("\n\nERROR COUNT: %d" % mismatch_count)

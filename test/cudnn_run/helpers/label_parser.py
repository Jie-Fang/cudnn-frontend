import os.path
import re

from collections import namedtuple
from helpers.utility     import is_ignored_line, split_and_strip, flags_from_descs_str, set_default_dir
from helpers.Flags       import Flags
from helpers.utility_py3 import *

from sys import exc_info

label_name = re.compile('\s*<(.*?)(?::(.*))?>\s*$')
import_re  = re.compile('^\s*import +([^\s#\/]+)\s*$') # matches import xxxx

def get_label_name(line):
    name_match = label_name.match(line)

    if name_match:
        return name_match.groups()[0]

    return None

def is_label_valid(label_filters_str, filter_db):
    # Empty configs are always valid
    if label_filters_str == None:
        return True

    is_included = False
    is_excluded = False

    any_include = False

    for label_filter in split_and_strip(label_filters_str, ','):
        # Ignore empty labels
        if not label_filter:
            continue

        label_filter_is_include = not label_filter.startswith('!')

        label_filter_name = label_filter if label_filter_is_include else label_filter[1:]

        if label_filter_is_include:
            any_include = True

        if label_filter_is_include and label_filter_name in filter_db:
            is_included = True

        if not label_filter_is_include and label_filter_name in filter_db:
            is_excluded = True

    return (not is_excluded) and (not any_include or is_included)

def get_labels_from_str(label_file_content, config_db, label_file_path, imported_file_map=None, labels=None):
    '''
    Generate labels to flag map, or update existing {labels} map, if provided,
    by reading {label_file_content} and referencing flag database {config_db}.

    {label_file_content} are contents from {label_file_path}
    {imported_file_map} is a map from imported file to the file-path and line where the import line occurs.

    so suppose imported_file_map['foo.label'] = ('bar.label', 3'), this means line 3 of 'bar.label' is 'import foo.label'
    It is used to track that each label file is only imported once.
    If during processing {label_file_content}, this finds 'import foo.label', with 'foo.label' already in {imported_file_map}
    and exception is raised.
    '''
    if imported_file_map is None:
        imported_file_map = {label_file_path : None} # map from imported file path to path and line where the first import occured
    if labels is None:
        labels = {}

    labels["exclude"] = [Flags()]
    labels["exclude"][0]["*exclude"] = ("",)

    lines = label_file_content.splitlines()

    cur_label_name = None
    cur_label_valid = None

    for (line_index, line) in enumerate(lines):
        # Ignore commented or empty lines
        if is_ignored_line(line):
            continue

        try:
            # Handle import another label file; minimal guard against recursive import
            import_match = import_re.match(line)
            if import_match:
                import_label_file = import_match.group(1)
                import_label_file = os.path.realpath(set_default_dir(import_label_file, os.path.dirname(label_file_path), force=True))
                if import_label_file in imported_file_map:
                    # NOTE: this does not completely guard against recursive import, e.g symbolic link
                    if imported_file_map[import_label_file] is None:
                        raise ValueError('cannot import {}: recursive import'.format(import_label_file))
                    else:
                        path, line = imported_file_map[import_label_file]
                        raise ValueError('cannot import {}: already seen in {}:{}'.format(import_label_file, path, line+1))
                imported_file_map[import_label_file] = (label_file_path, line_index)
                with open(import_label_file) as fp:
                    get_labels_from_str(fp.read(), config_db, import_label_file, imported_file_map=imported_file_map, labels=labels)
                continue

            # Handle case of label name
            label_name_match = label_name.match(line)
            if label_name_match:
                (cur_label_name, cur_label_filters) = label_name_match.groups()

                cur_label_valid = bool(is_label_valid(cur_label_filters, config_db))

                # Error out if it already exists
                if(label_name in labels):
                    raise Exception("Label \"%s\" is already defined" % label_name)

                # Initialize label generator
                labels[cur_label_name] = []

            # Handle case of flag descriptors/imports
            else:
                if cur_label_valid:
                    config_db.append(cur_label_name)
                    for new_flag in flags_from_descs_str(line, labels):
                        labels[cur_label_name].append(new_flag)

        except Exception as e:
            # Store traceback info (to find where real error spawned)
            t, v, tb = exc_info()

            exception = Exception("[LABEL PARSING] %s (at %s:%s)" % (str(e), label_file_path, line_index+1))
            reraise(t, exception, tb)

    return labels

def get_labels_from_file(label_file_path, config_db):
    '''Generate labels-to-flag map based on labels defined in {label_file_path} and tracked by {config_db}'''
    with open(label_file_path) as label_file:
        return get_labels_from_str(label_file.read(), config_db, label_file_path)

if __name__ == "__main__":
    test_str = '''
                   <label_import>
                   a: 2

                   <label_pre:filter_one>
                   b: 2 | label_import
                   c: 2 | label_import

                   <label_post>
                   d: 2 | label_pre'''

    print(get_labels_from_str(test_str, "test_file_name"))

from collections import namedtuple
from re import compile, search
from helpers.utility import is_ignored_line, flags_from_descs_str, cross_flags, get_flags_list_intersection, OrderedDefaultDict, flags_match_a_in_b_lists
from helpers.utility_py3 import *
from helpers.Flags import Flags
from sys import exc_info
import random

case_idx_pattern = compile('^Case Index: ([0-9]+)$')

Layer = namedtuple("Layer", "base_name test_name test_diff_flags flags case_idx")
BinaryMeta = namedtuple("BinaryMeta", "main_file_offset offset_encoding_bytes \
                        length_encoding_bytes segment_count all_layer_config_count \
                        layer_flag_file_offset_start \
                        engine_file_offset_start \
                        knob_file_offset_start \
                        layer_name_file_offset_start")

def read_metadata_from_binary_file(main_file):
    """read the metadata from bianry file and return a named tuple
    """

    # reset to the beginning of the binary file
    main_file.seek(0)

    # reserve the first 64 bytes for metedate
    main_file_offset = 64

    # the number of bytes used to encode offsets, 
    # 5 bytes -> 549G
    offset_encoding_bytes = 5

    # the number of bytes used to encode the length of flag string, 
    # 2 bytes -> 32K
    length_encoding_bytes = 2

    # how many segments in the binary file
    segment_count = 4

    # read the metadata: total number of layers in the binary file
    all_layer_config_count = int.from_bytes(main_file.read(5), byteorder="little")

    # read the metadata: where the layer flag segment start
    layer_flag_file_offset_start = int.from_bytes(main_file.read(5), byteorder="little")

    # read the metadata: where the engine segment start
    engine_file_offset_start = int.from_bytes(main_file.read(5), byteorder="little")

    # read the metadata: where the knob segment start
    knob_file_offset_start = int.from_bytes(main_file.read(5), byteorder="little")

    # read the metadata: where the layer name segment start
    layer_name_file_offset_start = int.from_bytes(main_file.read(5), byteorder="little")

    return BinaryMeta(main_file_offset, offset_encoding_bytes,
                      length_encoding_bytes, segment_count, all_layer_config_count,
                      layer_flag_file_offset_start, engine_file_offset_start,
                      knob_file_offset_start, layer_name_file_offset_start)

def read_case(main_file, binary_meta, case_idx_to_read):
    """read the case specified by case_idx_to_read from binary file (main_file)
    """

    # go to the start of offset list for the current case
    main_file.seek(binary_meta.main_file_offset + \
                   (binary_meta.offset_encoding_bytes * \
                   binary_meta.segment_count * case_idx_to_read))

    # read the offsets for each segment, respectively
    current_layer_flag_file_offset = int.from_bytes(main_file.read(binary_meta.offset_encoding_bytes), byteorder="little")
    current_engine_file_offset = int.from_bytes(main_file.read(binary_meta.offset_encoding_bytes), byteorder="little")
    current_knob_file_offset = int.from_bytes(main_file.read(binary_meta.offset_encoding_bytes), byteorder="little")
    current_layer_name_file_offset = int.from_bytes(main_file.read(binary_meta.offset_encoding_bytes), byteorder="little")

    # read layer name
    # e.g. MaskRCNN_backbone_00_Rconv_Pinh_Pouth_formatIn0_filtFormat0_formatOut0_n1
    main_file.seek(binary_meta.layer_name_file_offset_start + current_layer_name_file_offset)
    bytes_to_read = int.from_bytes(main_file.read(binary_meta.length_encoding_bytes), byteorder="little")
    layer_name = main_file.read(bytes_to_read).decode("UTF-8")

    # read layer flags 
    # e.g. R:conv * A:1 * B:0 * b: * Pin:h
    main_file.seek(binary_meta.layer_flag_file_offset_start + current_layer_flag_file_offset)
    bytes_to_read = int.from_bytes(main_file.read(binary_meta.length_encoding_bytes), byteorder="little")
    layer_descs = main_file.read(bytes_to_read).decode("UTF-8").strip()

    # read engine config and append to layer_desc
    # e.g. R:conv * A:1 * B:0 * b: * Pin:h
    #   -> R:conv * A:1 * B:0 * b: * Pin:h * backendEngine0
    main_file.seek(binary_meta.engine_file_offset_start + current_engine_file_offset)
    bytes_to_read = int.from_bytes(main_file.read(binary_meta.length_encoding_bytes), byteorder="little")
    engine_descs = main_file.read(bytes_to_read).decode("UTF-8").strip()

    # read knob config and append to layer_desc
    # e.g. R:conv * A:1 * B:0 * b: * Pin:h * backendEngine0
    #   -> R:conv * A:1 * B:0 * b: * Pin:h * backendEngine0 * knobUseTex=0
    main_file.seek(binary_meta.knob_file_offset_start + current_knob_file_offset)
    bytes_to_read = int.from_bytes(main_file.read(binary_meta.length_encoding_bytes), byteorder="little")
    knob_descs = main_file.read(bytes_to_read).decode("UTF-8").strip()

    # concatenate the layer_descs from engine and knob descs
    layer_descs += (" * " + engine_descs + " * " + knob_descs)

    return (layer_name, layer_descs, engine_descs, knob_descs)

def gen_flags_from_binary_file(binary_file_name, start_group_idx,
                        global_flags_list, partition_index, 
                        partition_count, randomize):
    """generate and read the binary file case by case
    """

    # random instance used in the binary file read
    rand_instance = random.Random()

    with open(binary_file_name, "r+b") as main_file:
        # read the metadata from binary file
        binary_meta = read_metadata_from_binary_file(main_file)

        # the cases are grouped by partition count
        # e.g. partition_count = 3, cases are [0,1,2,3,4,5,6,7,8,...]
        #   -> the cases willbe grouped as [[0,1,2],[3,4,5],[6,7,8],...] 
        #      the group id will be [0,1,2,...]
        # Note: the last group may not have the all partition_count cases
        for group_idx in range(start_group_idx, 
                               (binary_meta.all_layer_config_count // partition_count) + 1):

            # the global case idx (non-randomized)
            case_idx = (group_idx * partition_count) + partition_index

            # which case should I run? 
            #   - if not randomize, run exactly the original case id in each group
            #   - if randomize, shuffle within the group, get the real case id to read
            case_idx_to_read = case_idx

            if randomize:

                # ascending list from 0 to partition_count-1
                # e.g. [0,1,2] for partition_count == 3
                idx_list = list(range(partition_count))

                # random seed is determined by the group id
                # e.g. partition_count == 3, case_idx == 7 -> seed == 2
                rand_instance.seed(group_idx)

                # shuffle the original list using the new seed
                # e.g. [0,1,2] -> [2,0,1]
                rand_instance.shuffle(idx_list)

                # get the real case id to read and run
                # e.g. partition_index == 1, partition_count == 3, case_idx == 7
                #      -> case_idx_to_read = 6 + idx_list[1] = 6 + 0 = 6
                case_idx_to_read = (group_idx * partition_count) + idx_list[partition_index]

                # ignore the non-valid case idx (last group may not have all partition_count cases)
                if case_idx_to_read >= binary_meta.all_layer_config_count:
                    continue

            # read the case from binary
            (layer_name, layer_descs, engine_descs, knob_descs) = read_case(main_file, 
                                                                            binary_meta, 
                                                                            case_idx_to_read)

            # convert descs to Flag()
            layer_descs_flags_list = flags_from_descs_str(layer_descs)
            engine_knob_descs_flags_list = flags_from_descs_str(engine_descs + "*" + knob_descs)

            if not layer_descs_flags_list:
                continue

            # apply global_flags_list to the layer_descs_flags_list
            layer_descs_flags_list = cross_flags(layer_descs_flags_list,
                                                 global_flags_list)

            if not layer_descs_flags_list:
                continue

            # make sure the flag list contains exactly one Flag()
            assert len(layer_descs_flags_list) == len(engine_knob_descs_flags_list) == 1
            yield (layer_name, engine_knob_descs_flags_list[0], layer_descs_flags_list[0], case_idx)
    
    return

def gen_layers_from_binary_file(binary_file_path, start_group_idx, 
                         global_flags_list, dup_detector,
                         partition_index, partition_count, randomize):
    """read layers from binary file
    """

    gen_layers = gen_flags_from_binary_file(binary_file_path, start_group_idx, 
                                     global_flags_list,
                                     partition_index, partition_count, randomize)


    for (layer_name, test_diff_flags, flags, case_idx) in gen_layers:
        for sub_flag in flags.get_sub_flags():

            test_name = layer_name + test_diff_flags.get_str(
                prefix='_', delimiter='')

            # Skip duplicate flags
            if(dup_detector and not dup_detector.add_flags(sub_flag)):
                continue
            
            yield Layer(layer_name, test_name, test_diff_flags, sub_flag, case_idx)
        
    return

def get_next_group_idx_from_log(log_path, partition_count):
    """read the log file reversely to get the last case idx processed
       calculate the group idx and return the next group idx to start reading,
       if no case idx is detected, reset the start group idx to be 0
    """

    with open(log_path, "r+") as log:
        for line in reverse_readline(log):
            # check if this is the case idx line
            match_line = case_idx_pattern.match(line.rstrip())

            # reset the start group idx to be the next case
            if match_line:
                return (int(match_line.groups()[0]) // partition_count) + 1
    
    # start from 0 if no processed case found
    return 0

def reverse_readline(f, chunk_size=8192):
    """reversely read file line by line and in chunks
    """

    # go to the end of the file and get the file size
    f.seek(0, 2)
    file_size = f.tell()

    # partial line, the first line in each chunk may not complete
    partial_line = None

    # special case where the file is smaller than the chunk size
    if file_size < chunk_size:
        f.seek(0)
        lines = f.read(file_size).split('\n')
        for idx in range(len(lines)-1, -1, -1):
            yield lines[idx]

    # if the file is larger than the chunk size
    else:
        for seek_end_pos in range(file_size, 0, -1*chunk_size):

            # if the remaining bytes are more than chunk size
            if (seek_end_pos - chunk_size >= 0):
                f.seek(seek_end_pos - chunk_size)
                chunk = f.read(chunk_size)
            else:
                f.seek(0)
                chunk = f.read(seek_end_pos)
            lines = chunk.split('\n')
            
            # if there is remaining part in last iteration
            if partial_line is not None:
                if chunk[-1] == '\n':
                    yield partial_line
                else:
                    lines[-1] += partial_line

            # reset the partial line
            partial_line = lines[0]

            # return line reversely without the first line
            for idx in range(len(lines)-1, 0, -1):
                yield lines[idx]
        
        # clean the remaining partial line
        if partial_line is not None:
            yield partial_line

    return

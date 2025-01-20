# -*- coding: utf-8 -*-

import shelve
import os
import re
import shlex
import subprocess
from multiprocessing import current_process
import time
import shlex
from datetime import datetime
from copy import deepcopy
from collections import namedtuple
import sqlite3
import zlib
import hashlib


BinaryMeta = namedtuple("BinaryMeta", "main_file_offset offset_encoding_bytes \
                                        length_encoding_bytes segment_count all_layer_config_count \
                                        layer_flag_file_offset_start \
                                        engine_file_offset_start \
                                        knob_file_offset_start \
                                        layer_name_file_offset_start \
                                        binary_file_size")


class BinaryWriter:
    def __init__(self, binary_path, append=False):
        self.binary_path = binary_path
        self.append = append

        # how many cases in this file
        self.all_layer_config_count = 0

        # dictionary for engines and knobs
        self.layer_flag_to_offset = {}
        self.engine_to_offset = {}
        self.knob_to_offset = {}
        self.layer_name_to_offset = {}

        # leave 64 bytes for metadata like sum, engine offset, etc.
        self.header_offset = 64

        # use 5 bytes to encode integer, enough for 550GB oddset files
        self.offset_encoding_bytes = 5
        self.length_encoding_bytes = 2
        self.segment_count = 4

        # initial offsets for each file
        self.main_file_current_offset = self.header_offset
        self.layer_flag_file_current_offset = 0
        self.engine_file_current_offset = 0
        self.knob_file_current_offset = 0
        self.layer_name_file_current_offset = 0 

        if os.path.exists(self.binary_path) and self.append:
            tmp_reader = BinaryReader(self.binary_path)
            binary_meta = tmp_reader.read_metadata_from_binary_file()

            self.offset_encoding_bytes = binary_meta.offset_encoding_bytes
            self.length_encoding_bytes = binary_meta.length_encoding_bytes
            self.segment_count = binary_meta.segment_count
            self.all_layer_config_count = binary_meta.all_layer_config_count

            self.layer_flag_file_offset_start = binary_meta.layer_flag_file_offset_start
            self.engine_file_offset_start = binary_meta.engine_file_offset_start
            self.knob_file_offset_start = binary_meta.knob_file_offset_start
            self.layer_name_file_offset_start = binary_meta.layer_name_file_offset_start
            self.binary_file_size = binary_meta.binary_file_size

            tmp_reader.close()             

    def open_segment_files(self, file_name_postpend_str=""):
        """open the segment files in the output directory
        if append is True, read all segments in exisitng file into the segment files
        """

        self.output_dir = os.path.dirname(os.path.abspath(self.binary_path))
        self.layer_flag_file_path = os.path.join(self.output_dir, "layer_flag" + file_name_postpend_str + ".binary")
        self.engine_file_path = os.path.join(self.output_dir, "engine" + file_name_postpend_str + ".binary")
        self.knob_file_path = os.path.join(self.output_dir, "knob" + file_name_postpend_str + ".binary")
        self.layer_name_file_path = os.path.join(self.output_dir, "layer_name" + file_name_postpend_str + ".binary")

        # file descriptors
        self.layer_flag_file = open(self.layer_flag_file_path, "wb")
        self.engine_file = open(self.engine_file_path, "wb")
        self.knob_file = open(self.knob_file_path, "wb")
        self.layer_name_file = open(self.layer_name_file_path, "wb")

        # if we want to append to exisitng binary file, we first read everything from it and write to each segment
        if os.path.exists(self.binary_path) and self.append:
            self.main_file = open(self.binary_path, "r+b")
            print("Found exisitng binary file at %s, size: %s" % (self.binary_path, self.all_layer_config_count))

            self.rebuild_layer_flag_to_offset_dict(self.layer_flag_file_offset_start, self.engine_file_offset_start)
            self.main_file.seek(self.layer_flag_file_offset_start)
            existing_layer_flag_file_len = self.engine_file_offset_start - self.layer_flag_file_offset_start
            self.layer_flag_file.write(self.main_file.read(existing_layer_flag_file_len))
            self.layer_flag_file_current_offset = existing_layer_flag_file_len

            self.rebuild_engine_to_offset_dict(self.engine_file_offset_start, self.knob_file_offset_start)
            self.main_file.seek(self.engine_file_offset_start)
            existing_engine_file_len = self.knob_file_offset_start - self.engine_file_offset_start
            self.engine_file.write(self.main_file.read(existing_engine_file_len))
            self.engine_file_current_offset = existing_engine_file_len

            self.rebuild_knob_to_offset_dict(self.knob_file_offset_start, self.layer_name_file_offset_start)
            self.main_file.seek(self.knob_file_offset_start)
            existing_knob_file_len = self.layer_name_file_offset_start - self.knob_file_offset_start
            self.knob_file.write(self.main_file.read(existing_knob_file_len))
            self.knob_file_current_offset = existing_knob_file_len

            self.rebuild_layer_name_to_offset_dict(self.layer_name_file_offset_start, self.binary_file_size)
            self.main_file.seek(self.layer_name_file_offset_start)
            print("self.binary_file_size: ", self.binary_file_size)
            print("self.layer_name_file_offset_start: ", self.layer_name_file_offset_start)
            existing_layer_name_file_len = self.binary_file_size - self.layer_name_file_offset_start
            self.layer_name_file.write(self.main_file.read(existing_layer_name_file_len))
            self.layer_name_file_current_offset = existing_layer_name_file_len

            # main file delete everything after the main segment and go the the end of main segment
            self.main_file_current_offset = self.layer_flag_file_offset_start
            self.main_file.seek(self.layer_flag_file_offset_start)
        else:
            # seek to the start of main part
            self.main_file = open(self.binary_path, "wb")
            self.main_file.seek(self.header_offset)

    def rebuild_layer_flag_to_offset_dict(self, start_offset, end_offset):
        """reconstruct layer_flag_to_offset dict from exisitng binary file
        """

        current_layer_flag_file_offset = 0
        while (start_offset + current_layer_flag_file_offset) < end_offset:
            self.main_file.seek(start_offset + current_layer_flag_file_offset)
            bytes_to_read = int.from_bytes(self.main_file.read(self.length_encoding_bytes), byteorder="little")
            layer_flag_str = self.main_file.read(bytes_to_read).decode("UTF-8").strip()
            self.layer_flag_to_offset[layer_flag_str] = current_layer_flag_file_offset
            current_layer_flag_file_offset += (self.length_encoding_bytes + len(layer_flag_str))

    def rebuild_engine_to_offset_dict(self, start_offset, end_offset):
        """reconstruct engine_to_offset dict from exisitng binary file
        """

        current_engine_file_offset = 0
        while (start_offset + current_engine_file_offset) < end_offset:
            self.main_file.seek(start_offset + current_engine_file_offset)
            bytes_to_read = int.from_bytes(self.main_file.read(self.length_encoding_bytes), byteorder="little")
            engine_str = self.main_file.read(bytes_to_read).decode("UTF-8").strip()
            self.engine_to_offset[engine_str] = current_engine_file_offset
            current_engine_file_offset += (self.length_encoding_bytes + len(engine_str))

    def rebuild_knob_to_offset_dict(self, start_offset, end_offset):
        """reconstruct knob_to_offset dict from exisitng binary file
        """

        current_knob_file_offset = 0
        while (start_offset + current_knob_file_offset) < end_offset:
            self.main_file.seek((start_offset + current_knob_file_offset))
            bytes_to_read = int.from_bytes(self.main_file.read(self.length_encoding_bytes), byteorder="little")
            knob_str = self.main_file.read(bytes_to_read).decode("UTF-8").strip()
            self.knob_to_offset[knob_str] = current_knob_file_offset
            current_knob_file_offset += (self.length_encoding_bytes + len(knob_str))

    def rebuild_layer_name_to_offset_dict(self, start_offset, end_offset):
        """reconstruct layer_name_to_offset dict from exisitng binary file
        """

        current_layer_name_file_offset = 0
        while (start_offset + current_layer_name_file_offset) < end_offset:
            self.main_file.seek((start_offset + current_layer_name_file_offset))
            bytes_to_read = int.from_bytes(self.main_file.read(self.length_encoding_bytes), byteorder="little")
            layer_name_str = self.main_file.read(bytes_to_read).decode("UTF-8").strip()
            self.layer_name_to_offset[layer_name_str] = current_layer_name_file_offset
            current_layer_name_file_offset += (self.length_encoding_bytes + len(layer_name_str))

    def close_segment_files(self):
        """make sure the segment files are closed
        """

        self.main_file.close()
        self.layer_flag_file.close()
        self.engine_file.close()
        self.knob_file.close()
        self.layer_name_file.close()

    def write(self, flags_per_engine, test_flags, aug_layer_name):
        """write offsets list to the main file
           write contents to each segment
           write all flags of the all engines
        """ 

        for engine in sorted(flags_per_engine):
            for knob_str in flags_per_engine[engine]:

                self.write_one_case(test_flags.copy_without("backendQuery").get_descs_str(), 
                                    "backendEngine:" + str(engine), 
                                    knob_str.copy_without("backendEngine").get_descs_str(), 
                                    aug_layer_name)

    def write_one_case(self, layer_descs, engine_descs, knob_descs, layer_name):
        """write one case to the next location
        """

        # offset initialization
        offsets = [0, 0, 0, 0]

        if layer_descs not in self.layer_flag_to_offset:
            self.layer_flag_to_offset[layer_descs] = self.layer_flag_file_current_offset
            offsets[0] = self.layer_flag_file_current_offset
            self.layer_flag_file.write(len(layer_descs).to_bytes(self.length_encoding_bytes, byteorder="little"))
            self.layer_flag_file.write(layer_descs.encode("UTF-8"))
            self.layer_flag_file_current_offset += (self.length_encoding_bytes + len(layer_descs))
        else:
            offsets[0] = self.layer_flag_to_offset[layer_descs]

        if engine_descs not in self.engine_to_offset:
            self.engine_to_offset[engine_descs] = self.engine_file_current_offset
            offsets[1] = self.engine_file_current_offset
            self.engine_file.write(len(engine_descs).to_bytes(self.length_encoding_bytes, byteorder="little"))
            self.engine_file.write(engine_descs.encode("UTF-8"))
            self.engine_file_current_offset += (self.length_encoding_bytes + len(engine_descs))
        else:
            offsets[1] = self.engine_to_offset[engine_descs]

        if knob_descs not in self.knob_to_offset:
            self.knob_to_offset[knob_descs] = self.knob_file_current_offset
            offsets[2] = self.knob_file_current_offset
            self.knob_file.write(len(knob_descs).to_bytes(self.length_encoding_bytes, byteorder="little"))
            self.knob_file.write(knob_descs.encode("UTF-8"))
            # update knob_file_current_offset to point to next one
            self.knob_file_current_offset += (self.length_encoding_bytes + len(knob_descs))
        else:
            offsets[2] = self.knob_to_offset[knob_descs]

        if layer_name not in self.layer_name_to_offset:
            self.layer_name_to_offset[layer_name] = self.layer_name_file_current_offset
            offsets[3] = self.layer_name_file_current_offset
            self.layer_name_file.write(len(layer_name).to_bytes(self.length_encoding_bytes, byteorder="little"))
            self.layer_name_file.write(layer_name.encode("UTF-8"))
            self.layer_name_file_current_offset += (self.length_encoding_bytes + len(layer_name))
        else:
            offsets[3] = self.layer_name_to_offset[layer_name]

        # write offsets to main file
        for offset in offsets:
            self.main_file.write(offset.to_bytes(self.offset_encoding_bytes, byteorder="little"))
        self.main_file_current_offset += (self.offset_encoding_bytes * len(offsets))

        self.all_layer_config_count += 1
   
    def merge_segments(self):
        """merge segment files to the main file
        """
        with open(self.binary_path, "r+b") as main_f:

            # write number of configs
            main_f.seek(0)
            main_f.write(self.all_layer_config_count.to_bytes(self.offset_encoding_bytes, byteorder="little"))
            main_f.seek(self.offset_encoding_bytes)
            main_f.write(self.main_file_current_offset.to_bytes(self.offset_encoding_bytes, byteorder="little"))
            main_f.seek(self.offset_encoding_bytes*2)
            main_f.write((self.main_file_current_offset + self.layer_flag_file_current_offset)
                            .to_bytes(self.offset_encoding_bytes, byteorder="little"))
            main_f.seek(self.offset_encoding_bytes*3)
            main_f.write((self.main_file_current_offset + self.layer_flag_file_current_offset + self.engine_file_current_offset)
                            .to_bytes(self.offset_encoding_bytes, byteorder="little"))
            main_f.seek(self.offset_encoding_bytes*4)
            main_f.write((self.main_file_current_offset + self.layer_flag_file_current_offset + self.engine_file_current_offset + self.knob_file_current_offset)
                            .to_bytes(self.offset_encoding_bytes, byteorder="little"))
            main_f.seek(self.offset_encoding_bytes*5)
            main_f.write((self.main_file_current_offset + self.layer_flag_file_current_offset + self.engine_file_current_offset + self.knob_file_current_offset + self.layer_name_file_current_offset)
                            .to_bytes(self.offset_encoding_bytes, byteorder="little"))

            # cancatenate files
            main_f.seek(self.main_file_current_offset)
            with open(self.layer_flag_file_path, "rb") as seg_f:
                for piece in self.read_in_chunks(seg_f):
                    main_f.write(piece)
            os.remove(self.layer_flag_file_path)

            main_f.seek(self.main_file_current_offset + self.layer_flag_file_current_offset)
            with open(self.engine_file_path, "rb") as seg_f:
                for piece in self.read_in_chunks(seg_f):
                    main_f.write(piece)
            os.remove(self.engine_file_path)

            main_f.seek(self.main_file_current_offset + self.layer_flag_file_current_offset + self.engine_file_current_offset)
            with open(self.knob_file_path, "rb") as seg_f:
                for piece in self.read_in_chunks(seg_f):
                    main_f.write(piece)
            os.remove(self.knob_file_path)

            main_f.seek(self.main_file_current_offset + self.layer_flag_file_current_offset + self.engine_file_current_offset + self.knob_file_current_offset)
            with open(self.layer_name_file_path, "rb") as seg_f:
                for piece in self.read_in_chunks(seg_f):
                    main_f.write(piece)
            os.remove(self.layer_name_file_path)
    
    def read_in_chunks(self, f, buffer_size=1024):
        """read large file in chunks, pass the file descriptor
        """

        while True:
            buffer = f.read(buffer_size)
            if not buffer:
                break
            yield buffer
        return

    def get_size(self):
        return self.all_layer_config_count

class BinaryReader:
    def __init__(self, binary_file_path):
        self.main_file_path = binary_file_path
        self.main_file = open(self.main_file_path, "r+b")

    def read_case(self, binary_meta, case_idx_to_read):
        """read the case specified by case_idx_to_read from binary file (main_file)
        """

        # go to the start of offset list for the current case
        self.main_file.seek(binary_meta.main_file_offset + \
                    (binary_meta.offset_encoding_bytes * \
                    binary_meta.segment_count * case_idx_to_read))

        # read the offsets for each segment, respectively
        current_layer_flag_file_offset = int.from_bytes(self.main_file.read(binary_meta.offset_encoding_bytes), byteorder="little")
        current_engine_file_offset = int.from_bytes(self.main_file.read(binary_meta.offset_encoding_bytes), byteorder="little")
        current_knob_file_offset = int.from_bytes(self.main_file.read(binary_meta.offset_encoding_bytes), byteorder="little")
        current_layer_name_file_offset = int.from_bytes(self.main_file.read(binary_meta.offset_encoding_bytes), byteorder="little")

        # read layer name
        # e.g. MaskRCNN_backbone_00_Rconv_Pinh_Pouth_formatIn0_filtFormat0_formatOut0_n1
        self.main_file.seek(binary_meta.layer_name_file_offset_start + current_layer_name_file_offset)
        bytes_to_read = int.from_bytes(self.main_file.read(binary_meta.length_encoding_bytes), byteorder="little")
        layer_name = self.main_file.read(bytes_to_read).decode("UTF-8")

        # read layer flags 
        # e.g. R:conv * A:1 * B:0 * b: * Pin:h
        self.main_file.seek(binary_meta.layer_flag_file_offset_start + current_layer_flag_file_offset)
        bytes_to_read = int.from_bytes(self.main_file.read(binary_meta.length_encoding_bytes), byteorder="little")
        layer_descs = self.main_file.read(bytes_to_read).decode("UTF-8").strip()

        # read engine config and append to layer_desc
        # e.g. R:conv * A:1 * B:0 * b: * Pin:h
        #   -> R:conv * A:1 * B:0 * b: * Pin:h * backendEngine0
        self.main_file.seek(binary_meta.engine_file_offset_start + current_engine_file_offset)
        bytes_to_read = int.from_bytes(self.main_file.read(binary_meta.length_encoding_bytes), byteorder="little")
        engine_descs = self.main_file.read(bytes_to_read).decode("UTF-8").strip()

        # read knob config and append to layer_desc
        # e.g. R:conv * A:1 * B:0 * b: * Pin:h * backendEngine0
        #   -> R:conv * A:1 * B:0 * b: * Pin:h * backendEngine0 * knobUseTex=0
        self.main_file.seek(binary_meta.knob_file_offset_start + current_knob_file_offset)
        bytes_to_read = int.from_bytes(self.main_file.read(binary_meta.length_encoding_bytes), byteorder="little")
        knob_descs = self.main_file.read(bytes_to_read).decode("UTF-8").strip()

        return (layer_name, layer_descs, engine_descs, knob_descs)

    def read_metadata_from_binary_file(self):
        """read the metadata from bianry file and return a named tuple
        """

        # reset to the beginning of the binary file
        self.main_file.seek(0)

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
        all_layer_config_count = int.from_bytes(self.main_file.read(offset_encoding_bytes), byteorder="little")

        # read the metadata: where the layer flag segment start
        layer_flag_file_offset_start = int.from_bytes(self.main_file.read(offset_encoding_bytes), byteorder="little")

        # read the metadata: where the engine segment start
        engine_file_offset_start = int.from_bytes(self.main_file.read(offset_encoding_bytes), byteorder="little")

        # read the metadata: where the knob segment start
        knob_file_offset_start = int.from_bytes(self.main_file.read(offset_encoding_bytes), byteorder="little")

        # read the metadata: where the layer name segment start
        layer_name_file_offset_start = int.from_bytes(self.main_file.read(offset_encoding_bytes), byteorder="little")

        # read the metadata: the length of the complete file
        binary_file_size = int.from_bytes(self.main_file.read(offset_encoding_bytes), byteorder="little")

        return BinaryMeta(main_file_offset, offset_encoding_bytes,
                        length_encoding_bytes, segment_count, all_layer_config_count,
                        layer_flag_file_offset_start, engine_file_offset_start,
                        knob_file_offset_start, layer_name_file_offset_start,
                        binary_file_size)

    def close(self):
        self.main_file.close()

class ApiDbWriter():
    def __init__(self, api_db_path):
        self.api_db_path = os.path.abspath(api_db_path)

    def get_api_db_key_val_from_backendquery(self, test_flags, config_str, api_reverse_idx_dict_from_query):
        """get the complete flag string (the key of api db)
           and a list of graph capture strings (the value of api db)
           a complete flag string is (test_flags + engine_config_str)
           a list of graph capture strings is [api_0, api_1, ...]
        """

        # remove "backendQuery" from test_flag
        layer_flag_str = test_flags.copy_without(
            "backendQuery").get_descs_str()

        # seperate engine_config_str and graph_capture_str
        engine_config_str = config_str.split("|")[0]

        # default graph_capture_str_list is empty
        graph_capture_str_list = []
        if len(config_str.split("|")) >= 2:
            graph_capture_str_list = config_str.split("|")[1].split(";")


        # key of shelve has to be string
        api_db_key = ",".join(["[%s]" % str(self.indexer.get_key_index(layer_flag_str))] +
                              self.indexer.get_key_val_idx_list_from_flag_str(engine_config_str.strip(),
                                                                              item_seperator="*",
                                                                              key_val_seperator=":="))


        api_db_val = self.indexer.get_idx_tuple_from_graph_capture_str(
            graph_capture_str_list, api_reverse_idx_dict_from_query)

        return (api_db_key, api_db_val)

    def write(self, test_flags, config_str, api_reverse_idx_dict_from_query):
        """for a given sequence of test flags, write to api db in format:
           (test_flags + engine_config_str) -> [api_0, api_1, ...]
        """

        # get the key and val to write
        api_db_key, api_db_val = self.get_api_db_key_val_from_backendquery(
            test_flags, config_str, api_reverse_idx_dict_from_query)

        # Note: tuple cannot be used as keys in shelve
        # self.current_db.add({"flag_idx": api_db_key, "api_list": ",".join(list(api_db_val))})
        # self.current_db[api_db_key] = api_db_val

        self.to_insert_key_vals[api_db_key] = ",".join(list(api_db_val))

        return api_db_key, len(api_db_val)

    def get_flag_indexer(self):
        return self.flag_indexer

    def get_api_indexer(self):
        return self.api_indexer

    def get_indexer(self):
        return self.indexer

    def get_api_db_val_by_key(self, key):
        # return self.current_db[key]
        val = self.current_db.get(["api_list"], [Conditional("flag_idx", "=", key)])
        if val == None:
            return None
        else:
            return tuple(val.split(","))

    def open_and_load_indexers(self, db_mode='n', indexer=None):
        """open db for writing and load indexers
        """

        # write pending insert cases to mem first
        self.to_insert_key_vals = {}

        self.current_db_idx_table = PeresistentSQLDB(self.api_db_path, "idx_table",
                       elem_names=["idx", "val"],
                       type_names=["INTEGER PRIMARY KEY", "TEXT"], load_to_mem=False, mode=db_mode) 
        
        # use exisitng indexer if it is there
        if self.current_db_idx_table.get_size() > 0:
            self.indexer = FlagApiIndexer()
            key_to_index = {}
            index_to_key = {}
            for row in self.current_db_idx_table.get_gen(["idx", "val"]):
                key_to_index[row["val"]] = int(row["idx"])
                index_to_key[int(row["idx"])] = row["val"]
            self.indexer.load(key_to_index, index_to_key)
        else:

            # if there is no indexer in current db and no external indexer is given, create one
            if indexer == None:
                self.indexer = FlagApiIndexer()

            # if there is no indexer in current db but an external indexer is given, use external one
            else:
                # use json for deep copy thread safety
                self.indexer = deepcopy(indexer)
        
        # don't open multiple tables from same db at the same time
        self.current_db_idx_table.close()

        self.current_db = PeresistentSQLDB(self.api_db_path, "ConvData",
                                elem_names=["flag_idx", "api_list"],
                                type_names=["TEXT PRIMARY KEY", "TEXT"], load_to_mem=False, mode=db_mode)

    def close_and_save_indexers(self):

        start_time = time.time()
        print("[%s] start writing to api db" % (self.api_db_path))
        for key in self.to_insert_key_vals:
            self.current_db.add({"flag_idx": key, "api_list": self.to_insert_key_vals[key]})
        print("[%s] [%s cases] api db write time: %s sec" % (self.api_db_path, len(
            self.to_insert_key_vals.keys()), time.time() - start_time))
        # self.current_db["#indexer"] = self.indexer
        self.current_db.close()

        self.current_db_idx_table = PeresistentSQLDB(self.api_db_path, "idx_table",
                       elem_names=["idx", "val"],
                       type_names=["INTEGER PRIMARY KEY", "TEXT"], load_to_mem=False, mode='c')
        key_to_index_dict = self.indexer.get_key_to_index_dict()
        for key in key_to_index_dict:
            self.current_db_idx_table.replace({"idx": key_to_index_dict[key], "val": key})

        self.current_db_idx_table.close()

    def close(self):
        self.current_db.close()

    def print_all_keys(self):
        # for key in self.current_db:
        #     print(key)
        for row in self.current_db.get_gen(["api_list"]):
            print(row["api_list"])

    def get_db_size(self):
        return self.current_db.estimate_size()

class UniqueKeyIndexer():
    """assign each string in this indexer a unique idx
    used to replace long repeated strings with a single number
    """

    def __init__(self):
        self.key_to_index = {}
        self.index_to_key = {}

    def get_key_index(self, key):
        if key not in self.key_to_index:
            # print("indexing new key: %s" % (key))
            self.key_to_index[key] = len(self.key_to_index)
            self.index_to_key[self.key_to_index[key]] = key
        return self.key_to_index[key]

    def get_key_by_index(self, idx):
        return self.index_to_key[idx]

    def get_key_to_index_dict(self):
        return self.key_to_index

    def get_indexer_size(self):
        return len(self.key_to_index.keys())

    def load(self, key_to_index, index_to_key):
        self.key_to_index = key_to_index
        self.index_to_key = index_to_key

class FlagApiIndexer(UniqueKeyIndexer):
    """track and assign unqiue idx to test flags and eninge config strings
    """

    def __init__(self):
        UniqueKeyIndexer.__init__(self)

    def get_key_val_idx_list_from_flag_str(self, input_str, item_seperator, key_val_seperator):
        """the input_str has key value pairs seperated by `item_seperator`
        the key val pair is seperated by `key_val_seperator`
        will return a tuple in format (key1_idx, val1, key2_idx, val2, ...) sorted by key's idx
        """

        key_val_list = []
        for key_val in input_str.split(item_seperator):
            key_val = key_val.strip()
            splitted_key_val = key_val.split(key_val_seperator)
            if len(splitted_key_val) == 1:
                key_idx = self.get_key_index(splitted_key_val[0])
                key_val_list.append("[%s]" % str(key_idx))
                # key_val_list.append("?")
            elif len(splitted_key_val) == 2:
                key_idx = self.get_key_index(splitted_key_val[0])
                val = splitted_key_val[1]
                key_val_list.append("[%s]" % str(key_idx))
                key_val_list.append(val)
            else:
                raise RuntimeError("key value not seperable")
        
        return key_val_list

    def recover_from_flag_idx_str(self, idx_str, item_seperator, key_val_seperator):

        pattern_idx = re.compile(r'^\[(\d+)\]$')
        flag_list = []
        idx_list = idx_str.split(",")

        current_idx = 0
        while current_idx < len(idx_list):
            if pattern_idx.match(idx_list[current_idx]):
                index = int(pattern_idx.match(
                    idx_list[current_idx]).groups()[0])
                flag_list.append(self.get_key_by_index(index))
            else:
                if len(flag_list) > 0:
                    flag_list[-1] += (key_val_seperator + idx_list[current_idx])
                else:
                    raise RuntimeError("no key found for val %d at pos %s of flag_idx %s" % (
                        idx_list[current_idx], current_idx, idx_str))
            current_idx += 1

        return item_seperator.join(flag_list)

    def get_idx_tuple_from_graph_capture_str(self, graph_capture_str_list, api_reverse_idx_dict_from_query):
        # regex to match for kernel info
        pattern_kernel_and_memset = re.compile(
            r'^(\d+),(\d+)\((.*?)\)$')
        # pattern_memset = re.compile(
        #     r'^CudaGraphNodeMemset\((\w*),(\w*),(\w*),(\w*),\)$')

        api_idx_tuple = ()

        for graph_capture_str in graph_capture_str_list:

            if len(graph_capture_str) == 0:
                continue

            # if this is a kernel api invoke or memset
            if pattern_kernel_and_memset.match(graph_capture_str):

                # extract mangled kernel name and arguments
                api_idx, api_repeat_count, api_args = pattern_kernel_and_memset.match(graph_capture_str).groups()
                api_name = api_reverse_idx_dict_from_query[api_idx]
                # print("api_name: ", api_name)
                api_idx_tuple += ("[%d]" % self.get_key_index(api_name),)
                api_idx_tuple += (api_repeat_count,)
                for arg in api_args.split(","):
                    if arg != "":
                        api_idx_tuple += (arg.strip("()"),)

            else:
                raise RuntimeError("graph capture api string not parsable %s" % graph_capture_str)

        return api_idx_tuple

    def recover_from_idx_tuple(self, idx_tuple):

        pattern_idx = re.compile(r'^\[(\d+)\]$')
        api_list = []

        current_idx = 0
        while current_idx < len(idx_tuple):

            # if this is an index
            if pattern_idx.match(idx_tuple[current_idx]):
                index = int(pattern_idx.match(idx_tuple[current_idx]).groups()[0])
                api_list.append(self.get_key_by_index(index))
            else:
                api_list.append(idx_tuple[current_idx])
            current_idx += 1

        return api_list

    def get_kernel_name_list_from_idx_tuple(self, idx_tuple):

        pattern_idx = re.compile(r'^\[(\d+)\]$')
        kernel_name_list = []

        current_idx = 0
        while current_idx < len(idx_tuple):

            # if this is an index
            if pattern_idx.match(idx_tuple[current_idx]):
                index = int(pattern_idx.match(idx_tuple[current_idx]).groups()[0])
                if self.get_key_by_index(index) != "CudaGraphNodeMemset":
                    kernel_name_list.append(self.get_key_by_index(index))
            current_idx += 1
        
        return kernel_name_list

class FlagIndexer(UniqueKeyIndexer):
    """track and assign unqiue idx to test flags and eninge config strings
    """

    def __init__(self):
        UniqueKeyIndexer.__init__(self)

    def get_key_val_idx_list_from_str(self, input_str, item_seperator, key_val_seperator):
        """the input_str has key value pairs seperated by `item_seperator`
        the key val pair is seperated by `key_val_seperator`
        will return a tuple in format (key1_idx, val1, key2_idx, val2, ...) sorted by key's idx
        """

        key_val_list = []
        for key_val in input_str.split(item_seperator):
            splitted_key_val = key_val.split(key_val_seperator)
            if len(splitted_key_val) == 1:
                key_idx = self.get_key_index(splitted_key_val[0])
                key_val_list.append(str(key_idx))
                key_val_list.append("?")
            elif len(splitted_key_val) == 2:
                key_idx = self.get_key_index(splitted_key_val[0])
                val = splitted_key_val[1]
                key_val_list.append(str(key_idx))
                key_val_list.append(val)
            else:
                raise RuntimeError("key value not seperable")
        
        return key_val_list

    def recover_from_idx_str(self, idx_str, item_seperator, key_val_seperator):
        flag_list = []
        idx_list = idx_str.split(",")

        # make sure the length is even, in key val pairs
        assert len(idx_list)%2 == 0
        for i in range(0, len(idx_list), 2):
            tmp_str = self.get_key_by_index(int(idx_list[i]))
            if idx_list[i+1] != "?":
                tmp_str += (key_val_seperator + idx_list[i+1])
            flag_list.append(tmp_str)
        
        return item_seperator.join(flag_list)

class ApiIndexer(UniqueKeyIndexer):
    """track and assign unqiue idx to api mangled name and api arguments
    """

    def __init__(self):
        UniqueKeyIndexer.__init__(self)

    def get_idx_tuple_from_graph_capture_str(self, graph_capture_str_list, api_reverse_idx_dict_from_query):
        # regex to match for kernel info
        pattern_kernel_and_memset = re.compile(
            r'^(\d+),(\d+)\((.*?)\)$')

        api_idx_tuple = ()

        for graph_capture_str in graph_capture_str_list:

            if len(graph_capture_str) == 0:
                continue

            # if this is a kernel api invokeor memset
            if pattern_kernel_and_memset.match(graph_capture_str):

                # extract mangled kernel name and arguments
                api_idx, api_repeat_count, api_args = pattern_kernel_and_memset.match(graph_capture_str).groups()
                api_name = api_reverse_idx_dict_from_query[api_idx]
                # print("api_name: ", api_name)
                api_idx_tuple += (self.get_key_index(api_name),)
                api_idx_tuple += (int(api_repeat_count),)
                for arg in api_args.split(","):
                    if arg != "":
                        api_idx_tuple += (int(arg.strip("()")),)

            else:
                raise RuntimeError("graph capture api string not parsable %s" % graph_capture_str)

        return api_idx_tuple

    def recover_from_idx_str(self, idx_tuple):
        api_list = []

        current_idx = 0
        while current_idx < len(idx_tuple):

            # if this is Memset
            if self.get_key_by_index(idx_tuple[current_idx]) == "CudaGraphNodeMemset":
                repeat_count = idx_tuple[current_idx + 1]
                tmp_str = "CudaGraphNodeMemset,%d(%d,%d,%d,%d)" % (
                    repeat_count, idx_tuple[current_idx + 2], idx_tuple[current_idx + 3], idx_tuple[current_idx + 4], idx_tuple[current_idx + 5])
                current_idx += 5

            # if this is a kernel
            else:
                kernel_name = self.get_key_by_index(idx_tuple[current_idx])
                repeat_count = idx_tuple[current_idx + 1]
                tmp_str = "%s,%d<<<(%d,%d,%d),(%d,%d,%d),%d,%d>>>" % (kernel_name, repeat_count, idx_tuple[current_idx + 2], idx_tuple[current_idx + 3], idx_tuple[current_idx + 4],
                                                                      idx_tuple[current_idx + 5], idx_tuple[current_idx + 6], idx_tuple[current_idx + 7], idx_tuple[current_idx + 8], idx_tuple[current_idx + 9])
                current_idx += 9

            api_list.append(tmp_str)
            current_idx += 1

        return api_list

    def get_kernel_name_list_from_idx_str(self, idx_tuple):
        kernel_name_list = []

        current_idx = 0
        while current_idx < len(idx_tuple):

            # if this is Memset
            if self.get_key_by_index(idx_tuple[current_idx]) == "CudaGraphNodeMemset":
                current_idx += 5
            
            # if this is a kernel
            else:
                kernel_name_list.append(self.get_key_by_index(idx_tuple[current_idx]))
                current_idx += 9
            current_idx += 1
        
        return kernel_name_list

class Conditional:
    def __init__(self, name, comparator, value):
        self.name = name
        self.comparator = comparator
        self.value = value
    
    def get_str(self):

        # to use `in`, `value` should be a list
        if self.comparator == "in":
            return '"%s"%s%s' % (self.name, self.comparator, "(%s)" % (",".join(["?" for item in self.value])))
        return '"%s"%s?' % (self.name, self.comparator)
        
    def get_value(self):
        if self.value == "NULL":
            return None
            
        return self.value
    
def format_name(name):
    return '"%s"' % name
    
def unformat_name(name):
    return name[1:-1]
   
def get_array_as_str(array):
    return "\n".join([str(row) for row in array])
    

class PeresistentSQLDB:
    def __init__(self, file_name, table_name, elem_names=None, type_names=None, load_to_mem=False, mode='c', journal=None, pragma=None):
        """mode n: Always create a new, empty database, open for reading and writing
           mode c: Open database for reading and writing, creating it if it does not exist
           mode r: Open existing database for reading only
           mode w: Open existing database for reading and writing
        """

        self.table_name = table_name
        
        if elem_names != None:
            if type_names == None:
                raise Exception("[PeresistentSQLDB] Must provide type names if provided elem names")
                
            # Store formatted element names for SQL
            self.elem_names = [format_name(elem_name) for elem_name in elem_names]
            
            # Store type names
            self.type_names = type_names
            
            # Map for element to index in insertion
            self.elem_idx_map  = {}
            
            # Fill all maps for all elements
            for elem_name in self.elem_names:
                self.elem_idx_map[elem_name] = len(self.elem_idx_map)
                    
            # Get count of all elements
            self.elem_count    = len(self.elem_names)
            
            # Adding requires a string of '?' for each element
            self.qmarks_str = ",".join(['?'] * self.elem_count)
        
        self.file_name = file_name

        self.load_to_mem = load_to_mem

        # # Connect to DB
        # self.connect_db()

        # check file existance, set write permission, create table is necessary
        if mode == 'n':
            self.is_writeable = True
            if os.path.exists(file_name):
                os.remove(file_name)
            self.connect_db()
            self.create_table()
        elif mode == 'c':
            self.is_writeable = True
            if not os.path.exists(file_name):
                self.connect_db()
                self.create_table()
            else:
                self.connect_db()
                if not self.is_table_exist():
                    self.create_table()
        elif mode == 'r':
            self.is_writeable = False
            if not os.path.exists(file_name):
                raise Exception("[PeresistentSQLDB] db not exist: %s" % (file_name))
            else:
                self.connect_db()
                if not self.is_table_exist():
                    print("[PeresistentSQLDB] table \'%s\' not exists in db: %s" % (self.table_name, file_name))
        elif mode == 'w':
            self.is_writeable = True
            if not os.path.exists(file_name):
                raise Exception("[PeresistentSQLDB] db not exist: %s" % (file_name))
            else:
                self.connect_db()
                if not self.is_table_exist():
                    self.create_table()
        else:
            raise Exception("[PeresistentSQLDB] mode \'%s\' not supported" % (mode))

        if not self.is_table_exist():
            print("[PeresistentSQLDB] table \'%s\' not exists in db: %s" % (self.table_name, file_name))

        if elem_names == None:
            # the table exists, if user doesn't give element neames, get them from query
            self.cursor.execute("PRAGMA table_info(%s);" % (self.table_name))
            data = self.cursor.fetchall()

            self.elem_names = [format_name(elem[1]) for elem in data]
            self.type_names = [elem[2] for elem in data]

        # Map for element to index in insertion
        self.elem_idx_map  = {}

        # Fill all maps for all elements
        for elem_name in self.elem_names:
            self.elem_idx_map[elem_name] = len(self.elem_idx_map)

        # Get count of all elements
        self.elem_count    = len(self.elem_names)

        # Adding requires a string of '?' for each element
        self.qmarks_str = ",".join(['?'] * self.elem_count)

        # database synchronization is not used
        # https://sqlite.org/pragma.html
        self.pragma = pragma
        if self.pragma != None and self.pragma == "OFF":
            self.cursor.execute("PRAGMA synchronous = OFF;")

        # https://www.sqlite.org/wal.html
        self.journal = journal
        if self.journal != None:
            self.cursor.execute("PRAGMA journal_mode = %s;" % (self.journal))

    def enable_autocommit(self):
        self.conn.isolation_level = None

    def commit(self):
        self.conn.commit()

    def query_only(self):
        self.cursor.execute("PRAGMA query_only=1;")

    def read_uncommitted(self):
        self.cursor.execute("PRAGMA read_uncommitted=1;")

    def set_temp_store(self, val):
        self.cursor.execute("PRAGMA temp_store=%s;" % (val))
    
    def set_mmap_size(self, val):
        self.cursor.execute("PRAGMA mmap_size=%s;" % (val))
    
    def set_page_size(self, val):
        self.cursor.execute("PRAGMA page_size=%s;" % (val))

    def set_concurrent_read_only(self):
        self.query_only()
        self.read_uncommitted()
        self.set_temp_store("memory")
        self.set_page_size(32768)

    def begin_transaction(self):
        self.cursor.execute("BEGIN;")

    def is_table_exist(self):
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        table_list = [name_tuple[0] for name_tuple in self.cursor.fetchall()]
        return self.table_name in table_list

    def get_elem_names(self):
        if self.is_table_exist():
            return [s[1:-1] for s in self.elem_names]

    def get_type_names(self):
        if self.is_table_exist():
            return self.type_names
        
    def delete_db_file(self):
        if not self.is_writeable:
            raise Exception("[PeresistentSQLDB] Unable to call \"delete\" on non-writeable PeresistentSQLDB")
            
        if os.path.exists(self.file_name):
            os.remove(self.file_name)
            
    def reset(self):
        if not self.is_writeable:
            raise Exception("[PeresistentSQLDB] Unable to call \"reset\" on non-writeable PeresistentSQLDB")
            
        # Close connections
        self.close()

        # Delete old file
        self.delete_db_file()
            
        # Connect to new DB
        self.connect_db()

        # Create necessary table
        self.create_table()

    def connect_db(self):
        # load to mem for fast processing
        if self.load_to_mem:
            disk_conn = sqlite3.connect(self.file_name)
            self.conn = sqlite3.connect(':memory:')
            disk_conn.backup(self.conn)
            disk_conn.close()
        else:

            # Create/Load database
            self.conn = sqlite3.connect(self.file_name)

        # Create cursor within database
        self.cursor = self.conn.cursor()

    def reset_table(self):
        """delete content from existing table
        """
        if not self.is_writeable:
            raise Exception("[PeresistentSQLDB] Unable to call \"reset_table\" on non-writeable PeresistentSQLDB")
        print("[PeresistentSQLDB] reseting table %s for %s" % (self.table_name, self.file_name))
        self.cursor.execute("DELETE FROM %s;" % (self.table_name))
        
    def create_table(self):
        if not self.is_writeable:
            raise Exception("[PeresistentSQLDB] Unable to call \"create_table\" on non-writeable PeresistentSQLDB")
            
        # Create table string: "groupCount, srcDesc[dimA_0], error_msg"
        table_str = ", ".join(["%s %s" % (elem, type_name) for (elem, type_name) in zip(self.elem_names, self.type_names)])

        # Create table
        print("[PeresistentSQLDB] creating new table %s for %s" % (self.table_name, self.file_name))
        self.cursor.execute("CREATE TABLE %s (%s)" % (self.table_name, table_str))

    def delete(self, conditionals):
        # Initialize conditionals to empty (indicating no conditionals)
        cond_str = ""
        cond_values = []
        
        # If not none/empty, set conditional strings/values
        if conditionals != None and len(conditionals) > 0:
            # Create "WHERE "groupCount"    >=20 AND " string
            cond_str = "WHERE %s" % " AND ".join([cond.get_str() for cond in conditionals])
            
            # Get all conditional values
            cond_values = [cond.get_value() for cond in conditionals]
            
        query_str = "DELETE FROM %s %s" % (self.table_name, cond_str)
        
        self.cursor.execute(query_str, cond_values)

    def add(self, dict_of_elems):
        if not self.is_writeable:
            raise Exception("[PeresistentSQLDB] Unable to call \"add\" on non-writeable PeresistentSQLDB")
            
        # Any unspecified value is assumed to be None
        sqlite_elems = [None] * self.elem_count
        
        # Loop through all names provided as input
        for elem_name in dict_of_elems:
            # Check if element name exists in maps
            if format_name(elem_name) not in self.elem_idx_map:
                raise Exception("[SQLITE_INTERFACE] Unable to find element \"%s\"" % elem_name)
                
            # Get index of element (indicates where to store value in array)
            elem_idx = self.elem_idx_map[format_name(elem_name)]
        
            # Store formatted value in sqlite elems
            sqlite_elems[elem_idx] = dict_of_elems[elem_name]
        
        # Insert sqlite elems into database
        self.cursor.execute("INSERT OR REPLACE INTO %s VALUES (%s)" % (self.table_name, self.qmarks_str), tuple(sqlite_elems))

    def replace(self, dict_of_elems):
        if not self.is_writeable:
            raise Exception("[PeresistentSQLDB] Unable to call \"add\" on non-writeable PeresistentSQLDB")
            
        # Any unspecified value is assumed to be None
        sqlite_elems = [None] * self.elem_count
        
        # Loop through all names provided as input
        for elem_name in dict_of_elems:
            # Check if element name exists in maps
            if format_name(elem_name) not in self.elem_idx_map:
                raise Exception("[SQLITE_INTERFACE] Unable to find element \"%s\"" % elem_name)
                
            # Get index of element (indicates where to store value in array)
            elem_idx = self.elem_idx_map[format_name(elem_name)]
        
            # Store formatted value in sqlite elems
            sqlite_elems[elem_idx] = dict_of_elems[elem_name]
        
        # Insert sqlite elems into database
        self.cursor.execute("REPLACE INTO %s VALUES (%s)" % (self.table_name, self.qmarks_str), tuple(sqlite_elems))
    
    def __str__(self):
        return get_array_as_str(self.get())

    def exists(self, col_to_check, val_to_check):
        if col_to_check == None or val_to_check == None:
            raise Exception("[PeresistentSQLDB] no `col_to_check` or `val_to_check` provided for exists check")
        col_to_check = format_name(col_to_check)
        val_to_check = format_name(val_to_check)
        query_str = "SELECT EXISTS(SELECT 1 FROM %s WHERE %s = %s);" % (self.table_name, col_to_check, val_to_check)
        
        return_val = self.cursor.execute(query_str).fetchone()[0]

        if return_val == 1:
            return True
        else:
            return False

    def insert_from_db_file_name(self, other_db_file_name):
        self.conn.execute("ATTACH '%s' as dba" % other_db_file_name)

        self.conn.execute("BEGIN")

        self.conn.execute("INSERT INTO %s SELECT * FROM dba.%s" % (self.table_name, self.table_name))

        self.conn.commit()

        self.conn.execute("detach database dba")
        
    def get(self, retrieve_names=None, conditionals=None, order_by=None):
        # If none, set retrieve names to all names
        if retrieve_names == None:
            retrieve_names = self.elem_names
        else:
            retrieve_names = [format_name(elem_name) for elem_name in retrieve_names]
            
        # Create ""groupCount","destDesc[dimA_0]"" string
        retrieve_name_str = ",".join(retrieve_names)
        
        # Initialize conditionals to empty (indicating no conditionals)
        cond_str = ""
        cond_values = []
        
        # If not none/empty, set conditional strings/values
        if conditionals != None and len(conditionals) > 0:
            # Create "WHERE "groupCount"    >=20 AND " string
            cond_str = "WHERE %s" % " AND ".join([cond.get_str() for cond in conditionals])
            
            # Get all conditional values
            cond_values = [cond.get_value() for cond in conditionals]
            
        # Default to blank order by string
        order_str = ""
        
        if order_by:
            order_str = " ORDER BY %s" % format_name(order_by)
            
        query_str = "SELECT %s FROM %s %s%s" % (retrieve_name_str, self.table_name, cond_str, order_str)
        
        rows = self.cursor.execute(query_str, cond_values)
        
        column_names = [rows.description[idx][0] for idx in range(len(rows.description))]

        for column_name in column_names:
            if(column_name[0] == '"' and column_name[-1] == '"'):
                raise ValueError
        
        all_results = []
        
        for row in rows:
            result = {}

            for idx, elem_name in enumerate(retrieve_names):
                result[unformat_name(elem_name)] = row[idx]

            all_results.append(result)
            
        if len(all_results) == 0:
            return None
        
        return all_results

    def get_gen(self, retrieve_names=None, conditionals=None, order_by=None):
        # If none, set retrieve names to all names
        if retrieve_names == None:
            retrieve_names = self.elem_names
        else:
            retrieve_names = [format_name(elem_name) for elem_name in retrieve_names]

        # Create ""groupCount","destDesc[dimA_0]"" string
        retrieve_name_str = ",".join(retrieve_names)

        # Initialize conditionals to empty (indicating no conditionals)
        cond_str = ""
        cond_values = []

        # If not none/empty, set conditional strings/values
        if conditionals != None and len(conditionals) > 0:
            # Create "WHERE "groupCount"    >=20 AND " string
            cond_str = "WHERE %s" % " AND ".join([cond.get_str() for cond in conditionals])

            # Get all conditional values
            cond_values = [cond.get_value() for cond in conditionals]

        # Default to blank order by string
        order_str = ""

        if order_by:
            order_str = " ORDER BY %s" % ",".join(format_name(order_by_elem) for order_by_elem in order_by)

        query_str = "SELECT %s FROM %s %s%s" % (retrieve_name_str, self.table_name, cond_str, order_str)
        
        rows = self.cursor.execute(query_str, cond_values)

        for row in rows:
            result = {}

            for idx, elem_name in enumerate(retrieve_names):
                result[unformat_name(elem_name)] = row[idx]

            yield result

    def get_gen_in(self, retrieve_names, cond_col_name, cond_col_list):
        # If none, set retrieve names to all names
        retrieve_names = [format_name(elem_name) for elem_name in retrieve_names]

        # Create ""groupCount","destDesc[dimA_0]"" string
        retrieve_name_str = ",".join(retrieve_names)

        query_str = "SELECT %s FROM %s WHERE %s in (%s)" % (retrieve_name_str, self.table_name, cond_col_name, ",".join(cond_col_list))
        
        rows = self.cursor.execute(query_str)

        for row in rows:
            result = {}

            for idx, elem_name in enumerate(retrieve_names):
                result[unformat_name(elem_name)] = row[idx]

            yield result

    def get_size(self):
        tmp_size = self.cursor.execute("SELECT COUNT(%s) FROM %s;" % (self.elem_names[0], self.table_name)).fetchone()[0]
        if tmp_size == None:
            return 0
        else:
            return tmp_size

    def estimate_size(self):
        """estimate the size of the given table
        if there were no DELETE operations performed before, this estimation is accurate
        if there were DELETE operations performed before, some of the rows might be unavailable, so this is an estimation
        but this is FAST, much faster than COUNT operation
        """

        tmp_size = self.cursor.execute("SELECT MAX(_ROWID_) FROM %s LIMIT 1;" % (self.table_name)).fetchone()[0]
        if tmp_size == None:
            return 0
        else:
            return tmp_size
        
    def __enter__(self):
        return self
        
    def close(self):
        if self.load_to_mem:
            if os.path.exists(self.file_name):
                os.remove(self.file_name)
            disk_conn = sqlite3.connect(self.file_name)
            self.conn.backup(disk_conn)
            disk_conn.commit()
            disk_conn.close()
        self.conn.commit()
        self.conn.close()
        
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

def dump_so_to_cubin(cudnn_so_dir, cuda_bin_dir, cubin_save_dir, api_indexer=None):
    """main function to dump .so files to databases
    1. extract cubin files from .so
    2. dump cubin files and convert to dict
    3. diff the newly generated db with old one, create a dict with items only available in new
    """

    # record start time
    start_time = time.time()

    if api_indexer == None:
        api_indexer = ApiIndexer()

    # the pattern to match all cudnn*.so
    pattern_cudnn_so = re.compile('^libcudnn_[\w]*.so$')

    # dict for .so file nmae -> db file name
    dict_so_to_db = {}

    cudnn_so_file_list = []


    arch_cubin_dict = {}

    # get a dict of arch -> list of cubin files extracted from libcudnn*.so files
    # also perform the extraction, the extracted cubin files are located in same dir as the .so file
    for cudnn_so_file in os.listdir(cudnn_so_dir):

        if pattern_cudnn_so.match(cudnn_so_file):

            cudnn_so_file_list.append(
                os.path.join(cudnn_so_dir, cudnn_so_file))

            arch_cubin_dict = merge_two_dict(arch_cubin_dict,
                                             cuobjdump_extract_cubin_from_so(os.path.join(cudnn_so_dir, cudnn_so_file),
                                             cubin_save_dir,
                                                                             cuda_bin_dir + "/cuobjdump",
                                                                             "-xelf all"))

    print("Architectures dumped:")
    print(arch_cubin_dict.keys())

    return arch_cubin_dict

def merge_sassdb_mp(cubin_list):
    """merge the sassdbs of given architecture to a songle sassdb
    """

    p = current_process()
    pattern_arch = re.compile(r'.*.(sm_\d+).cubin')
    assert len(cubin_list) > 0
    arch = pattern_arch.match(cubin_list[0]).groups()[0]

    now = datetime.now()
    dateTimeStr = now.strftime("%m-%d-%Y_%H-%M")
    combined_sass_db_path = os.path.abspath("./sass_" + arch + "." + dateTimeStr + ".sqlite3")
    if os.path.exists(combined_sass_db_path):
        os.remove(combined_sass_db_path)
    count = 0
    conn = sqlite3.connect(combined_sass_db_path)
    c = conn.cursor()

    combined_sass_db = PeresistentSQLDB(combined_sass_db_path, "sass_table", 
                             elem_names=["fun_name", "sass_content"],
                             type_names=["TEXT PRIMARY KEY", "TEXT"], load_to_mem=False, mode='n')

    for db_name in cubin_list:
        count += 1
        db_name = db_name[:-5] + "sqlite3"

        # open partition sass db
        to_merge_db = PeresistentSQLDB(db_name, "sass_table", 
                            elem_names=["fun_name", "sass_content"],
                            type_names=["TEXT PRIMARY KEY", "TEXT"], load_to_mem=False, mode='r')

        for row in to_merge_db.get_gen(["fun_name", "sass_content"]):
            combined_sass_db.add({"fun_name": row["fun_name"], "sass_content": row["sass_content"]})
        to_merge_db.close()

        if count % 10 == 0:
            print("(%s)%s merged %s/%s" % (p._identity[0], arch, count, len(cubin_list)))

        if os.path.exists(db_name):
            os.remove(db_name)

    combined_sass_db.close()

    return combined_sass_db_path

def diff_sassdb_mp(arch, new_sassdb_arch_dict, old_sassdb_arch_dict):
    """find the `new` kernels in sassdb and create a sassdb with new cases only
    `new` means: 1. key change; 2. value change; 3. new key; 4. `CAL`ed kernel changed
    """

    # handle CAL instruction: callee -> caller dict
    # while diffing, if `CAL` instruction exists, add caller's key to caller dict
    # after diffing, for each callee, recursively add caller to diff if callee changed
    # e.g., a -> b -> c, a,b not changed, but c changed, the dict will be {b:[a], c:[b]}
    # the kernels needs to be updated will be [a,b]
    callee_to_caller_dict = {}
    pattern_CAL = re.compile(r'^.*CAL `\(\$(.*?)\).*$')

    diff_sassdb_path = os.path.join(os.path.dirname(
        new_sassdb_arch_dict[arch]), "diff_" + os.path.basename(new_sassdb_arch_dict[arch]).split(".")[0] + ".shelve")

    with explicit_shelve(new_sassdb_arch_dict[arch], "gnu", mode='r') as new_db_shelve:
        new_db = {}
        for key in new_db_shelve:
            new_db[key] = new_db_shelve[key]
        with explicit_shelve(diff_sassdb_path, "gnu", mode='n') as diff_db:
            # new arch added, not in old dbs
            # copy new db tp diff db
            if arch not in old_sassdb_arch_dict:
                for key in new_db:
                    diff_db[key] = None

            # if arch exists both in new and old dbs
            else:
                with explicit_shelve(old_sassdb_arch_dict[arch], "gnu", mode='r') as old_db_shelve:
                    old_db = {}
                    for key in old_db_shelve:
                        old_db[key] = old_db_shelve[key]

                    print("(%s) Diffing: %s" % (current_process()._identity[0], new_sassdb_arch_dict[arch]))
                    print("(%s) new db size: %s" % (current_process()._identity[0], len(new_db)))
                    print("(%s) diff db: %s" % (current_process()._identity[0], diff_sassdb_path))
                    print("(%s) old db: %s" % (current_process()._identity[0], old_sassdb_arch_dict[arch]))

                    count = 0
                    for key in new_db:
                        count += 1
                        if count % 100 == 0:
                            print("(%s) %d/%d" % (current_process()._identity[0], count, len(new_db)))

                        # check new key and key change
                        # if this is a new kernel or kernel signature changed, add directly to diff
                        if key not in old_db:
                            print("Found new key: %s" % (key))
                            diff_db[key] = None

                        # if key exists in both old and new, check for content
                        else:

                            # add to diff if content has different length
                            if len(new_db[key]) != len(old_db[key]):
                                print("Found key with diff length: %s" % (key))
                                diff_db[key] = None
                            else:
                                for line_idx, cubin_line in enumerate(new_db[key]):

                                    # if the contetn does not match, add the diff and break
                                    if old_db[key] != new_db[key]:
                                        print("Found key with diff content: %s" % (key))
                                        diff_db[key] = None
                                        break
                                    else:
                                        if pattern_CAL.match(cubin_line):
                                            callee = pattern_CAL.match(cubin_line).groups()[0]
                                            # print("Found CAL key: %s -> %s" % (key, callee))
                                            if callee not in callee_to_caller_dict:
                                                callee_to_caller_dict[callee] = []
                                            callee_to_caller_dict[callee].append(key)
            # handle CAL instruction
            for callee in callee_to_caller_dict:

                # if callee has been changed, get all affected kernels and add to diff
                if callee in diff_db:
                    for kernel_to_update in resolve_call_chain(callee, callee_to_caller_dict, []):
                        print("Found resolved call chain keys: %s" % (kernel_to_update))
                        if kernel_to_update not in diff_db:
                            diff_db[kernel_to_update] = None
    
    return diff_sassdb_path

def resolve_call_chain(callee, callee_to_caller_dict, visited):
    """callee is the kernel that has been changed
    given the `callee_to_caller_dict`, return all kernels that should be updated
    """

    visited.append(callee)
    update_list = []
    if callee in callee_to_caller_dict:
        for caller in callee_to_caller_dict[callee]:

            # if there is a cycle, stop and check next
            if caller in visited:
                continue
            update_list.append(caller)
            update_list.extend(resolve_call_chain(caller, callee_to_caller_dict, visited))
    return update_list

def parse_cubin_disasm_result_to_db(output, db_path):
    """given the output of nvdisasm from a cubin file
    parse the result to kernel_name -> kernel_content and write to db in `db_path`
    kernel_content is a dict section_name -> section_content
    section_content includes `.nv.info`, `.nv.constant`, `.text`
    """

    # remove exisitng db
    if os.path.exists(db_path):
        os.remove(db_path)

    # pattern to match header line like
    # `//--------------------- .text._ZN4xmma16setScaleInternalEP6__halfi --------------------------``
    pattern_section_head_line = re.compile(r'^//[\-]+ (.*)\.(\w+) [\-]+$')
    pattern_CAL = re.compile(r'^.*CAL `\(\$(.*?)\).*$')

    current_db = PeresistentSQLDB(db_path, "sass_table", 
                       elem_names=["fun_name", "sass_content"],
                       type_names=["TEXT PRIMARY KEY", "TEXT"], load_to_mem=False, mode='n')

    # write all contents to a in-mem dict
    tmp_dict = {}
    section_content = []
    section_name = ""
    fun_name = ""
    is_inside_section = False
    for stdoutLine in output.split("\n"):
        stdoutLine = stdoutLine.strip()

        # if this is a section start line
        if "//---------------------" in stdoutLine:

            # sort the section content if the previous section is not `text`
            if is_inside_section:
                if "text" not in section_name:
                    section_content.sort()
                tmp_dict[fun_name] += section_content
            is_inside_section = False

            # only extract sections realted to a function
            if pattern_section_head_line.match(stdoutLine):
                section_name = pattern_section_head_line.match(stdoutLine).group(1)
                fun_name = pattern_section_head_line.match(stdoutLine).group(2)
                # print("section : %s\nkernel :%s" % (section_name, fun_name))
                if fun_name not in tmp_dict:
                    tmp_dict[fun_name] = []
                if not is_inside_section:
                    is_inside_section = True
                    section_content = []

        if is_inside_section:
            section_content.append(stdoutLine)

    # copy to shelve
    for fun_name in tmp_dict:
        section_content = tmp_dict[fun_name]
        has_CAL = False
        for line in section_content:
            if pattern_CAL.match(line):
                has_CAL = True
                print("Found \'CAL\' line, compression ignored")
                break
        
        # only compress non_CAL kernels
        if has_CAL:
            current_db.add({"fun_name": fun_name, "sass_content": "\n".join(section_content)})
        else:
            section_content = "\n".join(section_content)
            compressed_val = zlib.compress(section_content.encode('utf-8'))
            current_db.add({"fun_name": fun_name, "sass_content": compressed_val})

    current_db.close()

def cuobjdump_extract_cubin_from_so(file_to_dump, cubin_save_dir, cuobjdump_path, arg_str):
    """use `cuobjdump -xelf` to extract all cubin files from a .so file
    aseemble these cubin file names into different list based on the architecture
    """
    
    arch_cubin_dict = {}
    lib_dir = os.path.abspath(os.path.dirname(file_to_dump))

    # regex to extract architecture
    pattern_arch = re.compile(r'^Extracting ELF file[ ]+[\d]+: (.*(sm_\d+).cubin)$')

    cmd = []
    cmd += [cuobjdump_path]
    cmd += shlex.split(arg_str)
    cmd += [file_to_dump]
    print(cmd)
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)

    for stdoutLine in iter(process.stdout.readline, b''):
        # line is a binary literal, should be decoded to str
        stdoutLine = stdoutLine.decode('utf-8').strip()

        if pattern_arch.match(stdoutLine):
            if len(pattern_arch.match(stdoutLine).groups()) == 2:
                cubin_name = pattern_arch.match(stdoutLine).groups()[0]
                arch = pattern_arch.match(stdoutLine).groups()[1]
                if arch not in arch_cubin_dict:
                    arch_cubin_dict[arch] = [os.path.join(cubin_save_dir, cubin_name)]
                else:
                    arch_cubin_dict[arch].append(os.path.join(cubin_save_dir, cubin_name))

    process.communicate()

    return arch_cubin_dict

def merge_two_dict(dict1, dict2):
    """merge dict1 to dict2, for duplicated keys, merge associated list
    """

    for key1 in dict1:
        if key1 in dict2:
            dict2[key1].extend(dict1[key1])
        else:
            dict2[key1] = dict1[key1]
    return dict2

def remove_existing_shelve_db(db_path):
    """remove all three artifacts of shelve db
    """
    db_path = os.path.abspath(db_path)
    for extention in ["", ".bak", ".dat", ".dir"]:
        if os.path.exists(db_path + extention):
            # print("Deleting: ", db_path + extentionq)
            os.remove(db_path + extention)

def map_partition(partition_count, target_script_path, flag_iter, args_to_iter, common_args_list, unique_str="", args_seperator=",", cwd=None):
    """launch `partition_count` instaces of `target_script_path`
    `args_to_iter` will be partitioned and dispatched to each script instance's flag `flag_iter`
    `common_args_list` will be directly dispatched to each script
    """

    # record start time
    start_time = time.time()

    if cwd != None:
        print("[%s] chaning directory to %s" % (os.path.dirname(target_script_path), cwd))
        os.chdir(cwd)

    # open a tmp file to store thr arguemnts
    # the iter args could be too large to be given directly on command line, e.g., 10000 will fail

    partition_list = []
    for partition_idx in range(partition_count):
        tmp_args_file_path = os.path.join(os.path.dirname(target_script_path), "tmp_args_" + unique_str + str(partition_idx) + ".txt")
        with open(tmp_args_file_path, "w") as tmp_file:
            p_cubin_list = []
            for i in range(partition_idx, len(args_to_iter), partition_count):
                p_cubin_list.append(args_to_iter[i])
            # print(args_seperator.join(p_cubin_list))
            tmp_file.write(args_seperator.join(p_cubin_list))
        
        cmd = ["/usr/bin/python3", "-u"]
        cmd += [target_script_path]
        cmd += [flag_iter]
        cmd += [tmp_args_file_path]
        cmd += common_args_list

        process = subprocess.Popen(cmd)
        partition_list.append((partition_idx, process,))
        
    for partition_idx, process in partition_list:
        process.wait()
        print("Done: ", partition_idx)
        tmp_args_file_path = os.path.join(os.path.dirname(target_script_path), "tmp_args_" + str(partition_idx) + ".txt")
        if os.path.exists(tmp_args_file_path):
            os.remove(tmp_args_file_path)

    print("[%s] [%s] map_partition (%s partitions) total time: %s seconds" % (target_script_path, print_used_mem(), partition_count, time.time() - start_time))

def explicit_shelve(db_path, backend, writeback=False, mode='r'):
    """original shelve db does not support explicitly set backend dbm system
    this wrapper enables specifying the backend
    """

    if backend == "gnu":
        import dbm.gnu
        db = dbm.gnu.open(db_path, mode)
    elif backend == "ndbm":
        import dbm.ndbm
        db = dbm.ndbm.open(db_path, mode)
    elif backend == "dumb":
        import dbm.dumb
        db = dbm.dumb.open(db_path, mode)
    else:
        raise RuntimeError("backend should be one of these: gnu, ndbm, dumb")

    return shelve.Shelf(db, writeback=writeback)

def merge_two_api_db(api_db_paths):
    """given two api dbs, merge to one (including indexers)
    """

    assert len(api_db_paths) == 2
    db_path_1 = api_db_paths[0]
    db_path_2 = api_db_paths[1]

    print("Merging %s and %s" % (db_path_1, db_path_2))

    # tracking idx change in db_2
    flag_old_idx_to_new_idx = {}
    api_old_idx_to_new_idx = {}

    with explicit_shelve(db_path_1, "gnu", mode="r") as db_1_shelve:

        # load db_1 to memory
        start_time = time.time()
        db_1 = {}
        for key in db_1_shelve:
            db_1[key] = db_1_shelve[key]
        print("loaded db to memory: %s, time: %s seconds" % (db_path_1, (time.time() - start_time)))

        if "#flag_indexer" in db_1:
            old_flag_indexer_1 = db_1["#flag_indexer"]
        else:
            raise RuntimeError("no flag_indexer in %s" % db_path_1)
        if "#api_indexer" in db_1:
            old_api_indexer_1 = db_1["#api_indexer"]
        else:
            raise RuntimeError("no api_indexer in %s" % db_path_1)

        with explicit_shelve(db_path_2, "gnu", mode="r") as db_2_shelve:

            # load db_2 to memory
            start_time = time.time()
            db_2 = {}
            for key in db_2_shelve:
                db_2[key] = db_2_shelve[key]
            print("loaded db to memory: %s, time: %s seconds" % (db_path_2, (time.time() - start_time)))

            if "#flag_indexer" in db_2:
                old_flag_indexer_2 = db_2["#flag_indexer"]
            else:
                raise RuntimeError("no flag_indexer in %s" % db_path_2)
            if "#api_indexer" in db_2:
                old_api_indexer_2 = db_2["#api_indexer"]
            else:
                raise RuntimeError("no api_indexer in %s" % db_path_2)

            flag_idx_time = 0
            api_idx_time = 0
            dict_write_time = 0
            key_count = 0
            db_2_len = len(db_2.keys()) -2
            # sweep, build old_idx_to_new_idx, and replace idx
            for flag in db_2:
                if flag == "#flag_indexer" or flag == "#api_indexer":
                    continue

                key_count += 1
                if (key_count % 1000 == 0):
                    print("traversed %d/%d" % (key_count, db_2_len))

                start_time = time.time()
                # old flag idx -> new flag idx
                flag_idx_list = flag.split(",")
                assert len(flag_idx_list) % 2 == 0
                for i in range(0, len(flag_idx_list), 2):
                    flag_str = old_flag_indexer_2.get_key_by_index(int(flag_idx_list[i]))
                    flag_idx_list[i] = str(old_flag_indexer_1.get_key_index(flag_str))
                new_flag_idx = ",".join(flag_idx_list)
                flag_idx_time += (time.time() - start_time)

                start_time = time.time()
                # old api idx -> new api idx
                idx_tuple = db_2[flag]
                idx_list = list(idx_tuple)
                current_idx = 0
                while current_idx < len(idx_list):

                    # if this is Memset
                    if old_api_indexer_2.get_key_by_index(idx_tuple[current_idx]) == "CudaGraphNodeMemset":
                        idx_list[current_idx] = old_api_indexer_1.get_key_index("CudaGraphNodeMemset")
                        current_idx += 5

                    # if this is a kernel
                    else:
                        api_str = old_api_indexer_2.get_key_by_index(idx_tuple[current_idx])
                        idx_list[current_idx] = old_api_indexer_1.get_key_index(api_str)
                        current_idx += 9
                    current_idx += 1
                api_idx_time += (time.time() - start_time)
                
                start_time = time.time()
                # write new item to db_1
                db_1[new_flag_idx] = tuple(idx_list)
                dict_write_time += (time.time() - start_time)

            db_1["#flag_indexer"] = old_flag_indexer_1
            db_1["#api_indexer"] = old_api_indexer_1
    
    start_time = time.time()
    with explicit_shelve(db_path_1, "gnu", mode="n") as db_1_shelve:
        for key in db_1:
            db_1_shelve[key] = db_1
    print("save db_1 to shelve: %s, time: %s seconds" % (db_path_1, (time.time() - start_time)))

    print("%s: flag_idx_time: %s" % (db_path_1, flag_idx_time))
    print("%s: api_idx_time: %s" % (db_path_1, api_idx_time))
    print("%s: dict_write_time: %s" % (db_path_1, dict_write_time))
    
    # verify_api_db_merge(db_path_1, db_path_2)
    remove_existing_shelve_db(db_path_2)

def verify_api_db_merge(db_path_1, db_path_2):
    """verify the merging is correct
    """

    with explicit_shelve(db_path_1, "gnu", mode="r") as db_1:
        with explicit_shelve(db_path_2, "gnu", mode="r") as db_2:
            db_1_key_list = {}
            old_flag_indexer_1 = db_1["#flag_indexer"]
            old_api_indexer_1 = db_1["#api_indexer"]
            for key in db_1:
                if key == "#flag_indexer" or key == "#api_indexer":
                    continue
                db_1_key_list[old_flag_indexer_1.recover_from_idx_str(key, ":", ",")] = ",".join(old_api_indexer_1.recover_from_idx_str(db_1[key]))

            db_2_key_list = {}
            old_flag_indexer_2 = db_2["#flag_indexer"]
            old_api_indexer_2 = db_2["#api_indexer"]
            for key in db_2:
                if key == "#flag_indexer" or key == "#api_indexer":
                    continue
                db_2_key_list[old_flag_indexer_2.recover_from_idx_str(key, ":", ",")] = ",".join(old_api_indexer_2.recover_from_idx_str(db_2[key]))

            for key in db_2_key_list:
                if key not in db_1_key_list:
                    print("%s not in db 1!" % key)
                if db_1_key_list[key] != db_2_key_list[key]:
                    print("diff val found")

def merge_two_binary(binary_paths):
    """given two api dbs, merge to one (including indexers)
    """

    assert len(binary_paths) == 2
    binary_path_1 = binary_paths[0]
    binary_path_2 = binary_paths[1]

    print("Merging %s and %s" % (binary_path_1, binary_path_2))

    tmp_binary_file_path = os.path.join(os.path.dirname(binary_path_1), "tmp_" + os.path.basename(binary_path_1))
    binary_writer = BinaryWriter(tmp_binary_file_path)
    binary_writer.open_segment_files(file_name_postpend_str=os.path.basename(binary_path_1).split(".")[0])

    # write everything in first binary file into the merged one
    binary_reader_1 = BinaryReader(binary_path_1)
    binary_meta_1 = binary_reader_1.read_metadata_from_binary_file()
    # print("binary_meta_1.all_layer_config_count: ", binary_meta_1.all_layer_config_count)
    for case_idx in range(binary_meta_1.all_layer_config_count):
        (layer_name, layer_descs, engine_descs, knob_descs) = binary_reader_1.read_case(binary_meta_1, case_idx)
        binary_writer.write_one_case(layer_descs, engine_descs, knob_descs, layer_name)
    binary_reader_1.close()
    if os.path.exists(binary_path_1):
        os.remove(binary_path_1)

    # write everything in second binary file into the merged one
    binary_reader_2 = BinaryReader(binary_path_2)
    binary_meta_2 = binary_reader_2.read_metadata_from_binary_file()
    # print("binary_meta_2.all_layer_config_count: ", binary_meta_2.all_layer_config_count)
    for case_idx in range(binary_meta_2.all_layer_config_count):
        (layer_name, layer_descs, engine_descs, knob_descs) = binary_reader_2.read_case(binary_meta_2, case_idx)
        binary_writer.write_one_case(layer_descs, engine_descs, knob_descs, layer_name)
    binary_reader_2.close()
    if os.path.exists(binary_path_2):
        os.remove(binary_path_2)

    # close the segment files
    binary_writer.close_segment_files()

    # merge the segment files
    binary_writer.merge_segments()

    # rename final merged file to the name of the first binary file
    os.rename(tmp_binary_file_path, binary_path_1)

    binary_reader = BinaryReader(binary_path_1)
    binary_meta = binary_reader.read_metadata_from_binary_file()
    print("merged.all_layer_config_count: ", binary_meta.all_layer_config_count)
    binary_reader.close()

def validate_binary_file(new_api_db_path, new_binary_path):
    """check if all cases writen in the binary file is in the api_db
    """

    print("Validating bnary file...")
    new_api_db_flag_list = []
    with explicit_shelve(new_api_db_path, "gnu", mode="r") as new_api_db:
        flag_indexer = new_api_db["#flag_indexer"]
        for key in new_api_db:
            if key == "#flag_indexer" or key == "#api_indexer":
                continue
            flag_list = sorted([flag.strip() for flag in flag_indexer.recover_from_idx_str(key, "*", ":=").split("*") if flag.strip() != ""])
            new_api_db_flag_list.append("".join(flag_list))

    binary_flag_list = []
    binary_reader = BinaryReader(new_binary_path)
    binary_meta = binary_reader.read_metadata_from_binary_file()
    for case_idx in range(binary_meta.all_layer_config_count):
        (layer_name, layer_descs, engine_descs, knob_descs) = binary_reader.read_case(binary_meta, case_idx)
        flag_list = sorted([flag.strip() for flag in ("*".join([layer_descs, engine_descs, knob_descs])).split("*") if flag.strip() != ""])
        binary_flag_list.append("".join(flag_list))
        # print(flag_list)
    binary_reader.close()

    found_invalid_key = False
    for flag in binary_flag_list:
        if flag not in new_api_db_flag_list:
            print("%s not in new_api_db_flag_list" % (flag))
            found_invalid_key = True

    if found_invalid_key:
        print("Failed validation")
        return False
    else:
        print("Pass validation")
        return True

def create_persistent_db(persistent_db_path):
    conn = sqlite3.connect(persistent_db_path)
    c = conn.cursor()

    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_list = [name_tuple[0] for name_tuple in c.fetchall()]
    print("all tables in current persistent db %s: %s" % (persistent_db_path, table_list))

    # create persistent db if not exists
    if "ConvData" not in table_list:
        print("creating persistent table 'ConvData' in %s" % (persistent_db_path))
        c.execute("""CREATE TABLE IF NOT EXISTS ConvData( 
            'flag_idx' TEXT, 
            'flag_idx.main_flag' INTEGER DEFAULT -1,
            'api_list' TEXT, 
            'update_time' TEXT, 
            'last_cl' TEXT,
            'processing.line' INTEGER, 
            'processing.flags' TEXT, 
            'gpu.sm' INTEGER, 
            'gpu.cap' INTEGER, 
            'gpu.clock' INTEGER, 
            'gpu.mem' INTEGER, 
            'test_flags' TEXT, 
            'layer_name' TEXT, 
            'unique_flags' TEXT, 
            'srcDesc.dimA_0' INTEGER, 
            'srcDesc.dimA_1' INTEGER, 
            'srcDesc.dimA_2' INTEGER, 
            'srcDesc.dimA_3' INTEGER, 
            'srcDesc.dimA_4' INTEGER, 
            'srcDesc.strideA_0' INTEGER, 
            'srcDesc.strideA_1' INTEGER, 
            'srcDesc.strideA_2' INTEGER, 
            'srcDesc.strideA_3' INTEGER, 
            'srcDesc.strideA_4' INTEGER, 
            'srcDesc.vect' INTEGER, 
            'srcDesc.packed' INTEGER, 
            'srcDesc.dataType' TEXT, 
            'filterDesc.dimA_0' INTEGER, 
            'filterDesc.dimA_1' INTEGER, 
            'filterDesc.dimA_2' INTEGER, 
            'filterDesc.dimA_3' INTEGER, 
            'filterDesc.dimA_4' INTEGER, 
            'filterDesc.format' INTEGER, 
            'destDesc.dimA_0' INTEGER, 
            'destDesc.dimA_1' INTEGER, 
            'destDesc.dimA_2' INTEGER, 
            'destDesc.dimA_3' INTEGER, 
            'destDesc.dimA_4' INTEGER, 
            'destDesc.strideA_0' INTEGER, 
            'destDesc.strideA_1' INTEGER, 
            'destDesc.strideA_2' INTEGER, 
            'destDesc.strideA_3' INTEGER, 
            'destDesc.strideA_4' INTEGER, 
            'destDesc.vect' INTEGER, 
            'destDesc.packed' INTEGER, 
            'destDesc.dataType' TEXT, 
            'convDesc.padA_0' INTEGER, 
            'convDesc.padA_1' INTEGER, 
            'convDesc.padA_2' INTEGER, 
            'convDesc.strideA_0' INTEGER, 
            'convDesc.strideA_1' INTEGER, 
            'convDesc.strideA_2' INTEGER, 
            'convDesc.dilationA_0' INTEGER, 
            'convDesc.dilationA_1' INTEGER, 
            'convDesc.dilationA_2' INTEGER, 
            'convDesc.mode' TEXT, 
            'convDesc.math' TEXT, 
            'convDesc.groupCount' INTEGER, 
            'convDesc.dataType' TEXT, 
            'algo.chosen_by' TEXT, 
            'algo.choice' INTEGER, 
            'workspace_size' INTEGER, 
            'query' TEXT, 
            'tune.gpu_id' INTEGER, 
            'tune.opset_id' INTEGER, 
            'tune.engine_id' INTEGER, 
            'tune.opset_params' TEXT, 
            'tune.knob_choices' TEXT, 
            'median_time' REAL, 
            'sol_gflops' REAL, 
            'mem_per_sec' REAL, 
            'time' REAL, 
            'max_abs_err.RelErr' REAL, 
            'max_abs_err.RelThreshold' REAL, 
            'max_abs_err.AbsErr' REAL, 
            'max_abs_err.DeadRegion' REAL, 
            'passed' TEXT, 
            'err_msg' TEXT, 
            'status.line' INTEGER, 
            'status.status' TEXT, 
            'stable_times' TEXT, 
            'stable_percent' REAL, 
            'conditional_stable_percent' REAL, 
            'execution_times' TEXT, 
            'launch_times' TEXT, 
            'runtime_error' TEXT, 
            'error' TEXT, 
            'time_stats.total' REAL, 
            'time_stats.total_execution' REAL, 
            'time_stats.total_launch' REAL, 
            'time_stats.sample_count' REAL, 
            'test_UID' TEXT );""")
        c.execute("CREATE INDEX IF NOT EXISTS flag_idx_idx ON ConvData(flag_idx);")
        c.execute("CREATE INDEX IF NOT EXISTS aug_layer_name ON ConvData(layer_name);")
        c.execute("CREATE INDEX IF NOT EXISTS main_flag_idx ON ConvData('flag_idx.main_flag');")
        c.execute("CREATE INDEX IF NOT EXISTS opset_params_idx ON ConvData('tune.opset_params');")
        
        conn.commit()
    conn.close()

def create_persistent_db_partition(persistent_db_path):
    hex_chars = [str(i) for i in range(10)] + [chr(i) for i in range(ord('a'), ord('f')+1)]
    for first_hex_char in hex_chars:
        for second_hex_char in hex_chars:
            hash_code = first_hex_char + second_hex_char
            persistent_db_partition_path = ".".join(persistent_db_path.split(".")[0:-1]) + "_" + hash_code + ".sqlite3"
            create_persistent_db(persistent_db_partition_path)

def delete_row_from_persistent_db(persistent_db_path, to_delete_map):
    """given the idx of those rows to be deleted from persistent db
    these rows are either removed or to be updated
    delete the rows from the persistent db
    """

    create_persistent_db(persistent_db_path)

    conn = sqlite3.connect(persistent_db_path)
    c = conn.cursor()

    print("deleteing rows in persistent db %s" % (persistent_db_path))

    deleted_to_update_count = 0
    deleted_to_delete_count = 0
    to_ignore_count = 0

    persistent_db_row_count = 0

    for aug_layer_name in to_delete_map.keys():
        print("[%s] querying aug_layer_name: %s" % (persistent_db_path, aug_layer_name))

        # traverse all cases in presistent db from this layer_name
        for row in gen_rows_for_delete(c, "ConvData", ["flag_idx"], "layer_name = \'%s\'" % (aug_layer_name)):
            persistent_db_row_count += 1
            persistent_db_flag_idx = row["flag_idx"]
            is_found = False
            for api_db_key, dirty_bit in to_delete_map[aug_layer_name]:

                # if there is a match, meaning it is either "to_update" or "to_ignore" case
                if persistent_db_flag_idx == api_db_key:

                    # delete the row in persitent db if it "to_update" case
                    if dirty_bit == "1":
                        print("to be updated: ", persistent_db_flag_idx)
                        deleted_to_update_count += 1

                        # this case is to be updated in following sweep -> delete
                        c.execute("DELETE FROM ConvData WHERE flag_idx = '%s';" % (persistent_db_flag_idx))
                    else:
                        to_ignore_count += 1

                    is_found = True
                    break

            # this is a "to_delete" case
            if not is_found:
                print("only in persistent: ", persistent_db_flag_idx)
                deleted_to_delete_count += 1

                # only exists in persistent db, not in new layer -> delete
                c.execute("DELETE FROM ConvData WHERE flag_idx = '%s';" % (persistent_db_flag_idx))

    print("[%s] deleted \'to_updated\' cases: %s" % (persistent_db_path, deleted_to_update_count))
    print("[%s] deleted \'to_delete\' cases: %s" % (persistent_db_path, deleted_to_delete_count))
    print("[%s] \'to_ignore\' cases: %s" % (persistent_db_path, to_ignore_count))
    print("[%s] persistent_db queried: %s" % (persistent_db_path, persistent_db_row_count))
    
    conn.commit()
    conn.close()

def delete_row_from_persistent_db_partition(persistent_db_path, to_delete_map):
    """given the idx of those rows to be deleted from persistent db
    these rows are either removed or to be updated
    delete the rows from the persistent db
    """

    create_persistent_db_partition(persistent_db_path)

    print("deleteing rows in persistent db %s" % (persistent_db_path))

    deleted_to_update_count = 0
    deleted_to_delete_count = 0
    to_ignore_count = 0

    persistent_db_row_count = 0

    aug_layer_name_count = 0
    aug_layer_name_len = len(to_delete_map.keys())
    for aug_layer_name in to_delete_map.keys():
        start_time = time.time()

        first_two_hash_code = hashlib.md5(aug_layer_name.encode('utf-8')).hexdigest()[0:2]
        persistent_db_partition_path = ".".join(persistent_db_path.split(".")[0:-1]) + "_" + first_two_hash_code + ".sqlite3"
        print("[%s/%s][%s] querying aug_layer_name: %s" % (aug_layer_name_count, aug_layer_name_len, 
                                                           persistent_db_partition_path, aug_layer_name))
        persistent_db_partition = PeresistentSQLDB(persistent_db_partition_path, "ConvData", load_to_mem=False, mode='w')

        to_delete_flag_idx_list = []

        # traverse all cases in presistent db from this layer_name
        for row in persistent_db_partition.get_gen(retrieve_names=["flag_idx"], conditionals=[Conditional("layer_name", "=", aug_layer_name)]):

            persistent_db_row_count += 1
            persistent_db_flag_idx = row["flag_idx"]
            is_found = False
            for api_db_key, dirty_bit in to_delete_map[aug_layer_name]:

                # if there is a match, meaning it is either "to_update" or "to_ignore" case
                if persistent_db_flag_idx == api_db_key:

                    # delete the row in persitent db if it "to_update" case
                    if dirty_bit == "1":
                        print("to be updated: ", persistent_db_flag_idx)
                        deleted_to_update_count += 1

                        # this case is to be updated in following sweep -> delete
                        to_delete_flag_idx_list.append(persistent_db_flag_idx)
                    else:
                        to_ignore_count += 1

                    is_found = True
                    break

            # this is a "to_delete" case 
            if not is_found:
                print("only in persistent: ", persistent_db_flag_idx)
                deleted_to_delete_count += 1

                # only exists in persistent db, not in new layer -> delete
                to_delete_flag_idx_list.append(persistent_db_flag_idx)

        # perform deletion
        for flag_idx in to_delete_flag_idx_list:
            persistent_db_partition.delete([Conditional("flag_idx", "=", flag_idx)])

        persistent_db_partition.close()

        print("[%s/%s][%s] done scanning and seletion in aug_layer_name: %s, time: %s" % 
              (aug_layer_name_count, aug_layer_name_len, 
              persistent_db_partition_path, aug_layer_name, time.time() - start_time))
        aug_layer_name_count += 1

    return deleted_to_update_count, deleted_to_delete_count, to_ignore_count, persistent_db_row_count

def gen_rows_for_delete(db_cursor, table_name, to_retrieve, filter_str):
    """retrieve the `to_retrieve` from `table_name`
    this func is used for retrieving rows for deletion, so we should fet them all at the beginning and then perform the deletion
    Note: do not query rows on the fly, since some of the rows probaboly have been deleted
    """

    rows = db_cursor.execute("SELECT %s FROM %s WHERE %s;" % (",".join(to_retrieve), table_name, filter_str))

    for row in rows.fetchall():
        result = {}

        for idx, elem_name in enumerate(to_retrieve):
            result[elem_name] = row[idx]

        yield result

def build_to_delete_map(dir_path):
    """save all `to_delete` files into a dict: aug_layer_name -> (api_db_key, dirty_bit)
    """

    to_delete_map = {}
    to_delete_file_list = []
    pattern_to_delete_file = re.compile(r'^to_delete_job.*\.log$')
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if pattern_to_delete_file.match(file):
                to_delete_file_list.append(os.path.join(dir_path, file))
                to_delete_lines = []
                with open(os.path.join(dir_path, file), 'r') as to_delete_file:
                    to_delete_lines = to_delete_file.readlines()
                for to_delete_line in to_delete_lines:
                    api_db_key, aug_layer_name, dirty_bit = to_delete_line.split(";")
                    if not aug_layer_name in to_delete_map:
                        to_delete_map[aug_layer_name] = []
                    to_delete_map[aug_layer_name].append([api_db_key, dirty_bit])

    # for f in to_delete_file_list:
    #     if os.path.exists(f):
    #         os.remove(f)
    return to_delete_map

def print_used_mem():
    """print the used memory out of the total memory
    in format `used/total G used`
    """

    os_free_results = os.popen('free -t -g').readlines()
    if os_free_results == None or len(os_free_results) == 0:
        return "??/??G Mem Used"
    tot_mem, used_mem, free_mem = map(int, os_free_results[-1].split()[1:])
    return "%s/%sG Mem Used" % (used_mem, tot_mem)

def add_persistent_db_additional_elem(parser_elem_names, parser_type_names):
    """add additional col name and type
    """

    elem_names = []
    type_names = []

    persistent_db_addtional_elem_type = {}
    persistent_db_addtional_elem_type['flag_idx'] = 'TEXT PRIMARY KEY UNIQUE'
    persistent_db_addtional_elem_type['flag_idx.main_flag'] = 'INTEGER DEFAULT -1'
    persistent_db_addtional_elem_type['api_list'] = 'TEXT'
    persistent_db_addtional_elem_type['update_time'] = 'TEXT'
    persistent_db_addtional_elem_type['last_cl'] = 'TEXT'

    for elem in persistent_db_addtional_elem_type:
        elem_names.append(elem)
        type_names.append(persistent_db_addtional_elem_type[elem])

    return elem_names + parser_elem_names, type_names + parser_type_names

def add_persistent_db_additional_content(all_parsed, indexer, new_api_db_path, last_cl):
    """add contents of additional cols of persistent db
    """

    test_flag = all_parsed["test_flags"].strip()

    layer_flag_str = test_flag.split("backendEngine")[0].strip(" *")
    engine_config_str = "*".join([flag.strip() for flag in test_flag.split(
        layer_flag_str)[1].strip(" *").split("*") if "Dprint_dbg" not in flag])
    flag_idx = ",".join(["[%s]" % str(indexer.get_key_index(layer_flag_str))] +
                              indexer.get_key_val_idx_list_from_flag_str(engine_config_str.strip(),
                                                                              item_seperator="*",
                                                                              key_val_seperator=":="))

    new_api_db_sqlite = PeresistentSQLDB(new_api_db_path, "ConvData",
                        elem_names=["flag_idx", "api_list"],
                        type_names=["TEXT PRIMARY KEY", "TEXT"], load_to_mem=False, mode='r')

    all_parsed["flag_idx"] = flag_idx
    all_parsed["flag_idx.main_flag"] = flag_idx.split(",")[0].strip("[]")
    rows = new_api_db_sqlite.get(retrieve_names=["api_list"], conditionals=[Conditional("flag_idx", "=", flag_idx)])
    if rows == None or len(rows) == 0:
        print("[new api db] no records for flag_idx: %s" % (flag_idx))
        all_parsed["api_list"] = "NULL"
    else:
        all_parsed["api_list"] = rows[0]["api_list"]
    all_parsed["update_time"] = str(int(time.time()))
    all_parsed["last_cl"] = str(last_cl)

    new_api_db_sqlite.close()

    return all_parsed

def get_indexer_from_new_api_db(new_api_db_path):
    """load the indexer from new api db
    """

    start_time = time.time()
    # read indexer from new api db
    new_idx_table_db = PeresistentSQLDB(new_api_db_path, "idx_table",
                        elem_names=["idx", "val"],
                        type_names=["INTEGER PRIMARY KEY", "TEXT"], load_to_mem=False, mode='r')
    if new_idx_table_db.is_table_exist():
        indexer = FlagApiIndexer()
        key_to_index = {}
        index_to_key = {}
        for row in new_idx_table_db.get_gen(["idx", "val"]):
            key_to_index[row["val"]] = int(row["idx"])
            index_to_key[int(row["idx"])] = row["val"]
        indexer.load(key_to_index, index_to_key)
    new_idx_table_db.close()
    print("loaded indexer from new api db, time: %s seconds" % (time.time() - start_time))

    return indexer

def get_db_from_new_api_db(new_api_db_path):
    """load the indexer from new api db
    """

    # read indexer from new api db
    new_idx_table_db = PeresistentSQLDB(new_api_db_path, "idx_table",
                        elem_names=["idx", "val"],
                        type_names=["INTEGER PRIMARY KEY", "TEXT"], load_to_mem=False, mode='r')
    if new_idx_table_db.is_table_exist():
        indexer = FlagApiIndexer()
        key_to_index = {}
        index_to_key = {}
        for row in new_idx_table_db.get_gen(["idx", "val"]):
            key_to_index[row["val"]] = int(row["idx"])
            index_to_key[int(row["idx"])] = row["val"]
        indexer.load(key_to_index, index_to_key)
    new_idx_table_db.close()

    start_time = time.time()

    new_api_db = {}
    flag_to_flag_idx = {}

    new_api_db_sqlite = PeresistentSQLDB(new_api_db_path, "ConvData",
                        elem_names=["flag_idx", "api_list"],
                        type_names=["TEXT PRIMARY KEY", "TEXT"], load_to_mem=False, mode='r')

    for row in new_api_db_sqlite.get_gen(["flag_idx", "api_list"]):
        key = row["flag_idx"]

        # get the original flag string from the flag idx
        original_flag = indexer.recover_from_flag_idx_str(key, item_seperator="*", key_val_seperator=":=")

        # sort the flags for uniqueness
        original_flag_list = [flag.strip() for flag in original_flag.split("*") if flag != ""]
        original_flag_list.sort()

        new_api_db[key] = row["api_list"]

        # sorted de-compressed flag -> flag_idx
        # this dict is created because the flags in combined db are in plain text string
        # we need a 1-to-1 mapping from plain text flags to flag_idx
        flag_to_flag_idx["*".join(original_flag_list)] = key
    new_api_db_sqlite.close()
    print("loaded api db, time: %s seconds" % (time.time() - start_time))

    return new_api_db, flag_to_flag_idx

def get_last_cl(bin_path):
    # read the last cl number from a file in cudnn_localbuild dir 
    # this script will be executed from `scripts` dir
    last_cl_file_path = os.path.join(os.path.abspath(bin_path), "last_cl.log")
    last_cl = ""
    if not os.path.exists(last_cl_file_path):
        print("last cl file not found at %s, using default empty cl" % (last_cl_file_path))
    else:
        with open(last_cl_file_path, "r") as last_cl_file:
            last_cl = last_cl_file.read().strip()
    return last_cl

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
import contextlib
import os.path
import collections
import re

# internal
from .Flags         import Flags
from .utility       import flags_from_descs_str
from .utility_py3   import *
from .log_extractor import LogExtractor

class LogParser:
    def __init__(self, to_extract):
        self.all_extractors = {}
        self.verbose_names = []
        self.compact_names = []
        self.type_names    = []

        for extract_value in to_extract:
            verbose_name = LogExtractor.getVerboseName(extract_value)

            compact_name = LogExtractor.getCompactName(verbose_name)

            self.verbose_names.append(verbose_name)
            self.compact_names.append(compact_name)
            self.type_names.append(LogExtractor.getType(verbose_name))

            name, field = verbose_name.rsplit('.', 1)

            if name not in self.all_extractors:
                self.all_extractors[name] = []

            self.all_extractors[name] += [field]

    def get_verbose_names(self):
        return self.verbose_names

    def get_compact_names(self):
        return self.compact_names

    def get_type_names(self):
        return self.type_names

    def parse_line(self, line):
        for name in self.all_extractors:
            extractor = LogExtractor.name2extractor_map[name]

            result_tuple = extractor.extractToTuple(line)

            if result_tuple != None:
                result = {}

                for field in result_tuple._fields:
                    if field in self.all_extractors[name]:
                        verbose_name = "%s.%s" % (name, field)
                        compact_name = LogExtractor.getCompactName(verbose_name)

                        result[compact_name] = getattr(result_tuple, field)

                return result

        return None

    def iter_cases(self, file_path):
        if not os.path.exists(file_path):
            return

        run_pattern = re.compile("\&\&\&\& RUNNING (.*)$")
        res_pattern = re.compile("\&\&\&\& (PASSED|WAIVED|FAILED) (.*)$")
        clear_pattern = re.compile("Printing all command line arguments")

        with open(file_path, 'r') as in_file:
            cur_case = None

            for line_idx, line in enumerate(in_file):
                # Check for clear pattern (indicates a new run)
                if clear_pattern.match(line):
                    cur_case = None
                    continue

                run_match = run_pattern.match(line)

                if run_match:
                    cur_case = LogCase(file_path, run_match.groups()[0], line_idx)
                    continue

                res_match = res_pattern.match(line)

                if res_match:
                    if cur_case == None:
                        raise Exception("[SPREADSHEET] Found end of pattern without beginning (at %s:%d)" % (file_path, line_idx))

                    final_status, final_flags_str = res_match.groups()

                    if cur_case.flags_str != final_flags_str:
                        raise Exception("[SPREADSHEET] Mismatch of flags \"%s\" (%s:%d) and \"%s\" (%s:%d)" % (cur_case.flags_str, file_path, cur_case.start_line_idx, final_flags_str, file_path, line_idx))

                    # Add to DB
                    yield cur_case.finish(final_status, line_idx)

                    cur_case = None

                    continue

                if cur_case:
                    parsed = self.parse_line(line)

                    if parsed:
                        cur_case.update(parsed)


class LogCase:
    def __init__(self, file_path, flags_str, start_line_idx):
        self.flags_str = flags_str
        self.start_line_idx = start_line_idx
        self.file_path = file_path
        self.all_parsed = {}

    def finish(self, final_status, end_line_idx):
        return self

    def update(self, parsed):
        self.all_parsed.update(parsed)

def parse_log_file(file_path, parser):
    return parser.iter_cases(file_path)

def use_parser_on_output(output, parser):
    all_parsed = {}

    for line in output.split('\n'):
        parsed = parser.parse_line(line)

        if parsed:
            all_parsed.update(parsed)

    return all_parsed

CompressedFlag = collections.namedtuple("CompressedFlag", "compressed_name compressed_value")


class LogFlagReader:
    def __init__(self, file_name, match_order):
        # Compress flag names/values by storing them in compressed form; e.g. "-Rconv" -> ("-R", "conv") -> (0, 1)
        self.compressor = {}
        self.uncompressor = {}

        # True/False if flag order matters in match
        self.match_order = match_order

        # Store all seen compressed flags
        self.seen = {}

        # Keep track of header_line, test_flags_line (append on finish)
        header_line = None
        test_flags_line = None

        for case in parse_log_file(file_name, LogParser(["test_flags"])):
            if "test_flags" not in case.all_parsed:
                continue

            flags = flags_from_descs_str(case.all_parsed["test_flags"])

            if len(flags) != 1:
                raise Exception("Invalid number of flags (expecting 1): %s" % str(flags))

            self.add_seen(flags[0], case.start_line_idx)

    def get_compressed(self, flag_name_or_val):
        if flag_name_or_val not in self.compressor:
            compressed_val = len(self.compressor)
            self.compressor[flag_name_or_val] = compressed_val
            self.uncompressor[compressed_val] = flag_name_or_val

        return self.compressor[flag_name_or_val]

    def get_uncompressed(self, compressed_name_or_val):
        if compressed_name_or_val not in self.uncompressor:
            raise Exception("[LogFlagReader] Attempting to uncompress value that hasn't been seen: %s (All Compressed: %s)" % (compressed_name_or_val, str(self.uncompressor)))

        return self.uncompressor[compressed_name_or_val]

    def compress_flags(self, uncompressed_flags):
        result = []

        if uncompressed_flags.is_multiple():
            raise Exception("[LogFlagReader] Unable to parse %s" % str(flags))

        for flag_name in uncompressed_flags:
            compressed_name = self.get_compressed(flag_name)
            compressed_val  = self.get_compressed(uncompressed_flags[flag_name])

            result.append(CompressedFlag(compressed_name, compressed_val))

        return tuple(result)

    def uncompress_flags(self, compressed_flags):
        result = Flags()

        for compressed_flag in compressed_flags:
            flag_name = self.get_uncompressed(compressed_flag.compressed_name)
            flag_val  = self.get_uncompressed(compressed_flag.compressed_value)

            result[flag_name] = flag_val

        return result

    def add_seen(self, flags, line_idx):
        compressed_flags = self.compress_flags(flags)

        if not self.match_order:
            compressed_flags = tuple(sorted(compressed_flags))

        self.seen[compressed_flags] = line_idx

    def has_seen(self, flags):
        compressed_flags = self.compress_flags(flags)

        if not self.match_order:
            compressed_flags = tuple(sorted(compressed_flags))

        return (compressed_flags in self.seen)

#  module serves as an interface between Python and cudnnTest/cublasTest

# To accomplish this, it runs cudnnTest and parses the output to obtain result info.
#   - Result info includes elapsed time, M, N, K, algo, etc...
#   - See "def parse(output)" for a definition of all parsed data.

# The main use of this module is the function "run_flags(flags)"
#   - This will run cudnnTest and return parsed output

from __future__ import print_function
import re  # For regular expressions
import subprocess  # To spawn processes (cudnnTest)
import collections  # For namedtuple
import functools
import os  # To check if file exists
import sys  # To check Platform
import random
import signal
import datetime
import errno
import time
from contextlib import contextmanager

from multiprocessing.dummy import Pool as ThreadPool
from math import ceil
from threading import Lock, current_thread, Thread, Event

# local imports
from .utility import split_space
from .utility_py3 import *
from .Flags import Flags
from .cutlass_interface import generate_cutlass_args
from .log_extractor import LogExtractor

try:
    from queue import Queue, Empty # python 3.x
except ImportError:
    from Queue import Queue, Empty  # python 2.x


#NonBlockingReader is a class that can be initialized with a stdout stream
#and it will read lines from that stream until it is ended (or the process terminates)
#It does so by calling the blocking readline() in a thread and enqueuing results
class NonBlockingReader:
    def __init__(self, stdout):
        self.out_stream = stdout
        self.q = Queue()
        self.thread = Thread(target=self.enqueue_output)
        self.thread.daemon = True #terminate thread with the program
        self.stdout_finished = Event()

    def start(self):
        self.thread.start()
    
    def stop(self):
        self.thread.join()

    #returns a tuple consisting of:
    # a bool indicating whether end of stdout has been reached and 
    # a line from stdout or None if no new lines have been added.
    def get_line(self):
        # read line without blocking
        try:  
            line = self.q.get_nowait()
        except Empty:
            return self.stdout_finished.is_set(), None
        else: # got line
            self.q.task_done()
            return False, line

    #do not call directly. This is called by the NonBlockingReader child thread to read from stdout
    def enqueue_output(self):
        for line in iter(self.out_stream.readline, b''):
            self.q.put(line)
        self.out_stream.close()
        self.stdout_finished.set()

#This approximates subprocess.wait(timeout) which does not exist in python 2
def process_wait_py2(process, timeout):
    start_time = time.time()
    retcode = None
    while retcode is None and (time.time() - start_time) < timeout:
        retcode = process.poll()
        time.sleep(.01)
    return retcode

from contextlib import contextmanager
import sys

@contextmanager
def open_or_stdout(path_to, filename):
    if path_to is None or path_to == '':
        yield None
    elif path_to == 'stdout':
        yield sys.stdout
    else:
        if path_to[-1] != '/':
            path_to = path_to + '/'
        try:
            with open(path_to + filename, 'a') as f:
                yield f
        except Exception as e:
            print("ERROR could not open file {} with error {}. No debug info logged".format(path_to + filename, e))
            yield None
    
#This function will take an open process, then use two timeout values
#First time out is the time to wait before gathering intermediate output
#Secod time out is the time to wait before killing the subprocess
#It will gather the output from the process and return it along with the return code.
#Returns a tuple
#  Status (bool): did process execute to completion (true) or terminate abnormally(false)
#  Retcode (int): return code from the process. Will be -1 if Status is false.
#  Output (str): Stdout from the process. May not be complete if Status is false.
def async_process_reader(process, io_timeout, process_timeout, output, extended_debug_path):
    print("INFO [{}] Process {} starting with timouts {} , {}".format(datetime.datetime.now().isoformat(), process.pid, io_timeout, process_timeout))

    # Start the 'testProcess' executable as a subprocess
    reader = NonBlockingReader(process.stdout)
    reader.start()

    print("INFO Process {} reader started".format(process.pid))

    #gather output using the non blocking reader
    #if the program does not finish in X seconds, start logging the output immediately.
    #  and continue waiting while logging output until the program terminates
    #else print all output to stdout
    retcode = None
    flag_process_timed_out = False
    start_time = time.time()

    #If we are doing IO timeouts, and the IO timeout happens before process timeout
    #call wait() with the io timeout
    if io_timeout > 0 and process_timeout > io_timeout:
        retcode = process_wait_py2(process, io_timeout)
        if retcode is None:
            print("[DebugOut.{}][{}] I/O Timeout".format(process.pid, datetime.datetime.now().isoformat()))
    #else if we are doing process timeouts, call wait() with the process timeout
    elif process_timeout > 0:
        retcode = process_wait_py2(process, process_timeout)
        if retcode is None:
            flag_process_timed_out = True
            process.terminate()
            retcode = -1
            #fall through to gather remaining lines.
    #else we are not using timeouts, so just block on wait()
    else:
        retcode = process.wait()

    #if retcode is not set, we ended early (ie io_timeout).
    #so we need to keep gathering intermidiate output
    #until either the process returns or we reach the process timeout
    if retcode is None and not flag_process_timed_out:
        with open_or_stdout(extended_debug_path, 'IODebug.' + str(process.pid) + '.log') as fh:
            while retcode is None and not flag_process_timed_out:
                _, line = reader.get_line()
                while line and not flag_process_timed_out:
                    flag_process_timed_out = process_timeout > 0 and time.time() - start_time > process_timeout
                    decoded_output = line.decode('utf-8').strip('\n')
                    if fh is not None:
                        fh.write('[DebugOut.{}][{}] {}\n'.format(process.pid, datetime.datetime.now().isoformat(), decoded_output))
                    output = output + line if output else line
                    _, line = reader.get_line()
                
                time.sleep(.01)
                retcode = process.poll()
                flag_process_timed_out = process_timeout > 0 and time.time() - start_time > process_timeout

    if flag_process_timed_out:
        print("[DebugOut.{}][{}] Process Timeout".format(process.pid, datetime.datetime.now().isoformat()))
        #it is possible we timed out just before the process terminated on its own, so check the retcode again.
        if not retcode:
            process.terminate()
        retcode = -1
        #fall through to gather remaining lines.

    print("INFO [{}] Process {} reader ending with code {}".format(datetime.datetime.now().isoformat(), process.pid, retcode if retcode is not None else "None"))

    #gather remaining output from the queue
    stat, line = reader.get_line()
    queue_start_time = time.time()
    queue_time_limit = 60
    # limit queue to 60 seconds
    # if it takes longer than a minute to get all the output, then truncate it
    while not stat:
        if (time.time() - queue_start_time) > queue_time_limit:
            print("INFO [{}] Process {} reader timed out gathering remaining stdout. Truncating stdout".format(datetime.datetime.now().isoformat(), process.pid))
            break
        if line is not None: 
            output = output + line if output else line
        stat, line = reader.get_line()

    print("INFO [{}] Process {} reader returning".format(datetime.datetime.now().isoformat(), process.pid))

    reader.stop()

    return ((not flag_process_timed_out), retcode, output)

class FileRange(collections.namedtuple("FileRange", "start end file_desc")):
    '''Nametuple for reading a range .start to .end of file descriptor .file_desc'''

    @classmethod
    def fromMatches(cls,
                    file_desc,
                    start_re,
                    stop_re,
                    max_segments=None,
                    ret_matches=False):
        '''
        Return list of instances for segments in {file_desc} between lines that matches {start_re} and the next one that matches {stop_re}

        if {max_segments} is an integer, stop read after {max_segments} has been generated.
        If {ret_matches} is True, each instance is accompanied by start and stop re match objects.
        '''
        ret = []
        count = 0
        start_match = None  # last start_re match
        # use this instead of iter(file_desc) to avoid the read-ahead buffe, which breaks .tell()
        while True:
            curr_pos = file_desc.tell()
            line = file_desc.readline()
            if not line:  # empty line == '\n'
                break
            match = start_re.match(line)
            if match:
                start_pos = curr_pos
                start_match = match
            match = stop_re.match(line)
            if start_match is not None and match:
                end_pos = file_desc.tell()
                inst = cls(file_desc=file_desc, start=start_pos, end=end_pos)
                if ret_matches:
                    ret.append((start_match, match, inst))
                else:
                    ret.append(inst)
                count += 1
                if max_segments is not None and count >= max_segments:
                    break
                start_match = None  # reset the start_match
        return ret

    def read(self):
        '''Read the range'''
        self.file_desc.seek(self.start)
        return self.file_desc.read(self.end - self.start)


# Create namedtuples for all parsed outputs
ParsedOutput = collections.namedtuple("ParsedOutput",
                                      LogExtractor.name2extractor_map.keys())

# ParsedTuples = [extractor.nt_type for extractor in LogExtractor.name2extractor_map.itervalues()]

# for ParsedTuple in ParsedTuples:
#   globals()[ParsedTuple.__name__] = ParsedTuple

# Create namedtuple for RunResult
RunResult = collections.namedtuple(
    "RunResult",
    "flags bin_path bin_name output error_msg return_code parsed interrupted")

# Create namedtuple for RunListResult
RunListResult = collections.namedtuple(
    "RunListResult",
    "flags bin_path bin_name outputs error_msg return_code parsed interrupted")

# Created namedtuple for CachedRunResult
CachedRunResult = collections.namedtuple("CachedRunResult",
                                         "cache_count run_result")


# Find the line strictly after lid in lines matching processing_pat
# Can throw ValueError if input is invalid.
def get_next(lid, lines):
    for lidNext in xrange(lid + 1, len(lines)):
        match = LogExtractor.name2extractor_map['processing'].pattern.search(
            lines[lidNext])
        if match:
            return (lidNext, int(match.groups(1)[0]))
    return (len(lines), -1)


# Remove all duplicates lines starting with 'HEURGEN: ', keeping original ordering
# Useful when running heuristics with -T > 1, to avoid extremely large outputs
def strip_heurgen_duplicates(string):
    if string == None:
        return None
    string_split = string.split('\n')
    seen = set()
    string_nodupheurgen_split = []
    for line in string_split:
        if line.startswith('HEURGEN: '):
            if line not in seen:
                seen.add(line)
                string_nodupheurgen_split.append(line)
        else:
            string_nodupheurgen_split.append(line)
    return '\n'.join(string_nodupheurgen_split)


def strip_duplicates(string):
    seen = set()

    result_lines = []

    for line in string.split('\n'):
        if line not in seen:
            result_lines.append(line)
            seen.add(line)

    return '\n'.join(result_lines)


# Output = complete output string
# Returns ([output1, output2, ...], [parsedoutput1, parsedoutput2, ...]) for each test of the -testsList
def parse_testsList(output, bin_path, bin_name, batch_method, is_interrupted,
                    stdout_log=sys.stdout, print_on_fail=False):

    print_to_log = lambda *args, **kwargs: print(
        file=stdout_log, *args, **kwargs)
    if batch_method == 0:
        try:
            if (output == None):
                return (None, None)

            if '$$$$ Reading from stdin' not in output:
                if print_on_fail:
                    print_to_log(
                        '[TEST EXECUTION] Critical error, expected -testsList output, could not find required data.')
                    print_to_log('[TEST EXECUTION] Output: ', output)
                return ([output], [None])

            lines = output.split('\n')
            (lid_current, begin_line) = get_next(-1, lines)
            if begin_line != 1:
                if print_on_fail:
                    print_to_log(
                        '[TEST EXECUTION] Critical error, expected to read line 1, got line {}'
                        .format(begin_line))
                return ([output], [None])
            (lid_next, next_line) = get_next(lid_current, lines)
            if next_line == -1:
                print('[TEST EXECUTION] WARNING, invalid result from get_next(). Attempting to continue...')
            parsed_result = []
            split_output = []
            expected_line = 0

            while lid_current < len(lines):
                expected_line += 1
                test_to_parse = lines[lid_current:lid_next]
                parsed_output = OutputParser('\n'.join(test_to_parse))
                if parsed_output and parsed_output["status"]:
                    status_line = int(parsed_output["status"].line)
                else:
                    if print_on_fail:
                        print_to_log(output)
                        print_to_log(
                            '[TEST EXECUTION] Critical error, parsing error for line {}'
                            .format(expected_line))
                    # If nothing to return (i.e., fails at first iteration), returns all the output so that we have something to print
                    if not split_output:
                        return ([output], [None])
                    else:
                        return (split_output, parsed_result)
                if begin_line != expected_line or status_line != expected_line:
                    if print_on_fail:
                        print_to_log(
                            '[TEST EXECUTION] Critical error, expected output from line {}, got output beginning line {} and ending line {}'
                            .format(expected_line, begin_line, status_line))
                    # If nothing to return (i.e., fails at first iteration), returns all the output so that we have something to print
                    if not split_output:
                        return ([output], [None])
                    else:
                        return (split_output, parsed_result)
                parsed_result.append(parsed_output)
                split_output.append(
                    strip_heurgen_duplicates('\n'.join(test_to_parse)))
                (lid_current, begin_line) = (lid_next, next_line)
                (lid_next, next_line) = get_next(lid_current, lines)

            return (split_output, parsed_result)
        except ValueError as e:
            print("[TEST EXECUTION] Critical Error reading parsing test output {}".format(e))
            return ([output], [None])
    else:
        pat_per_case = re.compile(
            r'\<\<\<\<\< \[.*?\] .*? \>\>\>\>\>.*?\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\-\n',
            re.MULTILINE | re.DOTALL)
        error_pat = re.compile(r'ERROR:(.*?)\n', re.IGNORECASE)

        result = []
        outputs = []

        for match in pat_per_case.findall(output):
            flags = None
            bin_path = bin_path
            bin_name = bin_name
            output_per = strip_duplicates(match)
            error_msg = None
            return_code = 0

            error_match = error_pat.search(output_per)

            if error_match:
                error_msg = error_match.groups()[0]
                return_code = 1
            outputs.append(output_per)

            result.append(
                RunResult(flags, bin_path, bin_name,
                          output_per, error_msg, return_code,
                          OutputParser(output_per), is_interrupted))

        return (outputs, result)


class OutputParser(object):
    '''Parser for cudnnTest output, instance[name] extracts namedtuple for mathcing name'''

    # list of valid --extract arguments
    def __init__(self, output):
        self.output = output
        self.cache = {}

    def __getitem__(self, name):
        if name in LogExtractor.name2extractor_map:
            extractor = LogExtractor.name2extractor_map[name]
            if self.cache.get(name) is None:
                self.cache[name] = extractor.extractToTuple(self.output)
            else:
                pass  # print("Cached %s" % name)

            return self.cache[name]
        raise Exception("Name %s is not parsable", name)


# Fill all values with None and return
def get_default_run():
    blank_run = RunResult(
        flags=None,
        bin_path=None,
        bin_name=None,
        output=None,
        error_msg=None,
        return_code=None,
        parsed=ParsedOutput(
            **{
                name: extractor.make_default_tuple(
                )  # extract namedtuples with all field set to default value None,
                for name, extractor in
                LogExtractor.name2extractor_map.iteritems()
            }),
        interrupted=None)
    return blank_run

# Runs given flags
def run_flags(plock, 
              flags,
              bin_path,
              bin_name,
              piped_input=None,
              pre_flags_str="",
              checkjit=False,
              debug_log=None,
              extended_logging=None):
    if debug_log:
        start = datetime.datetime.now()
        debug_log.write("Running Flags: %s\n" %
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Initialize all outputs in case everything fails
    return_code = None
    output = None
    error_msg = None

    shell_pre_flags = split_space(pre_flags_str)

    shell_bin_flag = ["%s/%s" % (bin_path, bin_name)]

    shell_flags = split_space(flags)

    shell_post_flags = []

    if bin_name == 'cutlass_profiler':
        if shell_flags != None and len(shell_flags) > 1:
            shell_post_flags = generate_cutlass_args(flags.flags_dict)
    else:
        shell_post_flags = shell_flags

    shell = shell_pre_flags + shell_bin_flag + shell_post_flags
    shell_str = ' '.join(shell)

    output = None
    return_code = None
    interrupted = False

    if checkjit:
        key = ""
        value = ""
        platform = sys.platform
        if platform == "win32":
            return_code = 2
            error_msg = "JIT checking not supported on Windows"
        else:
            if platform == "darwin":
                key = "DYLD_PRINT_BINDINGS"
                value = "1"
            elif platform == "qnx":
                key = "LD_DEBUG"
                value = "all"
            else:
                key = "LD_DEBUG"
                value = "bindings"
            _env = os.environ.copy()
            _env[key] = value
            process = subprocess.Popen(
                shell,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_env,
                shell=False)  # shell=False is required for the pid to be correct
            pid = process.pid
            if piped_input:
                print('@@@@ thread id ({}) spawning process PID ({}) with command `printf -- \"{}\" | {}`'
                    .format(current_thread().ident, pid, repr(piped_input), shell_str))
            else:
                print('@@@@ thread id ({}) spawning process PID ({}) with command `{}`'
                    .format(current_thread().ident, pid, shell_str))
            output, stderr = process.communicate(piped_input)

            if stderr.count("nvPTXCompilerCompile") > 0:
                return_code = 1
                error_msg = "JIT process is triggered"
            elif process.returncode not in [0, 1, 2]:  # unexpected error
                return_code = process.returncode
                error_msg = "Process returned %d" % return_code
            else:
                return_code = 0
    else:
        try:
            # Start process with requested flags (and piped output from stdout & stderr)
            # process.Popen is not thread safe so because we call it from multiple threads
            # we must protect it with a Lock
            process = None
            if plock is None:
                print("Using Stub plock. Should only happen with gpu and -g tests")
                plock = Lock() #stub

            # Acquire process lock and hold it through flushing stdin.
            with plock:
                while True:
                    try:    
                        process = subprocess.Popen(
                            shell,
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            shell=False)  # shell=False is required for the pid to be correct
                        pid = process.pid
                    except OSError as err:
                        # Note: ENOMEM only occurs when subprocess.Popen runs out of memory when trying to spawn a thread,
                        #       i.e. if cudnnTest internally runs out of memory, that would be dealt with by cudnnTest
                        #       and would not manifest as ENOMEM here
                        if err.errno == errno.ENOMEM:
                            time.sleep(0.5)
                            continue
                        else:
                            raise err
                    break

                if piped_input:
                    piped_input = piped_input.encode()
                    print('@@@@ [{}] thread id ({}) spawning process PID ({}) with (2) command `printf -- \"{}\" | {}`'
                        .format(datetime.datetime.now().isoformat(), current_thread().ident, pid, repr(piped_input), shell_str))
                    sys.stdout.flush()
                    # communicate handled the stdin, so we need to write it directly
                    process.stdin.write(piped_input)
                    process.stdin.flush()
                    process.stdin.close()

                else:
                    print('@@@@ [{}] thread id ({}) spawning process PID ({}) with command `{}`'
                        .format(datetime.datetime.now().isoformat(), current_thread().ident, pid, shell_str))

            #Release lock held through creation of process and flushing of stdin.
            try:                
                # If the process does not return in 600 seconds begin logging its output immediately
                # If the process does not return in 30 minutes (1800 seconds) terminate it
                batch_timeout = -1
                io_timeout = -1
                extended_debug_path_str = ''
                if extended_logging is not None:
                    batch_timeout = extended_logging['process_timeout']
                    io_timeout = extended_logging['io_timeout']
                    extended_debug_path_str = extended_logging['debug_path']

                print('  thread id ({}) process PID ({}) Starting stdout reader'.format(current_thread().ident, pid))
                sys.stdout.flush()
                status, return_code, output = async_process_reader(process, io_timeout, batch_timeout, output, extended_debug_path_str)
                print('  thread id ({}) process PID ({}) Continuing'.format(current_thread().ident, pid))
                sys.stdout.flush()
                if not status:
                    print("Test process {} exceeded {} seconds. Batch Terminated".format(pid, batch_timeout))
                    if return_code < 0:
                        return_code = 1 #Will cause CalledProcessError exception below.
                #else return_code and output will be valid. proceed.

            except KeyboardInterrupt:
                print(" thread id ({}) process PID ({}) Received Keyboard interrupt".format(current_thread().ident, pid))
                process.send_signal(signal.SIGINT)
                # output, unused_err = process.communicate()
                interrupted = True
                return_code = process.poll()
                raise KeyboardInterrupt

            # Error if return_code > 0
            if return_code:
                raise subprocess.CalledProcessError(
                    return_code, flags, output=output)

        except subprocess.CalledProcessError as e:
            # Grab error (guaranteed to be given)
            error_msg = str(e)

            if debug_log != None:
                debug_log.write(error_msg + '\n')
                debug_log.write("-- ERROR RUNNING BATCH OF FLAGS --\n")
                debug_log.write("-- FLAG VALUES --\n")
                debug_log.write(piped_input.decode('utf-8') + '\n')
                debug_log.write("-- OUTPUT OF EXECUTION --\n")
                if output:
                    debug_log.write(output.decode('utf-8') + '\n')
                debug_log.write("-- END OF OUTPUT --\n")

    if debug_log:
        debug_log.write("Done Running Flags ($?={:d}): {}\n".format(
            return_code,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        debug_log.write(
            "Total Time: %s\n" % str(datetime.datetime.now() - start))

    if output:
        output = output.decode('utf-8')
    if piped_input:
        print('@@@@ [{}] thread id ({}) completed process PID ({}) with command `printf -- \"{}\" | {}`'
            .format(datetime.datetime.now().isoformat(), current_thread().ident, pid, repr(piped_input), shell_str))
    else:
        print('@@@@ [{}] thread id ({}) completed process PID ({}) with command `{}`'
            .format(datetime.datetime.now().isoformat(), current_thread().ident, pid, shell_str))

    return RunResult(flags, bin_path, bin_name, output, error_msg, return_code,
                     OutputParser(output), interrupted)


def run_flags_wrapper(plock, args):
    return run_flags(plock, *args)


def create_pool(n_pool, func, arglist):
    pool = ThreadPool(n_pool)
    ret = pool.map(func, arglist)
    pool.close()
    pool.join()
    return ret


"""
merge_io() merges an list of StringIO() into a single StringIO()
PARAMETERS:
    list_of_stringio: List of StringIO() to be merged
    merge_to: StringIO() to be merged into
"""


def merge_io(list_of_stringio, merge_to):
    for idx, stringio in enumerate(list_of_stringio):
        merge_to.write(
            "======== START OF MERGED OUTPUT [%d] ==========\n" % idx)
        merge_to.write(stringio.getvalue())
        merge_to.write(
            "======== END OF MERGED OUTPUT [%d]   ==========\n" % idx)


"""
run_batch_of_flags() runs a batch of flags using run_flags() and separates individual results to emulate them being run individually by run_flags
PARAMETERS:
    batch_of_flags: "list(Flags)" containing all flags to run in this batch
    bin_path: "string" indicating where the binary is stored
    bin_name: "string" indicating what the binary name is (cublasTest or cudnnTest)
    pre_flags_str: "string" to be added before the binary (such as 'nvprof')
"""


def run_batch_of_flags(plock,
                        batch_of_flags,
                       bin_path,
                       bin_name,
                       device=['0'],
                       pre_flags_str="",
                       checkjit=False,
                       batch_method=0,
                       return_mode=0,
                       debug_log=None,
                       stdout_log=sys.stdout,
                       print_on_fail=False,
                       extended_logging=None):
    '''
    return_modde = 0 : default, 1: failed, 2: none
    '''
    # Set deterministic seed
    random.seed(1234)

    # Shuffle all but first element (because we NEED the first element)
    first_flag = batch_of_flags.pop(0)

    for i in xrange(10):
        random.shuffle(batch_of_flags)

    batch_of_flags = [first_flag] + batch_of_flags

    batch_len_per_gpu = int(ceil(float(len(batch_of_flags)) / len(device)))
    lists_of_batches = [
        batch_of_flags[i:i + batch_len_per_gpu]
        for i in xrange(0, len(batch_of_flags), batch_len_per_gpu)
    ]
    lists_of_piped_input = []
    for b, d in zip(lists_of_batches, device[:len(lists_of_batches)]):
        for i, ele in enumerate(b):
            ele['d'] = (str(d), )
            b[i] = ele
        if batch_method > 0:
            piped_input = get_piped_flags(clean_batch(b, bin_name))
        else:
            piped_input = get_piped_flags(b)
        lists_of_piped_input.append(piped_input)
    lists_of_batch_indicator_flags = []
    for d in device:
        batch_indicator_flags = Flags()
        batch_indicator_flags['d'] = (str(d), )
        if batch_method == 2:
            batch_indicator_flags['R'] = ('heur', )
        elif batch_method == 1:
            batch_indicator_flags['R'] = ('heurBackend', )
        else:
            batch_indicator_flags['testsList'] = ("1", )
            if return_mode < 2:
                batch_indicator_flags['returnOnFailed'] = ("1", )
            if return_mode < 1:
                batch_indicator_flags['returnOnWaived'] = ("1", )
        lists_of_batch_indicator_flags.append(batch_indicator_flags)

    arglist = []
    sub_debugs = [StringIO() for d in device]

    for idx in xrange(len(lists_of_piped_input)):
        _i = lists_of_batch_indicator_flags[:len(lists_of_piped_input)][idx]
        _p = lists_of_piped_input[idx]
        arglist.append((_i, bin_path, bin_name, _p, pre_flags_str, checkjit,
                        sub_debugs[idx], extended_logging))

    n_thread = len(arglist)
    if n_thread == 1:
        _results = [run_flags_wrapper(plock, arglist[0])]
    else:
        # equiv. to concurrently running: [run_flags_wrapper(plock, x) for x in arglist]
        func = functools.partial(run_flags_wrapper, plock)
        _results = create_pool(n_thread, func, arglist)

    merge_io(sub_debugs, debug_log)

    is_interrupted = False
    for _r in _results:
        if _r[-1]:
            is_interrupted = True
            break

    ret = []
    for _batch, (flags, _a, _b, output, error_msg, return_code, _c, _d) in zip(lists_of_batches, _results):
        split_outputs, parsed_outputs = parse_testsList(
            output, bin_path, bin_name, batch_method, is_interrupted, stdout_log, print_on_fail)
        ret.append([
            _batch,
            RunListResult(flags, bin_path, bin_name, split_outputs, error_msg,
                          return_code, parsed_outputs, is_interrupted)
        ])
    return ret


def print_error(error_type, error_message, flags, runlist_results, stdout_log=sys.stdout):
    print_to_log = lambda *args, **kwargs: print(
        file=stdout_log, *args, **kwargs)
    print_to_log(
        "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print_to_log("[{}] {}: {}".format(error_type, error_message,
                                      runlist_results.error_msg))
    if runlist_results.outputs == None:
        n_to_print = 0
    else:
        n_to_print = len(runlist_results.outputs)
    print_to_log(
        "While testing ... (only printing first {})".format(n_to_print))
    for f in flags[:n_to_print]:
        print_to_log(str(f))
    print_to_log(
        "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


"""
splice_wrap_around() returns all 'count' values starting from 'start' index with wrap around back to front elements
PARAMETERS:
    l: 'list' containing elements to splice
    start: 'int' indicating the index to start the splice
    count: 'int' indicating the number of elements to splice out of the list
EXAMPLES:
    splice_wrap_around([0, 1, 2, 3, 4], 0, 2) returns [0, 1]
    splice_wrap_around([0, 1, 2, 3, 4], 3, 2) returns [3, 4]
    splice_wrap_around([0, 1, 2, 3, 4], 2, 5) returns [2, 3, 4, 1, 2]
    splice_wrap_around([0, 1, 2, 3, 4], 2, 9) returns [2, 3, 4, 1, 2]
"""


def splice_wrap_around(l, start, count):
    # Return errors if trying to start at an index not in list
    if start >= len(l):
        raise Exception(
            "Unable to splice starting from %d in %s" % (start, str(l)))

    # Return all elements we can from 'start' to 'start+count'
    # Note this may return less than 'count' elements
    result = l[start:start + count]

    # If we weren't able to grab enough elements, wrap around to front to grab remaining
    if len(result) < count:
        # Calculate how many elements we should grab from beginning
        count_remaining = count - len(result)

        # Calculate last element to grab
        # Don't grab any elements after 'start' to avoid grabbing duplicate values (from our initial grab)
        last_element = min(start, count_remaining)

        # Grab values we calculated we should grab
        result += l[:last_element]

    # Return all grabbed values, note that this may have less elements than 'count'
    return result


"""
build_batch() builds a batch given flags that must run and suggested flags to also run
PARAMETERS:
    flags_must_run: 'Flags' class that must be run no matter what the batch size is (or what suggested_flags may contain)
    suggested_flags_dict: 'dict(str, Flags)' containing flags that should also run
        Keys are descriptor strings of the Flags class
        Value is the Flags class itself
    batch_size: 'int' The size of the batch to build
        Can be smaller than batch_size if we don't have enough flags in suggested_flags_dict
    bin_name: 'string' The binary name (must be 'cudnnTest' or "cublasTest")
"""


def build_batch(flags_must_run, suggested_flags_dict, batch_size):
    # Obtain list form of suggested_flags_dict
    suggested_flags_list = list(suggested_flags_dict.values())

    # Obtain key for the flags that must be run
    key_must_run = get_cache_key(flags_must_run)

    # If suggested_flags_dict has the flags that must be run, we start there and grab extra flags afterwards
    # For example, if we must run B and suggested_flags is [A, B, C] then we will run [B, C]
    # However, if we must run B and suggested flags is [A, C] then we will run [B] + [A] = [B, A]; see else statement
    if key_must_run in suggested_flags_dict:
        # Find index of flags that must run; this will be our 'start' index for the splice
        idx_must_run = list(suggested_flags_dict.keys()).index(key_must_run)

        # Grab batch_size elements starting from idx and going right (wrapping around at end)
        flags_to_run = splice_wrap_around(suggested_flags_list, idx_must_run,
                                          batch_size)

    # See above comments before initial if-statement
    else:
        # Return the flags that must run along with any others
        flags_to_run = [flags_must_run
                        ] + suggested_flags_list[:(batch_size - 1)]

    # Return error if batch is larger than batch size (shouldn't happen...)
    if len(flags_to_run) > batch_size:
        raise Exception(
            "[FLAG LIST PARSING] List of flags to run is larger than list length"
        )

    # Return copy that can be edited without concern of overwriting original copy
    return [flags.get_copy() for flags in flags_to_run]


"""
clean_batch() removes any flags that the batching method is unable to process
PARAMETERS:
    batch_of_flags: 'list(Flags)' containing the batch of flags
    bin_name: 'string' indicating if we are running "cublasTest" or 'cudnnTest'
"""


def clean_batch(batch_of_flags, bin_name):
    # Define flags to be scrubbed for cudnnTest
    if "cudnnTest" in bin_name:
        scrub_flags = ['d']

    # Define flags to be scrubbed for cublasTest
    elif bin_name == "cublasTest":
        scrub_flags = ['d', 'R', 'b']

    # Error if neither cudnnTest or cublasTest
    else:
        raise Exception("Binary '%s' is not supported" % bin_name)

    # Initialize empty list for our result
    result = []

    # Loop through all flags in given batch of flags
    for flags in batch_of_flags:
        # Create copy to ensure the original copy is not edited
        flags_copy = flags.get_copy()

        # Loop through all flags to be scrubbed
        for scrub_flag in scrub_flags:
            # If flag is defined in our copy
            if scrub_flag in flags_copy:
                # Scrub the flag from our copy
                del flags_copy[scrub_flag]

        # Append scrubbed flags to result
        result.append(flags_copy)

    # Return list of scrubbed flags
    return result


"""
get_piped_flags() converts of a list of flags into string form to be piped to a process
PARAMETERS:
    batch_of_flags: 'list(Flags)' containing the batch of flags
EXAMPLES:
    get_piped_flags([A, B, C]) returns "str(A)\nstr(B)\nstr(C)\n" such as
        such as "-Rconv -n1\n-Rdgrad -n2\n-Rwgrad -n3\n"
"""


def get_piped_flags(batch_of_flags):
    return "".join(["%s\n" % str(flags) for flags in batch_of_flags])


def get_cache_key(flags):
    _flags = flags.get_copy()
    del _flags['d']
    return _flags.get_descs_str()


class RunCache:
    def __init__(self, batch_size, max_cache):
        # Start with empty cache
        self.cache = collections.OrderedDict()
        self.cache_lock = Lock()
        self.plock = Lock() # This is a lock to protect subprocess.popen
        self.cache_count = 0
        self.batch_size = int(batch_size)
        self.max_cache = int(max_cache)

    def get(self,
            flags,
            suggested_flags,
            bin_path,
            bin_name,
            device=["0"],
            pre_flags_str="",
            checkjit=False,
            batch_method=0,
            return_mode=0,
            debug_log=None,
            stdout_log=sys.stdout,
            print_on_waive=True,
            print_on_fail=False,
            extended_logging=None):
        cache_key = get_cache_key(flags)
        run_idx = 0
        while True:
            with self.cache_lock:
                if cache_key in self.cache:
                    return self.cache[cache_key].run_result

            # Print out if a re-run was necessary
            if run_idx > 0:
                if self.batch_size == 1:
                    raise Exception(
                        "Attempting to re-run test while batch size is 1; this shouldn't happen."
                    )
                print("Rerunning run due to unfound key %s (run idx %d)" %
                      (str(cache_key), run_idx), file=stdout_log)
            self.run(
                flags,
                suggested_flags,
                bin_path,
                bin_name,
                device=device,
                pre_flags_str=pre_flags_str,
                checkjit=checkjit,
                batch_method=batch_method,
                return_mode=return_mode,
                debug_log=debug_log,
                stdout_log=stdout_log,
                print_on_waive=print_on_waive,
                print_on_fail=print_on_fail,
                extended_logging=extended_logging)
            run_idx += 1

    def run(self,
            flags,
            suggested_flags,
            bin_path,
            bin_name,
            device=["0"],
            pre_flags_str="",
            checkjit=False,
            batch_method=0,
            return_mode=0,
            debug_log=None,
            stdout_log=sys.stdout,
            print_on_waive=True,
            print_on_fail=False,
            extended_logging=None):
        print_to_log = lambda *args, **kwargs: print(
            file=stdout_log, *args, **kwargs)
        # Initialize batch to None (assume batch_size is 1)
        batch_of_flags = None
        # If batch_size > 1, we should try to build a batch
        if self.batch_size > 1 or batch_method > 0:
            batch_of_flags = build_batch(
                flags, suggested_flags, self.batch_size)
            print_to_log(
                "RunCache[{cache_size}] running batch of flags: {batch_size}/{batch_cap}"
                .format(
                    cache_size=len(self.cache),
                    batch_size=len(batch_of_flags),
                    batch_cap=self.batch_size))

        # Skip batching if method is 0 (meaning only batch if appropriate) and batch_of_flags is valid & size > 1
        if batch_method == 0 and (batch_of_flags is None or len(batch_of_flags) == 1):
            _flags = flags.get_copy()
            if "d" not in flags:
                flags["d"] = (str(device[0]), )
            self.insert(
                _flags,
                run_flags(
                    self.plock,
                    flags,
                    bin_path,
                    bin_name,
                    pre_flags_str=pre_flags_str,
                    checkjit=checkjit,
                    extended_logging=extended_logging))
        # If batch is valid, run all flags as a single batch
        else:
            tmp_log = StringIO()
            ret = run_batch_of_flags(
                self.plock,
                batch_of_flags,
                bin_path,
                bin_name,
                device=device,
                pre_flags_str=pre_flags_str,
                checkjit=checkjit,
                batch_method=batch_method,
                return_mode=return_mode,
                debug_log=tmp_log,
                stdout_log=stdout_log,
                print_on_fail=print_on_fail,
                extended_logging=extended_logging)

            interrupted = False
            for _batch_of_flags, runlist_results in ret:
                if runlist_results.interrupted:
                    interrupted = True
                    break

            debug_log.write(tmp_log.getvalue())
            # Parse output

            # If some tests did not run or failed, we output the lists of tests leading to that failure. This may not be a critical issue.
            # We couldn't parse anything, this is a critical error and we skip this test
            tmp = 0
            try:
                tmp = sum([len(r[1].outputs) for r in ret])
            except TypeError as e:
                print('[CRITICAL ERROR] Error while processing output from test {}'.format(e))

            print("***** processed {}".format(tmp))
            if tmp == 0:
                print("Test failed. No output")
                self.insert(flags,
                            RunResult(flags, bin_path, bin_name,
                                      None, None, 1, None, interrupted))  # Failed
                return

            for _batch_of_flags, runlist_results in ret:

                error_msg = runlist_results.error_msg

                if runlist_results.return_code == 1 and checkjit:  # jit is triggered, return
                    self.insert(
                        flags,
                        RunResult(flags, bin_path, bin_name, runlist_results.outputs,
                                  error_msg, 1, None, interrupted))
                    continue

                if batch_method == 0:
                    # Verify flags match with what -testsList runs: this should not fail for single thread execution, otherwise something is really wrong in either cudnnTest -testsList or in the parsing
                    # Failures under multithread execution are caused by insufficient workspace memory
                    for (id, (parsed, flags_expected)) in enumerate(
                            zip(runlist_results.parsed, _batch_of_flags)):
                        if parsed is None:  # Failure caused by memory
                            continue
                        flags_match = False
                        flags_expected_str = str(flags_expected)
                        flags_actual_str = 'Flags not found'
                        line = 'Line not found'
                        if parsed["processing"]:
                            flags_actual_str = parsed["processing"].flags
                            line = int(parsed["processing"].line)
                            if (flags_actual_str.strip() == flags_expected_str.strip()) and (
                                    line == id + 1):
                                flags_match = True
                        if not flags_match:
                            if print_on_fail:
                                print_error(
                                    'TEST EXECUTION',
                                    'Critical error, flags do not match or line is wrong: expected (flag, line) ({}, {}), found ({}, {})'
                                    .format(flags_expected_str, id + 1, flags_actual_str, line),
                                    batch_of_flags, runlist_results, stdout_log)
                                print_to_log(
                                    "[TEST EXECUTION] Skipping tests {} with status FAILED".format(flags))
                            self.insert(flags,
                                        RunResult(
                                            flags, bin_path, bin_name, None, None, 1, None, interrupted))  # Failed
                            continue

                    # Go through files, add them to cache if result is considered valid
                    first_ok = False
                    for (output, parsed, flags_ran) in zip(
                            runlist_results.outputs, runlist_results.parsed, _batch_of_flags):
                        if 'd' in flags_ran:
                            del flags_ran['d']
                        if parsed is None:  # Failure caused by memory
                            self.insert(flags_ran,
                                        RunResult(
                                            flags_ran, bin_path, bin_name, output, error_msg, 1, None, interrupted))
                            continue
                        # We check if the test is good (reliable) as long as PASSED or WAIVED / FAILED with specific error codes
                        # We always accept the first one
                        if first_ok:
                            if parsed["status"]:  # We have a status
                                # It's not passed
                                if parsed["status"].status.strip() != 'CUDNNBATCH_PASSED':
                                    # It is assumed the following list of non-success status are not intermitant. That is, one does not expect the status code to change by rerunning the same flags.
                                    cases_ok = {
                                        "CUDNN_STATUS_NOT_SUPPORTED",
                                        "CUDNN_STATUS_ARCH_MISMATCH",
                                        "CUDNN_STATUS_BAD_PARAM",
                                        "waived: Unsupported data type:",
                                        "requires device version >=",
                                        "insufficient for persistent mode",
                                        "too large for persistent rnn",
                                    }
                                    if (not parsed["err_msg"]):
                                        break
                                    err_msg = parsed["err_msg"].err_msg
                                    if not any(err_msg_ok in err_msg for err_msg_ok in cases_ok):
                                        if (parsed["status"].status.strip() == 'CUDNNBATCH_WAIVED' and print_on_waive) or print_on_fail:
                                            print_to_log(
                                                'flag not inserted due to error: "{}"'.format(err_msg))
                                        break
                            else:  # We have no proper status
                                break
                        first_ok = True
                        if parsed["status"]:
                            if (parsed["status"].status.strip() ==
                                    'CUDNNBATCH_PASSED'):
                                return_code = 0
                            elif (parsed["status"].status.strip() ==
                                  'CUDNNBATCH_WAIVED'):
                                return_code = 2
                            else:
                                return_code = 1
                        else:  # If we couldn't find a status, we return Failed.
                            print_to_log("Parser failed to find a status, so setting return code to 1")
                            return_code = 1
                        if return_code != 0:
                            test_result = RunResult(
                                flags=flags_ran,
                                bin_path=bin_path,
                                bin_name=bin_name,
                                output=output,
                                error_msg=error_msg,
                                return_code=return_code,
                                parsed=parsed,
                                interrupted=interrupted)
                        else:
                            test_result = RunResult(
                                flags=flags_ran,
                                bin_path=bin_path,
                                bin_name=bin_name,
                                output=output,
                                error_msg=None,
                                return_code=return_code,
                                parsed=parsed,
                                interrupted=interrupted)

                        # The test is good, so we 1) add it to the cache 2) remove it from the suggested list (if it was in it)
                        self.insert(flags_ran, test_result)
                        key_ran = flags_ran.get_descs_str()
                        if key_ran in suggested_flags:
                            del suggested_flags[key_ran]

                else:  # if batch_method > 0
                    run_lists = runlist_results.parsed

                    for run_idx, run_result in enumerate(run_lists):
                        idx_pat_per_case = re.compile(
                            r'\<\<\<\<\< \[(.*?)\] .*? \>\>\>\>\>')

                        actual_idx = int(
                            idx_pat_per_case.match(
                                run_result.output).groups()[0])

                        cur_run = run_result._replace(
                            flags=_batch_of_flags[actual_idx].get_copy())

                        if 'd' in cur_run.flags:
                            del cur_run.flags['d']

                        self.insert(cur_run.flags, cur_run)
                        key_ran = cur_run.flags.get_descs_str()
                        if key_ran in suggested_flags:
                            del suggested_flags[key_ran]

    def insert(self, flags, run):
        cache_key = get_cache_key(flags)
        with self.cache_lock:
            if cache_key in self.cache:
                self.cache[cache_key] = self.cache[cache_key]._replace(
                    cache_count=self.cache_count)
                self.cache_count += 1
                return True

            while len(self.cache) >= self.max_cache:
                # Pop first item from cache
                prev_first_key, prev_first_cached_run = self.cache.popitem(
                    last=False)
                # Disallow removing any values that were run in this batch
                if self.cache_count - prev_first_cached_run.cache_count < self.batch_size:
                    self.cache[prev_first_key] = prev_first_cached_run

            self.cache[cache_key] = CachedRunResult(self.cache_count, run)
            self.cache_count += 1
            return True

    # suggested_flags = [['-n10','-r3',...],['-n25','-r5',...],...]
    def get_suggested_flags(self, suggested_layers):
        suggested_flags = collections.OrderedDict()
        for layer in suggested_layers:
            flags = layer.flags
            key = get_cache_key(flags)
            with self.cache_lock:
                if key in self.cache:
                    self.cache[key] = self.cache[key]._replace(
                        cache_count=self.cache_count)
                    self.cache_count += 1
                elif key not in suggested_flags:
                    suggested_flags[key] = flags.get_copy()
        return suggested_flags

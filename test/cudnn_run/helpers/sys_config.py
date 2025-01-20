import sys, os
import collections # For NamedTuple

from helpers.cudnn_interface import run_flags
from helpers.Flags           import Flags
import re

platform = sys.platform

# Fix Python 2.x naming (incorrectly includes linux2, linux3, so on...)
if(platform.startswith('linux')):
    platform = 'linux'

# Android is basically linux style
if(platform == "android"):
    platform = 'linux'

if(platform == "qnx"):
    platform = 'linux'

# Sometimes "android" returns "unknown" due to a bug in the makefile
if(platform == "unknown"):
    platform = 'linux'

# Get name of OS library path based on OS
os_lib_paths = {'win32': 'PATH', 'linux': 'LD_LIBRARY_PATH', 'darwin': 'DYLD_LIBRARY_PATH'}
os_lib_path = os_lib_paths[platform]

# Get separator of OS library path based on OS
os_lib_seps = {'win32': ';', 'linux': ':', 'darwin': ':'}
os_lib_sep = os_lib_seps[platform]

# Disable windows error popups
if(platform == 'win32'):
    import ctypes

    SEM_FAILCRITICALERRORS = 1
    SEM_NOGPFAULTERRORBOX  = 2
    SEM_NOOPENFILEERRORBOX = 0x8000

    ctypes.windll.kernel32.SetErrorMode(SEM_FAILCRITICALERRORS |
                                        SEM_NOGPFAULTERRORBOX  |
                                        SEM_NOOPENFILEERRORBOX)

def augment_lib(lib_path):
    previous = ""

    if(os_lib_path in os.environ):
        previous = os_lib_sep + os.environ[os_lib_path]

    # Add lib_path to OS library path
    os.environ[os_lib_path] = lib_path + previous

def enable_logging():
    os.environ["CUDNN_LOGINFO_DBG"] = "1"
    os.environ["CUDNN_LOGDEST_DBG"] = "stdout"


# Redirect glibc backtraces to stderr
def redirect_glibc_backtraces():

    if platform == 'linux':

        os.environ['LIBC_FATAL_STDERR_'] = '1'

def print_lib():
    if(os_lib_path in os.environ):
        print(os.environ[os_lib_path])
    else:
        print("%s not set" % os_lib_path)

def get_gpu_info(device, bin_path, bin_name):
    """Execute `cudnnTest -gpu0` and parse the output to collect GPU attributes

    If multiple devices are specified for cudnn_run.py, always check the first device specified.
    """
    flags = Flags()

    parsed_gpu_field = None
    gpu_device_str = str(device)
    gpu_device_str = gpu_device_str.split(",")[0]
    flags["gpu"] = (gpu_device_str, )
    parsed_gpu_field = "gpu"

    test_name_str = "%s %s" % (bin_name, str(flags))

    print("&&&& RUNNING %s" % test_name_str)

    # Specify test being run
    print("Running test GPU : '%s/%s %s'\n" % (bin_path, bin_name, str(flags)))

    gpu_query = run_flags(None, flags, bin_path, bin_name)

    passed = True

    # Print output
    if(gpu_query.output == None):
        print("No output detected\n")
        passed = False
    else:
        print(gpu_query.output)
        print("")

    # Detect any errors (print if so)
    if(gpu_query.error_msg != None):
        print("[GPU DETECTION] Error Detected: %s" % gpu_query.error_msg)
        passed = False

    if(gpu_query.parsed[parsed_gpu_field] == None):
        print("[GPU DETECTION] Unable to detect GPU")
        passed = False

    # If PASSED, print PASSED and return GPU info
    if(passed):
        # Print detected GPU info
        print("\nGPU DETECTED: %s\n" % str(gpu_query.parsed[parsed_gpu_field]))

        print("&&&& PASSED %s" % test_name_str)

        return gpu_query.parsed[parsed_gpu_field]

    # We haven't passed, print FAILED and return None
    print("&&&& FAILED %s" % test_name_str)
    return None

def print_general_info(bin_path, bin_name):
    flags = Flags()
    id2gpu=dict()
    if "cudnnTest" in bin_name:
        # cudnnTest flags
        flags['g'] = ("", )

    else:
        # cublasTest flags
        flags['v'] = ("", )

    test_name_str = "%s %s" % (bin_name, str(flags))

    print("&&&& RUNNING %s" % test_name_str)

    # Specify test being run
    print("Running test general_info : \'%s/%s %s\'\n" % (bin_path, bin_name, str(flags)))

    gpu_query = run_flags(None, flags, bin_path, bin_name)

    passed = True

    # Print output
    cudart = None
    if(gpu_query.output == None):
        print("No output detected\n")
        passed = False
    else:
        for line in gpu_query.output.split("\n"):
            m=re.search('device (\d+) : Sms=.*Product=\'(.+)\'',line)
            if m:
                id2gpu[m.group(1)]=m.group(2)

            cudart_regex_res=re.search('CUDART_VERSION : (\d+)', line)
            if cudart_regex_res:
                cudart = str(cudart_regex_res.group(1))
        print(gpu_query.output)
        print("")

    # Detect any errors (print if so)
    if(gpu_query.error_msg != None):
        print("[GPU DETECTION] Error Detected: %s" % gpu_query.error_msg)
        passed = False

    # If PASSED, print PASSED and return GPU info
    if(passed):
        print("&&&& PASSED %s" % test_name_str)
        return True,id2gpu,cudart

    # We haven't passed, print FAILED and return None
    print("&&&& FAILED %s" % test_name_str)
    return False,id2gpu,cudart

def get_gpu_filter(gpu):
    result = []

    if(gpu == None):
        return 'UNKNOWN'

    gpu_cap = int(gpu.cap)
    gpu_mem = int(gpu.mem)
    
    if(gpu_cap == 53 or gpu_cap == 62):
        return 'SLOW'

    if(gpu_mem <= 3000 or gpu_cap < 35 or platform != "linux"):
        return 'MID'

    return 'FAST'

    return speed

def get_nvidia_smi_cmd():
    is_win32 = (platform == 'win32')
    smi_filename = 'nvidia-smi.exe' if is_win32 else 'nvidia-smi'

    sys_paths = os.environ["PATH"].split(os.pathsep)
    if is_win32:
        # DVS Windows machines sometimes have trouble picking up nvidia-smi from PATH
        # By inspection, DVS Windows machines have nvidia-smi in 1 of these locations 99% of the time
        # (Win paths are case insensitive)
        win_extra_paths = [os.path.join('C:', os.sep, 'Program Files', 'NVIDIA Corporation', 'NVSMI'),
                              os.path.join('C:', os.sep, 'Windows', 'system32')]
        sys_paths += win_extra_paths

    smi_valid_paths = [os.path.join(path, smi_filename) for path in sys_paths if os.path.isfile(os.path.join(path, smi_filename))]
    if smi_valid_paths:
        print("\nFound nvidia-smi at: " + ", ".join(smi_valid_paths) + " (using first entry)\n")
        return smi_valid_paths[0]

    print("\nCould not locate nvidia-smi\n")
    return None


#!/usr/bin/python3 -u

# Parse the cubin files extracted from cudnn libraries
# Save the SASS content of each kernel into sqlite3 dbs

from heur_diff_utility import parse_cubin_disasm_result_to_db
import argparse
import subprocess
import os

parser = argparse.ArgumentParser(
    description='An automation tool to build, run, integrate, and test cuDNN Heuristics.')
parser.add_argument('-args_file_path', '--args_file_path',
                    metavar='args_file_path',
                    dest='args_file_path',
                    required=True,
                    help='Specify the tmp file that stores a list of cubin files to dump.')
parser.add_argument('-nvdisasm_path', '--nvdisasm_path',
                    metavar='nvdisasm_path',
                    dest='nvdisasm_path',
                    required=True,
                    help='Specify the path to nvdisasm')
args = parser.parse_args()

# get partition idx from tmp file name
partition_idx = os.path.basename(args.args_file_path).split(".")[0].split("_")[-1]

cubin_list = []
with open(args.args_file_path, "r") as tmp_file:
    cubin_list = tmp_file.read().split(",")
assert(len(cubin_list) > 0)


# ***********************************************
# dump cubin to separate sass db
# ***********************************************

for file_idx, file_to_disasm in enumerate(cubin_list):
    if file_to_disasm == "":
        continue
    cmd = []
    cmd += [args.nvdisasm_path]
    cmd += ["-raw"]
    cmd += [file_to_disasm]
    # print(cmd)
    process = subprocess.Popen(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

    output, _ = process.communicate()

    try:
        parse_cubin_disasm_result_to_db(output.decode('utf-8'), file_to_disasm[:-5] + "sqlite3")
    except RuntimeError as e:
        print("Failed to pase cubin files")

    if os.path.exists(file_to_disasm):
        os.remove(file_to_disasm)

    print("(%s) Done (%d/%d): %s" % (partition_idx, file_idx+1, len(cubin_list), file_to_disasm))
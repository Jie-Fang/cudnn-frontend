# Extract the SASS info of all kernels from the lastest cudnn libraries
# Compare the lastest SASS db with the previous one (diff)
# Save the changed kernels into diff SASS dbs

from heur_diff_utility import dump_so_to_cubin, map_partition, merge_sassdb_mp
from multiprocessing import Pool, cpu_count
import os
import time
import re
import argparse

parser = argparse.ArgumentParser(
    description='Convert cudnnlib .so files to sass_db and diff with the previous sass db.')
parser.add_argument('-libpath', '--libpath',
                    metavar='libpath',
                    dest='libpath',
                    required=True,
                    help='Specify the path to cudnn library .so files.')
parser.add_argument('-cuda_bin_dir', '--cuda_bin_dir',
                    metavar='cuda_bin_dir',
                    dest='cuda_bin_dir',
                    required=True,
                    help='Specify the path to cuda binaries, for nvdisasm and cuobjdump')
parser.add_argument('-log_path', '--log_path',
                    metavar='log_path',
                    dest='log_path',
                    required=True,
                    help='Specify the path to save logs ans artifacts')
args = parser.parse_args()

current_dir = os.path.abspath("./")


# ***********************************************
# extract libcudnn*.so -> multiple cubin files
# ***********************************************

# record start time
start_time = time.time()

# the cuda utility `cuobjdump` will extract all cubin files to current dir, so we change dir first
print("Chaning dir to %s" % (args.log_path))
os.chdir(args.log_path)

# extract each .so file to multiple cubin files and save to a dict
# arch_cubin_dict: arch -> (all cubin files from this arch)
arch_cubin_dict = dump_so_to_cubin(
    args.libpath, args.cuda_bin_dir, args.log_path)

print("Chaning dir to %s" % (current_dir))
os.chdir(current_dir)


# ***********************************************
# dump the cubin files to sass db, one per architecture
# ***********************************************

# cubin's abspath is used
all_cubin_files = []
for arch in arch_cubin_dict:
    for cubin_path in arch_cubin_dict[arch]:
        all_cubin_files.append([cubin_path, os.path.getsize(cubin_path)])

# sort from large to small based on size
all_cubin_files.sort(key=lambda tup: tup[1], reverse=True)
all_cubin_files = [item[0] for item in all_cubin_files]

# launch cpu_count() instances to dump cubin files to separate sass db in parallel
try:
    map_partition(cpu_count(), "./persistent_perf/dump_cubin_to_sass_db.py", "-args_file_path",
                all_cubin_files, ["-nvdisasm_path", args.cuda_bin_dir + "/nvdisasm"])
except RuntimeError as e:
    print("Failed to dump cubin to sass db")

# prepare a list of cubin file paths for each arch
cubin_list = []
new_sassdb_paths = []
for key in arch_cubin_dict:
    cubin_list.append(arch_cubin_dict[key])

print("Chaning dir to %s" % (args.log_path))
os.chdir(args.log_path)

# merge shelve dbs of same arch to one
p = Pool(len(cubin_list))
new_sassdb_paths = p.map(merge_sassdb_mp, cubin_list)
p.close()

print("Chaning dir to %s" % (current_dir))
os.chdir(current_dir)

print("[%s] sass db merge completed" % (os.path.basename(__file__)))
print("[%s] dump new sass db time: %s seconds" % (os.path.basename(__file__), time.time() - start_time))


# ***********************************************
# diff sass db with old ones, generate `new only` sass db
# ***********************************************

start_time = time.time()

# walk through the dir to find all old sass db
pattern_old_sass = re.compile(r'^sass_(sm_\d+)\.sqlite3$')
old_sassdb_paths = []
old_sassdb_arch_dict = {}
for root, dirs, files in os.walk(args.log_path):
    for file in files:
        if pattern_old_sass.match(file):
            old_sassdb_paths.append(os.path.join(args.log_path, file))
            arch = pattern_old_sass.match(file).groups()[0]
            old_sassdb_arch_dict[arch] = os.path.join(args.log_path, file)

# prepare the data for sass diffing:
#   search for all archs we currently have on new sass db
#   try to match each one with an old sass db with same arch
#   if no old sass db for this arch is found, meaning this is a new arch
pattern_new_sass = re.compile(r'^.*sass_(sm_\d+)\.[\w\d\-_]+\.sqlite3$')
new_sassdb_arch_dict = {}
arg_list = []
for new_sassdb_path in new_sassdb_paths:
    arch = pattern_new_sass.match(new_sassdb_path).groups()[0]
    new_sassdb_arch_dict[arch] = new_sassdb_path

    # if this is a new arch, the old sass db path will be None
    arg_list.append(
        ",".join([arch, new_sassdb_path, str(old_sassdb_arch_dict.get(arch))]))

# launch multiple instances (the number of new arch) in parallel, each instance diff one arch
common_args_list = []

try:
    map_partition(len(new_sassdb_arch_dict), "./persistent_perf/diff_sass_db.py", "-args_file_path", arg_list, common_args_list)
except RuntimeError as e:
    print("Failed to diff sass db")

print("[%s] diff sass db time: %s seconds" % (os.path.basename(__file__), time.time() - start_time))

# rename `new_sassdb_paths` for next run
for new_sassdb_path in new_sassdb_paths:
    if os.path.exists(new_sassdb_path):
        new_sassdb_path_dir = os.path.dirname(new_sassdb_path)
        new_sassdb_path_basename = os.path.basename(new_sassdb_path)
        os.rename(new_sassdb_path, os.path.join(new_sassdb_path_dir, new_sassdb_path_basename.split(".")[0] + ".sqlite3"))

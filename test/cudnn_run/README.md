# Run tests with cudnn_run.py 

`cudnn_run/` folder is copied from cudnn/scripts in order to reuse cudnn_run.py (and associated label/layer files) to run pycudnnTest.py tests.

The directory structure:

```bash
$ tree -L 1 cudnn_run/
cudnn_run/
├── cudnnLogToTestCmd.py
├── cudnn_run.py
├── helpers
├── persistent_perf
```

Example:

```bash
$python cudnn_run.py -bin_name=pycudnnTest.py -binpath="/path/to/cudnn_frontend/test" -layer_file="example.layer" -label_file="example.label" -global_flags="jsonPath=:/path/to/fusionGraphTests.json" -device=0 -threads=8  -no_rerun -dump_to_log="exampl.log"

RESULT
Failures      : 1 // pycudnnTest.py -gpu0 return error
Successes     : 9
Waived        : 0
Basic Sanity  : 90.00%
Total Time    : 10.01 sec
```
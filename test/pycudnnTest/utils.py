import os, sys
from enum import Enum
import torch
import cupy

# Module scope variables -- can be set from pycudnnTest.py
LOG_RUNTIME = False
DISABLE_CUPTI = True

if LOG_RUNTIME:
    import time

    g_clk_id = time.CLOCK_MONOTONIC_RAW


class ImplementationError(Exception):
    def __init__(self, reason):
        self.reason = reason


class StatusCode(Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    WAIVED = "WAIVED"


def getFwdConvInputDims(outputTensorDim, pad, filterDim, stride, dilation):
    inputTensorDim = [0] * len(outputTensorDim)
    inputTensorDim[0] = outputTensorDim[0]
    inputTensorDim[1] = filterDim[1]
    for dim in range(2, len(inputTensorDim)):
        inputTensorDim[dim] = getSingleFwdConvInputDim(
            outputTensorDim[dim],
            pad[dim - 2],
            filterDim[dim],
            stride[dim - 2],
            dilation[dim - 2],
        )

    return inputTensorDim


def getSingleFwdConvInputDim(outputTensorDim, pad, filterDim, stride, dilation):
    paddedTensorDim = (outputTensorDim - 1) * stride + getSingleFwdConvDilatedFilterDim(
        filterDim, dilation
    )
    inputTensorDim = getSingleFwdConvImageDimFromPadded(paddedTensorDim, pad)
    return int(inputTensorDim)


def getSingleFwdConvImageDimFromPadded(paddedTensorDim, pad):
    tensorDim = paddedTensorDim - (2 * pad)
    return tensorDim


def getSingleFwdConvDilatedFilterDim(filterDim, dilation):
    return ((filterDim - 1) * dilation) + 1


def getFwdConvDilatedFilterDim(filterDim, dilation):
    return ((filterDim - 1) * dilation) + 1


def getFwdConvPaddedImageDim(tensorDim, pad):
    return tensorDim + (2 * pad)


def getFwdConvOutputDim(tensorDim, pad, filterDim, stride, dilation):
    p = (
        getFwdConvPaddedImageDim(tensorDim, pad)
        - getFwdConvDilatedFilterDim(filterDim, dilation)
    ) / stride + 1
    return int(p)


def computeStrideNdTransposedPacked(nbDims, dims, axesOrder):
    inverseTranspose = dict()
    for i in range(nbDims):
        inverseTranspose[axesOrder[i]] = i

    strides = [1] * nbDims
    strides[inverseTranspose[nbDims - 1]] = 1
    for dim in range(nbDims - 2, -1, -1):
        strides[inverseTranspose[dim]] = (
            dims[inverseTranspose[dim + 1]] * strides[inverseTranspose[dim + 1]]
        )

    return strides


def reportCurrentTime(msg):
    if LOG_RUNTIME:
        cur_time = time.clock_gettime_ns(g_clk_id)
        print("[MB_PROFILE] {} {}".format(msg, cur_time))


def create_nhwc_strides(dims):
    nbDims = len(dims)
    # Only support NCHW and NCDHW
    assert nbDims > 3 and nbDims < 6
    stride = [1] * nbDims

    dim_N = 0
    dim_C = 1
    dim_W = nbDims - 1
    dim_H = dim_W - 1
    if nbDims == 5:
        dim_D = dim_H - 1

    stride[dim_C] = 1
    stride[dim_W] = dims[dim_C] * stride[dim_C]
    stride[dim_H] = stride[dim_W] * dims[dim_W]

    if nbDims == 5:
        stride[dim_D] = stride[dim_H] * dims[dim_H]
        stride[dim_N] = stride[dim_D] * dims[dim_D]
    else:
        stride[dim_N] = stride[dim_H] * dims[dim_H]

    return stride


def filter_outliers_iqr(data_list, multiplier=1.5):
    """
    Filter outliers from a list using the Interquartile Range (IQR) method.

    Args:
        data_list: List of float values to filter
        multiplier: IQR multiplier for defining outlier bounds (default: 1.5)
                   - 1.5 is standard for outliers
                   - 3.0 is more conservative, removes only extreme outliers

    Returns:
        tuple: (filtered_list, removed_count, lower_bound, upper_bound)
            - filtered_list: List with outliers removed
            - removed_count: Number of outliers removed
            - lower_bound: Lower threshold used
            - upper_bound: Upper threshold used

    Example:
        >>> data = [1, 2, 3, 4, 5, 100]  # 100 is an outlier
        >>> filtered, count, lower, upper = filter_outliers_iqr(data)
        >>> print(filtered)  # [1, 2, 3, 4, 5]
        >>> print(count)     # 1
    """
    import numpy as np

    if len(data_list) == 0:
        return [], 0, 0, 0

    if len(data_list) < 4:
        # Not enough data points for IQR, return original
        return data_list, 0, min(data_list), max(data_list)

    # Calculate quartiles
    q1 = np.percentile(data_list, 25)
    q3 = np.percentile(data_list, 75)
    iqr = q3 - q1

    # Calculate bounds
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    # Filter data
    filtered_list = [x for x in data_list if lower_bound <= x <= upper_bound]
    removed_count = len(data_list) - len(filtered_list)

    return filtered_list, removed_count, lower_bound, upper_bound


# Launch a delay kernel for 2ms to cover the overhead of the kernel launch
def launch_cupy_delay_kernel_2ms():
    torch_stream = torch.cuda.current_stream()
    cupy_stream = cupy.cuda.ExternalStream(torch_stream.cuda_stream)

    delay_kernel = cupy.RawKernel(
        r"""
        extern "C" __global__
        void delay_kernel(unsigned long long wait_cycles) {
            unsigned long long start = clock64();
            while (clock64() - start < wait_cycles) {;}
        }
        """,
        "delay_kernel",
    )
    wait_cycles = int(2e6)

    with cupy_stream:
        delay_kernel((1,), (1,), (wait_cycles,))


def measure_gpu_runtime_with_events(execution_callback, timingLoop):
    """
    Measure GPU runtime using CUDA Events for precise kernel timing.
    This method is lightweight and suitable for quick performance measurements.

    Args:
        execution_callback: Function to execute the GPU kernel/graph
        timingLoop: Number of iterations to measure

    Returns:
        float: (avg_time) in microseconds
    """
    import numpy as np

    batch_size = min(10, timingLoop)
    time_list = []
    num_batches = (timingLoop + batch_size - 1) // batch_size

    # Warmup the GPU
    for _ in range(5):
        execution_callback()
    torch.cuda.synchronize()
    cupy.cuda.Stream.null.synchronize()

    for batch_idx in range(num_batches):
        # Determine how many iterations in this batch
        remaining = timingLoop - batch_idx * batch_size
        current_batch_size = min(batch_size, remaining)

        start_event_batch = [
            torch.cuda.Event(enable_timing=True) for _ in range(current_batch_size)
        ]
        end_event_batch = [
            torch.cuda.Event(enable_timing=True) for _ in range(current_batch_size)
        ]

        # Submit all kernels in the batch without intermediate synchronization
        for i in range(current_batch_size):
            start_event = start_event_batch[i]
            end_event = end_event_batch[i]

            launch_cupy_delay_kernel_2ms()
            start_event.record()
            execution_callback()
            end_event.record()

            start_event.synchronize()
            end_event.synchronize()

            elapsed = start_event.elapsed_time(end_event)
            time_list.append(elapsed)

        # Synchronize each measurement
        torch.cuda.synchronize()
        cupy.cuda.Stream.null.synchronize()

    # Log environment info for reproducibility (only once)
    if not hasattr(measure_gpu_runtime_with_events, "_env_logged"):
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_capability = torch.cuda.get_device_capability(0)
            print(f"[CUDA_EVENT] GPU: {gpu_name}, Compute Capability: {gpu_capability}")
            print(
                f"[CUDA_EVENT] PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}"
            )
            print(f"[CUDA_EVENT] Runs: {timingLoop}")
        except Exception as e:
            print(f"[CUDA_EVENT] Warning: Could not log environment info: {e}")
        measure_gpu_runtime_with_events._env_logged = True
    # Apply IQR filtering to remove outliers
    filtered_list, removed_count, lower_bound, upper_bound = filter_outliers_iqr(
        time_list, multiplier=1.5
    )

    if removed_count > 0:
        print(
            f"[CUDA_EVENT] IQR filtering: removed {removed_count} outliers (lower={lower_bound*1000:.3f} us, upper={upper_bound*1000:.3f} us)"
        )
        print(
            f"[CUDA_EVENT] Filtered data: {len(filtered_list)}/{len(time_list)} measurements retained"
        )
    else:
        print(f"[CUDA_EVENT] IQR filtering: no outliers detected")
        filtered_list = time_list

    # Use filtered data for final statistics
    min_time = min(filtered_list)
    max_time = max(filtered_list)
    median_time = np.median(filtered_list)
    max_time_ratio = (max_time - min_time) / max_time
    avg_time = np.mean(filtered_list)
    std_time = np.std(filtered_list)
    cv = std_time / avg_time if avg_time > 0 else 0

    print(
        f"[CUDA_EVENT] Summary (min {min_time* 1000:.3f} us, avg {avg_time* 1000:.3f} us, max {max_time* 1000:.3f} us, median {median_time* 1000:.3f} us, std {std_time*1000:.3f} us, CV {cv*100:.2f}%, (max-min)/max {max_time_ratio:.3f})"
    )

    # Return the results in microseconds
    return avg_time * 1000


def measure_gpu_runtime(execution_callback, timingLoop):
    import torch

    # If CUPTI is disabled, still run the graph timingLoop times, just don't profile it here
    # This can be useful in case we want to run through nsys
    if DISABLE_CUPTI:
        for i in range(timingLoop):
            execution_callback()
        return (-1, -1, -1)

    cupti_runtimes = []
    kernel_times = {}

    # Callback function to process a profile
    # This will update cupti_runtimes and kernel_times
    def process_profile(prof):
        # Calculate runtime for the graph we just executed
        end_time = 0
        start_time = max(e.time_range.start for e in prof.events())
        trace_started = False
        # The calculation needs to take into account that some kernels may overlap in time.
        # Also, we want to avoid measuring launch latency.
        # Therefore we 1) skip anything before the first cuLaunchKernel, and 2) find the earliest kernel start time
        # and latest kernel end time
        for event in prof.events():
            # Any event before this is not counted (this avoids measuring cuda memsets and the very first launch latency)
            if "LaunchKernel" in event.name:
                trace_started = True
            if not trace_started:
                continue
            if event.device_time > 0:
                if not event.name in kernel_times:
                    kernel_times[event.name] = []
                kernel_times[event.name].append(event.device_time)
                if event.time_range.end > end_time:
                    end_time = event.time_range.end
                if event.time_range.start < start_time:
                    start_time = event.time_range.start
        # We discard the first, and potentially the last run.
        # First: always seems to be much longer than the rest, even if we set skip_first=N
        # Last: If we ended the profiling loop with a warmup run, no useful kernels will have been profiled (trace_started will never be set)
        if process_profile.first_iteration or not trace_started:
            process_profile.first_iteration = False
            return

        cupti_runtimes.append(end_time - start_time)

    process_profile.first_iteration = True

    # The profile schedule works as follows:
    # 1) the first skip_first runs in the loop are not profiled
    # 2) From then on, we run sets of (warmup+active) runs of which the first warmup runs are not profiled
    # 3) Data on active runs is processed by the on_trace_ready callback function
    warmup = 1
    skip_first = 2
    active = 1
    total_runs = timingLoop + skip_first
    prof_schedule = torch.profiler.schedule(
        wait=0, skip_first=skip_first, warmup=warmup, active=active
    )

    # We need to have at least 2 non-skipped cycles.
    # Otherwise the profile will be empty
    assert total_runs >= skip_first + 2 * (warmup + active + skip_first)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=process_profile,
    ) as prof:
        for i in range(total_runs):
            execution_callback()
            prof.step()

    # lambda function for quick stats
    min_avg_max_ratio = lambda L: (
        (min(L), sum(L) / len(L), max(L), (max(L) - min(L)) / max(L))
        if L
        else (0, 0, 0, 0)
    )

    cupti_runtime_stats = min_avg_max_ratio(cupti_runtimes)
    print(
        "[MB_TIME] Summary (num kernels, min (us), avg (us), max (us), (max-min)/max): {}, {}, {}, {}, {}".format(
            len(kernel_times), *cupti_runtime_stats
        )
    )
    print(
        "[MB_TIME] Found {} kernels found in this cudnn graph. Min/Avg/Max/Ratio (including overlapped times):".format(
            len(kernel_times)
        )
    )
    for kernel in kernel_times:
        print("[MB_TIME]", kernel, min_avg_max_ratio(kernel_times[kernel]))
    return (cupti_runtime_stats[0], cupti_runtime_stats[1], cupti_runtime_stats[2])


class OutputGrabber(object):
    escape_char = "\b"

    def __init__(self, stream=sys.stdout):
        self.origstream = stream
        self.capturedtext = ""
        self.origstreamfd = self.origstream.fileno()
        self.pipe_out, self.pipe_in = os.pipe()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        self.capturedtext = ""
        self.streamfd = os.dup(self.origstreamfd)
        os.dup2(self.pipe_in, self.origstreamfd)

    def stop(self):
        self.origstream.write(self.escape_char)
        self.origstream.flush()
        self.readOutput()
        os.close(self.pipe_in)
        os.close(self.pipe_out)
        os.dup2(self.streamfd, self.origstreamfd)
        os.close(self.streamfd)

    def readOutput(self):
        while True:
            char = os.read(self.pipe_out, 1).decode(self.origstream.encoding)
            if len(char) == 0 or char == "" or not char or self.escape_char in char:
                break
            self.capturedtext += char

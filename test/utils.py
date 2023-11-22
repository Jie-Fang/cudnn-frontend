# Module scope variables -- can be set from pycudnnTest.py
LOG_RUNTIME = False
DISABLE_CUPTI = True

if LOG_RUNTIME:
    import time 
    g_clk_id = time.CLOCK_MONOTONIC_RAW

class ImplementationError(Exception):
    def __init__(self, reason):
        self.reason = reason

def getFwdConvInputDims(outputTensorDim, pad, filterDim, stride, dilation):
    inputTensorDim = [0] * len(outputTensorDim)
    inputTensorDim[0] = outputTensorDim[0]
    inputTensorDim[1] = filterDim[1]
    for dim in range(2, len(inputTensorDim)):
        inputTensorDim[dim] = getSingleFwdConvInputDim(outputTensorDim[dim], pad[dim-2], filterDim[dim], stride[dim - 2], dilation[dim -2])

    return inputTensorDim

def getSingleFwdConvInputDim(outputTensorDim, pad, filterDim, stride, dilation):
    paddedTensorDim = (outputTensorDim - 1) * stride + getSingleFwdConvDilatedFilterDim(filterDim, dilation)
    inputTensorDim  = getSingleFwdConvImageDimFromPadded(paddedTensorDim, pad)
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
    p = (getFwdConvPaddedImageDim(tensorDim, pad) - getFwdConvDilatedFilterDim(filterDim, dilation)) / stride + 1
    return int(p)

def computeStrideNdTransposedPacked(nbDims, dims, axesOrder):
    inverseTranspose = dict()
    for i in range(nbDims):
        inverseTranspose[axesOrder[i]] = i

    print (dims)
    
    strides = [1] * nbDims
    strides[inverseTranspose[nbDims - 1]] = 1
    for dim in range(nbDims - 2, -1, -1):
        strides[inverseTranspose[dim]] = dims[inverseTranspose[dim + 1]] * strides[inverseTranspose[dim + 1]]
    
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

def measure_gpu_runtime(cudnn_graph, variant_pack, workspace, timingLoop):
    import torch
    # For now we will run two methodologies to measure the time (with warm caches)

    # Methodology using CUPTI
    cupti_runtimes = []
    kernel_times = {}

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
            print(event.cuda_time, event.name, event.time_range.start, event.time_range.end)
            # Any event before this is not counted (this avoids measuring cuda memsets and the very first launch latency)
            if "LaunchKernel" in event.name:
                trace_started = True
            if not trace_started:
                continue
            if event.cuda_time > 0:
                if not event.name in kernel_times:
                    kernel_times[event.name] = []
                kernel_times[event.name].append(event.cuda_time)
                if event.time_range.end > end_time:
                    end_time = event.time_range.end
                if event.time_range.start < start_time:
                    start_time = event.time_range.start

        # If we ended the profiling loop with a warmup run, no useful kernels will have been profiled
        if not trace_started:
            return
        print(end_time-start_time)
        cupti_runtimes.append(end_time-start_time)
        
    if DISABLE_CUPTI:
         for i in range(timingLoop):
            cudnn_graph.execute(variant_pack, workspace)
    else:
        warmup = 1
        skip_first = 2
        total_runs = timingLoop + skip_first
        total_runs += total_runs % 2
        prof_schedule = torch.profiler.schedule(wait=0, skip_first=skip_first, warmup=warmup, active=1)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA], schedule=prof_schedule, on_trace_ready=process_profile) as prof:
            for i in range(total_runs):
                cudnn_graph.execute(variant_pack, workspace)
                prof.step()
            

    # TODO(@mbreughe): Remove this
    # Methodology with events and delay kernel
    # For ease of implementation, we will fake the delay kernel by running the graph "warmup_runs_using_events" number of times
    # We use as single measurement to estimate the number of iterations to fit in 1 ms (the time used by heuristics today.)
    delay_kernel_time_us = 1000

    start=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)

    start.record()
    cudnn_graph.execute(variant_pack, workspace)
    end.record()
    torch.cuda.synchronize()
    estimate_kernel_us = start.elapsed_time(end) * 1000
    
    if estimate_kernel_us >= delay_kernel_time_us:
        warmup_runs_using_events = 1
    else:
        warmup_runs_using_events=int(delay_kernel_time_us/estimate_kernel_us)+1

    events_time_us = []
    for i in range(timingLoop):
        # Run the fake delay kernel
        #TODO(@mbreughe): Make this a different kernel
        for warmup_run in range(warmup_runs_using_events):
            cudnn_graph.execute(variant_pack, workspace)
        start.record()
        cudnn_graph.execute(variant_pack, workspace)
        end.record()
        torch.cuda.synchronize()
        events_time_us.append(start.elapsed_time(end) * 1000)
    
    min_avg_max_ratio = lambda L: (min(L), sum(L)/len(L), max(L), (max(L)-min(L))/max(L))
    
    events_runtimes = min_avg_max_ratio(events_time_us)
    
    print("[MB_TIME] delay_kernel (us):", *events_runtimes)

    if not DISABLE_CUPTI:
        cupti_runtime_stats = min_avg_max_ratio(cupti_runtimes)
        print("[MB_TIME] cupti (us):", *cupti_runtime_stats)
        print("[MB_TIME] Summary: {}, {}, {}, {}, {}, {}, {}, {}, {}".format(len(kernel_times), *cupti_runtime_stats, *events_runtimes))
        print("[MB_TIME] Found {} kernels found in this cudnn graph. Min/Avg/Max/Ratio (including overlapped times):".format(len(kernel_times)))
        for kernel in kernel_times:
            print(kernel, min_avg_max_ratio(kernel_times[kernel]))
        return (cupti_runtime_stats[0], cupti_runtime_stats[1], cupti_runtime_stats[2])
    else:
        return (-1,-1,-1)

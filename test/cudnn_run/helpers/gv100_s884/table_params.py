'''
 GV100 / CUDA9.2 heuristic table parameters for s884 algorithms.  Other algorithms will require updating this
file's algo2CTAPerSM and a2kp variables.
'''

algos = [ i for i in xrange(16) ]
npts  = [ 32,64,128,256,512 ]
kpts  = [ (1<<i) for i in xrange(8,17) ]
smc   = 80

# For algo.'s with 1 CTA (block) per SM; those that can support more CTAs per SM are sampled more (by that factor).
# For example, an algo. needing to 2 CTAs per SM for best performance would have 20 sample points vice just 10
# for an algo. where best perf. required just 1 CTA per SM.
# Changing occ_samples will require updating the c/c++ code using the tables.
occ_samples = 10
cta_step = smc / occ_samples

# cublasTest defaults
T=100
niter=10

#
# algo2CTAPerSM and a2kp specific to s884 algorithms / cuda 9.2
algo2CTAPerSM = [4, 3, 2, 2, 1, 2, 1, 1, 4, 3, 2, 2, 1, 2, 1, 1]

# tuple ordering is as follows: rows, cols, splitK, threads, smem, registers.
a2kp = {
    0 : (   64 ,   64 , 0 ,  128 ,  16640  ,  104),
    1 : (   64 ,  128 , 0 ,  128 ,  24832  ,  136),
    2 : (  128 ,   64 , 0 ,  128 ,  32768  ,  184),
    3 : (  128 ,  128 , 0 ,  128 ,  33024  ,  248),
    4 : (  128 ,  256 , 0 ,  256 ,  65536  ,  248),
    5 : (  256 ,   64 , 0 ,  128 ,  41216  ,  248),
    6 : (  256 ,  128 , 0 ,  256 ,  65536  ,  248),
    7 : (  512 ,   64 , 0 ,  256 ,  73984  ,  248),
    8 : (   64 ,   64 , 1 ,  128 ,  16640  ,  104),
    9 : (   64 ,  128 , 1 ,  128 ,  24832  ,  136),
    10 : (  128 ,   64 , 1 ,  128 ,  32768  ,  184),
    11 : (  128 ,  128 , 1 ,  128 ,  33024  ,  248),
    12 : (  128 ,  256 , 1 ,  256 ,  65536  ,  248),
    13 : (  256 ,   64 , 1 ,  128 ,  41216  ,  248),
    14 : (  256 ,  128 , 1 ,  256 ,  65536  ,  248),
    15 : (  512 ,   64 , 1 ,  256 ,  73984  ,  248)
}

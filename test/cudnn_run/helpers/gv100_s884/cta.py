'''
The model of algo,m,n,k to #CTAs launched is derived from CUDA / cuBLAS 9.2.  Supporting other toolkit versions
may (will) require updating this file carefully.
'''

import table_params as tp

def divideUp(x,d): return (x+d-1)/d

def get_splitk_z(k):
    slice_k = min(512, (k+1)/2)
    return divideUp(k, slice_k)

def get_algo2CTAcount(m,n,k,algo):
    ''' For a given problem, set the number of ctas each algo. will launch. Algo. is actually
    100 + algo. within. '''
    ret = None
    if 0 == algo:
        ret = divideUp(m,  64) * divideUp(n,  64)
    elif 1 == algo:
        ret = divideUp(m,  64) * divideUp(n, 128)
    elif 2 == algo:
        ret = divideUp(m, 128) * divideUp(n,  64)
    elif 3 == algo:
        ret = divideUp(m, 128) * divideUp(n, 128)
    elif 4 == algo:
        ret = divideUp(m, 128) * divideUp(n, 256)
    elif 5 == algo:
        ret = divideUp(m, 256) * divideUp(n,  64)
    elif 6 == algo:
        ret = divideUp(m, 256) * divideUp(n, 128)
    elif 7 == algo:
        ret = divideUp(m, 512) * divideUp(n,  64)
    elif 8 == algo:
        ret = divideUp(m,  64) * divideUp(n,  64) * get_splitk_z(k)
    elif 9 == algo:
        ret = divideUp(m,  64) * divideUp(n, 128) * get_splitk_z(k)
    elif 10 == algo:
        ret = divideUp(m, 128) * divideUp(n,  64) * get_splitk_z(k)
    elif 11 == algo:
        ret = divideUp(m, 128) * divideUp(n, 128) * get_splitk_z(k)
    elif 12 == algo:
        ret = divideUp(m, 128) * divideUp(n, 256) * get_splitk_z(k)
    elif 13 == algo:
        ret = divideUp(m, 256) * divideUp(n,  64) * get_splitk_z(k)
    elif 14 == algo:
        ret = divideUp(m, 256) * divideUp(n, 128) * get_splitk_z(k)
    elif 15 == algo:
        ret = divideUp(m, 512) * divideUp(n,  64) * get_splitk_z(k)
    else:
        raise Exception("Unsupported algo. (%d)."(algo))
    #
    return ret

def get_m_ncta(algo, n, k, ncta):
    ''' return first m yielding ncta (or more) ctas for given algo.'''
    m = tp.a2kp[algo][0]
    a2cta = get_algo2CTAcount(m,n,k,algo)
    while a2cta < ncta:
        m += tp.a2kp[algo][0]
        a2cta = get_algo2CTAcount(m,n,k,algo)
    return m,a2cta

def get_rows(algo):
    return tp.a2kp[algo][0]

def next_cta(algo,n,k,rcta):
    '''Given n,k, and the requested number of ctas, compute the m yielding that number
    or greater of ctas.  Given that output, compute the next number of ctas to request
    so that we produce a different m value than the just returned.
    '''
    m,ncta = get_m_ncta(algo, n, k, rcta)
    if ncta > rcta:
        rcta = ncta
        mtmp,nctatmp = get_m_ncta(algo, n, k, rcta)
        if nctatmp == ncta:
            rcta += tp.cta_step
    else:
        rcta += tp.cta_step
    return rcta

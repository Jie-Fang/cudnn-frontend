'''
Helper functions for the table builder top level code.  Mostly text munging,
but pay attention to cublastest_get_perf if new toolkit or architecture support
is being added.
'''
import sys
import os.path
import subprocess as sp
import tempfile as tf

import table_params as tp
import cta

def read_trim(fname):
    ''' read fname's contents into a list and trim leading- and trailing-whitespace'''
    if not os.path.exists(fname):
        return []
    f = open(fname,"r")
    t = f.readlines()
    t = [x.lstrip().rstrip() for x in t]
    f.close()
    return t

def parse_line(logline,table_no):
    ''' parse intermediate output lines from prep_[p1x|p2x].'''
    # format supported is algo:  5 m:    10240 n: 256 k:    16384 cta_per_sm: 2 cta_this_measurment:    160 perf:       76802.744056
    u = logline.split()
    if 14 != len(u):
        return None
    else:
        algo = int(u[1])
        m    = int(u[3])
        n    = int(u[5])
        k    = int(u[7])
        cta_per_sm = int(u[9])
        cta_this_measurement = int(u[11])
        if 1 == table_no:
            cta_lower = 0
            cta_upper = tp.smc * cta_per_sm
        elif 2 == table_no:
            cta_lower = tp.smc * cta_per_sm
            cta_upper = 2 * tp.smc * cta_per_sm
        else:
            raise Exception("Unsupported table_no value (%d) -- should be 1 or 2."%(table_no))
        #
        if cta_this_measurement < cta_lower:
            return None
        elif cta_this_measurement > cta_upper:
            # algo. 9 can be an offender here
            print >>sys.stderr, "truncating occ_idx on logline:",logline
            occ_idx = (tp.smc * cta_per_sm)/tp.cta_step - 1
        else:
            # table_no 1 is the set of perf. measurements for ctas up to smc * cta_per_sm
            # table_no 2 is the analogous, but for measurements smc*cta_per_sm up to
            # 2*smc*cta_per_sm.  But we would like the tables to have the same dimensions,
            # so we subtract off the offset (which is cta_lower).
            occ_idx = cta.divideUp(cta_this_measurement - cta_lower, tp.cta_step)-1
        #
        perf = float(u[13])
        #
        return (algo,m,n,k,cta_per_sm,cta_this_measurement,occ_idx,perf)

def populate_table(tfname, table_no, verboseF):
    ''' organize the intermediate output into tables of (algo,n,k) by list of performance
    where such lists are variable length.'''
    tables = {}
    t = read_trim(tfname)
    for logline in t:
        v = parse_line(logline, table_no)
        (algo,m,n,k,cta_per_sm,cta_this_measurement,occ_idx,perf) = v
        ank_tabkey = (algo,n,k)
        if not tables.has_key(ank_tabkey):
            tables[ank_tabkey] = [0]*((tp.smc * cta_per_sm)/tp.cta_step) # some algo don't exactly fit (like algo 9)
        if perf > tables[ank_tabkey][occ_idx]:
            tables[ank_tabkey][occ_idx] = perf
    for algo in tp.algos:
        for n in tp.npts:
            for k in tp.kpts:
                ank_tabkey = (algo,n,k)
                if not tables.has_key(ank_tabkey):
                    tables[ank_tabkey] = [0]*((tp.algo2CTAPerSM[algo]*tp.smc)/tp.cta_step)
                tab = tables[ank_tabkey]
                if verboseF:
                    print "populate_table, algo: %2d n: %3d k: %6d perf"%(algo,n,k),
                nocc = (tp.algo2CTAPerSM[algo]*tp.smc)/tp.cta_step
                pstr = ", ".join(["%18.6f"%(tab[occ_idx]) for occ_idx in xrange(nocc)])
                if verboseF:
                    print pstr
    return tables
    
def cublastest_get_perf(m,n,k,algo,device,cpath,Tval,numAvg,verboseF):
    ''' execute & average cublasTtest ; configured for s884 kernels at present and 
    may require generalization later. '''
    path = os.path.join(cpath, "cublasTest")
    cmd  = "%s -d%d -Rgemm -b -T%d -Ps -Pinh -Pouth"%(path,device,Tval)
    cmd += " -m%d -n%d -k%d -transA0 -transB0 -algorithm%d -mathMode1 | grep sgemm | sort -rnk +8"%(m,n,k,algo+100)
    perfl = []
    for iter_no in xrange(numAvg):
        if (0 == iter_no) and verboseF:
            print "issuing cmd:",cmd
        f = sp.Popen(cmd, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
        (outline,errline) = f.communicate()
        outlines = outline.split("\n")
        errlines = errline.split("\n")
        best_perf = 0.
        for ol in outlines:
            u = ol.split()
            if 13 != len(u): continue
            perf = float(u[-2])
            if perf > best_perf:
                best_perf = perf
        perfl.append( best_perf )
    return sum(perfl)/len(perfl)

def prep_p1x(dev, cpath, T, niter, lfname, verboseF):
    ''' Create perf. input for table with occupancy < 1 full usage of the GPU. '''
    f = open(lfname,"w")
    ncublas = 0
    for n in tp.npts:
        for k in tp.kpts:
            for algo in tp.algos:
                cta_per_sm = tp.algo2CTAPerSM[ algo ]
                target_cta = cta_per_sm * tp.smc
                m_tmp      = cta.get_rows(algo)
                min_cta    = cta.get_algo2CTAcount(m_tmp,n,k,algo)
                rcta       = max( tp.cta_step, min_cta )
                while rcta <= target_cta:
                    m,ncta = cta.get_m_ncta(algo, n, k, rcta)
                    if verboseF:
                        print "prep_p1x (%6d) n: %4d k: %6d m: %8d algo: %2d cta_per_sm: %d rcta: %6d ncta: %6d"%(ncublas,n,k,m,algo,cta_per_sm,rcta,ncta)
                    #
                    perf = cublastest_get_perf(m,n,k,algo,dev,cpath,T,niter,verboseF)
                    print >>f,"algo: %2d m: %8d n: %3d k: %8d cta_per_sm: %d cta_this_measurment: %6d perf: %18.6f"%(algo,m,n,k,cta_per_sm,ncta,perf)
                    # NOTE: we can ask for the m corresponding to rcta's, given n,k but aren't guaranteed to get an m corresponding to that
                    # requested number of ctas.
                    # We can receive something larger & will sample perf. at that m value.  However, we will want the next m, rcta value to be different
                    # than the one just used.
                    rcta = cta.next_cta(algo,n,k,rcta)
                    ncublas += 1
    f.close()
    if verboseF:
        print "prep_p1x population of %s complete; %d calls to cuBLAS."%(lfname,ncublas)

def prep_p2x(dev, cpath, T, niter, lfname, verboseF):
    ''' Create perf. input for table with occupancy 1 < occ < 2x full usage of the GPU. '''
    f = open(lfname,"w")
    ncublas = 0
    for k in tp.kpts:
        for algo in tp.algos:
            for n in tp.npts:
                cta_per_sm = tp.algo2CTAPerSM[ algo ]
                target_cta = cta_per_sm * tp.smc
                m_tmp      = cta.get_rows(algo)
                min_cta    = cta.get_algo2CTAcount(m_tmp,n,k,algo)
                rcta       = max( tp.cta_step, min_cta )
                # step to rcta >= target_cta
                while rcta < target_cta:
                    rcta = cta.next_cta(algo,n,k,rcta)
                #
                if rcta < target_cta:
                    raise Exception("Unable to advance rcta to correct starting point on algo %d m %d n %d k %d"%(algo,m,n,k))
                #
                while rcta <= 2*target_cta:
                    m,ncta = cta.get_m_ncta(algo, n, k, rcta)
                    if verboseF:
                        print "prep_p2x (%6d) n: %4d k: %6d m: %8d algo: %2d cta_per_sm: %d rcta: %6d ncta: %6d"%(ncublas,n,k,m,algo,cta_per_sm,rcta,ncta)
                    perf = cublastest_get_perf(m,n,k,algo,dev,cpath,T,niter,verboseF)
                    print >>f,"algo: %2d m: %8d n: %3d k: %8d cta_per_sm: %d cta_this_measurment: %6d perf: %18.6f"%(algo,m,n,k,cta_per_sm,ncta,perf)
                    rcta = cta.next_cta(algo,n,k,rcta)
                    ncublas += 1
    f.close()
    if verboseF:    
        print "prep_p2x population of %s complete; %d calls to cuBLAS."%(lfname,ncublas)

def get_rand_fname(pfx,sfx):
    tmpf = tf.NamedTemporaryFile(prefix=pfx,suffix=sfx,delete=False)
    return tmpf.name

def write_preamble(f,stem):
    print >>f, "// autogenerated by %s"%(sys.argv[0])
    print >>f, "#pragma once"
    print >>f, "int ank_%s_npts[] = {%s};"%(stem,",".join([str(x) for x in tp.npts]))
    print >>f, "int ank_%s_kpts[] = {%s};"%(stem,",".join([str(x) for x in tp.kpts]))

def write_table(tH,stem,f):
    print >>f, "double ank_%s_perf[] = {"%(stem)
    ct = 0
    for algo in tp.algos:
        af = (algo == tp.algos[-1])
        for n in tp.npts:
            nf = (n == tp.npts[-1])
            for k in tp.kpts:
                kf = (k == tp.kpts[-1])
                comment = "ct: %8d algo: %2d n: %6d k: %6d perf:"%(ct,algo,n,k)
                cta_per_sm = tp.algo2CTAPerSM[algo]
                ank_tabkey = (algo,n,k)
                perfstr = ", ".join( ["%12.6f"%(x) for x in  tH[ank_tabkey]] )
                ct += len( tH[ank_tabkey] )
                print >>f, perfstr,
                if not (af and nf and kf):
                    print >>f,", /* %s */"%(comment)
                else:
                    print >>f,"/* %s */"%(comment)
    print >>f,"};"


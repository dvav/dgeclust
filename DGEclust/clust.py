## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import DGEclust         as cl
import DGEclust.models  as mdl
import numpy            as np
    
################################################################################

def clust(countData, outDir = 'DGEclust_output', 
    niters = 100000, nlog = 10, pars = [100., 1., 10.], K0 = 100, K = 100, model = mdl.NegBinom, nthreads = 1, extendSim = False, updatePars = True):

    ## load data and prepare output
    mtr = cl.Monitor(outDir, nlog, extendSim)   

    ################################################################################

    ## prepare HDP object
    if extendSim is True:        
        X0   = np.loadtxt(mtr.fX0)
        lw0  = np.loadtxt(mtr.flw0)                        
        LW   = np.loadtxt(mtr.fLW)
        C    = np.loadtxt(mtr.fC, dtype='int')
        Z    = np.loadtxt(mtr.fZ, dtype='int')            
        eta0 = np.loadtxt(mtr.feta)[-1,1]
        eta  = np.loadtxt(mtr.feta)[-1,2:]
        pars = np.loadtxt(mtr.fpars)[-1,2:]        
    else:
        ngenes  = countData.ngenes
        ngroups = countData.ngroups

        pars = np.r_[np.log(countData.counts.mean()), pars]
    
        X0   = model.rPrior(K0, *pars); 
        lw0  = np.tile(-np.log(K0), K0)
        LW   = np.tile(-np.log(K), (ngroups,K))    
        C    = np.random.randint(0, K0, (ngroups,K))        # [ np.zeros(K, dtype = 'int') for i in range(M) ] 
        Z    = np.random.randint(0, K,  (ngroups,ngenes))   # [ np.zeros(N, dtype = 'int') for i in range(M) ]   
        eta0 = 1.
        eta  = np.ones(ngroups)

    hdp  = cl.HDP(X0, lw0, LW, C, Z, eta0, eta, pars)

    ## execute
    sampler = cl.GibbsSampler(nthreads)
    sampler.loop(niters, updatePars, countData, model, hdp, mtr)
    

# Copyright (C) 2012-2013 Dimitrios V. Vavoulis
# Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
# Department of Computer Science
# University of Bristol

################################################################################

import sys
import numpy            as np
import multiprocessing  as mp
import utils            as ut
import stats            as st

################################################################################

def loop(T, tpars, data, model, hdp, rec, nthreads):    
    ## multiprocessing
    nthreads = nthreads if nthreads > 0 else mp.cpu_count() 
    pool     = mp.Pool(processes = nthreads)
        
    ## save initial state ...
    Z, Ko, Ka = hdp.getClusterInfo()
    if rec.t0 == 0:   ## ... only if this is not an extension of a previous simulation 
        rec.update(0, hdp, Z, Ka)    
    
    ## loop
    print >> sys.stderr, 'Iteration {0}/{1} -- {2}/{3} active clusters'.format(0, T, Ka, Ko.size)         
    for t in range(1, T+1):
        ## sample
        _sample(t, pool, data, model, hdp)
        Z, Ko, Ka = hdp.getClusterInfo()

        ## sample parameters     
        if (tpars > 0) and not (t % tpars):
            hdp.pars = model.rParams(hdp.X0[Ko > 0], *hdp.pars)
        
        ## save state and log progress 
        rec.update(t, hdp, Z, Ka)        
        print >> sys.stderr, 'Iteration {0}/{1} -- {2}/{3} active clusters'.format(t, T, Ka, Ko.size)             

################################################################################

def _sample(t, pool, data, model, hdp):
    '''
    sample
    '''
    
    ## sample LW, C and Z
    M = data.counts.shape[0]
    args = zip(hdp.LW, hdp.C, hdp.Z, hdp.eta, data.counts, data.exposures, (hdp.lw0,) * M, (hdp.X0,) * M, (model.dLogLik,) * M)
    hdp.LW, hdp.C, hdp.Z, hdp.eta = zip(*pool.map(_sampleWCZ, args))

    ## get cluster info
    K0         = hdp.lw0.size
    Ko, Ki, Ka = ut.getClusterInfo(K0, np.asarray(hdp.C).ravel()) 
    idxs       = Ki.nonzero()[0] 
            
    ## sample lw0 
    hdp.lw0 = st.stick(Ko, hdp.eta0) 
    
    ## sample X0        
    hdp.X0[~Ki] = model.rPrior(K0 - Ka, *hdp.pars)                      ## sample inactive clusters from the prior
    args = zip(hdp.X0[Ki], idxs, (hdp.C,) * Ka, (hdp.Z,) * Ka, (data,) * Ka, (hdp.pars,) * Ka, (model.rPost,) * Ka)
    hdp.X0[Ki] = pool.map(_sampleX0, args)                          ## sample active clusters from the posterior
    
    ## sample eta0
    hdp.eta0 = st.gamma(6.,0.2) # st.rEta(hdp.eta0, Ka, hdp.Z[0].size * M)
    
################################################################################

def _sampleX0(args):
    x0, idx, C, Z, data, pars, rPost = args
    return rPost(x0, idx, C, Z, data, *pars)
    
################################################################################

def _sampleWCZ(args): 
    lw, c, z, eta, counts, exposure, lw0, X0, dLogLik = args
    
    ## compute log-likelihood
    loglik = dLogLik(X0, counts, exposure)

    ## sample z
    logw = lw + loglik[:,c]
    logw = ut.normalizeLogWeights(logw.T)
    z    = st.categorical(np.exp(logw))

    ## get cluster info
    K           = c.size
    Ko, Ki, Ka  = ut.getClusterInfo(K, z)
    idxs        = Ki.nonzero()[0]
            
    ## sample lw
    lw = st.stick(Ko, eta) 
            
    ## sample c
    c[~Ki] = st.categorical(np.exp(lw0), Ko.size - Ka)                           ## sample inactive clusters from the prior   
    logw   = lw0 + np.asarray([ loglik[z == idx].sum(0) for idx in idxs ])       ## sample active clusters from the posterior
    logw   = ut.normalizeLogWeights(logw.T)
    c[Ki]  = st.categorical(np.exp(logw))  

    ## sample eta
    eta = st.gamma(6.,0.2) #st.rEta(eta, Ka, z.size)
                    
    ## return
    return lw, c, z, eta

################################################################################
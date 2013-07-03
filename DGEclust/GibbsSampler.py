## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import sys
import numpy            as np
import multiprocessing  as mp
import utils            as ut
import stats.rand       as rn

################################################################################

class GibbsSampler(object):
    def __init__(self, nthreads):
        ## multiprocessing
        nthreads  = nthreads if nthreads > 0 else mp.cpu_count() 
        self.pool = mp.Pool(processes = nthreads)

    ############################################################################
        
    ## main loop
    def loop(self, T, U, data, model, hdp, mtr):        
        ## prepare data
        counts    = data.values.astype('double').T
        exposures = data.exposures.values 

        ## save initial state ...
        Z, Ko, Ka = hdp.getClusterInfo()
        if mtr.t0 == 0:   ## ... only if this is not an extension of a previous simulation 
            mtr.update(0, hdp, Z, Ka)    
    
        ## loop
        print >> sys.stderr, 'Iteration {0}/{1} -- {2}/{3} active clusters'.format(0, T, Ka, Ko.size)         
        for t in range(1, T+1):
            ## sample
            self._sample(t, counts, exposures, model, hdp)
            Z, Ko, Ka = hdp.getClusterInfo()
    
            ## sample parameters     
            if U: hdp.pars = model.rParams(hdp.X0[Ko > 0], *hdp.pars)
            
            ## save state and log progress 
            mtr.update(t, hdp, Z, Ka)        
            print >> sys.stderr, 'Iteration {0}/{1} -- {2}/{3} active clusters'.format(t, T, Ka, Ko.size)             

    ############################################################################

    def _sample(self, t, counts, exposures, model, hdp):
        '''
        sample
        '''
        
        ## sample LW, C and Z
        M = counts.shape[0]
        args = zip(hdp.LW, hdp.C, hdp.Z, hdp.eta, counts, exposures, (hdp.lw0,) * M, (hdp.X0,) * M, (model.dLogLik,) * M)

        hdp.LW, hdp.C, hdp.Z, hdp.eta = zip(*self.pool.map(_sampleWCZ, args))
    
        ## get cluster info
        K0         = hdp.lw0.size
        Ko, Ki, Ka = ut.getClusterInfo(K0, np.asarray(hdp.C).ravel()) 
        idxs       = Ki.nonzero()[0] 
                
        ## sample lw0 
        hdp.lw0 = rn.stick(Ko, hdp.eta0) 
        
        ## sample X0        
        hdp.X0[~Ki] = model.rPrior(K0 - Ka, *hdp.pars)                      ## sample inactive clusters from the prior
        args = zip(hdp.X0[Ki], idxs, (hdp.C,) * Ka, (hdp.Z,) * Ka, (counts,) * Ka, (exposures,) * Ka, (hdp.pars,) * Ka, (model.rPost,) * Ka)
        hdp.X0[Ki] = self.pool.map(_sampleX0, args)                          ## sample active clusters from the posterior
        
        ## sample eta0
        hdp.eta0 = rn.gamma(6.,0.2) # st.rEta(hdp.eta0, Ka, hdp.Z[0].size * M)
    
################################################################################

def _sampleX0(args):
    x0, idx, C, Z, counts, exposures, pars, rPost = args
    return rPost(x0, idx, C, Z, counts, exposures, *pars)
    
################################################################################

def _sampleWCZ(args): 
    lw, c, z, eta, counts, exposure, lw0, X0, dLogLik = args
    
    ## compute log-likelihood
    loglik = dLogLik(X0, counts, exposure)

    ## sample z
    logw = lw + loglik[:,c]
    logw = ut.normalizeLogWeights(logw.T)
    z    = rn.categorical(np.exp(logw))

    ## get cluster info
    K           = c.size
    Ko, Ki, Ka  = ut.getClusterInfo(K, z)
    idxs        = Ki.nonzero()[0]
            
    ## sample lw
    lw = rn.stick(Ko, eta) 
            
    ## sample c
    c[~Ki] = rn.categorical(np.exp(lw0), Ko.size - Ka)                           ## sample inactive clusters from the prior   
    logw   = lw0 + np.asarray([ loglik[z == idx].sum(0) for idx in idxs ])       ## sample active clusters from the posterior
    logw   = ut.normalizeLogWeights(logw.T)
    c[Ki]  = rn.categorical(np.exp(logw))  

    ## sample eta
    eta = rn.gamma(6.,0.2) #st.rEta(eta, Ka, z.size)
                    
    ## return
    return lw, c, z, eta

################################################################################
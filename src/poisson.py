'''
Created on Apr 20, 2013

@author: dimitris
'''
################################################################################

import numpy as np
import stats as st

################################################################################
        
def rParams(beta):    
    return st.normal_mean_var(beta)
    
################################################################################
    
def rPrior(N, mean, var):    
    return st.normal(mean, var, N)
        
################################################################################
   
def rPost(beta, idx, C, Z, data, *pars):     
    # # make proposal
    beta_ = beta * np.exp(0.01 * st.normal(N=1)) 

    # # compute posterior densities
    lp  = _dLogPost(beta,  idx, C, Z, data, *pars)
    lp_ = _dLogPost(beta_, idx, C, Z, data, *pars)        
    
    # # do Metropolis step
    if (lp_ > lp) or (st.uniform() < np.exp(lp_ - lp)):
        beta = beta_

    # return        
    return beta

################################################################################
    
def dLogLik(beta, counts, exposure):        
    return st.dLogPoisson(counts.reshape(-1,1), exposure * np.exp(beta))
    
################################################################################
        
def _dLogPrior(beta, mean, var):        
    return st.dLogNormal(beta, mean, var) 
    
################################################################################
    
def _dLogPost(beta, idx, C, Z, data, *pars):
    logprior = _dLogPrior(beta, *pars)
    loglik = [ dLogLik(beta, counts[c[z] == idx], exposure).sum() for c, z, counts, exposure in zip(C, Z, data.counts, data.exposures) ]
        
    return logprior + np.sum(loglik)
    
################################################################################
    
def generateData(N = 1000, M = 2, L = 4, ):
    pass

################################################################################
        

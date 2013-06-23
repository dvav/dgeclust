'''
Created on Apr 20, 2013

@author: dimitris
'''
################################################################################

import numpy as np
import stats as st
import utils as ut
import stats as st

################################################################################

_GAMMA_SHAPE = 1.

################################################################################
        
def rParams(x0, mean, var, scale):
    alpha = x0[:, 0]
    beta  = x0[:, 1] 

    scale = st.gamma_scale(alpha, shape = _GAMMA_SHAPE)	
    mean, var = st.normal_mean_var(beta)
    
    return mean, var, scale

################################################################################
    
def rPrior(N, mean, var, scale):    
    alpha = st.gamma(shape = _GAMMA_SHAPE, scale = scale, N = (N, 1)) 
    beta  = st.normal(mean, var, (N, 1))
    
    return np.hstack((alpha, beta))
    
################################################################################
   
def rPost(x0, idx, C, Z, data, *pars):     
    # # make proposal
    x0_ = x0 * np.exp(0.01 * st.normal(N=2)) 

    # # compute posterior densities
    lp  = _dLogPost(x0,  idx, C, Z, data, *pars)
    lp_ = _dLogPost(x0_, idx, C, Z, data, *pars)        
    
    # # do Metropolis step
    if (lp_ > lp) or (st.uniform() < np.exp(lp_ - lp)):
        x0 = x0_

    # return        
    return x0

################################################################################
    
def dLogLik(X0, counts, exposure):
    X0 = np.atleast_2d(X0)
    
    alpha = X0[:,0] 
    beta  = X0[:,1]
    
    mu = exposure * np.exp(beta)
    p  = alpha / (alpha + mu)
    
    return st.dLogNegBinomial(counts.reshape(-1,1), alpha, p)
    
################################################################################
        
def _dLogPrior(x0, mean, var, scale):
    alpha, beta = x0
    
    logprior_alpha = st.dLogGamma(alpha, shape = _GAMMA_SHAPE, scale = scale)
    logprior_beta  = st.dLogNormal(beta, mean, var)
    
    return logprior_alpha + logprior_beta 
    
################################################################################
    
def _dLogPost(x0, idx, C, Z, data, *pars):
    logprior = _dLogPrior(x0, *pars)
    loglik = [ dLogLik(x0, counts[c[z] == idx], exposure).sum() for c, z, counts, exposure in zip(C, Z, data.counts, data.exposures) ]
        
    return logprior + np.sum(loglik)
    
################################################################################
    
def generateData(N = 1000, M = 2, L = 4, ):
    pass

################################################################################
        

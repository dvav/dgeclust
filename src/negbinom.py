'''
Created on Apr 20, 2013

@author: dimitris
'''
################################################################################

import numpy as np
import stats as st
import utils as ut

################################################################################

_GAMMA_SHAPE = 0.8

################################################################################
        
def rParams(x0, mean, var, scale):
    phi   = x0[:, 0]
    beta  = x0[:, 1] 

    scale = st.gamma_scale(phi, shape = _GAMMA_SHAPE)	
    mean, var = st.normal_mean_var(beta)
    
    return mean, var, scale

################################################################################
    
def rPrior(N, mean, var, scale):    
    phi   = st.gamma(shape = _GAMMA_SHAPE, scale = scale, N = (N, 1)) 
    beta  = st.normal(mean, var, (N, 1))
    
    return np.hstack((phi, beta))
    
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
    
    phi  = X0[:,0] 
    beta = X0[:,1]
    
    alpha = 1. / phi

    mu = exposure * np.exp(beta)
    p  = alpha / (alpha + mu)
    
    return st.dLogNegBinomial(counts.reshape(-1,1), alpha, p)
    
################################################################################
        
def _dLogPrior(x0, mean, var, scale):
    phi, beta = x0
    
    logprior_phi   = st.dLogGamma(phi, shape = _GAMMA_SHAPE, scale = scale)
    logprior_beta  = st.dLogNormal(beta, mean, var)
    
    return logprior_phi + logprior_beta 
    
################################################################################
    
def _dLogPost(x0, idx, C, Z, data, *pars):
    logprior = _dLogPrior(x0, *pars)
    loglik = [ dLogLik(x0, counts[c[z] == idx], exposure).sum() for c, z, counts, exposure in zip(C, Z, data.counts, data.exposures) ]
        
    return logprior + np.sum(loglik)
    
################################################################################
    
def generateData(N = 1000, M = 2, L = 4, ):
    pass

################################################################################
        

# Copyright (C) 2012-2013 Dimitrios V. Vavoulis
# Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
# Department of Computer Science
# University of Bristol

################################################################################

import numpy as np
import stats as st
import utils as ut

################################################################################

PHI_MIN = 1e-6

################################################################################
        
def rParams(x0, mean, var, shape, scale):
    phi   = x0[:, 0]
    beta  = x0[:, 1] 

    # shape, scale = st.gamma_shape_scale(phi, shape, scale)    
    scale        = st.gamma_scale(phi, shape)    
    mean, var    = st.normal_mean_var(beta)
    
    return mean, var, shape, scale  

################################################################################

def rPrior(N, mean, var, shape, scale):    
    phi   = st.gamma(shape, scale, (N, 1)) + PHI_MIN;       ## make sure phi never becomes zero   
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
        
def _dLogPrior(x0, mean, var, shape, scale):
    phi, beta = x0
    
    logprior_phi   = st.dLogGamma(phi, shape, scale)
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
        

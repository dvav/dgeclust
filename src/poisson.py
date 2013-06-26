# Copyright (C) 2012-2013 Dimitrios V. Vavoulis
# Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
# Department of Computer Science
# University of Bristol

################################################################################

import numpy as np
import stats as st

################################################################################
        
# def rParams(beta):    
#     return st.normal_mean_var(beta)
#     
# ################################################################################
#     
# def rPrior(N, mean, var):    
#     return st.normal(mean, var, N)
#         
# ################################################################################
#    
# def rPost(beta, idx, C, Z, data, *pars):     
#     # # make proposal
#     beta_ = beta * np.exp(0.01 * st.normal(N=1)) 
# 
#     # # compute posterior densities
#     lp  = _dLogPost(beta,  idx, C, Z, data, *pars)
#     lp_ = _dLogPost(beta_, idx, C, Z, data, *pars)        
#     
#     # # do Metropolis step
#     if (lp_ > lp) or (st.uniform() < np.exp(lp_ - lp)):
#         beta = beta_
# 
#     # return        
#     return beta
# 
# ################################################################################
#     
# def dLogLik(beta, counts, exposure):        
#     return st.dLogPoisson(counts.reshape(-1,1), exposure * np.exp(beta))
#     
# ################################################################################
#         
# def _dLogPrior(beta, mean, var):        
#     return st.dLogNormal(beta, mean, var) 
#     
# ################################################################################
#     
# def _dLogPost(beta, idx, C, Z, data, *pars):
#     logprior = _dLogPrior(beta, *pars)
#     loglik = [ dLogLik(beta, counts[c[z] == idx], exposure).sum() for c, z, counts, exposure in zip(C, Z, data.counts, data.exposures) ]
#         
#     return logprior + np.sum(loglik)
#     
# ################################################################################
    
def generateData(N = 1000, M = 2, L = 4, ):
    pass

################################################################################

def rParams(rate, shape, scale):    
    return st.gamma_shape_scale(rate, shape, scale)
    
################################################################################
    
def rPrior(N, shape, scale):    
    return st.gamma(shape, scale, N)
        
################################################################################
   
def rPost(rate, idx, C, Z, data, *pars):     
    # # make proposal
    rate_ = rate * np.exp(0.01 * st.normal(N=1)) 

    # # compute posterior densities
    lp  = _dLogPost(rate,  idx, C, Z, data, *pars)
    lp_ = _dLogPost(rate_, idx, C, Z, data, *pars)        
    
    # # do Metropolis step
    if (lp_ > lp) or (st.uniform() < np.exp(lp_ - lp)):
        rate = rate_

    # return        
    return rate
    
################################################################################
    
def dLogLik(rate, counts, exposure):        
    return st.dLogPoisson(counts.reshape(-1,1), exposure * rate)
    
################################################################################
        
def _dLogPrior(rate, shape, scale):        
    return st.dLogGamma(rate, shape, scale) 
    
################################################################################
    
def _dLogPost(rate, idx, C, Z, data, *pars):
    logprior = _dLogPrior(rate, *pars)
    loglik = [ dLogLik(rate, counts[c[z] == idx], exposure).sum() for c, z, counts, exposure in zip(C, Z, data.counts, data.exposures) ]
        
    return logprior + np.sum(loglik)
    
################################################################################
        

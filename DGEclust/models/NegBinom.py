## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import numpy               as np
import DGEclust.stats.rand as rn
import DGEclust.stats.conj as cj
import DGEclust.stats.dist as ds
import DGEclust.utils      as ut

################################################################################

PHI_MIN = 1e-6

################################################################################

def rParams(x0, mean, var, shape, scale, dsum, d2sum, N, ldsum, dsum2, N2):
    phi   = x0[:,0]
    beta  = x0[:,1] 
    
    mean,  var,   dsum,  d2sum, N  = cj.normal_mean_var(beta,  dsum,  d2sum, N)   
    shape, scale, ldsum, dsum2, N2 = cj.gamma_shape_scale(phi, shape, scale, ldsum, dsum2, N2) 
    
    return mean, var, shape, scale, dsum, d2sum, N, ldsum, dsum2, N2  
        
################################################################################
        
def rPrior(N, mean, var, shape, scale, *args):    
    phi   = rn.gamma(shape, scale, (N, 1)) + PHI_MIN;       ## make sure phi never becomes zero   
    beta  = rn.normal(mean, var, (N, 1))
        
    return np.hstack((phi, beta))
    
################################################################################
   
def rPost(x0, idx, C, Z, counts, exposures, *pars):     
    ## make proposal
    x0_ = x0 * np.exp(0.01 * rn.normal(N=2)) 

    ## compute posterior densities
    lp  = _dLogPost(x0,  idx, C, Z, counts, exposures, *pars)
    lp_ = _dLogPost(x0_, idx, C, Z, counts, exposures, *pars)        
    
    # # do Metropolis step
    if (lp_ > lp) or (rn.uniform() < np.exp(lp_ - lp)):
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
    
    return ds.dLogNegBinomial(counts.reshape(-1,1), alpha, p)
    
################################################################################
        
def _dLogPrior(x0, mean, var, shape, scale, *args):
    phi, beta = x0
    
    logprior_phi   = ds.dLogGamma(phi, shape, scale)
    logprior_beta  = ds.dLogNormal(beta, mean, var)
    
    return logprior_phi + logprior_beta 
    
################################################################################
    
def _dLogPost(x0, idx, C, Z, counts, exposures, *pars):
    logprior = _dLogPrior(x0, *pars)
    loglik = [ dLogLik(x0, cnts[c[z] == idx], exposure).sum() for c, z, cnts, exposure in zip(C, Z, counts, exposures) ]
        
    return logprior + np.sum(loglik)
    
################################################################################
    
def generateData(N = 1000, M = 2, L = 4, ):
    pass

################################################################################
        

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

def rParams(x0, mean, var, shape, scale):
    phi   = x0[:,0]
    beta  = x0[:,1] 
    
    # mean,  var,   _,  _, _  = cj.normal_mean_var(beta)   
    # shape, scale, _, _, _   = cj.gamma_shape_scale(phi, shape, scale) 

    var,   _,  _, _  = cj.normal_var(beta, mean)   
    scale, _, _      = cj.gamma_scale(phi, shape) 
    
    return mean, var, shape, scale  
        
################################################################################
        
def rPrior(N, mean, var, shape, scale):    
    phi   = rn.gamma(shape, scale, (N, 1)) + PHI_MIN;       ## make sure phi never becomes zero   
    beta  = rn.normal(mean, var, (N, 1))
        
    return np.hstack((phi, beta))
    
################################################################################
   
def rPost(x0, idx, C, Z, countData, *pars):     
    ## make proposal
    x0_ = x0 * np.exp(0.01 * rn.normal(N=2)) 

    ## compute posterior densities
    lp  = _dLogPost(x0,  idx, C, Z, countData, *pars)
    lp_ = _dLogPost(x0_, idx, C, Z, countData, *pars)        
    
    # # do Metropolis step
    if (lp_ > lp) or (rn.uniform() < np.exp(lp_ - lp)):
        x0 = x0_

    # return        
    return x0

################################################################################
    
def dLogLik(X0, counts, exposures):
    X0        = np.atleast_2d(X0)
    counts    = np.atleast_2d(counts)
    exposures = np.atleast_2d(exposures)
    
    ## read X0
    phi   = X0[:,0] 
    beta  = X0[:,1]
    alpha = 1. / phi
    
    ## compute mu and p 
    mu     = exposures.reshape(-1,1) * np.exp(beta)
    p      = alpha / (alpha + mu)

    ## compute loglik
    loglik = [ ds.dLogNegBinomial(cnts.reshape(-1,1), alpha, pi) for cnts, pi in zip(counts, p) ]
    
    return np.sum(loglik, 0)
    
################################################################################
        
def _dLogPrior(x0, mean, var, shape, scale):
    phi, beta = x0
    
    logprior_phi   = ds.dLogGamma(phi, shape, scale)
    logprior_beta  = ds.dLogNormal(beta, mean, var)
    
    return logprior_phi + logprior_beta 
    
################################################################################
    
def _dLogPost(x0, idx, C, Z, countData, *pars):
    counts    = countData.counts
    exposures = countData.exposures
    groups    = countData.groups
    
    logprior = _dLogPrior(x0, *pars)
    loglik = [ dLogLik(x0, counts[group][:, c[z] == idx], exposures[group]).sum() for c, z, group in zip(C, Z, groups)]
        
    return logprior + np.sum(loglik)
    
################################################################################
    
def generateData(N = 1000, M = 2, L = 4, ):
    pass

################################################################################
        

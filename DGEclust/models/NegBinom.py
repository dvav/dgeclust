## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import numpy               as np
import scipy.special       as sp
import DGEclust.stats.rand as rn
import DGEclust.stats.conj as cj
import DGEclust.stats.dist as ds
import DGEclust.utils      as ut

################################################################################

def rParams(x0, phi_min, phi_max, a = 1., b = 1.):
    phi = x0[:,0]
    phi_max = rn.pareto(a + phi.size, max(np.r_[phi, b]))
    
    ## return
    return phi_min, phi_max
    
################################################################################
        
def rPrior(N, phi_min, phi_max):    
    ## sample alpha and p
    phi = rn.uniform(phi_min, phi_max, (N, 1))  ## rn.gamma(shape, scale, (N,1))         
    p   = rn.uniform(0, 1,  (N, 1))
    
    ## compute mu
    mu = (1 - p) / (p * phi)
    
    ## return    
    return np.hstack((phi, mu))
    
################################################################################

def _dLogPostAlpha(alpha, counts, ncounts, cntsum, shape, scale):
    # beta = 1. / scale ## the scale parameter of the inverse gamma
    
    lp  = sp.betaln(ncounts * alpha + 1., cntsum + 1.)
    lp += ( sp.gammaln(counts + alpha) - sp.gammaln(alpha) - sp.gammaln(counts + 1.) ).sum()
    # lp += shape * np.log(beta) - sp.gammaln(shape) - (shape + 1) * np.log(alpha) - beta / alpha
 
    ##
    return lp    
## 
        
def rPost(x0, idx, C, Z, countData, *pars):     
    phi   = x0[0]
    alpha =  1 / phi
    
    ## read data
    counts = countData.countsNorm
    groups = countData.groups
    
    counts  = np.hstack([ counts[group][:, c[z] == idx].ravel() for c, z, group in zip(C, Z, groups) ])
    cntsum  = counts.sum()
    ncounts = counts.size
    
    ## sample alpha
    alpha_ = alpha * np.exp(0.01 * rn.normal())    ## make proposal
    
    lp     = _dLogPostAlpha(alpha,  counts, ncounts, cntsum, *pars)  ## compute posterior density for alpha
    lp_    = _dLogPostAlpha(alpha_, counts, ncounts, cntsum, *pars)  ## compute posterior density for alpha
    
    if (lp_ > lp) or (rn.uniform() < np.exp(lp_ - lp)): ## do Metropolis step
        alpha = alpha_
         
    ## sample p
    p = rn.beta(ncounts * alpha + 1., cntsum + 1.)     

    ## compute mu
    phi = 1. / alpha
    mu  = alpha * (1. - p) / p
    
    # return            
    return (phi, mu)
   
################################################################################
    
def dLogLik(X0, counts):
    X0       = np.atleast_2d(X0)
    counts   = np.atleast_2d(counts)
    nsamples = counts.shape[0]
    
    ## read X0
    phi   = X0[:,0] 
    mu    = X0[:,1]
    alpha = 1. / phi
    
    ## compute mu and p 
    mu = np.tile(mu,(nsamples,1))
    p  = alpha / (alpha + mu)

    ## compute loglik
    loglik = [ ds.dLogNegBinomial(cnts.reshape(-1,1), alpha, pi) for cnts, pi in zip(counts, p) ]
    
    ## return
    return np.sum(loglik, 0)
    
################################################################################
        
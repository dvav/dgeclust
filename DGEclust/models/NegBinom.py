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

PHI_MIN = 1e-12

################################################################################

def rParams(x0, shape, scale):
    phi = x0[:,0]

    # scale, _, _ = cj.gamma_scale(phi, shape)
    shape, scale, _, _, _ = cj.gamma_shape_scale(phi, shape, scale)
    
    ## return
    return shape, scale
    
################################################################################
        
def rPrior(N, shape, scale):    
    ## sample alpha and p
    phi = rn.gamma(shape, scale, (N,1)) + PHI_MIN        
    p   = rn.uniform(0, 1,  (N, 1))
    
    ## compute mu
    mu  = (1 - p) / (p * phi)
    
    ## return    
    return np.hstack((phi, mu))
    
################################################################################

def _logBinomCoeff(counts, phi):
    alpha = 1. / phi
    coeff = sp.gammaln(counts + alpha) - sp.gammaln(alpha) - sp.gammaln(counts + 1.)
    
    ##
    return coeff

def rPost(x0, idx, C, Z, countData, *pars):     
    phi = x0[0]
    
    ## read data
    counts = countData.countsNorm
    groups = countData.groups
    
    counts  = np.hstack([ counts[group][:, c[z] == idx].ravel() for c, z, group in zip(C, Z, groups) ])
    cntsum  = counts.sum()
    ncounts = counts.size
    
    ## sample phi using Metropolis
    phi_ = phi * np.exp(0.01 * rn.normal())    ## make proposal
    
    lp  = sp.betaln(ncounts / phi  + 1., cntsum + 1.) + _logBinomCoeff(counts, phi).sum()   + ds.dLogGamma(phi,  *pars)     ## posterior density for phi
    lp_ = sp.betaln(ncounts / phi_ + 1., cntsum + 1.) + _logBinomCoeff(counts, phi_).sum()  + ds.dLogGamma(phi_, *pars)     ## posterior density for phi_

    if (lp_ > lp) or (rn.uniform() < np.exp(lp_ - lp)): ## do Metropolis step
        phi = phi_
         
    ## sample p
    p = rn.beta(ncounts / phi + 1., cntsum + 1.)     

    ## compute mu
    mu = (1. - p) / (p * phi)
    
    ## return            
    return phi, mu
    
################################################################################

def dLogLik(X0, counts):
    X0       = np.atleast_2d(X0)
    counts   = np.atleast_2d(counts)
    ngenes   = counts.shape[1]
    
    ## read X0
    phi   = X0[:,[0]] 
    mu    = X0[:,[1]]
    
    ## compute p 
    p  = 1. / (1. + mu * phi)

    ## compute loglik
    loglik = ds.dLogNegBinomial(counts[:,np.newaxis,:], phi, p).sum(0).T
    
    ## return
    return loglik
        
################################################################################
        
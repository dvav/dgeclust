## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import numpy         as np
import numpy.random  as rn
import scipy.stats   as st

################################################################################

def uniform(low = 0., high = 1., N = None):
    return st.uniform.rvs(low, high - low, N)
    
################################################################################

def normal(mean = 0., var = 1., N = None):
    return st.norm.rvs(mean, np.sqrt(var), N)
    
################################################################################

def beta(a = 1., b = 1., N = None):
    return st.beta.rvs(a, b, 0., 1., N)

################################################################################

def exponential(rate = 1., N = None):
    return st.expon.rvs(0., 1. / rate, N)

################################################################################

def gamma(shape = 2., scale = 1., N = None):
    return st.gamma.rvs(shape, 0., scale, N)

################################################################################

def invgamma(shape = 2., scale = 1., N = None):
    return st.invgamma.rvs(shape, 0., scale, N)

################################################################################

def categorical(w, N = 1):
    ws     = w.cumsum(0)
    ws[-1] = 1. 

    if w.ndim == 1:        
        idxs = ( ws < rn.uniform( size = (N,1) ) ).sum(1) 
    else:  ## assume w.ndim == 2
        idxs = ( ws < rn.uniform( size = w.shape[1] ) ).sum(0) 
        
    ## return
    return idxs
        
################################################################################
 
def stick(Ko, eta): 
    ## compute the cumulative sum of Ko
    cs = Ko.cumsum()
    
    ## generate beta variates 
    v     = rn.beta(1. + Ko, eta + cs[-1] - cs);
    v[-1] = 1.   ## this ensures that sum(w) = 1. 

    lv    = np.log(v)
    lcp   = np.log1p(-v[:-1]).cumsum()
    
    ## compute weights
    lw = np.r_[lv[0], lv[1:] + lcp]
    
    ## return        
    return lw
    
################################################################################

def rEta(eta, K, N, a = 2., b = 1.):
    ## compute x, r and p
    x  = beta(eta + 1., N)
    lx = np.log(x)
    r  = (a + K - 1.) / ( N * (b - lx) )
    p  = r / (r + 1.)
    
    ## update eta
    rnd     = uniform()
    eta_out = gamma(a + K, b - lx) if rnd < p else gamma(a + K - 1., b - lx)  
    
    ## return
    return eta_out 
    
################################################################################

## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import numpy         as np
import scipy.special as sp
import rand          as rn

################################################################################

def gamma_scale(data, shape, dsum = 0., N = 0, a0 = 2., b0 = 1.):
    ## update dsum and N
    dsum += data.sum()
    N    += data.size
    
    ## compute shape and rate    
    a = a0 + N * shape;
    b = b0 + dsum;

    ## return
    return 1. / rn.gamma(a, 1. / b), (dsum, N);

################################################################################

def gamma_shape_scale(data, shape, scale, ldsum, lsum, N, lp0 = 0., q0 = 0., r0 = 0., s0 = 0.):
    ## compute rate
    rate = 1. / scale
    
    ## update ldsum, lsum and N
    ldsum += np.log(data).sum()
    lsum  += data.sum() 
    N     += data.size
    
    ## compute log(p), q, r, s
    lp = lp0 + ldsum
    q  = q0  + lsum 
    r  = r0  + N
    s  = s0  + N
        
    ## proposals    
    shape_ = shape * np.exp(0.01 * np.random.randn())
    rate_  = rate  * np.exp(0.01 * np.random.randn())
    
    ## compute log-densities
    ll  = (shape  - 1.) * lp - rate  * q - r * sp.gammaln(shape)  + shape  * s * np.log(rate) 
    ll_ = (shape_ - 1.) * lp - rate_ * q - r * sp.gammaln(shape_) + shape_ * s * np.log(rate_)

    ## make Metropolis step     
    if ( ll_ > ll ) or ( np.random.rand() < np.exp(ll_ - ll) ):
        shape = shape_
        scale = 1. / rate_
    
    ## return
    return shape, scale, (ldsum, lsum, N)

################################################################################

def normal_mean_var(data, dsum, d2sum, N, mu0 = -10., n0 = 1., a0 = 2., s0 = 1.):
    ## update dsum, dsum2 and N
    N     += data.size
    dsum  += data.sum()
    d2sum += (data ** 2).sum()
    
    ## compute mu, n, a, s
    avg = dsum / N
    dot = d2sum - 2. * avg * dsum + N * avg ** 2

    mu = (n0 * mu0 + dsum) / (n0 + N);
    n  = n0 + N;
    a  = a0 + N;
    s  = s0 + dot + N * n0 / (N + n0) * (avg - mu0) * (avg - mu0);    
    
    ## compute var and mean
    var  = rn.invgamma(a * 0.5, s * 0.5);
    mean = rn.normal(mu, var / n);

    ## return
    return mean, var, (dsum, d2sum, N)

 
################################################################################

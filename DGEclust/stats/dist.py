## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import numpy         as np
import scipy.stats   as st
import scipy.special as sp

################################################################################

def dLogNegBinomial(x, phi, p):
    alpha = 1. / phi
    
    return sp.gammaln(x + alpha) - sp.gammaln(alpha) - sp.gammaln(x + 1.) + alpha * np.log(p) + x * np.log1p(-p)

################################################################################

def dLogPoisson(x, mu = 1.):
    return x * np.log(mu) - mu - sp.gammaln(x + 1.)
    
################################################################################

def dLogNormal(x, mean = 0., var = 1.):
    return st.norm.logpdf(x, mean, np.sqrt(var))
    
################################################################################
    
def dLogExponential(x, rate = 1.):
    return st.expon.logpdf(x, 0., 1. / rate)
    
################################################################################

def dLogGamma(x, shape = 2., scale = 1.):
    return st.gamma.logpdf(x, shape, 0., scale)

################################################################################

def dLogInvGamma(x, shape = 2., scale = 1.):
    return st.invgamma.logpdf(x, shape, 0., scale)

################################################################################

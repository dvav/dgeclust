from __future__ import division

import numpy as np
import numpy.random as rn
import scipy.special as sp
import scipy.stats as st

########################################################################################################################


def betaln(x, a=1, b=1):
    """Returns the log-density of the beta distribution at x"""

    ## return
    return sp.gammaln(a+b) - sp.gammaln(a) - sp.gammaln(b) + (a-1) * np.log(x) + (b-1) * np.log(1-x)

########################################################################################################################


def normalln(x, mean=0, var=1):
    """Returns the log-density of the normal distribution at x"""

    ## return
    return - 0.5 * (x - mean)**2 / var - 0.5 * np.log(2 * np.pi * var)

########################################################################################################################


def lognormalln(x, mean=0, var=1):
    """Returns the log-density of the lognormal distribution at x"""

    ## return
    return -np.log(x) - 0.5 * (np.log(x) - mean)**2 / var - 0.5 * np.log(2 * np.pi * var)

########################################################################################################################


def poissonln(x, theta):
    """Returns the log-density of the Poisson distribution at x"""

    ## return
    return x * np.log(theta) - sp.gammaln(x + 1) - theta

########################################################################################################################


def binomln(x, p, n):
    """Returns the log-density of the binomial distribution at x"""

    ## return
    return sp.gammaln(n + 1) - sp.gammaln(x + 1) - sp.gammaln(n - x + 1) + x * np.log(p) + (n - x) * np.log(1 - p)

########################################################################################################################


def nbinomln(x, alpha, p):
    """Returns the log-density of the negative binomial distribution at x"""

    ## return
    return sp.gammaln(x + alpha) - sp.gammaln(alpha) - sp.gammaln(x + 1) + alpha * np.log(p) + x * np.log(1 - p)

########################################################################################################################


def sample_normal_mean_var(s1, s2, n, mu0=0, k0=1e-3, a0=1, s0=1e-3):
    """Samples the mean and variance of a normal distribution given data with sufficient statistics s1, s2 and n"""

    ## compute mu, k, a, s
    avg = s1 / n
    dot = s2 - 2 * avg * s1 + n * avg**2

    mu = (k0 * mu0 + s1) / (k0 + n)
    k = k0 + n
    a = a0 + n
    s = s0 + dot + n * k0 / (n + k0) * (avg - mu0) * (avg - mu0)
    
    ## compute var and mean
    var = st.invgamma.rvs(a * 0.5, 0, s * 0.5)
    mean = rn.randn() * np.sqrt(var / k) + mu

    ## return
    return mean, var

########################################################################################################################


def sample_gamma_shape_scale(suma, logsuma, ndata, shape, scale, lp0=0, q0=1, r0=1, s0=1):
    """Samples the shape and scale of the gamma distribution from their posterior"""

    ## compute updated params
    lp = lp0 + logsuma
    q = q0 + suma
    r = r0 + ndata
    s = s0 + ndata

    ## make proposals
    shape_, scale_ = (shape, scale) * np.exp(0.01 * rn.randn(2))

    ## compute lp and lp_
    logpost = (shape - 1) * lp - q / scale - r * sp.gammaln(shape) - shape * s * np.log(scale)
    logpost_ = (shape_ - 1) * lp - q / scale_ - r * sp.gammaln(shape_) - shape_ * s * np.log(scale_)

    ## do Metropolis step
    if logpost_ > logpost or rn.rand() < np.exp(logpost_ - logpost):
        shape = shape_
        scale = scale_

    ## return
    return shape, scale

########################################################################################################################


def sample_gamma_scale(suma, ndata, shape, a0=1, b0=1):
    """Samples the scale the gamma distribution from its posterior, when shape is known"""

    ## return
    return 1 / rn.gamma(a0 + ndata * shape, 1 / (b0 + suma))

########################################################################################################################


def sample_categorical(w, n=1):
    """Samples from the categorical distribution with matrix of weights w"""

    ws = w.cumsum(0)
    ws[-1] = 1                                      # sum of ws along rows should be equal to 1

    if w.ndim == 1:                                 # w is 1D
        idxs = np.sum(ws < rn.rand(n, 1), 1)
    else:                                           # assume w is 2D
        idxs = np.sum(ws < rn.rand(w.shape[1]), 0)
        
    ## return
    return idxs
        
########################################################################################################################


def sample_stick(cluster_occupancies, eta):
    """Samples random stick lengths (in logarithmic scale) given a vector of cluster occupancies"""

    ## compute the cumulative sum of the count vector
    cs = cluster_occupancies.cumsum()
    
    ## generate beta variates 
    v = rn.beta(1 + cluster_occupancies, eta + cs[-1] - cs)
    v[-1] = 1                  # this ensures that sum(w) = 1

    ## compute weights
    lv = np.log(v)
    lcp = np.log(1-v).cumsum()
    
    lw = np.r_[lv[0], lv[1:] + lcp[:-1]]
    
    ## return        
    return lw, lv
    
########################################################################################################################


def sample_eta(lw, a=1, b=1e-3):
    """Samples the concentration parameter eta given a vector of mixture log-weights"""

    ## return
    return rn.gamma(lw.size - 1 + a, 1 / (b - np.min(lw)))     # 1 / (b - lw[-1])

########################################################################################################################

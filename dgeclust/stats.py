from __future__ import division

import numpy as np
import numpy.random as rn
import scipy.special as sp
import scipy.stats as st

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


def nbinomln(x, alpha, p):
    """Returns the log-density of the negative binomial distribution at x"""

    ## return
    return sp.gammaln(x + alpha) - sp.gammaln(alpha) - sp.gammaln(x + 1) + alpha * np.log(p) + x * np.log(1 - p)

########################################################################################################################


def sample_normal_meanvar(s1, s2, n, mu0=0, k0=1, a0=2, s0=1):
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


def sample_eta(lw, a=2, b=1):
    """Samples the concentration parameter eta given a vector of mixture log-weights"""

    ## return
    return rn.gamma((lw.size - 1) + a, 1 / (b - np.min(lw)))     ## correct: 1 / (b - lw[-1])
      
########################################################################################################################

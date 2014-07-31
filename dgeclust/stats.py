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


def poissonln(x, rate=1):
    """Returns the log-density of the Poisson distribution at x"""

    ## return
    return x * np.log(rate) - sp.gammaln(x + 1) - rate

########################################################################################################################


def binomln(x, n=1, p=0.5):
    """Returns the log-density of the binomial distribution at x"""

    ## return
    return sp.gammaln(n + 1) - sp.gammaln(x + 1) - sp.gammaln(n - x + 1) + x * np.log(p) + (n - x) * np.log(1 - p)

########################################################################################################################


def nbinomln(x, alpha=1, p=0.5):
    """Returns the log-density of the negative binomial distribution at x"""

    ## return
    return sp.gammaln(x + alpha) - sp.gammaln(alpha) - sp.gammaln(x + 1) + alpha * np.log(p) + x * np.log(1 - p)

########################################################################################################################


def bbinomln(x, n=1, alpha=0.5, beta=0.5):
    """Returns the log-density of the beta binomial distribution at x"""

    ## compute intermediate quantities
    c1 = sp.gammaln(n + 1) - sp.gammaln(x + 1) - sp.gammaln(n - x + 1)
    c2 = sp.gammaln(x + alpha) + sp.gammaln(n - x + beta) - sp.gammaln(n + alpha + beta)
    c3 = sp.gammaln(alpha + beta) - sp.gammaln(alpha) - sp.gammaln(beta)

    ## return
    return c1 + c2 + c3

########################################################################################################################


def sample_normal_mean_var(s1, s2, ndata, mu=0, k=0, shape=0, rate=0):
    """Samples the mean and variance of a normal distribution given data with sufficient statistics s1, s2 and n"""

    ## update mu, k, shape, rate
    avg = s1 / ndata
    dot = s2 - 2 * avg * s1 + ndata * avg * avg + 1e-12     # do not let dot become zero

    mu_ = (k * mu + s1) / (k + ndata)
    k_ = k + ndata
    shape_ = shape + ndata * 0.5
    rate_ = rate + 0.5 * dot + 0.5 * ndata * k / (ndata + k) * (avg - mu) * (avg - mu)
    
    ## compute var and mean
    var = 1 / rn.gamma(shape_, 1 / rate_)
    mean = rn.randn() * np.sqrt(var / k_) + mu_

    ## return
    return mean, var

########################################################################################################################


def sample_normal_mean_var_jeffreys(s1, s2, ndata):
    """Samples the mean and variance of a normal distribution given data with sufficient statistics s1, s2 and n"""

    ## update mu, k, shape, rate
    avg = s1 / ndata
    dot = s2 - 2 * avg * s1 + ndata * avg * avg + 1e-12     # do not let dot become zero

    ## sample var and mean
    mean = st.t.rvs(ndata + 1, avg, dot / (ndata*ndata + ndata))
    var = 1 / rn.gamma((ndata + 1) * 0.5, 2 / dot)

    ## return
    return mean, var

########################################################################################################################


def sample_normal_var_jeffreys(s1, s2, ndata):
    """Samples the mean and variance of a normal distribution given data with sufficient statistics s1, s2 and n"""

    ## update mu, k, shape, rate
    avg = s1 / ndata
    dot = s2 - 2 * avg * s1 + ndata * avg * avg + 1e-12     # do not let dot become zero

    ## sample var and mean
    var = 1 / rn.gamma(ndata * 0.5, 2 / dot)

    ## return
    return var

########################################################################################################################


def sample_gamma_shape_scale(suma, logsuma, ndata, shape, scale, lp0=0, q0=0, r0=0, s0=0):
    """Samples the shape and scale of the gamma distribution from their posterior"""

    ## compute updated params
    lp = lp0 + logsuma
    q = q0 + suma
    r = r0 + ndata
    s = s0 + ndata

    ## make proposals
    shape_, scale_ = (shape, scale) * np.exp(0.01 * rn.randn(2))

    ## compute logpost and logpost_
    logpost = (shape - 1) * lp - q / scale - r * sp.gammaln(shape) - shape * s * np.log(scale)
    logpost_ = (shape_ - 1) * lp - q / scale_ - r * sp.gammaln(shape_) - shape_ * s * np.log(scale_)

    ## do Metropolis step
    if logpost_ > logpost or rn.rand() < np.exp(logpost_ - logpost):
        shape = shape_
        scale = scale_

    ## return
    return shape, scale

########################################################################################################################


def sample_gamma_scale(suma, ndata, shape, a0=0, b0=0):
    """Samples the scale of the gamma distribution from its posterior, when shape is known"""

    ## return
    return 1 / rn.gamma(a0 + ndata * shape, 1 / (b0 + suma))

########################################################################################################################


def sample_gamma_shape(logsuma, ndata, shape, scale, lp0=0, r0=0, s0=0):
    """Samples the shape of the gamma distribution from its posterior, when scale is known"""

    ## compute updated params
    lp = lp0 + logsuma
    r = r0 + ndata
    s = s0 + ndata

    ## make proposal
    shape_ = shape * np.exp(0.01 * rn.randn())

    ## compute logpost and logpost_
    logpost = (shape - 1) * lp - r * sp.gammaln(shape) - shape * s * np.log(scale)
    logpost_ = (shape_ - 1) * lp - r * sp.gammaln(shape_) - shape_ * s * np.log(scale)

    ## do Metropolis step
    if logpost_ > logpost or rn.rand() < np.exp(logpost_ - logpost):
        shape = shape_

    ## return
    return shape

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


def sample_eta_ishwaran(lw, a=0, b=0):
    """Samples the concentration parameter eta given a vector of mixture log-weights"""

    ## return
    return rn.gamma(lw.size + a - 1, 1 / (b - np.min(lw)))

########################################################################################################################


def sample_eta_west(eta, nact, n0, a=0, b=0):
    """Samples the concentration parameter eta"""

    ## compute x, r and p
    x = rn.beta(eta + 1, n0)
    lx = np.log(x)
    r = (a + nact - 1) / (n0 * (b - lx))
    p = r / (r + 1)

    ## return
    return rn.gamma(a + nact, 1 / (b - lx)) if rn.rand() < p else rn.gamma(a + nact - 1, 1 / (b - lx))

########################################################################################################################


def sample_eta(eta, nact, n0, a=0, b=0):
    """Samples the concentration parameter eta"""

    ## proposal
    eta_ = eta * np.exp(rn.randn() * 0.01)

    ## posterior densities
    lp = sp.gammaln(eta) - sp.gammaln(eta + n0) + (nact + a - 1) * np.log(eta) - eta * b
    lp_ = sp.gammaln(eta_) - sp.gammaln(eta_ + n0) + (nact + a - 1) * np.log(eta_) - eta_ * b

    ## return
    return eta_ if lp_ > lp or rn.rand() < np.exp(lp_ - lp) else eta

########################################################################################################################

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
    return sp.gammaln(x + alpha) - sp.gammaln(alpha) - sp.gammaln(x + 1) + alpha * np.log(p) + x * np.log1p(-p)

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


def exponentialln(x, scale=1):
    """Returns the log-density of the exponential distribution at x"""

    ##
    return -np.log(scale) - x / scale

########################################################################################################################


def sample_normal_mean(s1, ndata, prec, m0=0, t0=0):
    """Samples the mean of a normal distribution"""

    ##
    t = t0 + prec * ndata
    m = (t0 * m0 + prec * s1) / t

    ## return
    return rn.normal(m, 1 / np.sqrt(t))

########################################################################################################################


def sample_normal_prec(s1, s2, ndata, mean, a0=0, b0=0):
    """Samples the precision of a normal distribution"""

    ##
    dot = s2 - 2 * mean * s1 + ndata * mean**2

    a = a0 + ndata * 0.5
    b = b0 + 0.5 * dot

    ## return
    return rn.gamma(a, 1 / b)

########################################################################################################################


def sample_normal_mean_prec(s1, s2, ndata, m0=0, l0=0, a0=0, b0=0):
    """Samples the precision of a normal distribution"""

    ##
    avg = s1 / ndata
    dot = s2 - 2 * avg * s1 + ndata * avg**2

    l = l0 + ndata
    m = (l0 * m0 + s1) / l
    a = a0 + 0.5 * ndata
    b = b0 + 0.5 * dot + 0.5 * l0 * ndata * (avg - m0)**2 / l

    ##
    prec = rn.gamma(a, 1 / b)
    mean = rn.normal(m, 1 / np.sqrt(l * prec))

    ## return
    return mean, prec

########################################################################################################################


def sample_normal_mean_jeffreys(s1, ndata, prec):
    """Samples the mean of a normal distribution"""

    ##
    return rn.normal(s1 / ndata, 1 / np.sqrt(prec * ndata))

########################################################################################################################


def sample_normal_prec_jeffreys(s1, s2, ndata):
    """Samples the precision of a normal distribution"""

    ##
    avg = s1 / ndata
    dot = s2 - 2 * avg * s1 + ndata * avg**2

    ##
    return rn.gamma(ndata * 0.5, 2 / dot)

########################################################################################################################


def sample_normal_mean_prec_jeffreys(s1, s2, ndata):
    """Samples the precision of a normal distribution"""

    ##
    avg = s1 / ndata
    dot = s2 - 2 * avg * s1 + ndata * avg**2

    mean = st.t.rvs(ndata+1, avg, dot / (ndata * ndata + ndata))
    prec = rn.gamma((ndata+1) * 0.5, 2 / dot)

    ##
    return mean, prec

########################################################################################################################


def sample_gamma_rate(s, ndata, shape, a0=0, b0=0):
    """Samples the scale of the gamma distribution from its posterior, when shape is known"""

    ## return
    return rn.gamma(a0 + ndata * shape, 1 / (b0 + s))

########################################################################################################################


def sample_gamma_shape(sl, ndata, shape, rate, la0=0, b0=0, c0=0):
    """Samples the shape of the gamma distribution from its posterior, when scale is known"""

    ## compute updated params
    la = la0 + sl
    b = b0 + ndata
    c = c0 + ndata

    ## make proposal
    shape_ = shape * np.exp(0.01 * rn.randn())

    ## compute logpost and logpost_
    logpost = (shape - 1) * la + shape * c * np.log(rate) - b * sp.gammaln(shape)
    logpost_ = (shape_ - 1) * la + shape_ * c * np.log(rate) - b * sp.gammaln(shape_)

    ## do Metropolis step
    if logpost_ > logpost or rn.rand() < np.exp(logpost_ - logpost):
        shape = shape_

    ## return
    return shape

########################################################################################################################


def sample_dirichlet(a):
    """Sample from multiple Dirichlet distributions given the matrix of concentration parameter columns a"""

    x = rn.gamma(a, 1)
    w = x / np.sum(x, 0)

    ##
    return w

########################################################################################################################


def sample_categorical(w, nsamples=1):
    """Samples from the categorical distribution with matrix of weight columns w"""

    _, ncols = w.shape

    ws = w.cumsum(0)
    ws[-1] = 1                                      # sum of ws along rows should be equal to 1

    idxs = np.sum(ws[:, :, np.newaxis] < rn.rand(ncols, nsamples), 0)

    ## return
    return idxs.T

########################################################################################################################


def sample_stick(cluster_occupancies, eta):
    """Samples random stick lengths (in logarithmic scale) given a vector of cluster occupancies"""

    ## compute the cumulative sum of the count vector
    cs = cluster_occupancies.cumsum()

    ## generate beta variates
    v = rn.beta(1 + cluster_occupancies, eta + cs[-1] - cs)
    v[-1] = 1                  # this ensures that sum(w) = 1
    v = np.clip(v, 1e-12, 1 - 1e-12)

    ## compute weights
    lv = np.log(v)
    lcp = np.log(1-v).cumsum()

    lw = np.r_[lv[0], lv[1:] + lcp[:-1]]

    ## return
    return lw, lv

########################################################################################################################


def sample_eta_ishwaran(lw, eta, a=0, b=0):
    """Samples the concentration parameter eta given a vector of mixture log-weights"""

    eta = rn.gamma(lw.size + a - 1, 1 / (b - lw[-1])) if np.isfinite(lw[-1]) else eta

    ##
    return eta

########################################################################################################################


def sample_eta_west(eta, nact, n0, a=1, b=0):
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
    eta_ = eta * np.exp(0.01 * rn.randn())

    ## posterior densities
    lp = sp.gammaln(eta) - sp.gammaln(eta + n0) + (nact + a - 1) * np.log(eta) - eta * b
    lp_ = sp.gammaln(eta_) - sp.gammaln(eta_ + n0) + (nact + a - 1) * np.log(eta_) - eta_ * b

    ## return
    return eta_ if lp_ > lp or rn.rand() < np.exp(lp_ - lp) else eta

########################################################################################################################

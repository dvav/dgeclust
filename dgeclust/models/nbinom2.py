from __future__ import division

import numpy as np
import numpy.random as rn
import scipy.special as sp

import dgeclust.stats as st

########################################################################################################################


def compute_loglik(j, data, state):
    """Computes the log-likelihood of each element of counts for each element of theta"""

    ## read data
    counts = data.counts_norm[:, data.groups[j]].T
    counts = np.atleast_2d(counts)
    counts = counts[:, :, np.newaxis]

    ## read theta
    phi = state.theta[:, 0]
    p = state.theta[:, 1]

    ## compute loglik
    loglik = st.nbinomln(counts, 1 / phi, p)

    ## return
    return loglik

########################################################################################################################


def sample_prior(size, mean, var):
    """Samples phi and p from their priors, log-normal and uniform, respectively"""

    ## sample phi and p
    phi = np.exp(rn.randn(size, 1) * np.sqrt(var) + mean)
    p = rn.rand(size, 1)

    ## return
    return np.hstack((phi, p))
    
########################################################################################################################


def sample_params(theta, mean, var):
    """Samples the mean and var of the log-normal from the posterior, given phi"""

    data = np.log(theta[:, 0])

    ## update S1, S2 and N
    n = data.size
    s1 = data.sum()
    s2 = np.sum(data**2)

    mean, var = st.sample_normal_meanvar(s1, s2, n)

    ## return
    return mean, var

########################################################################################################################


def sample_posterior(idx, data, state):
    """Sample phi and p from their posterior, given counts"""

    ## fetch all data points that belong to cluster idx
    counts = [data.counts_norm[:, group][zz == idx].ravel() for group, zz in zip(data.groups, state.zz)]
    counts = np.hstack(counts)

    s = counts.sum()
    n = counts.size

    ## read params
    mean, var = state.pars

    ## read theta
    phi, _ = state.theta[idx]
    alpha = 1 / phi

    ## sample alpha using Metropolis
    phi_ = phi * np.exp(0.01 * rn.randn())    # make proposal
    alpha_ = 1 / phi_

    coeff = np.sum(sp.gammaln(counts + alpha) - sp.gammaln(alpha))
    coeff_ = np.sum(sp.gammaln(counts + alpha_) - sp.gammaln(alpha_))

    lp = sp.betaln(n * alpha + 1, s + 1) + coeff + st.lognormalln(phi,  mean, var)     # posterior density of alpha
    lp_ = sp.betaln(n * alpha_ + 1, s + 1) + coeff_ + st.lognormalln(phi_, mean, var)  # posterior density of alpha_

    if (lp_ > lp) or (rn.rand() < np.exp(lp_ - lp)):    # do Metropolis step
        alpha = alpha_
        phi = 1 / alpha_

    ## sample p given the new alpha
    p = rn.beta(n * alpha + 1, s + 1)

    ## return
    return phi, p

########################################################################################################################

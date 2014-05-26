from __future__ import division

import numpy as np
import numpy.random as rn
import scipy.stats as sp

import dgeclust.stats as st

########################################################################################################################


def compute_loglik(j, data, state):
    """Computes the log-likelihood of each element of counts for each element of theta"""

    ## read data
    counts = data.counts[:, data.groups[j]]
    counts = counts.T
    counts = np.atleast_2d(counts)
    counts = counts[:, :, np.newaxis]

    ## read theta
    mean = state.theta[:, 0]
    var = state.theta[:, 1]

    ## return
    return st.normalln(counts, mean, var)

########################################################################################################################


def sample_prior(size, mu0, k0, a0, s0):
    """Samples mean and var from their priors, normal and inverse gamma, respectively"""

    ## sample mean and var
    var = sp.invgamma.rvs(a0 * 0.5, 0, s0 * 0.5, size=(size, 1))
    mean = rn.randn(size, 1) * np.sqrt(var / k0) + mu0

    ## return    
    return np.hstack((mean, var))
    
########################################################################################################################


def sample_params(theta, mu0, k0, a0, s0):
    """Samples the mean and var of the log-normal from the posterior, given theta"""

    ## return
    return mu0, k0, a0, s0
    
########################################################################################################################


def sample_posterior(idx, data, state):
    """Sample mean and var from their posterior, given counts"""

    ## fetch all data points that belong to cluster idx
    counts = [data.counts[:, group][zz == idx].ravel() for group, zz in zip(data.groups, state.zz)]
    counts = np.hstack(counts)

    ## compute sufficient statistics
    n = counts.size
    s1 = counts.sum()
    s2 = np.sum(counts**2)

    ## read parameters
    mu0, k0, a0, s0 = state.pars

    ## read theta
    mean, var = st.sample_normal_mean_var(s1, s2, n, mu0, k0, a0, s0)

    ## return
    return mean, var
    
########################################################################################################################

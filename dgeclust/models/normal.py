from __future__ import division

import numpy as np
import numpy.random as rn
import scipy.stats as sp

import dgeclust.stats as st

########################################################################################################################


def compute_loglik(j, data, state):
    """Computes the log-likelihood of each element of counts for each element of theta"""

    ## read data
    group = data.groups.values()[j]
    counts = data.counts[group].values.T
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

    ## read parameters
    mean = theta[:, 0]
    prec = 1 / theta[:, 1]

    ## compute sufficient statistics
    n = prec.size

    s = prec.sum()
    ls = np.log(prec).sum()

    s1 = mean.sum()
    s2 = np.sum(mean**2)

    ## sample a0 and b0
    a0 = st.sample_gamma_shape(ls, n, a0, 1 / s0)
    b0 = st.sample_gamma_scale(s, n, a0)

    ## sample mu0 and k0
    mu0, var = st.sample_normal_mean_var(s1, s2, n)

    ## return
    return mu0, var * b0, a0, 1 / b0
    
########################################################################################################################


def sample_posterior(idx, data, state):
    """Sample mean and var from their posterior, given counts"""

    ## fetch all data points that belong to cluster idx
    groups = data.groups.values()
    counts = [data.counts[group][zz == idx].values.ravel() for group, zz in zip(groups, state.zz)]
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

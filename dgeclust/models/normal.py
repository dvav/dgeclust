from __future__ import division

import numpy as np
import numpy.random as rn

import dgeclust.stats as st

########################################################################################################################


def compute_loglik(j, data, state):
    """Computes the log-likelihood of each element of counts for each element of theta"""

    ## read data
    group = data.groups.values()[j]
    counts = data.counts[group].values.T
    counts = counts[:, :, np.newaxis]

    ## read theta
    mean = state.pars[:, 0]
    var = state.pars[:, 1]

    ## return
    return st.normalln(counts, mean, var)

########################################################################################################################


def sample_prior(size, mu, k, shape, rate):
    """Samples mean and var from their priors, normal and inverse gamma, respectively"""

    ## sample mean and var
    var = 1 / rn.gamma(shape, 1 / rate, size=(size, 1))
    mean = rn.randn(size, 1) * np.sqrt(var / k) + mu

    ## return    
    return np.hstack((mean, var))
    
########################################################################################################################


def sample_hpars(pars, mu, k, shape, rate):
    """Samples the mean and var of the log-normal from the posterior, given theta"""

    ## read parameters
    mean = pars[:, 0]
    var = pars[:, 1]
    prec = 1 / var

    ndata = len(pars)

    ## sample shape and rate
    shape = st.sample_gamma_shape(np.log(prec).sum(), ndata, shape, 1 / rate)
    scale = st.sample_gamma_scale(np.sum(prec), ndata, shape)

    ## sample mu and k
    mu, s2 = st.sample_normal_mean_var(mean.sum(), np.sum(mean**2), ndata)
    k = np.mean(var) / s2   # double-check this !!!

    ## return
    return mu, k, shape, 1 / scale
    
########################################################################################################################


def sample_posterior(idx, data, state):
    """Sample mean and var from their posterior, given counts"""

    ## fetch all data points that belong to cluster idx
    groups = data.groups.values()
    counts = [data.counts[group][zz == idx].values.ravel() for group, zz in zip(groups, state.zz)]
    counts = np.hstack(counts)

    ## sample mean and var
    mean, var = st.sample_normal_mean_var(np.sum(counts), np.sum(counts**2), counts.size, *state.hpars)

    ## return
    return mean, var
    
########################################################################################################################

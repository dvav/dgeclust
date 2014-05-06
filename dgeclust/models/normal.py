from __future__ import division

import numpy as np
import numpy.random as rn
import scipy.stats as sp

import stats as st

########################################################################################################################


def compute_loglik(counts, theta):
    """Computes the log-likelihood of each element of counts for each element of theta"""

    counts = np.atleast_2d(counts)
    theta = np.atleast_2d(theta)
    
    ## read theta
    mean = theta[:, 0]
    var = theta[:, 1]
    
    ## compute loglik
    counts = counts[:, :, np.newaxis]
    loglik = st.normalln(counts, mean, var)
        
    ## return
    return loglik

########################################################################################################################


def sample_prior(size, mu0, k0, a0, s0):
    """Samples mean and var from their priors, normal and inverse gamma, respectively"""

    ## sample mean and var
    var = sp.invgamma.rvs(a0, 0, s0, size=(size, 1))
    mean = rn.randn(size, 1) * np.sqrt(var / k0) + mu0

    ## return    
    return np.hstack((mean, var))
    
########################################################################################################################


def sample_params(theta, mu0, k0, a0, s0):
    """Samples the mean and var of the log-normal from the posterior, given theta"""

    ## return
    return mu0, k0, a0, s0
    
########################################################################################################################


def sample_posterior(theta, idx, c, z, counts, mu0, k0, a0, s0):
    """Sample mean and var from their posterior, given counts"""

    ## fetch all data points that belong to cluster idx
    counts = [np.atleast_2d(cnts)[:, ci[zi] == idx].ravel() for cnts, ci, zi in zip(counts, c, z)]
    counts = np.hstack(counts)

    n = counts.size
    s1 = counts.sum()
    s2 = np.sum(counts**2)

    ## read theta
    mean, var = st.sample_normal_meanvar(s1, s2, n, mu0, k0, a0, s0)

    ## return
    return mean, var
    
########################################################################################################################

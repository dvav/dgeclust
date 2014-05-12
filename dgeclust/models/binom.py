from __future__ import division

import numpy as np
import numpy.random as rn

import stats as st

########################################################################################################################


def _compute_loglik(theta, counts, lib_sizes):
    """Computes the log-likelihood of each element of counts for each element of phi and mu"""

    ## prepare data
    counts = counts.T
    counts = np.atleast_2d(counts)
    counts = counts[:, :, np.newaxis]

    lib_sizes = np.atleast_2d(lib_sizes).T
    lib_sizes = lib_sizes[:, :, np.newaxis]

    ## return
    return st.binomln(counts, theta, lib_sizes)


########################################################################################################################


def compute_loglik(j, data, state):
    """Computes the log-likelihood of each element of counts for each element of theta"""

    ## read data
    counts = data.counts[:, data.groups[j]]
    lib_sizes = data.library_sizes[data.groups[j]]

    ## return
    return _compute_loglik(state.theta, counts, lib_sizes)

########################################################################################################################


def sample_prior(size, a, b):
    """Samples theta from its prior"""

    ## return    
    return rn.beta(a, b, size)
    
########################################################################################################################


def sample_params(theta, a, b):
    """Samples the parameters of the beta distribution from its posterior, given theta"""

    ## return
    return a, b
    
########################################################################################################################


def sample_posterior(idx, data, state):
    """Sample theta from its posterior, given counts. It does not work when replicas are present!!!!!"""

    ## fetch all data points that belong to cluster idx
    counts = [data.counts[:, group][zz == idx] for group, zz in zip(data.groups, state.zz)]
    lib_sizes = [data.library_sizes[group] for group in data.groups]

    s = np.sum([cnts.sum() for cnts in counts])
    n = np.sum([cnts.size * libsz for cnts, libsz in zip(counts, lib_sizes)])

    ## parameters
    a, b = state.pars

    ## return
    return rn.beta(a + s, b + n - s)
    
########################################################################################################################

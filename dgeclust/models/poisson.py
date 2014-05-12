from __future__ import division

import numpy as np
import numpy.random as rn

import dgeclust.stats as st

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
    return st.poissonln(counts, lib_sizes * theta)


########################################################################################################################


def compute_loglik(j, data, state):
    """Computes the log-likelihood of each element of counts for each element of theta"""

    ## read data
    counts = data.counts[:, data.groups[j]]
    lib_sizes = data.library_sizes[data.groups[j]]

    ## return
    return _compute_loglik(state.theta, counts, lib_sizes)

########################################################################################################################


def sample_prior(size, shape, scale):
    """Samples theta from its prior"""

    ## return    
    return rn.gamma(shape, scale, size)
    
########################################################################################################################


def sample_params(theta, shape, _, a0=1, b0=1):
    """Samples the shape and scale of the gamma distribution from its posterior, given theta"""

    ## sample scale
    s = theta.sum()
    n = theta.size

    rate = rn.gamma(a0 + n * shape, 1 / (b0 + s))

    ## return
    return shape, 1 / rate
    
########################################################################################################################


def sample_posterior(idx, data, state):
    """Sample theta from its posterior, given counts. It does not work when replicas are present!!!!!"""

    ## fetch all data points that belong to cluster idx
    counts = [data.counts[:, group][zz == idx] for group, zz in zip(data.groups, state.zz)]
    lib_sizes = [data.library_sizes[group] for group in data.groups]

    s = np.sum([cnts.sum() for cnts in counts])
    n = np.asarray([cnts.size for cnts in counts])

    m = np.sum(lib_sizes * n)

    ## parameters
    shape, scale = state.pars

    ## return
    return rn.gamma(shape + s, scale / (m * scale + 1))
    
########################################################################################################################

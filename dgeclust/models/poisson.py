from __future__ import division

import numpy as np
import numpy.random as rn

import dgeclust.stats as st

########################################################################################################################


def _compute_loglik(theta, counts, norm_factors):
    """Computes the log-likelihood of each element of counts for each element of phi and mu"""

    ## prepare data
    counts = counts.T
    counts = np.atleast_2d(counts)
    counts = counts[:, :, np.newaxis]

    norm_factors = np.atleast_2d(norm_factors).T
    norm_factors = norm_factors[:, :, np.newaxis]

    ## return
    return st.poissonln(counts, norm_factors * theta)


########################################################################################################################


def compute_loglik(j, data, state):
    """Computes the log-likelihood of each element of counts for each element of theta"""

    ## read data
    counts = data.counts[:, data.groups[j]]
    norm_factors = data.norm_factors[data.groups[j]]

    ## return
    return _compute_loglik(state.theta, counts, norm_factors)

########################################################################################################################


def sample_prior(size, shape, scale):
    """Samples theta from its prior"""

    ## return    
    return rn.gamma(shape, scale, size)
    
########################################################################################################################


def sample_params(theta, shape, scale):
    """Samples the shape and scale of the gamma distribution from its posterior, given theta"""

    ## sample scale
    s = theta.sum()
    ls = np.log(theta).sum()
    n = theta.size

    ## sample scale, then sample shape
    shape = st.sample_gamma_shape(ls, n, shape, scale)
    scale = st.sample_gamma_scale(s, n, shape)

    ## return
    return shape, scale


########################################################################################################################


def sample_posterior(idx, data, state):
    """Sample theta from its posterior, given counts"""

    ## fetch all data points that belong to cluster idx
    counts = [data.counts[:, group][zz == idx] for group, zz in zip(data.groups, state.zz)]
    norm_factors = [np.sum(data.norm_factors[group]) for group in data.groups]

    s = np.sum([cnts.sum() for cnts in counts])
    n = np.asarray([cnts.size for cnts in counts])

    m = np.sum(norm_factors * n)

    ## parameters
    shape, scale = state.pars

    ## return
    return rn.gamma(shape + s, scale / (m * scale + 1))
    
########################################################################################################################

from __future__ import division

import numpy as np
import numpy.random as rn

import stats as st

########################################################################################################################


def compute_loglik(counts, theta):
    """Computes the log-likelihood of each element of counts for each element of theta"""

    counts = np.atleast_2d(counts)

    ## compute loglik
    counts = counts[:, :, np.newaxis]
    loglik = st.poissonln(counts, theta)
        
    ## return
    return loglik

########################################################################################################################


def sample_prior(size, shape, scale):
    """Samples theta from its prior"""

    ## return    
    return rn.gamma(shape, scale, size)
    
########################################################################################################################


def sample_params(theta, shape, scale):
    """Samples the shape and scale of the gamma distribution from its posterior, given theta"""

    ## return
    return shape, scale
    
########################################################################################################################


def sample_posterior(theta, idx, c, z, counts, shape, scale):
    """Sample theta from its posterior, given counts"""

    ## fetch all data points that belong to cluster idx
    counts = [np.atleast_2d(cnts)[:, ci[zi] == idx].ravel() for cnts, ci, zi in zip(counts, c, z)]
    counts = np.hstack(counts)

    s = counts.sum()
    n = counts.size

    ## return
    return rn.gamma(shape + s, scale / (n * scale + 1))
    
########################################################################################################################

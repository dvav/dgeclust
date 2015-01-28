from __future__ import division

import numpy as np
import scipy.misc as ms
import matplotlib.pylab as pl

########################################################################################################################


def compute_occupancies_2d(nclusters, z):
    """Compute cluster occupancies per row of matrix z"""

    labels = np.arange(nclusters)
    occ = z[:, :, np.newaxis] == labels
    occ = np.sum(occ, 1)

    ##
    return occ

########################################################################################################################


def normalize_log_weights(lw):
    """Normalises a matrix of log-weights, row-wise"""

    ref = lw.max(0)
    lsum = ms.logsumexp(lw - ref, 0) + ref      # more stable than np.log(np.exp(lw - ref).sum(0)) + ref

    ## return
    return lw - lsum
 
########################################################################################################################


def plot_ra(x, y, idxs=None):
    """Computes the RA plot of two groups of samples"""

    ## compute log2 values
    l1 = np.log2(x)
    l2 = np.log2(y)

    ## compute A and R
    r = l1 - l2
    a = np.r_[(l1 + l2) * 0.5]

    h = pl.figure()
    if idxs is None:
        pl.plot(a, r, '.k', markersize=2)
    else:
        pl.plot(a[~idxs], r[~idxs], '.k', markersize=2)
        pl.plot(a[idxs], r[idxs], '.r')

    pl.plot(pl.gca().get_xlim(), [0, 0], '--k')

    ## return
    return h

########################################################################################################################

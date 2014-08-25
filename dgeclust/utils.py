from __future__ import division

import numpy as np
import scipy.misc as ms
import matplotlib.pylab as pl

########################################################################################################################


def normalize_log_weights(lw):
    """Normalises a matrix of log-weights, row-wise"""

    ref = lw.max(0)
    lsum = ms.logsumexp(lw - ref, 0) + ref      # more stable than np.log(np.exp(lw - ref).sum(0)) + ref

    ## return
    return lw - lsum
 
########################################################################################################################


def plot_ra(x, y, idxs=None):
    """Computes the RA plot of two groups of samples. Use the returned arrays to actually plot the diagram"""

    ## compute log2 values
    l1 = np.log2(x)
    l2 = np.log2(y)

    ## compute A and R
    r = l1 - l2
    a = (l1 + l2) * 0.5

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

from __future__ import division

import numpy as np
import scipy.misc as ms
import matplotlib.pylab as pl

########################################################################################################################


def compute_cluster_occupancies(nclusters, cluster_indicators):
    """Calculates cluster occupancies, given a vector of cluster indicators"""

    labels = np.arange(nclusters).reshape(-1, 1)               # cluster labels
    occupancy_matrix = (cluster_indicators == labels)          # N x M occupancy matrix

    ## return
    return np.sum(occupancy_matrix, 1), occupancy_matrix

########################################################################################################################


def get_cluster_info(nclusters, cluster_indicators):
    """Returns various cluster info, including cluster occupancies"""

    cluster_occupancies, occupancy_matrix = compute_cluster_occupancies(nclusters, cluster_indicators)
    iactive = cluster_occupancies > 0           # indicators of active clusters
    nactive = np.count_nonzero(iactive)         # number of active clusters

    ## return
    return cluster_occupancies, iactive, nactive, occupancy_matrix

########################################################################################################################


def normalize_log_weights(lw):
    """Normalises a matrix of log-weights, row-wise"""

    ref = lw.max(0)
    lsum = ms.logsumexp(lw - ref, 0) + ref      # more stable than np.log(np.exp(lw - ref).sum(0)) + ref

    ## return
    return lw - lsum
 
########################################################################################################################


def plot_fitted_model(sample, state, data, model, xmin=-1, xmax=12, npoints=1000, nbins=100, epsilon=0.5):
    """Computes the fitted model"""

    ## fetch group
    group = [i for i, item in enumerate(data.groups.items()) if sample in item[1]][0]

    ## fetch clusters
    z = state.z
    delta = state.delta[:, [group]]
    pars = state.pars

    occ, _ = compute_cluster_occupancies(len(pars), z)

    ## fetch data
    counts = data.counts[sample].values.astype('float')
    counts[counts < 1] = epsilon
    counts = np.log(counts)

    lib_size = data.lib_sizes[sample].values.ravel()

    ## compute fitted model
    x = np.reshape(np.linspace(xmin, xmax, npoints), (-1, 1))
    xx = np.exp(x)
    y = xx * np.exp(model.compute_loglik(([xx[:, :, np.newaxis]], [lib_size]), pars[z], delta).sum(-1))

    ## groups
    idxs = np.nonzero(occ)[0]
    yg = np.asarray([np.sum(y[:, z == idx], 1) / z.size for idx in idxs]).T

    ## plot
    pl.hist(counts, nbins, histtype='stepfilled', linewidth=0, normed=True, color='gray')
    pl.plot(x, yg, 'k', x, yg.sum(1), 'r')

    ## return
    return x, y

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

from __future__ import division

import collections as cl
import numpy as np
import scipy.misc as ms
import matplotlib.pylab as pl
import pandas as pd

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


def plot_fitted_model(sample, res, data, model, xmin=-1, xmax=12, npoints=1000, nbins=100, log_scale=True, epsilon=0.5):
    """Computes the fitted model"""

    ## fetch group
    group = [k for k, v in data.groups.items() if sample in v][0]

    ## compute cluster occupancies
    occ, iact, _, _ = get_cluster_info(len(res.pars), res.zz[group].values)
    occ = occ[iact]                     # keep occupancies of active clusters, only

    ## compute fitted model
    x = np.reshape(np.linspace(xmin, xmax, npoints), (-1, 1))
    state = cl.namedtuple('FakeGibbsState', 'pars')(res.pars[iact].values)        # wrapper object
    counts = data.counts[sample]
    lib_sizes = data.lib_sizes[sample]
    if log_scale is True:
        xx = np.exp(x)
        fakedata = cl.namedtuple('FakeCountData', 'counts, groups, lib_sizes')(
            pd.DataFrame(xx), {0: [0]}, pd.DataFrame([lib_sizes]))
        y = xx * np.exp(model.compute_loglik(0, fakedata, state).sum(0))
        counts[counts < 1] = epsilon
        counts = np.log(counts)
    else:
        fakedata = cl.namedtuple('FakeCountData', 'counts, groups, lib_sizes')(
            pd.DataFrame(x), {0: [0]}, pd.DataFrame([lib_sizes]))
        y = np.exp(model.compute_loglik(0, fakedata, state).sum(0))
    y = y * occ / len(res.zz)                             # notice the normalisation of y

    ## plot
    pl.hist(counts, nbins, histtype='stepfilled', linewidth=0, normed=True, color='gray')
    pl.plot(x, y, 'k', x, np.sum(y, 1), 'r')

    ## return
    return x, y

########################################################################################################################


def compute_ra_plot(samples1, samples2, epsilon=0.5):
    """Computes the RA plot of two groups of samples. Use the returned arrays to actually plot the diagram"""

    ## set zero elements to epsilon
    samples1[samples1 < 1] = epsilon
    samples2[samples2 < 1] = epsilon

    ## compute log2 values
    l1 = np.log2(samples1)
    l2 = np.log2(samples2)

    ## compute A and R
    r = l2 - l1
    a = (l2 + l1) * 0.5

    ## return
    return r, a

########################################################################################################################

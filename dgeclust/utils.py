from __future__ import division

import collections as cl
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


def estimate_norm_factors(counts, locfcn=np.median):
    """Estimates normalization factors, using the same method as DESeq"""

    ## compute geometric mean over genes
    lsum = np.log(counts).sum(1)
    means = np.exp(lsum / counts.shape[1])

    ## divide samples by geometric means
    counts = counts.T / means

    ## get median (or other central tendency metric) of samples excluding features with 0 mean
    norm_factors = locfcn(counts[:, means > 0], 1)

    ## return
    return norm_factors

########################################################################################################################


def plot_fitted_model(isample, igroup, res, data, model, xmin=-1, xmax=12, npoints=1000, nbins=100, log_scale=True):
    """Computes the fitted model"""

    ## compute cluster occupancies
    cluster_occupancies, iactive, _, _ = get_cluster_info(len(res.theta), res.zz[igroup])
    cluster_occupancies = cluster_occupancies[iactive]                     # keep occupancies of active clusters, only

    ## compute fitted model
    x = np.reshape(np.linspace(xmin, xmax, npoints), (-1, 1))
    state = cl.namedtuple('FakeGibbsState', 'theta')(res.theta[iactive])        # wrapper object
    if log_scale is True:
        xx = np.exp(x)
        fakedata = cl.namedtuple('FakeCountData', 'counts, groups, norm_factors, lib_sizes')(
            xx, [0], [data.norm_factors[isample]], [data.lib_sizes[isample]])
        y = xx * np.exp(model.compute_loglik(0, fakedata, state).sum(0))
    else:
        fakedata = cl.namedtuple('FakeCountData', 'counts, groups, norm_factors, lib_sizes')(
            x, [0], [data.norm_factors[isample]], [data.lib_sizes[isample]])
        y = np.exp(model.compute_loglik(0, fakedata, state).sum(0))
    y = y * cluster_occupancies / res.zz[igroup].size                             # notice the normalisation of y

    ## plot
    pl.hist(np.log(data.counts[:, isample]), nbins, histtype='stepfilled', linewidth=0, normed=True, color='gray')
    pl.plot(x, y, 'k', x, y.sum(1), 'r')

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

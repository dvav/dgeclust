from __future__ import division

import collections as cl
import numpy as np
import scipy.misc as ms

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


def compute_fitted_model(isample, igroup, res, data, model, xmin=-1, xmax=12, npoints=1000, log_scale=True):
    """Computes the fitted model"""

    ## compute cluster occupancies
    cluster_occupancies, iactive, _, _ = get_cluster_info(len(res.theta), res.zz[igroup])
    cluster_occupancies = cluster_occupancies[iactive]                     # keep occupancies of active clusters, only

    ## compute fitted model
    x = np.linspace(xmin, xmax, npoints).reshape(-1, 1)
    state = cl.namedtuple('FakeGibbsState', 'theta')(res.theta[iactive])        # wrapper object
    if log_scale is True:
        xx = np.exp(x)
        data = cl.namedtuple('FakeCountData', 'counts, groups, library_sizes')(xx, [0], data.library_sizes[[isample]])
        y = xx * np.exp(model.compute_loglik(0, data, state).sum(0))
    else:
        data = cl.namedtuple('FakeCountData', 'counts, groups, library_sizes')(x, [0], data.library_sizes[[isample]])
        y = np.exp(model.compute_loglik(0, data, state).sum(0))
    y = y * cluster_occupancies / res.zz[igroup].size                                   # notice the normalisation of y

    ## return
    return x, y

########################################################################################################################


def compute_ratio_average(samples1, samples2, epsilon=1):
    """Computes the RA diagram of two groups of samples. Use the returned arrays to actually plot the diagram"""

    ## set zero elements to epsilon
    samples1[samples1 < 1] = epsilon
    samples2[samples2 < 1] = epsilon

    ## compute means
    lmeans1 = np.mean(np.log2(samples1), 0)
    lmeans2 = np.mean(np.log2(samples2), 0)

    ## compute A and R
    ratio = (lmeans2 + lmeans1) * 0.5
    average = lmeans2 - lmeans1

    ## return
    return average, ratio

########################################################################################################################

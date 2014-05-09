from __future__ import division

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

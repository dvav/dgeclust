from __future__ import division

import collections as cl
import numpy as np
import scipy.misc as ms
import pandas as pd

################################################################################


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

################################################################################


def read_count_data(file_name, norm_factors=None, groups=None, locfcn=np.median):
    """Reads a data file containing a matrix of count data"""

    ## read data file
    data = pd.read_table(file_name).astype(np.uint32)

    ## fetch counts
    counts = data.values

    ## names of features and samples
    sample_names = data.columns.tolist()
    feature_names = data.index.tolist()

    ## number of features and samples
    nfeatures, nsamples = counts.shape

    ## group information
    groups = np.arange(nsamples) if groups is None else groups
    ngroups = np.size(groups)
    nreplicas = np.asarray([np.size(group) for group in groups])

    ## compute normalisation factors
    norm_factors = estimate_norm_factors(counts, locfcn) if norm_factors is None else norm_factors
    
    ## return as a named tuple
    return cl.namedtuple('CountData', 'counts, sample_names, feature_names, norm_factors, groups, ngroups, nreplicas, '
                                      'nfeatures, nsamples')(counts, sample_names, feature_names, norm_factors, groups,
                                                             ngroups, nreplicas, nfeatures, nsamples)
            
################################################################################


def compute_cluster_occupancies(nclusters, cluster_indicators):
    """Calculates cluster occupancies, given a vector of cluster indicators"""

    labels = np.arange(nclusters).reshape(-1, 1)               # cluster labels
    occupancy_matrix = (cluster_indicators == labels)          # N x M occupancy matrix

    ## return
    return np.sum(occupancy_matrix, 1), occupancy_matrix

################################################################################


def get_cluster_info(nclusters, cluster_indicators):
    """Returns various cluster info, including cluster occupancies"""

    cluster_occupancies, occupancy_matrix = compute_cluster_occupancies(nclusters, cluster_indicators)
    iactive = cluster_occupancies > 0           # indicators of active clusters
    nactive = np.count_nonzero(iactive)         # number of active clusters

    ## return
    return cluster_occupancies, iactive, nactive, occupancy_matrix

################################################################################


def normalize_log_weights(lw):
    """Normalises a matrix of log-weights, row-wise"""

    ref = lw.max(0)
    lsum = ms.logsumexp(lw - ref, 0) + ref      # more stable than np.log(np.exp(lw - ref).sum(0)) + ref

    ## return
    return lw - lsum
 
################################################################################

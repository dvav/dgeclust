from __future__ import division

import numpy as np
import pandas as pd

########################################################################################################################


class CountData(object):
    """Represents a counts data set"""

    def __init__(self, counts, sample_names, feature_names, groups, ngroups, nreplicas, nfeatures, nsamples, lib_sizes):
        """Initialise state from raw data"""

        self.counts = counts
        self.sample_names = sample_names
        self.feature_names = feature_names
        self.groups = groups
        self.ngroups = ngroups
        self.nreplicas = nreplicas
        self.nfeatures = nfeatures
        self.nsamples = nsamples
        self.lib_sizes = lib_sizes

    ####################################################################################################################

    @classmethod
    def load(cls, file_name, norm_method='Quantile', groups=None):
        """Reads a data file containing a matrix of count data"""

        ## read data file
        data = pd.read_table(file_name, index_col=0)  # .astype(np.uint32)

        ## fetch counts
        counts = data.values

        ## names of features and samples
        sample_names = data.columns.tolist()
        feature_names = data.index.tolist()

        ## number of features and samples
        nfeatures, nsamples = counts.shape

        ## group information
        groups = range(nsamples) if groups is None else groups
        ngroups = len(groups)
        nreplicas = np.asarray([np.size(group) for group in groups])

        ## compute normalisation factors and library sizes
        # norm_factors = estimate_norm_factors(counts, locfcn) if norm_factors is None else norm_factors
        lib_sizes = {
            'Total': lambda x: np.sum(x, 0),
            'Quantile': lambda x: estimate_lib_sizes_quantile(x),
            'DESeq': lambda x: estimate_lib_sizes_deseq(x)  # notice that we divide by norm factors
        }[norm_method](counts)

        ## return
        return cls(counts, sample_names, feature_names, groups, ngroups, nreplicas, nfeatures, nsamples, lib_sizes)

    ####################################################################################################################


def estimate_lib_sizes_quantile(counts, quant=75):
    """Estimate library sizes using the quantile method"""

    ## Consider only features smaller that the 75% quantile of non-zero counts
    counts = [sample[sample > 0] for sample in counts.T]
    counts = [sample[sample <= np.percentile(sample, quant)] for sample in counts]
    lib_sizes = [sample.sum() for sample in counts]

    ## return
    return np.asarray(lib_sizes)

########################################################################################################################


def estimate_lib_sizes_deseq(counts, locfcn=np.median):
    """Estimates normalization factors, using the same method as DESeq"""

    ## compute geometric mean of each row in log-scale
    logcounts = np.log(counts.T)
    logmeans = np.mean(logcounts, 0)

    ## take the ratios
    logcounts -= logmeans

    ## get median (or other central tendency metric) of ratios excluding rows with 0 mean
    logcounts = logcounts[:, np.isfinite(logmeans)]
    lib_sizes = np.exp(locfcn(logcounts, 1))

    ## return
    return lib_sizes

########################################################################################################################

from __future__ import division

import numpy as np
import pandas as pd

import dgeclust.utils as ut

########################################################################################################################


class CountData(object):
    """Represents a counts data set"""

    def __init__(self, counts, sample_names, feature_names, groups, ngroups, nreplicas, nfeatures, nsamples,
                 norm_factors, lib_sizes):
        """Initialise state from raw data"""

        self.counts = counts
        self.sample_names = sample_names
        self.feature_names = feature_names
        self.groups = groups
        self.ngroups = ngroups
        self.nreplicas = nreplicas
        self.nfeatures = nfeatures
        self.nsamples = nsamples
        self.norm_factors = norm_factors
        self.lib_sizes = lib_sizes

    ####################################################################################################################

    @classmethod
    def load(cls, file_name, norm_factors=None, groups=None, locfcn=np.median):
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
        norm_factors = ut.estimate_norm_factors(counts, locfcn) if norm_factors is None else norm_factors
        lib_sizes = counts.sum(0)

        ## return
        return cls(counts, sample_names, feature_names, groups, ngroups, nreplicas, nfeatures, nsamples,
                   norm_factors, lib_sizes)

    ####################################################################################################################

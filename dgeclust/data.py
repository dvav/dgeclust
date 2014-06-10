from __future__ import division

import numpy as np
import pandas as pd
import collections as cl

########################################################################################################################


class CountData(object):
    """Represents a counts data set"""

    def __init__(self, counts, groups, lib_sizes):
        """Initialise state from raw data"""

        self.counts = counts
        self.groups = groups
        self.lib_sizes = lib_sizes

    ####################################################################################################################

    @classmethod
    def load(cls, file_name, norm_method='Quantile', groups=None):
        """Reads a data file containing a matrix of count data"""

        ## read data file
        counts = pd.read_table(file_name, index_col=0)  # .astype(np.uint32)

        ## group information
        groups = counts.columns.tolist() if groups is None else groups
        labels = cl.OrderedDict.fromkeys(groups).keys()     # get unique elements, preserve order
        groups = [[col for col, group in zip(counts.columns, groups) if label == group] for label in labels]
        groups = cl.OrderedDict(zip(labels, groups))

        ## compute library sizes
        norm_method = {
            'Total': lambda x: np.sum(x, 0),
            'Quantile': lambda x: estimate_lib_sizes_quantile(x),
            'DESeq': lambda x: estimate_lib_sizes_deseq(x)
        }[norm_method]
        lib_sizes = norm_method(counts.values)

        lib_sizes = pd.DataFrame(lib_sizes, index=counts.columns, columns=['Library sizes']).T

        ## return
        return cls(counts, groups, lib_sizes)


########################################################################################################################


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

import numpy as np
import pandas as pd
import collections as cl

########################################################################################################################


class CountData(object):
    """Represents a counts data set"""

    def __init__(self, counts, lib_sizes=None, groups=None):
        """Initialise state from raw data"""

        # group information
        groups = counts.columns.tolist() if groups is None else groups
        if len(groups) != len(counts.columns.tolist()):
            raise Exception("The list of groups is not the same length as the list of samples!")

        labels = cl.OrderedDict.fromkeys(groups).keys()     # get unique elements, preserve order
        groups = [[col for col, group in zip(counts.columns, groups) if label == group] for label in labels]
        groups = cl.OrderedDict(zip(labels, groups))

        # compute library sizes
        # norm_method = {
        #     'DESeq': estimate_lib_sizes_deseq,
        #     'Quantile': estimate_lib_sizes_quantile
        # }[norm_method]
        # lib_sizes = norm_method(counts.values) if lib_sizes is None else lib_sizes
        lib_sizes = estimate_lib_sizes_deseq(counts.values) if lib_sizes is None else lib_sizes
        lib_sizes = pd.DataFrame(lib_sizes, index=counts.columns, columns=['sizes']).T

        # compute number of replicas per group
        nreplicas = [np.size(val) for val in groups.values()]
        nreplicas = cl.OrderedDict(zip(groups.keys(), nreplicas))

        ##
        self.counts = counts
        self.lib_sizes = lib_sizes
        self.counts_norm = self.counts / self.lib_sizes.values.ravel()
        self.groups = groups
        self.nreplicas = nreplicas

########################################################################################################################


def estimate_lib_sizes_quantile(counts, quant=75):
    """Estimate library sizes using the quantile method"""

    # Consider only features smaller that the quant quantile of non-zero counts
    counts = [sample[sample > 0] for sample in counts.T]
    counts = [sample[sample <= np.percentile(sample, quant)] for sample in counts]
    lib_sizes = [sample.sum() for sample in counts]

    # return
    return np.asarray(lib_sizes)

########################################################################################################################


def estimate_lib_sizes_deseq(counts, locfcn=np.median):
    """Estimates normalization factors, using the same method as DESeq"""

    # compute geometric mean of each row in log-scale
    logcounts = np.log(counts.T)
    logmeans = np.mean(logcounts, 0)

    # take the ratios
    logcounts -= logmeans

    # get median (or other central tendency metric) of ratios excluding rows with 0 mean
    logcounts = logcounts[:, np.isfinite(logmeans)]
    lib_sizes = np.exp(locfcn(logcounts, 1))

    # return
    return lib_sizes

########################################################################################################################

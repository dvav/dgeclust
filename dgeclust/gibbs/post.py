from __future__ import division

import os
import itertools as it
import numpy as np
import pandas as pd

########################################################################################################################


def _compute_pvals(args):
    """Given a particular sample, identify differential expression between features"""
    sample_name, (indir, group1, group2) = args

    ## read sample and identify differentially expressed features
    z = np.loadtxt(os.path.join(indir, str(sample_name)), dtype='uint32')
    p = z[group1] == z[group2]

    ## return
    return p

##


def compute_pvals(indir, fname, t0, tend, dt, group1, group2, pool):
    """For each feature, compute the posterior probability of differential expression between group1 and group2"""

    ## read and sort contents of input directory
    sample_names = os.listdir(indir)
    sample_names = np.sort(np.asarray(sample_names, dtype='uint32'))      # ordered list of sample file names

    ## keep every dt-th sample between t0 and tend
    idxs = (sample_names >= t0) & (sample_names <= tend) & (np.arange(sample_names.size) % dt == 0)
    sample_names = sample_names[idxs]

    ## compute un-normalized values of posteriors
    args = zip(sample_names, it.repeat((indir, group1, group2)))
    p = pool.map(_compute_pvals, args)
    nsamples = len(p)

    ## normalise and adjust posteriors
    p = np.mean(p, 0)
    ii = p.argsort()
    tmp = p[ii].cumsum() / np.arange(1, p.size+1)
    padj = np.zeros(p.shape)
    padj[ii] = tmp

    ## fetch feature names
    feature_names = np.loadtxt(fname, dtype='str')

    ## return
    return pd.DataFrame(np.vstack((p, padj)).T, columns=('pval', 'padj'), index=feature_names).sort(
        columns = 'padj'), nsamples

########################################################################################################################


def _compute_similarity_matrix(args):
    """Given a sample, calculate feature- or group-wise similarity matrix"""
    sample_name, (path, compare_features) = args

    ## read sample
    z = np.loadtxt(os.path.join(path, str(sample_name)), dtype='uint32')
    z = z.T if compare_features is True else z
    # z = z if idxs is None else z[idxs]

    ## calculate un-normalised similarity matrix
    nrows, ncols = z.shape
    dist = np.zeros((nrows, nrows))
    for i in range(nrows):
        dist[i, i:] = np.sum(z[i] == z[i:], 1)
        dist[i:, i] = dist[i, i:]

    ## return
    return dist / ncols


def compute_similarity_matrix(indir, t0, tend, dt, compare_features, pool):
    """Calculate feature- or sample-wise similarity matrix"""

    ## read and sort contents of path
    sample_names = os.listdir(indir)
    sample_names = np.sort(np.asarray(sample_names, dtype='uint32'))      # ordered list of sample file names

    ## keep every dt-th sample between t0 and tend
    ii = (sample_names >= t0) & (sample_names <= tend) & (np.arange(sample_names.size) % dt == 0)
    sample_names = sample_names[ii]

    ## compute intermediate values of p for each sample
    args = zip(sample_names, it.repeat((indir, compare_features)))
    mat = pool.map(_compute_similarity_matrix, args)
    nsamples = len(mat)

    ## return
    return np.mean(mat, 0), nsamples

########################################################################################################################

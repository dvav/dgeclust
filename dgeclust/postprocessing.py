from __future__ import division

import os
import json
import itertools as it
import collections as cl
import multiprocessing as mp

import numpy as np
import pandas as pd

import dgeclust.config as cfg

########################################################################################################################


def _compute_pvals(args):
    """Given a particular sample, identify differential expression between features"""
    sample_name, (indir, igroup1, igroup2) = args

    ## read sample and identify differentially expressed features
    z = np.loadtxt(os.path.join(indir, str(sample_name)), dtype='int', usecols=(igroup1, igroup2)).T
    p = z[0] == z[1]

    ## return
    return p

########################################################################################################################


def compare_groups(indir, group1, group2, t0=5000, tend=10000, dt=1, nthreads=0):
    """For each feature, compute the posterior probability of differential expression between group1 and group2"""

    full_indir = os.path.join(indir, cfg.fnames['z'])

    ## read and sort contents of input directory
    sample_names = os.listdir(full_indir)
    sample_names = np.sort(np.asarray(sample_names, dtype='int'))      # ordered list of sample file names

    ## keep every dt-th sample between t0 and tend
    idxs = (sample_names >= t0) & (sample_names <= tend) & (np.arange(sample_names.size) % dt == 0)
    sample_names = sample_names[idxs]

    ## fetch feature names and groups
    with open(os.path.join(indir, cfg.fnames['config'])) as f:
        config = json.load(f, object_pairs_hook=cl.OrderedDict)
        feature_names = config['featureNames']
        groups = config['groups'].keys()  # order is preserved here

    igroup1 = groups.index(group1)
    igroup2 = groups.index(group2)

    ## prepare for multiprocessing
    nthreads = nthreads if nthreads > 0 else mp.cpu_count()
    pool = mp.Pool(processes=nthreads)

    ## compute un-normalized values of posteriors
    args = zip(sample_names, it.repeat((full_indir, igroup1, igroup2)))
    p = pool.map(_compute_pvals, args)
    nsamples = len(p)

    ## compute posteriors, FDR and FWER
    post = np.mean(p, 0)
    ii = post.argsort()

    tmp = post[ii].cumsum() / np.arange(1, post.size+1)
    fdr = np.zeros(post.shape)
    fdr[ii] = tmp

    # pro = post / post.sum()
    # tmp = pro[ii].cumsum() / pro.size
    # fwer = np.zeros(pro.shape)
    # fwer[ii] = tmp

    ## return
    return pd.DataFrame(np.vstack((post, fdr)).T, columns=('Posteriors', 'FDR'), index=feature_names), nsamples

########################################################################################################################


def _compute_similarity_matrix(args):
    """Given a sample, calculate feature- or group-wise similarity matrix"""
    sample_name, (indir, compare_features) = args

    ## read sample
    z = np.loadtxt(os.path.join(indir, str(sample_name)), dtype='int').T
    z = z.T if compare_features is True else z

    ## calculate un-normalised similarity matrix
    nrows, ncols = z.shape
    dist = np.zeros((nrows, nrows))
    for i in range(nrows):
        dist[i, i:] = np.sum(z[i] == z[i:], 1)
        dist[i:, i] = dist[i, i:]

    ## return
    return dist / ncols


def compute_similarity_matrix(indir, t0=5000, tend=10000, dt=1, compare_features=False, nthreads=0):
    """Calculate feature- or sample-wise similarity matrix"""

    ## read and sort contents of path
    sample_names = os.listdir(os.path.join(indir, cfg.fnames['z']))
    sample_names = np.sort(np.asarray(sample_names, dtype='int'))      # ordered list of sample file names

    ## keep every dt-th sample between t0 and tend
    idxs = (sample_names >= t0) & (sample_names <= tend) & (np.arange(sample_names.size) % dt == 0)
    sample_names = sample_names[idxs]

    ## prepare for multiprocessing
    nthreads = nthreads if nthreads > 0 else mp.cpu_count()
    pool = mp.Pool(processes=nthreads)

    ## compute similarity matrices for each sample
    args = zip(sample_names, it.repeat((indir, compare_features)))
    mat = pool.map(_compute_similarity_matrix, args)
    nsamples = len(mat)

    ## return
    return np.mean(mat, 0), nsamples

########################################################################################################################

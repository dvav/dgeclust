from __future__ import division

import os
import json
import itertools as it
import collections as cl
import multiprocessing as mp

import numpy as np
import pandas as pd

import config as cfg

########################################################################################################################


def _compute_pvals(args):
    """Given a number of samples, compute posterior probabilities of non-differential expression between groups"""

    samples, (indir, igroup1, igroup2) = args

    ## read sample and identify differentially expressed features
    p = 0
    for sample in samples:
        fname = os.path.join(indir, str(sample))
        z = np.loadtxt(fname, dtype='int', usecols=(igroup1, igroup2)).T
        p += z[0] == z[1]

    ## return
    return p

########################################################################################################################


def compare_groups(indir, group1, group2, t0=5000, tend=10000, dt=1, nthreads=None):
    """For each gene, compute the posterior probability of non-differential expression between group1 and group2"""

    full_indir = os.path.join(indir, cfg.fnames['z'])

    ## fetch feature names and groups
    with open(os.path.join(indir, cfg.fnames['config'])) as f:
        config = json.load(f, object_pairs_hook=cl.OrderedDict)
        gene_names = config['geneNames']
        groups = config['groups'].keys()  # order is preserved here

    igroup1 = groups.index(group1)
    igroup2 = groups.index(group2)

    ## prepare for multiprocessing
    nthreads = mp.cpu_count() if nthreads is None or nthreads <= 0 else nthreads
    pool = None if nthreads == 1 else mp.Pool(processes=nthreads)

    ## prepare samples
    samples = np.asarray(os.listdir(full_indir), dtype='int')
    idxs = (samples >= t0) & (samples <= tend) & (np.arange(samples.size) % dt == 0)
    samples = samples[idxs]
    nsamples = samples.size

    ## compute un-normalized values of posteriors
    chunk_size = int(nsamples / nthreads + 1)
    chunks = [samples[i:i+chunk_size] for i in range(0, nsamples, chunk_size)]
    args = zip(chunks, it.repeat((full_indir, igroup1, igroup2)))
    if pool is None:
        p = map(_compute_pvals, args)
    else:
        p = pool.map(_compute_pvals, args)

    ## compute posteriors, FDR and FWER
    post = np.sum(p, 0) / nsamples
    ii = post.argsort()

    tmp = post[ii].cumsum() / np.arange(1, post.size+1)
    fdr = np.zeros(post.shape)
    fdr[ii] = tmp

    # pro = post / post.sum()
    # tmp = pro[ii].cumsum() / pro.size
    # fwer = np.zeros(pro.shape)
    # fwer[ii] = tmp

    ## return
    return pd.DataFrame(np.vstack((post, fdr)).T, columns=('Posteriors', 'FDR'), index=gene_names), nsamples

########################################################################################################################


def _compute_similarity_vector(args):
    """Given a sample, calculate gene- or group-wise similarity matrix"""

    samples, (indir, inc, compare_genes) = args

    ## read sample
    sim_vec = 0
    for sample in samples:
        fname = os.path.join(indir, str(sample))
        z = np.loadtxt(fname, dtype='int')
        z = z if compare_genes is True else z.T
        z = z[inc] if inc is not None else z

        ## calculate un-normalised similarity matrix
        nrows, ncols = z.shape
        sim = [np.sum(z[i] == z[i+1:], 1) for i in range(nrows-1)]
        sim_vec += np.hstack(sim) / ncols

    ## return
    return sim_vec / samples.size


def compute_similarity_vector(indir, t0=5000, tend=10000, dt=1, inc=None, compare_genes=False, nthreads=None):
    """Calculate gene- or group-wise similarity matrix"""

    full_indir = os.path.join(indir, cfg.fnames['z'])

    ## prepare for multiprocessing
    nthreads = mp.cpu_count() if nthreads is None or nthreads <= 0 else nthreads
    pool = None if nthreads == 1 else mp.Pool(processes=nthreads)

    ## prepare samples
    samples = np.asarray(os.listdir(full_indir), dtype='int')
    idxs = (samples >= t0) & (samples <= tend) & (np.arange(samples.size) % dt == 0)
    samples = samples[idxs]
    nsamples = samples.size

    ## compute similarity matrices for each sample
    chunk_size = int(nsamples / nthreads + 1)
    chunks = [samples[i:i+chunk_size] for i in range(0, nsamples, chunk_size)]
    args = zip(chunks, it.repeat((full_indir, inc, compare_genes)))
    if pool is None:
        vec = map(_compute_similarity_vector, args)
    else:
        vec = pool.map(_compute_similarity_vector, args)

    ## return
    return np.mean(vec, 0), nsamples

########################################################################################################################

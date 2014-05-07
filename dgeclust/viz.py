from __future__ import division

import os
import collections as cl

import numpy as np
import matplotlib.pyplot as pl

import utils as ut
import config as cfg

########################################################################################################################


def read_simulation_results(indir):
    """Reads the results of a previously executed simulation from the disk"""

    ## read parameters
    pars = np.loadtxt(os.path.join(indir, cfg.fnames['pars']))
    eta = np.loadtxt(os.path.join(indir, cfg.fnames['eta']))
    pars = cl.namedtuple('Params', 't, Ka, mu, s2, eta')(pars[:, 0], pars[:, 1], pars[:, 2], pars[:, 3], eta[:, 1:])

    ## read cluster centers and cluster indicators
    theta = np.loadtxt(os.path.join(indir, cfg.fnames['theta']))
    c = np.loadtxt(os.path.join(indir, cfg.fnames['c']), dtype='uint32')
    z = np.loadtxt(os.path.join(indir, cfg.fnames['z']), dtype='uint32')
    zz = np.asarray([ci[zi] for ci, zi in zip(c, z)])

    ## return
    return cl.namedtuple('Results', 'pars, theta, c, z, zz')(pars, theta, c, z, zz)

########################################################################################################################


def plot_ratio_average_plot(samples1, samples2, idxs=None, epsilon=1, xlab='log2 mean', ylab='log2 fold change'):
    """Plots the RA diagram of two groups of samples, optionally flagging features indicated by idxs"""

    ## set zero elements to epsilon
    samples1[samples1 < 1] = epsilon
    samples2[samples2 < 1] = epsilon

    ## compute means
    lmeans1 = np.mean(np.log2(samples1), 0)
    lmeans2 = np.mean(np.log2(samples2), 0)

    ## compute A and R
    ratio = (lmeans2 + lmeans1) * 0.5
    average = lmeans2 - lmeans1

    ## generate RA plot
    if idxs is not None:
        pl.plot(average[~idxs], ratio[~idxs], 'k.')
        pl.plot(average[idxs],  ratio[idxs],  'r.')
    else:
        pl.plot(average, ratio, 'k.')

    pl.plot(pl.gca().get_xlim(), (0, 0), '--', color='k')
    pl.xlabel(xlab)
    pl.ylabel(ylab)

    ## return
    return average, ratio

########################################################################################################################


def compute_fitted_model(theta, cluster_indicators, compute_loglik, xmin=-1, xmax=12, npoints=1000, log_scale=True):
    """Computes the fitted model"""

    ## compute cluster occupancies
    cluster_occupancies, iactive, _, _ = ut.get_cluster_info(len(theta), cluster_indicators)
    cluster_occupancies = cluster_occupancies[iactive]                     # keep occupancies of active clusters, only

    ## read active alpha and beta, compute mu and p
    x = np.linspace(xmin, xmax, npoints)
    if log_scale is True:
        y = np.exp(x).reshape(-1, 1) * np.exp(compute_loglik(np.exp(x), theta[iactive]).sum(0))
    else:
        y = np.exp(compute_loglik(x, theta[iactive]).sum(0))
    y = y * cluster_occupancies / cluster_indicators.size          # notice the normalisation of y

    ## return
    return x, y

########################################################################################################################


def plot_fitted_model(isample, igroup, res, data, model, nbins=100, histcolor='grey', linescolor='black',
                      linecolor='red', xlab='log (# counts)', ylab='density', log_scale=True):
    """Plots the histogram of log-counts for a sample, along with the corresponding fitted model and components"""

    ## compute fitted model
    sample = (data.counts / data.norm_factors)[:, isample]
    x, y = compute_fitted_model(res.theta, res.zz[igroup], model.compute_loglik, log_scale=log_scale)

    ## plot fitted model
    if log_scale is True:
        pl.hist(np.log(sample+0.1), nbins, normed=True, histtype='stepfilled', color=histcolor, linewidth=0)
    else:
        pl.hist(sample, nbins, normed=True, histtype='stepfilled', color=histcolor, linewidth=0)

    pl.plot(x, y, color=linescolor)
    pl.plot(x, y.sum(1), color=linecolor)
    pl.xlabel(xlab)
    pl.ylabel(ylab)

    ## return
    return x, y

########################################################################################################################

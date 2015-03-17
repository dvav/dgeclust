# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals
#
# from builtins import *

##

import os
import pickle as pkl
import abc
import itertools as it

##

import numpy as np
import numpy.random as rn

##

import dgeclust.config as cfg
import dgeclust.stats as st
import dgeclust.utils as ut

########################################################################################################################


class Model(metaclass=abc.ABCMeta):
    """Abstract class representing a model"""

    # constructor
    def __init__(self, nfeatures, nsamples, ngroups, ntrunc=(100, 50), outdir=cfg.fnames['outdir']):
        # file names
        if os.path.exists(outdir):
            raise Exception("Directory '{}' already exists!".format(outdir))
        else:
            outdir = os.path.abspath(outdir)
            self.fnames = {
                'outdir': outdir,
                'state': os.path.join(outdir, cfg.fnames['state']),
                'pars': os.path.join(outdir, cfg.fnames['pars']),
                'z': os.path.join(outdir, cfg.fnames['z'])
            }
            os.makedirs(self.fnames['z'])

        # various parameters
        self.nfeatures = nfeatures
        self.nsamples = nsamples
        self.ngroups = ngroups

        # current iteration
        self.iter = 0

        # concentration parameters
        self.eta = 1
        self.zeta = np.ones(self.ngroups)

        # weights
        self.lw = np.tile(-np.log(ntrunc[0]), ntrunc[0])
        self.lu = np.tile(-np.log(ntrunc[1]), (ngroups, ntrunc[1]))

        # indicators
        self.c = rn.choice(self.lw.size, (ngroups, ntrunc[1]), p=np.exp(self.lw))
        self.d = np.asarray([rn.choice(lu.size, nfeatures, p=np.exp(lu)) for lu in self.lu])
        self.c[0, :] = 0
        self.d[0, :] = 0
        self.c[:, 0] = 0

        self.z = np.asarray([c[d] for c, d in zip(self.c, self.d)])

        # cluster statistics
        self.occ = np.bincount(self.z[1:].ravel(), minlength=self.lw.size)
        self.iact = self.occ > 0
        self.nact = np.sum(self.iact)

    ##
    def dump(self, fname):
        """Save current model state"""

        with open(fname, 'wb') as f:
            pkl.dump(self, f)

    ##
    @staticmethod
    def load(indir):
        """Initializes model state from file"""

        # sanity check
        if not os.path.exists(indir):
            raise Exception("Directory '{}' does not exist!".format(indir))

        # load state
        indir = os.path.abspath(indir)
        with open(os.path.join(indir, cfg.fnames['state']), 'rb') as f:
            state = pkl.load(f)

        # correct, in case the original output dir was moved
        if indir != state.fnames['outdir']:
            print('Original output directory has changed! Updating model state...')
            state.fnames = {
                'outdir': indir,
                'state': os.path.join(indir, cfg.fnames['state']),
                'pars': os.path.join(indir, cfg.fnames['pars']),
                'z': os.path.join(indir, cfg.fnames['z'])
            }

        # return
        return state

    ##
    @abc.abstractmethod
    def save(self):
        """Save current model state and state traces"""
        pass

    ##
    @abc.abstractmethod
    def plot_fitted_model(self, sample, data, fig=None, xmin=-1, xmax=12, npoints=1000, nbins=100, epsilon=0.25):
        """Plot fitted model"""
        pass

    ##
    @abc.abstractmethod
    def plot_clusters(self, fig=None, npoints=100):
        """Plot LFC clusters"""
        pass

    ##
    @abc.abstractmethod
    def plot_progress(self, fig=None):
        """Plot simulation progress"""
        pass

    ##
    @abc.abstractmethod
    def update(self, data, pool):
        """Implements a single step of the blocked Gibbs sampler"""
        pass

    ##
    def _update_hdp(self, counts_norm, common_args, compute_loglik, pool):
        # update group-specific variables
        common_args = it.repeat((common_args, compute_loglik, self.lw))
        args = zip(self.c[1:], self.d[1:], self.lu[1:], self.zeta[1:], counts_norm[1:], common_args)

        if pool is None:
            self.c[1:], self.d[1:], self.z[1:], self.lu[1:], self.zeta[1:] = zip(*map(_update_group_vars, args))
        else:
            self.c[1:], self.d[1:], self.z[1:], self.lu[1:], self.zeta[1:] = zip(*pool.map(_update_group_vars, args))

        # update occupancies
        self.occ[:] = np.bincount(self.z[1:].ravel(), minlength=self.lw.size)
        self.iact[:] = self.occ > 0
        self.nact = np.sum(self.iact)

        # update eta
        self.eta = st.sample_eta_west(self.eta, self.nact, self.occ.sum())

        # update weights
        self.lw[:], _ = st.sample_stick(self.occ, self.eta)

########################################################################################################################


def _update_group_vars(args):
    c, _, lu, zeta, counts_norm, (args, compute_loglik, lw) = args

    ##
    nfeatures, nreplicas = counts_norm.shape

    ##
    loglik = compute_loglik(counts_norm[:, :, np.newaxis], args).sum(1)

    # update d
    logw = loglik[:, c] + lu
    logw = ut.normalize_log_weights(logw.T)
    d = st.sample_categorical(np.exp(logw)).ravel()

    # update d: merge null
    d_ = np.zeros(nfeatures, dtype='int')

    ll = compute_loglik(counts_norm, args, c[d].reshape(-1, 1)).sum(-1)
    ll_ = compute_loglik(counts_norm, args, c[d_].reshape(-1, 1)).sum(-1)

    idxs = (ll_ >= ll) | (rn.rand(nfeatures) < np.exp(ll_ - ll))
    d[idxs] = d_[idxs]

    ##
    occ = np.bincount(d, minlength=lu.size)
    iact = occ > 0
    kact = np.sum(iact)

    # update c
    logw = np.vstack([loglik[d == k].sum(0) for k in np.nonzero(iact)[0]]) + lw
    logw = ut.normalize_log_weights(logw.T)
    c[iact] = st.sample_categorical(np.exp(logw)).ravel()
    c[~iact] = rn.choice(lw.size, c.size-kact, p=np.exp(lw))
    c[0] = 0

    # update zeta
    zeta = st.sample_eta_west(zeta, kact, nfeatures)

    # update lu
    lu[:], _ = st.sample_stick(occ, zeta)

    ##
    return c, d, c[d], lu, zeta

########################################################################################################################

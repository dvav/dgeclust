# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals
#
# from builtins import *

##

import os
import itertools as it

##

import numpy as np
import numpy.random as rn
import matplotlib.pylab as pl

##

from dgeclust.models import Model

import dgeclust.stats as st
import dgeclust.config as cfg
import dgeclust.utils as ut

########################################################################################################################


class NBinomModel(Model):
    """Class representing a negative binomial model"""

    # constructor
    def __init__(self, data, outdir=cfg.fnames['outdir'], ntrunc=(100, 50), hpars=(0, 1)):
        # call super
        super(NBinomModel, self).__init__(data, outdir)

        # data mean and var
        mean = np.mean(np.log(data.counts_norm.values + 1))
        var = np.var(np.log(data.counts_norm.values + 1))

        # initial hyper-parameter values
        self.mu, self.tau = np.log(np.abs(var - mean) / mean**2), 1
        self.m0, self.t0 = hpars

        # initial log-values for phi and mu
        self.log_phi = rn.normal(self.mu, 1 / np.sqrt(self.tau), self.nfeatures)
        self.log_mu = rn.normal(mean, np.sqrt(var), self.nfeatures)

        # concentration parameters
        self.eta = 1
        self.zeta = np.ones(self.ngroups)

        # weights
        self.lw = np.tile(-np.log(ntrunc[0]), ntrunc[0])
        self.lu = np.tile(-np.log(ntrunc[1]), (self.ngroups, ntrunc[1]))

        # initial cluster centers
        self.beta = np.r_[0, rn.normal(self.m0, 1/np.sqrt(self.t0), self.lw.size-1)]

        # indicators
        self.c = rn.choice(self.lw.size, (self.ngroups, ntrunc[1]), p=np.exp(self.lw))
        self.d = np.asarray([rn.choice(lu.size, self.nfeatures, p=np.exp(lu)) for lu in self.lu])
        self.c[0, :] = 0
        self.d[0, :] = 0
        self.c[:, 0] = 0

        self.z = np.asarray([c[d] for c, d in zip(self.c, self.d)])

        # cluster statistics
        self.occ = np.bincount(self.z[1:].ravel(), minlength=self.lw.size)
        self.iact = self.occ > 0
        self.nact = np.sum(self.iact)

    ##
    def save(self):
        """Save current model state and state traces"""

        # save state
        self.dump(self.fnames['state'])

        # save chains
        pars = np.hstack([self.iter, self.nact, self.eta, self.mu, self.tau, self.m0, self.t0])
        with open(self.fnames['pars'], 'a') as f:
            np.savetxt(f, np.atleast_2d(pars), fmt='%d\t%d' + '\t%f' * 5)

        # save z
        fout = os.path.join(self.fnames['z'], str(self.iter))
        with open(fout, 'w') as f:
            np.savetxt(f, self.z.T, fmt='%d', delimiter='\t')

    ##
    def plot_fitted_model(self, sample, data, fig=None, xmin=-1, xmax=12, npoints=1000, nbins=100, epsilon=0.25):
        """Plot fitted model"""

        # fetch group
        group = [i for i, item in enumerate(data.groups.items()) if sample in item[1]][0]

        # fetch data
        counts = data.counts_norm[sample].values.astype('float')
        counts[counts < 1] = epsilon
        counts = np.log(counts)

        # compute fitted model
        x = np.reshape(np.linspace(xmin, xmax, npoints), (-1, 1))
        xx = np.exp(x)
        loglik = _compute_loglik(xx, self.log_phi, self.log_mu, self.beta[self.z[group]])
        y = xx * np.exp(loglik) / self.nfeatures

        # plot
        fig = pl.figure() if fig is None else fig
        pl.figure(fig.number)

        pl.hist(counts, nbins, histtype='stepfilled', linewidth=0, normed=True, color='gray')
        pl.plot(x, np.sum(y, 1), 'r')

        pl.grid()
        pl.xlabel('log counts')
        pl.ylabel('density')
        pl.legend(['model', 'data'], loc=0)
        pl.tight_layout()

    ##
    def plot_clusters(self, fig=None, npoints=100):
        """Plot LFC clusters"""

        # data
        beta = self.beta[self.iact]
        occ = self.occ[self.iact]
        x = np.linspace(beta.min()-1, beta.max()+1, npoints)
        y = np.exp(st.normalln(x, self.m0, 1 / np.sqrt(self.t0)))

        # plot
        fig = pl.figure() if fig is None else fig
        pl.figure(fig.number)

        pl.plot(x, y)
        pl.axvline(0, linestyle='--', color='k')
        pl.vlines(beta[1:], [0], occ[1:] / occ[1:].sum(), color='r')

        pl.grid()
        pl.xlabel('LFC')
        pl.ylabel('density')
        pl.legend(['LFC prior', 'null cluster', 'non-null clusters'], loc=0)

        pl.tight_layout()

    ##
    def plot_progress(self, fig=None):
        """Plot simulation progress"""

        # load data
        pars = np.loadtxt(self.fnames['pars'])

        t = pars[:, [0]]
        nact = pars[:, [1]]
        eta = pars[:, [2]]
        mu = pars[:, [3]]
        tau = pars[:, [4]]
        m0 = pars[:, [5]]
        t0 = pars[:, [6]]

        # plot
        fig = pl.figure() if fig is None else fig
        pl.figure(fig.number)

        pl.subplot(2, 2, 1)
        pl.plot(t, nact)
        pl.ylim([0, self.lw.size])
        pl.grid()
        pl.xlabel('# of iterations')
        pl.ylabel('# of LFC clusters')

        pl.subplot(2, 2, 2)
        pl.plot(t, eta)
        pl.grid()
        pl.xlabel('# of iterations')
        pl.ylabel('global concentration parameter')

        pl.subplot(2, 2, 3)
        pl.plot(t, np.c_[mu, 1/tau])
        pl.grid()
        pl.xlabel('# of iterations')
        pl.ylabel('log-dispersion prior')
        pl.legend(['mean', 'variance'], loc=0)

        pl.subplot(2, 2, 4)
        pl.plot(t, np.c_[m0, 1/t0])
        pl.grid()
        pl.xlabel('# of iterations')
        pl.ylabel('LFC prior')
        pl.legend(['mean', 'variance'], loc=0)

        pl.tight_layout()

    ##
    def update(self, data, pool):
        """Implements a single step of the blocked Gibbs sampler"""

        ##
        self.iter += 1

        ##
        # self._update_phi_global(data)
        # self._update_phi_local(data)
        # self._update_mu(data)
        # self._update_beta_global(data)
        # self._update_beta_local(data)

        if rn.rand() < 0.5:
            _update_phi_global(self, data)
            _update_mu(self, data)
            _update_beta_global(self, data)
        else:
            _update_phi_local(self, data)
            _update_mu(self, data)
            _update_beta_local(self, data)

        # update group-specific variables
        counts_norm, _ = data
        common_args = it.repeat((self.log_phi, self.log_mu, self.beta, self.lw))
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

        # update hyper-parameters
        _update_hpars(self)

########################################################################################################################


def _update_phi_local(model, data):

    #
    counts_norm, nreplicas = data
    counts_norm = np.hstack(counts_norm)

    # proposal
    log_phi_ = model.log_phi * np.exp(0.01 * rn.randn(model.nfeatures))

    # log-likelihood
    beta = np.repeat(model.beta[model.z.T], nreplicas, axis=1)
    loglik = _compute_loglik(counts_norm, model.log_phi.reshape(-1, 1), model.log_mu.reshape(-1, 1), beta).sum(-1)
    loglik_ = _compute_loglik(counts_norm, log_phi_.reshape(-1, 1), model.log_mu.reshape(-1, 1), beta).sum(-1)

    # log-prior
    logprior = st.normalln(model.log_phi, model.mu, 1 / model.tau)
    logprior_ = st.normalln(log_phi_, model.mu, 1 / model.tau)

    # log-posterior
    logpost = loglik + logprior
    logpost_ = loglik_ + logprior_

    # update
    idxs = (logpost_ >= logpost) | (rn.rand(model.nfeatures) < np.exp(logpost_ - logpost))
    model.log_phi[idxs] = log_phi_[idxs]


########################################################################################################################

def _update_phi_global(model, data):

    ##
    counts_norm, nreplicas = data
    counts_norm = np.hstack(counts_norm)

    ##
    log_phi_ = rn.normal(model.mu, 1/np.sqrt(model.tau), model.nfeatures).reshape(-1, 1)

    ##
    beta = np.repeat(model.beta[model.z.T], nreplicas, axis=1)
    loglik = _compute_loglik(counts_norm, model.log_phi.reshape(-1, 1), model.log_mu.reshape(-1, 1), beta).sum(-1)
    loglik_ = _compute_loglik(counts_norm, log_phi_.reshape(-1, 1), model.log_mu.reshape(-1, 1), beta).sum(-1)

    ##
    idxs = (loglik_ >= loglik) | (rn.rand(model.nfeatures) < np.exp(loglik_ - loglik))
    model.log_phi[idxs] = log_phi_[idxs]


########################################################################################################################

def _update_mu(model, data):

    ##
    counts_norm, nreplicas = data
    counts_norm = np.hstack(counts_norm)

    ##
    beta = np.exp(model.beta)[model.z.T]
    beta = np.repeat(beta, nreplicas, axis=1)

    ##
    c1 = model.nsamples / np.exp(model.log_phi)
    c2 = (counts_norm / beta).sum(-1)

    ##
    p = rn.beta(0.5 + c1, 0.5 + c2)
    model.log_mu[:] = np.log1p(-p) - np.log(p) - model.log_phi


########################################################################################################################

def _update_beta_global(model, data):

    ##
    counts_norm, nreplicas = data
    counts_norm = np.hstack(counts_norm)

    ##
    beta_ = np.r_[0, rn.normal(model.m0, 1/np.sqrt(model.t0), model.lw.size-1)]

    ##
    beta1 = np.repeat(model.beta[model.z.T], nreplicas, axis=1)
    beta2 = np.repeat(beta_[model.z.T], nreplicas, axis=1)
    loglik = _compute_loglik(counts_norm, model.log_phi.reshape(-1, 1), model.log_mu.reshape(-1, 1), beta1)
    loglik_ = _compute_loglik(counts_norm, model.log_phi.reshape(-1, 1), model.log_mu.reshape(-1, 1), beta2)

    z = np.repeat(model.z.T, nreplicas, axis=1)
    loglik = np.bincount(z.ravel(), loglik.ravel(), minlength=model.lw.size)
    loglik_ = np.bincount(z.ravel(), loglik_.ravel(), minlength=model.lw.size)

    ##
    idxs = (loglik_ >= loglik) | (rn.rand(model.lw.size) < np.exp(loglik_ - loglik))
    model.beta[model.iact & idxs] = beta_[model.iact & idxs]

    ##
    model.beta[~model.iact] = beta_[~model.iact]


########################################################################################################################

def _update_beta_local(model, data):

    ##
    counts_norm, nreplicas = data
    counts_norm = np.hstack(counts_norm)

    ##
    beta_ = np.r_[0, model.beta[1:] * np.exp(0.01 * rn.randn(model.lw.size-1))]

    ##
    beta1 = np.repeat(model.beta[model.z.T], nreplicas, axis=1)
    beta2 = np.repeat(beta_[model.z.T], nreplicas, axis=1)
    loglik = _compute_loglik(counts_norm, model.log_phi.reshape(-1, 1), model.log_mu.reshape(-1, 1), beta1)
    loglik_ = _compute_loglik(counts_norm, model.log_phi.reshape(-1, 1), model.log_mu.reshape(-1, 1), beta2)

    z = np.repeat(model.z.T, nreplicas, axis=1)
    loglik = np.bincount(z.ravel(), loglik.ravel(), minlength=model.lw.size)
    loglik_ = np.bincount(z.ravel(), loglik_.ravel(), minlength=model.lw.size)

    logprior = st.normalln(model.beta, model.m0, 1/model.t0)
    logprior_ = st.normalln(beta_, model.m0, 1/model.t0)

    logpost = loglik + logprior
    logpost_ = loglik_ + logprior_

    ##
    idxs = (logpost_ >= logpost) | (rn.rand(model.lw.size) < np.exp(logpost_ - logpost))
    model.beta[model.iact & idxs] = beta_[model.iact & idxs]

    ##
    model.beta[~model.iact] = rn.normal(model.m0, 1/np.sqrt(model.t0), model.lw.size-model.nact)


########################################################################################################################

def _update_hpars(model):

    # sample first group of hyper-parameters
    s1 = np.sum(model.log_phi)
    s2 = np.sum(model.log_phi**2)
    n = model.log_phi.size
    model.mu, model.tau = st.sample_normal_mean_prec_jeffreys(s1, s2, n)

    # sample second group of hyper-parameters
    beta = model.beta[model.iact]
    s1 = np.sum(beta)
    s2 = np.sum(beta**2)
    n = beta.size
    model.m0, model.t0 = st.sample_normal_mean_prec_jeffreys(s1, s2, n) if n > 2 else (model.m0, model.t0)
    # self.t0 = st.sample_normal_prec_jeffreys(s1, s2, n) if n > 2 else self.t0


########################################################################################################################

def _update_group_vars(args):
    c, _, lu, zeta, counts_norm, (log_phi, log_mu, beta, lw) = args

    ##
    nfeatures, nreplicas = counts_norm.shape

    ##
    loglik = _compute_loglik(counts_norm[:, :, np.newaxis], log_phi.reshape(-1, 1, 1), log_mu.reshape(-1, 1, 1), beta)
    loglik = loglik.sum(1)

    # update d
    logw = loglik[:, c] + lu
    logw = ut.normalize_log_weights(logw.T)
    d = st.sample_categorical(np.exp(logw)).ravel()

    # update d: merge null
    d_ = np.zeros(nfeatures, dtype='int')

    ll = _compute_loglik(counts_norm, log_phi.reshape(-1, 1), log_mu.reshape(-1, 1), beta[c[d]].reshape(-1, 1)).sum(-1)
    ll_ = _compute_loglik(counts_norm, log_phi.reshape(-1, 1), log_mu.reshape(-1, 1), beta[c[d_]].reshape(-1, 1)).sum(-1)

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


def _compute_loglik(counts_norm, log_phi, log_mu, beta):

    ##
    alpha = 1 / np.exp(log_phi)
    p = alpha / (alpha + np.exp(log_mu + beta))

    ##
    return st.nbinomln(counts_norm, alpha, p)

########################################################################################################################

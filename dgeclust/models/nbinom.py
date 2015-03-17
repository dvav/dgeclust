# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals
#
# from builtins import *

##

import os

##

import numpy as np
import numpy.random as rn
import matplotlib.pylab as pl

##

from dgeclust.models.model import Model

import dgeclust.stats as st
import dgeclust.config as cfg

########################################################################################################################


class NBinomModel(Model):
    """Class representing a negative binomial model"""

    # constructor
    def __init__(self, data, outdir=cfg.fnames['outdir'], ntrunc=(100, 50), hpars=(0, 1)):
        # call super
        super().__init__(data.counts.shape[0], data.counts.shape[1], len(data.groups), ntrunc, outdir)

        # data mean and var
        mean = np.mean(np.log(data.counts_norm.values + 1))
        var = np.var(np.log(data.counts_norm.values + 1))

        # initial hyper-parameter values
        self.mu, self.tau = np.log(np.abs(var - mean) / mean**2), 1
        self.m0, self.t0 = hpars

        # initial log-values for phi and mu
        self.log_phi = self.mu + rn.randn(self.nfeatures) / np.sqrt(self.tau)
        self.log_mu = mean + rn.randn(self.nfeatures) * np.sqrt(var)

        # initial cluster centers
        self.beta = np.r_[0, self.m0 + rn.randn(self.lw.size-1) / np.sqrt(self.t0)]

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
        x = np.reshape(np.linspace(xmin, xmax, npoints))
        xx = np.exp(x)
        loglik = _compute_loglik(xx, (self.log_phi, self.log_mu, self.beta), z=self.z.T[:, group])
        y = xx * np.exp(loglik) / self.nfeatures

        # plot
        fig = pl.figure() if fig is None else fig
        pl.figure(fig.number)

        pl.hist(counts, nbins, histtype='stepfilled', linewidth=0, normed=True, color='gray')
        pl.plot(x, np.sum(y, 0), 'r')

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
        if rn.rand() < 0.5:
            self._update_phi_global(data)
            self._update_mu(data)
            self._update_beta_global(data)
        else:
            self._update_phi_local(data)
            self._update_mu(data)
            self._update_beta_local(data)

        # update hdp
        self._update_hdp(data['norm_counts'], (self.log_phi, self.log_mu, self.beta), _compute_loglik, pool)

        # update hyper-parameters
        self._update_hpars()

    ##
    def _update_phi_local(self, data):

        ##
        nreplicas = data['nreplicas']
        counts_norm = np.hstack(data['counts_norm'])

        # proposal
        log_phi_ = self.log_phi * np.exp(0.01 * rn.randn(self.nfeatures))

        # log-likelihood
        z = np.repeat(self.z.T, nreplicas, axis=1)
        loglik = _compute_loglik(counts_norm, (self.log_phi, self.log_mu, self.beta), z).sum(-1)
        loglik_ = _compute_loglik(counts_norm, (log_phi_, self.log_mu, self.beta), z).sum(-1)

        # log-prior
        logprior = st.normalln(self.log_phi, self.mu, 1 / self.tau)
        logprior_ = st.normalln(log_phi_, self.mu, 1 / self.tau)

        # log-posterior
        logpost = loglik + logprior
        logpost_ = loglik_ + logprior_

        # update
        idxs = (logpost_ >= logpost) | (rn.rand(self.nfeatures) < np.exp(logpost_ - logpost))
        self.log_phi[idxs] = log_phi_[idxs]

    ##
    def _update_phi_global(self, data):

        ##
        nreplicas = data['nreplicas']
        counts_norm = np.hstack(data['counts_norm'])

        ##
        log_phi_ = (self.mu + rn.randn(self.nfeatures) / np.sqrt(self.tau)).reshape(-1, 1)

        ##
        z = np.repeat(self.z.T, nreplicas, axis=1)
        loglik = _compute_loglik(counts_norm, (self.log_phi, self.log_mu, self.beta), z).sum(-1)
        loglik_ = _compute_loglik(counts_norm, (log_phi_, self.log_mu, self.beta, z)).sum(-1)

        ##
        idxs = (loglik_ >= loglik) | (rn.rand(self.nfeatures) < np.exp(loglik_ - loglik))
        self.log_phi[idxs] = log_phi_[idxs]

    ##
    def _update_mu(self, data):

        ##
        nreplicas = data['nreplicas']
        counts_norm = np.hstack(data['counts_norm'])

        ##
        beta = np.exp(self.beta)[self.z.T]
        beta = np.repeat(beta, nreplicas, axis=1)

        ##
        c1 = self.nsamples / np.exp(self.log_phi)
        c2 = (counts_norm / beta).sum(-1)

        ##
        p = rn.beta(0.5 + c1, 0.5 + c2)
        self.log_mu[:] = np.log1p(-p) - np.log(p) - self.log_phi

    ##
    def _update_beta_global(self, data):

        ##
        nreplicas = data['nreplicas']
        counts_norm = np.hstack(data['counts_norm'])

        ##
        beta_ = np.r_[0, self.m0 + rn.randn(self.lw.size-1) / np.sqrt(self.t0)]

        ##
        z = np.repeat(self.z.T, nreplicas, axis=1)
        loglik = _compute_loglik(counts_norm, (self.log_phi, self.log_mu, self.beta), z)
        loglik_ = _compute_loglik(counts_norm, (self.log_phi, self.log_mu, beta_), z)

        loglik = np.bincount(z.ravel(), loglik.ravel(), minlength=self.lw.size)
        loglik_ = np.bincount(z.ravel(), loglik_.ravel(), minlength=self.lw.size)

        ##
        idxs = (loglik_ >= loglik) | (rn.rand(self.lw.size) < np.exp(loglik_ - loglik))
        self.beta[self.iact & idxs] = beta_[self.iact & idxs]

        ##
        self.beta[~self.iact] = beta_[~self.iact]

    ##
    def _update_beta_local(self, data):

        ##
        nreplicas = data['nreplicas']
        counts_norm = np.hstack(data['counts_norm'])

        ##
        beta_ = np.r_[0, self.beta[1:] * np.exp(0.01 * rn.randn(self.lw.size-1))]

        ##
        z = np.repeat(self.z.T, nreplicas, axis=1)
        loglik = _compute_loglik(counts_norm, (self.log_phi, self.log_mu, self.beta), z)
        loglik_ = _compute_loglik(counts_norm, (self.log_phi, self.log_mu, beta_), z)

        loglik = np.bincount(z.ravel(), loglik.ravel(), minlength=self.lw.size)
        loglik_ = np.bincount(z.ravel(), loglik_.ravel(), minlength=self.lw.size)

        logprior = st.normalln(self.beta, self.m0, 1/self.t0)
        logprior_ = st.normalln(beta_, self.m0, 1/self.t0)

        logpost = loglik + logprior
        logpost_ = loglik_ + logprior_

        ##
        idxs = (logpost_ >= logpost) | (rn.rand(self.lw.size) < np.exp(logpost_ - logpost))
        self.beta[self.iact & idxs] = beta_[self.iact & idxs]

        ##
        self.beta[~self.iact] = self.m0 + rn.randn(self.lw.size-self.nact) / np.sqrt(self.t0)

    ##
    def _update_hpars(self):
        # sample first group of hyper-parameters
        s1 = np.sum(self.log_phi)
        s2 = np.sum(self.log_phi**2)
        n = self.log_phi.size
        self.mu, self.tau = st.sample_normal_mean_prec_jeffreys(s1, s2, n)

        # sample second group of hyper-parameters
        beta = self.beta[self.iact]
        s1 = np.sum(beta)
        s2 = np.sum(beta**2)
        n = beta.size
        self.m0, self.t0 = st.sample_normal_mean_prec_jeffreys(s1, s2, n) if n > 2 else (self.m0, self.t0)
        # self.t0 = st.sample_normal_prec_jeffreys(s1, s2, n) if n > 2 else self.t0


########################################################################################################################


def _compute_loglik(counts_norm, args, z=None):

    ##
    log_phi, log_mu, beta = args

    ##
    if z is not None:
        beta = beta[z]

    log_phi = log_phi.reshape(-1, 1)
    log_mu = log_mu.reshape(-1, 1)

    ##
    alpha = 1 / np.exp(log_phi)
    p = alpha / (alpha + np.exp(log_mu + beta))

    ##
    return st.nbinomln(counts_norm, alpha, p)

########################################################################################################################

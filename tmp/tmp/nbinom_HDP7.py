from __future__ import division

import os
import pickle as pkl
import itertools as it

import numpy as np
import numpy.random as rn
import matplotlib.pylab as pl

import dgeclust.stats as st
import dgeclust.config as cfg
import dgeclust.utils as ut

########################################################################################################################


class NBinomModel(object):
    """Class representing a negative binomial model"""

    ## constructor
    def __init__(self, data, ntrunc=1000, hpars=(0, 1)):
        """Initializes model from raw data"""

        ## various parameters
        self.ngroups = len(data.groups)
        self.nfeatures, self.nsamples = data.counts.shape

        ## iterations
        self.iter = 0

        ## initial hyper-parameter values
        dmean = np.mean(np.log(data.counts.values+1))
        dvar = np.var(np.log(data.counts.values+1))

        self.mu, self.tau = np.log(np.abs(dvar - dmean) / dmean**2), 1        # hyper-parameters
        self.m0, self.t0 = hpars

        ## initial log-values for phi and mu
        self.log_phi = rn.normal(self.mu, 1/np.sqrt(self.tau), self.nfeatures)
        self.log_mu = rn.normal(dmean, np.sqrt(dvar), self.nfeatures)

        ## concentration parameters
        self.eta = np.log(ntrunc)

        ## weights
        self.lw, _ = st.sample_stick(np.zeros(ntrunc), self.eta)

        ## initial cluster centers
        self.beta = np.r_[0, rn.normal(self.m0, 1/np.sqrt(self.t0), self.lw.size-1)]

        ## indicators
        self.z = rn.choice(self.lw.size, (self.nfeatures, self.ngroups), p=np.exp(self.lw))
        self.z[:, 0] = 0

        ## cluster statistics
        self.occ = np.bincount(self.z.ravel(), minlength=self.lw.size)
        self.iact = self.occ > 0
        self.nact = np.sum(self.iact)

    ##
    def dump(self, fname):
        """Save current model state"""

        with open(fname, 'wb') as f:
            pkl.dump(self, f)

    ##
    def save(self, outdir):
        """Saves the state of the Gibbs sampler"""

        ## save state
        self.dump(os.path.join(outdir, cfg.fnames['state']))

        ## save chains
        pars = np.hstack([self.iter, self.nact, self.eta, self.mu, self.tau, self.m0, self.t0])
        with open(os.path.join(outdir, cfg.fnames['pars']), 'a') as f:
            np.savetxt(f, np.atleast_2d(pars), fmt='%d\t%d' + '\t%f' * 5)

        ## save z
        fout = os.path.join(outdir, cfg.fnames['z'], str(self.iter))
        with open(fout, 'w') as f:
            np.savetxt(f, self.z, fmt='%d', delimiter='\t')

    ##
    def plot_fitted_model(self, sample, data, fig=None, xmin=-1, xmax=12, npoints=1000, nbins=100, epsilon=0.5):
        """Computes the fitted model"""

        ## fetch group
        group = [i for i, item in enumerate(data.groups.items()) if sample in item[1]][0]

        ## fetch data
        counts = data.counts_norm[sample].values.astype('float')
        counts[counts < 1] = epsilon
        counts = np.log(counts)

        ## compute fitted model
        x = np.reshape(np.linspace(xmin, xmax, npoints), (-1, 1))
        xx = np.exp(x)
        loglik = _compute_loglik(xx, self.log_phi, self.log_mu, self.beta[self.z[:, group]])
        y = xx * np.exp(loglik) / self.nfeatures

        ## plot
        fig = pl.figure() if fig is None else fig
        pl.figure(fig.number)

        pl.hist(counts, nbins, histtype='stepfilled', linewidth=0, normed=True, color='gray')
        pl.plot(x, np.sum(y, 1), 'r')

        ## return
        return x, y

    ##
    def update(self, data, pool):
        """Implements a single step of the blocked Gibbs sampler"""

        ##
        self.iter += 1

        ##
        self._update_phi_global(data)
        self._update_mu(data)
        self._update_beta_local(data)
        self._update_z(data)

        ## update occupancies
        self.occ[:] = np.bincount(self.z.ravel(), minlength=self.lw.size)
        self.iact[:] = self.occ > 0
        self.nact = np.sum(self.iact)

        ## update eta
        self.eta = st.sample_eta_west(self.eta, self.nact, self.occ.sum())

        ## update weights
        self.lw[:], _ = st.sample_stick(self.occ, self.eta)

        ## update hyper-parameters
        self._update_hpars()

    ##
    def _update_phi_local(self, data):

        ##
        counts_norm, nreplicas = data
        counts_norm = np.hstack(counts_norm)

        ## proposal
        log_phi_ = self.log_phi * np.exp(0.01 * rn.randn(self.nfeatures))

        ## log-likelihood
        beta = np.repeat(self.beta[self.z], nreplicas, axis=1)
        loglik = _compute_loglik(counts_norm, self.log_phi.reshape(-1, 1), self.log_mu.reshape(-1, 1), beta).sum(-1)
        loglik_ = _compute_loglik(counts_norm, log_phi_.reshape(-1, 1), self.log_mu.reshape(-1, 1), beta).sum(-1)

        ## log-prior
        logprior = st.normalln(self.log_phi, self.mu, 1 / self.tau)
        logprior_ = st.normalln(log_phi_, self.mu, 1 / self.tau)

        ## log-posterior
        logpost = loglik + logprior
        logpost_ = loglik_ + logprior_

        ## update
        idxs = (logpost_ >= logpost) | (rn.rand(self.nfeatures) < np.exp(logpost_ - logpost))
        self.log_phi[idxs] = log_phi_[idxs]

    ##
    def _update_phi_global(self, data):

        ##
        counts_norm, nreplicas = data
        counts_norm = np.hstack(counts_norm)

        ##
        log_phi_ = rn.normal(self.mu, 1/np.sqrt(self.tau), self.nfeatures).reshape(-1, 1)

        ##
        beta = np.repeat(self.beta[self.z], nreplicas, axis=1)
        loglik = _compute_loglik(counts_norm, self.log_phi.reshape(-1, 1), self.log_mu.reshape(-1, 1), beta).sum(-1)
        loglik_ = _compute_loglik(counts_norm, log_phi_.reshape(-1, 1), self.log_mu.reshape(-1, 1), beta).sum(-1)

        ##
        idxs = (loglik_ >= loglik) | (rn.rand(self.nfeatures) < np.exp(loglik_ - loglik))
        self.log_phi[idxs] = log_phi_[idxs]

    ##
    def _update_mu(self, data):

        ##
        counts_norm, nreplicas = data
        counts_norm = np.hstack(counts_norm)

        ##
        beta = np.exp(self.beta)[self.z]
        beta = np.repeat(beta, nreplicas, axis=1)

        ##
        c1 = self.nsamples / np.exp(self.log_phi)
        c2 = (counts_norm / beta).sum(-1)

        ##
        p = rn.beta(0.5 + c1, 0.5 + c2)
        self.log_mu[:] = np.log1p(-p) - np.log(p) - self.log_phi

    ##
    def _update_beta_global(self, data):
        """Propose matrix of indicators c and corresponding delta"""

        ##
        counts_norm, nreplicas = data
        counts_norm = np.hstack(counts_norm)

        ##
        beta_ = np.r_[0, rn.normal(self.m0, 1/np.sqrt(self.t0), self.lw.size-1)]

        ##
        beta1 = np.repeat(self.beta[self.z], nreplicas, axis=1)
        beta2 = np.repeat(beta_[self.z], nreplicas, axis=1)
        loglik = _compute_loglik(counts_norm, self.log_phi.reshape(-1, 1), self.log_mu.reshape(-1, 1), beta1)
        loglik_ = _compute_loglik(counts_norm, self.log_phi.reshape(-1, 1), self.log_mu.reshape(-1, 1), beta2)

        z = np.repeat(self.z, nreplicas, axis=1)
        loglik = np.bincount(z.ravel(), loglik.ravel(), minlength=self.lw.size)
        loglik_ = np.bincount(z.ravel(), loglik_.ravel(), minlength=self.lw.size)

        ##
        idxs = (loglik_ >= loglik) | (rn.rand(self.lw.size) < np.exp(loglik_ - loglik))
        self.beta[self.iact & idxs] = beta_[self.iact & idxs]

        ##
        self.beta[~self.iact] = beta_[~self.iact]

    ##
    def _update_beta_local(self, data):
        """Propose matrix of indicators c and corresponding delta"""

        ##
        counts_norm, nreplicas = data
        counts_norm = np.hstack(counts_norm)

        ##
        beta_ = np.r_[0, self.beta[1:] * np.exp(0.01 * rn.randn(self.lw.size-1))]

        ##
        beta1 = np.repeat(self.beta[self.z], nreplicas, axis=1)
        beta2 = np.repeat(beta_[self.z], nreplicas, axis=1)
        loglik = _compute_loglik(counts_norm, self.log_phi.reshape(-1, 1), self.log_mu.reshape(-1, 1), beta1)
        loglik_ = _compute_loglik(counts_norm, self.log_phi.reshape(-1, 1), self.log_mu.reshape(-1, 1), beta2)

        z = np.repeat(self.z, nreplicas, axis=1)
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
        self.beta[~self.iact] = rn.normal(self.m0, 1/np.sqrt(self.t0), self.lw.size-self.nact)

    def _update_z(self, data):
        """Propose matrix of indicators c and corresponding delta"""

        ##
        counts_norm, nreplicas = data
        counts_norm = np.hstack(counts_norm)

        ##
        z_ = rn.choice(self.lw.size, (self.nfeatures, self.ngroups), p=np.exp(self.lw))

        ##
        beta1 = np.repeat(self.beta[self.z], nreplicas, axis=1)
        beta2 = np.repeat(self.beta[z_], nreplicas, axis=1)
        loglik = _compute_loglik(counts_norm, self.log_phi.reshape(-1, 1), self.log_mu.reshape(-1, 1), beta1)
        loglik_ = _compute_loglik(counts_norm, self.log_phi.reshape(-1, 1), self.log_mu.reshape(-1, 1), beta2)

        idxs = np.cumsum(nreplicas)[:-1]
        loglik = np.asarray([item.sum(-1) for item in np.hsplit(loglik, idxs)]).T
        loglik_ = np.asarray([item.sum(-1) for item in np.hsplit(loglik_, idxs)]).T

        ##
        idxs = (loglik_ >= loglik) | (rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.z[idxs] = z_[idxs]

        ##
        z_ = np.zeros((self.nfeatures, self.ngroups), dtype='int')

        ##
        beta1 = np.repeat(self.beta[self.z], nreplicas, axis=1)
        beta2 = np.repeat(self.beta[z_], nreplicas, axis=1)
        loglik = _compute_loglik(counts_norm, self.log_phi.reshape(-1, 1), self.log_mu.reshape(-1, 1), beta1)
        loglik_ = _compute_loglik(counts_norm, self.log_phi.reshape(-1, 1), self.log_mu.reshape(-1, 1), beta2)

        idxs = np.cumsum(nreplicas)[:-1]
        loglik = np.asarray([item.sum(-1) for item in np.hsplit(loglik, idxs)]).T
        loglik_ = np.asarray([item.sum(-1) for item in np.hsplit(loglik_, idxs)]).T

        ##
        idxs = (loglik_ >= loglik) | (rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.z[idxs] = z_[idxs]


    ##
    def _update_hpars(self):
        """Samples the mean and var of the log-normal from the posterior, given phi"""

        ## sample first group of hyper-parameters
        s1 = np.sum(self.log_phi)
        s2 = np.sum(self.log_phi**2)
        n = self.log_phi.size
        self.mu, self.tau = st.sample_normal_mean_prec_jeffreys(s1, s2, n)

        ## sample second group of hyper-parameters
        beta = self.beta[self.iact]
        s1 = np.sum(beta)
        s2 = np.sum(beta**2)
        n = beta.size
        self.m0, self.t0 = st.sample_normal_mean_prec_jeffreys(s1, s2, n) if n > 2 else (self.m0, self.t0)
        # self.t0 = st.sample_normal_prec_jeffreys(s1, s2, n) if n > 2 else self.t0

    ##
    @staticmethod
    def load(indir):
        """Initializes model state from file"""

        with open(os.path.join(indir, cfg.fnames['state']), 'rb') as f:
            state = pkl.load(f)

        ## return
        return state

    ##
    @staticmethod
    def plot_progress(indir, fig=None, npoints=100):
        """Plot simulation progress"""

        ## load data
        pars = np.loadtxt(os.path.join(indir, cfg.fnames['pars']))
        model = NBinomModel.load(indir)

        t = pars[:, [0]]
        nact = pars[:, [1]]
        eta = pars[:, [2]]
        mu = pars[:, [3]]
        tau = pars[:, [4]]
        m0 = pars[:, [5]]
        t0 = pars[:, [6]]

        ## plot
        fig = pl.figure() if fig is None else fig
        pl.figure(fig.number)

        pl.subplot(3, 2, 1)
        pl.plot(t, nact)
        pl.grid()

        pl.subplot(3, 2, 2)
        pl.plot(t, eta)
        pl.grid()

        pl.subplot(3, 2, 3)
        pl.plot(t, np.c_[mu, 1/tau])
        pl.grid()

        pl.subplot(3, 2, 4)
        pl.plot(t, np.c_[m0, 1/t0])
        pl.grid()

        pl.subplot(3, 2, 5)
        beta = model.beta[model.iact]
        occ = model.occ[model.iact]
        pl.vlines(beta[1:], [0], occ[1:] / occ[1:].sum(), color='r')
        pl.axvline(0, linestyle='--', color='k')
        x = np.linspace(beta.min()-1, beta.max()+1, npoints)
        y = np.exp(st.normalln(x, model.m0, 1 / np.sqrt(model.t0)))
        pl.plot(x, y)
        pl.grid()

########################################################################################################################


def _compute_loglik(counts_norm, log_phi, log_mu, beta):
    """Computes the log-likelihood of each element of counts for each element of theta"""

    ##
    alpha = 1 / np.exp(log_phi)
    p = alpha / (alpha + np.exp(log_mu + beta))

    ##
    return st.nbinomln(counts_norm, alpha, p)

########################################################################################################################

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
    def __init__(self, data, ntrunc=(100, 50), hpars=(0, 1)):
        """Initializes model from raw data"""

        ## various parameters
        self.ngroups = len(data.groups)
        self.nfeatures, self.nsamples = data.counts.shape

        ## iterations
        self.iter = 0

        ## data mean and var
        mean = np.mean(np.log(data.counts_norm.values + 1))
        var = np.var(np.log(data.counts_norm.values + 1))

        ## initial hyper-parameter values
        self.mu1, self.tau1 = np.log(np.abs(var - mean) / mean**2), 1
        self.mu2, self.tau2 = mean, 1 / var
        self.m0, self.t0 = hpars

        ## initial log-values for phi and mu
        self.log_phi = rn.normal(self.mu1, 1 / np.sqrt(self.tau1), (self.ngroups, self.nfeatures))
        self.log_mu = rn.normal(self.mu2, 1 / np.sqrt(self.tau2), self.nfeatures)

        ## concentration parameters
        self.eta = 1
        self.zeta = np.ones(self.ngroups)

        ## weights
        self.lw = np.tile(-np.log(ntrunc[0]), ntrunc[0])
        self.lu = np.tile(-np.log(ntrunc[1]), (self.ngroups, ntrunc[1]))

        ## initial cluster centers
        self.beta = np.r_[0, rn.normal(self.m0, 1/np.sqrt(self.t0), self.lw.size-1)]

        ## indicators
        self.c = rn.choice(self.lw.size, (self.ngroups, ntrunc[1]), p=np.exp(self.lw))
        self.d = np.asarray([rn.choice(lu.size, self.nfeatures, p=np.exp(lu)) for lu in self.lu])
        self.c[0, :] = 0
        self.d[0, :] = 0
        self.c[:, 0] = 0

        self.z = np.asarray([c[d] for c, d in zip(self.c, self.d)])

        ## cluster statistics
        self.occ = np.bincount(self.z[1:].ravel(), minlength=self.lw.size)
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
        pars = np.hstack([self.iter, self.nact, self.eta, self.mu1, self.tau1, self.mu2, self.tau2, self.m0, self.t0])
        with open(os.path.join(outdir, cfg.fnames['pars']), 'a') as f:
            np.savetxt(f, np.atleast_2d(pars), fmt='%d\t%d' + '\t%f' * 7)

        ## save z
        fout = os.path.join(outdir, cfg.fnames['z'], str(self.iter))
        with open(fout, 'w') as f:
            np.savetxt(f, self.z.T, fmt='%d', delimiter='\t')

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
        loglik = _compute_loglik(xx, self.log_phi[group], self.log_mu, self.beta[self.z[group]])
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
        # self._update_phi_global(data)
        # self._update_phi_local(data)
        # self._update_mu(data)
        # self._update_beta_global(data)
        # self._update_beta_local(data)

        if rn.rand() < 0.5:
            self._update_phi_global(data)
            self._update_mu_global(data)
            self._update_beta_global(data)
        else:
            self._update_phi_local(data)
            self._update_mu_local(data)
            self._update_beta_local(data)

        ## update group-specific variables
        counts_norm, _ = data
        common_args = it.repeat((self.log_mu, self.beta, self.lw))
        args = zip(self.log_phi[1:], self.c[1:], self.d[1:], self.lu[1:], self.zeta[1:], counts_norm[1:], common_args)

        if pool is None:
            self.c[1:], self.d[1:], self.z[1:], self.lu[1:], self.zeta[1:] = zip(*map(_update_group_vars, args))
        else:
            self.c[1:], self.d[1:], self.z[1:], self.lu[1:], self.zeta[1:] = zip(*pool.map(_update_group_vars, args))

        ## update occupancies
        self.occ[:] = np.bincount(self.z[1:].ravel(), minlength=self.lw.size)
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
        log_phi_ = self.log_phi * np.exp(0.01 * rn.randn(self.ngroups, self.nfeatures))

        ## log-likelihood
        beta = np.repeat(self.beta[self.z.T], nreplicas, axis=1)
        log_phi_1 = np.repeat(self.log_phi.T, nreplicas, axis=1)
        log_phi_2 = np.repeat(log_phi_.T, nreplicas, axis=1)
        loglik = _compute_loglik(counts_norm, log_phi_1, self.log_mu.reshape(-1, 1), beta).sum(-1)
        loglik_ = _compute_loglik(counts_norm, log_phi_2, self.log_mu.reshape(-1, 1), beta).sum(-1)

        ## collapse

        ## log-prior
        logprior = st.normalln(self.log_phi, self.mu1, 1 / self.tau1)
        logprior_ = st.normalln(log_phi_, self.mu1, 1 / self.tau1)

        ## log-posterior
        logpost = loglik + logprior
        logpost_ = loglik_ + logprior_

        ## update
        idxs = (logpost_ >= logpost) | (rn.rand(self.ngroups, self.nfeatures) < np.exp(logpost_ - logpost))
        self.log_phi[idxs] = log_phi_[idxs]

    ##
    def _update_phi_global(self, data):

        ##
        counts_norm, nreplicas = data
        counts_norm = np.hstack(counts_norm)

        ##
        log_phi_ = rn.normal(self.mu1, 1/np.sqrt(self.tau1), (self.ngroups, self.nfeatures))

        ##
        beta = np.repeat(self.beta[self.z.T], nreplicas, axis=1)
        log_phi_1 = np.repeat(self.log_phi.T, nreplicas, axis=1)
        log_phi_2 = np.repeat(log_phi_.T, nreplicas, axis=1)
        loglik = _compute_loglik(counts_norm, log_phi_1, self.log_mu.reshape(-1, 1), beta)
        loglik_ = _compute_loglik(counts_norm, log_phi_2, self.log_mu.reshape(-1, 1), beta)

        ## collapse
        idxs = []
        loglik = np.hsplit()

        ##
        idxs = (loglik_ >= loglik) | (rn.rand(self.ngroups, self.nfeatures) < np.exp(loglik_ - loglik))
        self.log_phi[idxs] = log_phi_[idxs]

    ##
    def _update_mu_global(self, data):

        ##
        counts_norm, nreplicas = data
        counts_norm = np.hstack(counts_norm)

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
        """Propose matrix of indicators c and corresponding delta"""

        ##
        counts_norm, nreplicas = data
        counts_norm = np.hstack(counts_norm)

        ##
        beta_ = np.r_[0, rn.normal(self.m0, 1/np.sqrt(self.t0), self.lw.size-1)]

        ##
        beta1 = np.repeat(self.beta[self.z.T], nreplicas, axis=1)
        beta2 = np.repeat(beta_[self.z.T], nreplicas, axis=1)
        loglik = _compute_loglik(counts_norm, self.log_phi.reshape(-1, 1), self.log_mu.reshape(-1, 1), beta1)
        loglik_ = _compute_loglik(counts_norm, self.log_phi.reshape(-1, 1), self.log_mu.reshape(-1, 1), beta2)

        z = np.repeat(self.z.T, nreplicas, axis=1)
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
        beta1 = np.repeat(self.beta[self.z.T], nreplicas, axis=1)
        beta2 = np.repeat(beta_[self.z.T], nreplicas, axis=1)
        loglik = _compute_loglik(counts_norm, self.log_phi.reshape(-1, 1), self.log_mu.reshape(-1, 1), beta1)
        loglik_ = _compute_loglik(counts_norm, self.log_phi.reshape(-1, 1), self.log_mu.reshape(-1, 1), beta2)

        z = np.repeat(self.z.T, nreplicas, axis=1)
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
        pl.ylim([0, model.lw.size])
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


def _update_group_vars(args):
    c, _, lu, zeta, counts_norm, (log_phi, log_mu, beta, lw) = args

    ##
    nfeatures, nreplicas = counts_norm.shape

    ##
    loglik = _compute_loglik(counts_norm[:, :, np.newaxis], log_phi.reshape(-1, 1, 1), log_mu.reshape(-1, 1, 1), beta)
    loglik = loglik.sum(1)

    ## update d
    logw = loglik[:, c] + lu
    logw = ut.normalize_log_weights(logw.T)
    d = st.sample_categorical(np.exp(logw)).ravel()

    ## update d: merge null
    d_ = np.zeros(nfeatures, dtype='int')

    ll = _compute_loglik(counts_norm, log_phi.reshape(-1, 1), log_mu.reshape(-1, 1), beta[c[d]].reshape(-1, 1)).sum(-1)
    ll_ = _compute_loglik(counts_norm, log_phi.reshape(-1, 1), log_mu.reshape(-1, 1), beta[c[d_]].reshape(-1, 1)).sum(-1)

    idxs = (ll_ >= ll) | (rn.rand(nfeatures) < np.exp(ll_ - ll))
    d[idxs] = d_[idxs]

    ##
    occ = np.bincount(d, minlength=lu.size)
    iact = occ > 0
    kact = np.sum(iact)

    ## update c
    logw = np.vstack([loglik[d == k].sum(0) for k in np.nonzero(iact)[0]]) + lw
    logw = ut.normalize_log_weights(logw.T)
    c[iact] = st.sample_categorical(np.exp(logw)).ravel()
    c[~iact] = rn.choice(lw.size, c.size-kact, p=np.exp(lw))
    c[0] = 0

    ## update zeta
    zeta = st.sample_eta_west(zeta, kact, nfeatures)

    ## update lu
    lu[:], _ = st.sample_stick(occ, zeta)

    ##
    return c, d, c[d], lu, zeta

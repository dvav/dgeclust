from __future__ import division

import os
import pickle as pkl

import numpy as np
import numpy.random as rn
import matplotlib.pylab as pl

import dgeclust.stats as st
import dgeclust.config as cfg

_LARGE_FLOAT = 1e8
_SMALL_FLOAT = 1e-8

########################################################################################################################


class NBinomModel(object):
    """Class representing a negative binomial model"""

    ## constructor
    def __init__(self, data, ntrunc=100, init_pars=(0, 10, 1, 0.1), eta=None, thr=0.3):
        """Initializes model from raw data"""

        ## various parameters
        self.ngroups = len(data.groups)
        self.nfeatures, self.nsamples = data.counts.shape
        self.thr = thr

        ## basic model state
        self.iter = 0

        dmean = np.mean(np.log(data.counts.values+1))     # add 1 to avoid numerical errors
        dvar = np.var(np.log(data.counts.values+1))
        self.mu1, self.tau1 = np.log(np.abs(dvar - dmean) / dmean**2), 1        # hyper-parameters
        self.mu2, self.tau2 = dmean, 1 / dvar

        self.log_phi = rn.normal(self.mu1, 1 / np.sqrt(self.tau1), self.nfeatures)                 # cluster centers
        self.log_mu = rn.normal(self.mu2, 1 / np.sqrt(self.tau2), self.nfeatures)

        self.eta = np.log(ntrunc) if eta is None else eta
        self.lw, _ = st.sample_stick(np.zeros(ntrunc), self.eta)
        self.z = rn.choice(self.lw.size, (self.nfeatures, self.ngroups), p=np.exp(self.lw))
        self.z[:, 0] = 0

        self.m0, self.t0, self.a0, self.b0 = init_pars
        self.beta_precs = rn.gamma(self.a0, 1 / self.b0, self.lw.size)
        self.beta_means = rn.normal(self.m0, 1 / np.sqrt(self.t0), self.lw.size)
        self.beta_means[0] = 0    # the null cluster

        self.beta = rn.normal(self.beta_means[self.z], 1 / np.sqrt(self.beta_precs[self.z]))  # matrix of fold-changes
        self.beta[:, 0] = 0

        self.z[np.abs(self.beta) < self.thr] = 0

        self.occ = np.bincount(self.z[:, 1:].ravel(), minlength=self.lw.size)   # omit the first column
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
        pars = np.hstack([self.iter, self.nact, self.eta,
                          self.mu1, self.tau1, self.mu2, self.tau2,
                          self.m0, self.t0, self.a0, self.b0])
        with open(os.path.join(outdir, cfg.fnames['pars']), 'a') as f:
            np.savetxt(f, np.atleast_2d(pars), fmt='\t%d' * 2 + '\t%f' * 9)

        ## save z
        path = os.path.join(outdir, cfg.fnames['z'])
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, str(self.iter)), 'w') as f:
            np.savetxt(f, self.z, fmt='%d', delimiter='\t')

    ##
    def plot_fitted_model(self, sample, data, fig=None, xmin=-1, xmax=12, npoints=1000, nbins=100, epsilon=0.5):
        """Computes the fitted model"""

        ## fetch group
        group = [i for i, item in enumerate(data.groups.items()) if sample in item[1]][0]

        ## fetch clusters
        beta = self.beta[:, [group]]
        log_phi = self.log_phi
        log_mu = self.log_mu

        ## fetch data
        counts = data.counts[sample].values.astype('float')
        counts[counts < 1] = epsilon
        counts = np.log(counts)

        lib_size = data.lib_sizes[sample].values.ravel()

        ## compute fitted model
        x = np.reshape(np.linspace(xmin, xmax, npoints), (-1, 1))
        xx = np.exp(x)
        loglik = self._compute_loglik((xx[:, :, np.newaxis], lib_size, 1), log_phi, log_mu, beta).squeeze()
        y = xx * np.exp(loglik) / self.nfeatures

        ## plot
        fig = pl.figure() if fig is None else fig
        pl.figure(fig.number)

        pl.hist(counts, nbins, histtype='stepfilled', linewidth=0, normed=True, color='gray')
        # if plot_components is True:
        #     pl.plot(x, y, 'k')
        pl.plot(x, np.sum(y, 1), 'r')

        ## return
        return x, y

    ##
    def update(self, data, _):
        """Implements a single step of the blocked Gibbs sampler"""

        self.iter += 1

        self._update_phi(data)
        self._update_mu(data)

        self._update_z_beta(data)
        self._update_pars()
        self._update_hpars()

        ## update occupancies, eta and log-weights
        # self.eta = st.sample_eta_ishwaran(self.lw[self.iact], self.eta)
        self.eta = st.sample_eta_west(self.eta, self.nact, self.occ.sum())
        self.lw[:], _ = st.sample_stick(self.occ, self.eta)

    ##
    def _update_phi(self, data):

        ## propose
        log_phi_ = rn.normal(self.mu1, 1/np.sqrt(self.tau1), self.nfeatures)

        ## compute log-likelihoods
        loglik = self._compute_loglik(data, self.log_phi, self.log_mu, self.beta).sum(-1)
        loglik_ = self._compute_loglik(data, log_phi_, self.log_mu, self.beta).sum(-1)

        ## do Metropolis step
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.log_phi[idxs] = log_phi_[idxs]

    # ##
    # def _update_mu(self, data):
    #
    #     ##
    #     counts, lib_sizes, nreplicas = data
    #     beta = np.repeat(self.beta, nreplicas, axis=1)
    #
    #     ##
    #     c1 = self.nsamples / np.exp(self.phi)
    #     c2 = (counts / np.exp(np.log(lib_sizes) + beta)).sum(-1)
    #
    #     ##
    #     # p = rn.beta(1 + c1, 1 + c2)
    #     p = c1 / (c1 + c2)
    #     self.mu[:] = np.log(1 - p) - np.log(p) - self.phi

    ##
    def _update_mu(self, data):

        ## propose
        log_mu_ = rn.normal(self.mu2, 1/np.sqrt(self.tau2), self.nfeatures)

        ## compute log-likelihoods
        loglik = self._compute_loglik(data, self.log_phi, self.log_mu, self.beta).sum(-1)
        loglik_ = self._compute_loglik(data, self.log_phi, log_mu_, self.beta).sum(-1)

        ## do Metropolis step
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.log_mu[idxs] = log_mu_[idxs]

    ##
    def _update_z_beta(self, data):
        """Propose matrix of indicators c and corresponding delta"""

        ##
        z_ = rn.choice(self.lw.size, (self.nfeatures, self.ngroups), p=np.exp(self.lw))
        z_[:, 0] = 0

        ##
        beta_ = rn.normal(self.beta_means[z_], 1/np.sqrt(self.beta_precs[z_]))
        beta_[:, 0] = 0

        z_[np.abs(beta_) < self.thr] = 0

        ##
        loglik = self._compute_loglik(data, self.log_phi, self.log_mu, self.beta)
        loglik_ = self._compute_loglik(data, self.log_phi, self.log_mu, beta_)

        _, _, nreplicas = data
        idxs = np.cumsum(nreplicas)[:-1]
        loglik = np.asarray([item.sum(-1) for item in np.hsplit(loglik, idxs)]).T
        loglik_ = np.asarray([item.sum(-1) for item in np.hsplit(loglik_, idxs)]).T

        # loglik = self._compute_loglik(data, self.log_phi, self.log_mu, self.beta).sum(-1)
        # loglik_ = self._compute_loglik(data, self.log_phi, self.log_mu, beta_).sum(-1)

        ##
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.z[idxs] = z_[idxs]
        self.beta[idxs] = beta_[idxs]

        ##
        self.occ[:] = np.bincount(self.z[:, 1:].ravel(), minlength=self.lw.size)   # omit first column
        self.iact[:] = self.occ > 0
        self.nact = np.sum(self.iact)

    ##
    def _update_pars(self):

        ##
        z = self.z[:, 1:].ravel()
        beta = self.beta[:, 1:].ravel()

        ##
        s1 = np.bincount(z, beta, minlength=self.lw.size)[self.iact]
        s2 = np.bincount(z, beta**2, minlength=self.lw.size)[self.iact]
        n = self.occ[self.iact]

        t0 = st.sample_normal_prec_jeffreys(s1[0], s2[0], n[0])
        m = st.sample_normal_mean(s1[1:], n[1:], self.beta_precs[self.iact][1:], self.m0, self.t0)
        t = st.sample_normal_prec(s1[1:], s2[1:], n[1:], m, self.a0, self.b0)

        self.beta_means[self.iact] = np.r_[0, m]
        self.beta_precs[self.iact] = np.r_[t0, t]

        ninact = self.lw.size - self.nact
        self.beta_means[~self.iact] = rn.normal(self.m0, 1/np.sqrt(self.t0), ninact)
        self.beta_precs[~self.iact] = rn.gamma(self.a0, 1 / self.b0, ninact)

        self.beta_precs[self.beta_precs > _LARGE_FLOAT] = _LARGE_FLOAT     # set limits

    ##
    def _update_hpars(self):
        """Samples the mean and var of the log-normal from the posterior, given phi"""

        ## sample first group of hyper-parameters
        s1 = np.sum(self.log_phi)
        s2 = np.sum(self.log_phi**2)
        n = self.log_phi.size
        self.mu1, self.tau1 = st.sample_normal_mean_prec_jeffreys(s1, s2, n)

        s1 = np.sum(self.log_mu)
        s2 = np.sum(self.log_mu**2)
        n = self.log_mu.size
        self.mu2, self.tau2 = st.sample_normal_mean_prec_jeffreys(s1, s2, n)

        ## sample second group of hyper-parameters
        beta = self.beta[self.z > 0]
        beta_mean = np.mean(beta)
        beta_var = max(np.var(beta), _SMALL_FLOAT)

        self.m0 = beta_mean
        self.t0 = 1 / beta_var
        self.a0 = 1
        self.b0 = beta_var

        # bmeans = self.beta_means[self.iact][1:]
        # bprecs = self.beta_precs[self.iact][1:]
        #
        # s1 = np.sum(bmeans)
        # s2 = np.sum(bmeans**2)
        # n = self.nact - 1
        # self.m0 = st.sample_normal_mean(s1, n, self.t0, beta_mean, beta_prec)
        # self.t0 = st.sample_normal_prec(s1, s2, n, self.m0, 1, beta_var)
        #
        # # s1 = np.sum(bprecs)
        # # n = self.nact - 1
        # # self.a0 = 1
        # # self.b0 = st.sample_gamma_rate(s1, n, self.a0, 1, beta_prec)
        #
        # s1 = np.sum(bprecs)
        # sl = np.sum(np.log(bprecs))
        # n = self.nact - 1
        # self.a0 = st.sample_gamma_shape(sl, n, self.a0, self.b0 + 1e-12)
        # w = rn.gamma(self.a0 * n + 1, 1 / (beta_prec + self.a0 * s1) + 1e-12)
        # self.b0 = self.a0 * w

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
    def _compute_loglik(data, log_phi, log_mu, beta):
        """Computes the log-likelihood of each element of counts for each element of theta"""

        ##
        counts, lib_sizes, nreplicas = data

        ##
        log_phi = log_phi.reshape(-1, 1)
        log_mu = log_mu.reshape(-1, 1)
        beta = np.repeat(beta, nreplicas, axis=1)

        ##
        alpha = 1 / np.exp(log_phi)
        p = alpha / (alpha + lib_sizes * np.exp(log_mu + beta))

        ##
        return st.nbinomln(counts, alpha, p)

    ##
    @staticmethod
    def plot_progress(indir, fig=None):
        """Plot simulation progress"""

        ## load data
        pars = np.loadtxt(os.path.join(indir, cfg.fnames['pars']))

        t = pars[:, [0]]
        nact = pars[:, [1]]
        eta = pars[:, [2]]
        mu1 = pars[:, [3]]
        tau1 = pars[:, [4]]
        mu2 = pars[:, [5]]
        tau2 = pars[:, [6]]
        m0 = pars[:, [7]]
        t0 = pars[:, [8]]
        a0 = pars[:, [9]]
        b0 = pars[:, [10]]

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
        pl.plot(t, np.c_[mu1, 1/tau1])
        pl.grid()

        pl.subplot(3, 2, 4)
        pl.plot(t, np.c_[mu2, 1/tau2])
        pl.grid()

        pl.subplot(3, 2, 5)
        pl.plot(t, np.c_[m0, 1/t0])
        pl.grid()

        pl.subplot(3, 2, 6)
        pl.plot(t, np.c_[a0, b0])
        pl.grid()

########################################################################################################################

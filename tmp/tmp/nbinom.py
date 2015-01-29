from __future__ import division

import os
import pickle as pkl

import numpy as np
import numpy.random as rn
import matplotlib.pylab as pl

import dgeclust.stats as st
import dgeclust.config as cfg

########################################################################################################################


class NBinomModel(object):
    """Class representing a negative binomial model"""

    ## constructor
    def __init__(self, data, ntrunc=100, hpars=(0, 10), eta=None, thr=0.3):
        """Initializes model from raw data"""

        ## various parameters
        self.ngroups = len(data.groups)
        self.nfeatures, self.nsamples = data.counts.shape
        self.thr = thr

        ## iterations
        self.iter = 0

        ## hyper-parameters
        dmean = np.mean(np.log(data.counts.values+1))     # add 1 to avoid infinities
        dvar = np.var(np.log(data.counts.values+1))

        self.m0, self.t0 = hpars                         # hyper-parameters

        self.mu1, self.tau1 = np.log(np.abs(dvar - dmean) / dmean**2), 1
        self.mu2, self.tau2 = dmean, 1 / dvar

        ## log-values of mu and phi
        self.log_phi = rn.normal(self.mu1, 1/np.sqrt(self.tau1), self.nfeatures)                 # cluster centers
        self.log_mu = rn.normal(self.mu2, 1/np.sqrt(self.tau2), self.nfeatures)

        ## concentration parameter, log-weights and cluster centers
        self.eta = np.log(ntrunc) if eta is None else eta
        self.lw, _ = st.sample_stick(np.zeros(ntrunc), self.eta)
        self.beta = np.r_[0, rn.normal(self.m0, 1/np.sqrt(self.t0), self.lw.size-1)]  # matrix of fold-changes

        ## indicator variables
        self.z = np.c_[
            np.zeros((self.nfeatures, 1), dtype='int'),
            rn.choice(self.lw.size, (self.nfeatures, self.ngroups-1), p=np.exp(self.lw))
        ]

        self.z[np.abs(self.beta[self.z]) < self.thr] = 0

        ## cluster info
        self.occ = np.bincount(self.z[:, 1:].ravel(), minlength=self.lw.size)
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
            np.savetxt(f, np.atleast_2d(pars), fmt='\t%d' * 2 + '\t%f' * 7)

        ## save z
        fout = os.path.join(outdir, cfg.fnames['z'], str(self.iter))
        with open(fout, 'w') as f:
            np.savetxt(f, self.z, fmt='%d', delimiter='\t')

    ##
    def plot_fitted_model(self, sample, data, fig=None, xmin=-1, xmax=12, npoints=1000, nbins=100, epsilon=0.5):
        """Computes the fitted model"""

        ## fetch group
        group = [i for i, item in enumerate(data.groups.items()) if sample in item[1]][0]

        ## fetch clusters
        beta = self.beta[self.z[:, [group]]]

        ## fetch data
        counts = data.counts[sample].values.astype('float')
        counts[counts < 1] = epsilon
        counts = np.log(counts)

        lib_size = data.lib_sizes[sample].values.ravel()

        ## compute fitted model
        x = np.reshape(np.linspace(xmin, xmax, npoints), (-1, 1))
        xx = np.exp(x)
        loglik = self._compute_loglik((xx[:, :, np.newaxis], lib_size, 1), self.log_phi, self.log_mu, beta).squeeze()
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
    def update(self, data):
        """Implements a single step of the blocked Gibbs sampler"""

        self.iter += 1

        ## update mu, phi
        self._update_phi(data)
        self._update_mu(data)

        ## update beta
        self._update_beta(data)

        ## update z
        self._update_z(data)

        ## update occupancies, eta and log-weights
        # self.eta = st.sample_eta_ishwaran(self.lw, self.eta)
        self.eta = st.sample_eta_west(self.eta, self.nact, self.occ.sum())
        self.lw, _ = st.sample_stick(self.occ, self.eta)

        ## update hyper-parameters
        self._update_hpars()

    ##
    def _update_phi(self, data):

        ## propose
        log_phi_ = rn.normal(self.mu1, 1/np.sqrt(self.tau1), self.nfeatures)

        ## compute log-likelihoods
        loglik = self._compute_loglik(data, self.log_phi, self.log_mu, self.beta[self.z]).sum(-1)
        loglik_ = self._compute_loglik(data, log_phi_, self.log_mu, self.beta[self.z]).sum(-1)

        ## do Metropolis step
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.log_phi[idxs] = log_phi_[idxs]

    ##
    def _update_mu(self, data):

        ## propose
        log_mu_ = rn.normal(self.mu2, 1/np.sqrt(self.tau2), self.nfeatures)

        ## compute log-likelihoods
        loglik = self._compute_loglik(data, self.log_phi, self.log_mu, self.beta[self.z]).sum(-1)
        loglik_ = self._compute_loglik(data, self.log_phi, log_mu_, self.beta[self.z]).sum(-1)

        ## do Metropolis step
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.log_mu[idxs] = log_mu_[idxs]

    ##
    def _update_z(self, data):
        """Propose matrix of indicators c and corresponding delta"""

        ##
        z_ = np.c_[
            np.zeros((self.nfeatures, 1), dtype='int'),
            rn.choice(self.lw.size, (self.nfeatures, self.ngroups-1), p=np.exp(self.lw))
        ]

        z_[np.abs(self.beta[z_]) < self.thr] = 0

        ##
        loglik = self._compute_loglik(data, self.log_phi, self.log_mu, self.beta[self.z])
        loglik_ = self._compute_loglik(data, self.log_phi, self.log_mu, self.beta[z_])

        _, _, nreplicas = data
        idxs = np.cumsum(nreplicas)[:-1]
        loglik = np.asarray([item.sum(-1) for item in np.hsplit(loglik, idxs)]).T
        loglik_ = np.asarray([item.sum(-1) for item in np.hsplit(loglik_, idxs)]).T

        # loglik = self._compute_loglik(data, self.phi, self.mu, self.beta[self.z]).sum(-1)
        # loglik_ = self._compute_loglik(data, self.phi, self.mu, self.beta[z_]).sum(-1)

        ##
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.z[idxs] = z_[idxs]

        ##
        self.occ[:] = np.bincount(self.z[:, 1:].ravel(), minlength=self.lw.size)
        self.iact[:] = self.occ > 0
        self.nact = np.sum(self.iact)

    ##
    def _update_beta(self, data):

        ##
        beta_ = np.r_[0, rn.normal(self.m0, 1/np.sqrt(self.t0), self.lw.size-1)]

        ##
        loglik = self._compute_loglik(data, self.log_phi, self.log_mu, self.beta[self.z])
        loglik_ = self._compute_loglik(data, self.log_phi, self.log_mu, beta_[self.z])

        _, _, nreplicas = data
        z = np.repeat(self.z, nreplicas, axis=1)
        loglik = np.bincount(z.ravel(), loglik.ravel(), minlength=self.lw.size)
        loglik_ = np.bincount(z.ravel(), loglik_.ravel(), minlength=self.lw.size)

        ##
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        idxs = np.logical_and(idxs, np.abs(beta_) > self.thr)
        self.beta[idxs] = beta_[idxs]

        ##
        self.beta[~self.iact] = beta_[~self.iact]

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
        beta = self.beta[self.z][self.z > 0]
        s1 = np.sum(beta)
        s2 = np.sum(beta**2)
        n = beta.size
        self.m0, self.t0 = st.sample_normal_mean_prec_jeffreys(s1, s2, n)

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
    def _compute_loglik(data, phi, mu, beta):
        """Computes the log-likelihood of each element of counts for each element of theta"""

        ##
        counts, lib_sizes, nreplicas = data

        ##
        phi = phi.reshape(-1, 1)
        mu = mu.reshape(-1, 1)
        beta = np.repeat(beta, nreplicas, axis=1)

        ##
        alpha = 1 / np.exp(phi)
        p = alpha / (alpha + lib_sizes*np.exp(mu + beta))

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

########################################################################################################################

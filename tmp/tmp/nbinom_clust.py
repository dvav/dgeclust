from __future__ import division

import os
import pickle as pkl

import numpy as np
import numpy.random as rn
import matplotlib.pylab as pl

import dgeclust.stats as st
import dgeclust.config as cfg

########################################################################################################################

THR = 0.5

class NBinomModel(object):
    """Class representing a negative binomial model"""

    ## constructor
    def __init__(self, data, ntrunc=1000, hpars=(-4, 1, 4, 10, 0, 1)):
        """Initializes model from raw data"""

        ## various parameters
        self.ngroups = len(data.groups)
        self.nfeatures, self.nsamples = data.counts.shape
        self.ntrunc = ntrunc

        ## basic model state
        self.t = 0

        self.m1, self.v1, self.m2, self.v2, self.m0, self.v0 = hpars                            # hyper-parameters

        self.phi = rn.normal(self.m1, np.sqrt(self.v1), self.nfeatures)                         # cluster centers
        self.mu = rn.normal(self.m2, np.sqrt(self.v2), self.nfeatures)
        self.beta = np.c_[
            np.zeros((self.ntrunc, 1)),
            rn.normal(self.m0, np.sqrt(self.v0), (self.ntrunc, self.ngroups-1))   # matrix of fold-changes
        ]

        self.eta = np.log(self.ntrunc)
        self.lw, _ = st.sample_stick(np.zeros(self.ntrunc), self.eta)
        self.c = rn.choice(self.ntrunc, self.nfeatures, p=np.exp(self.lw))                  # matrix of indicators

        self.occ = np.bincount(self.c, minlength=self.ntrunc)
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
        pars = np.hstack([self.t, self.nact, self.eta, self.m1, self.v1, self.m2, self.v2, self.m0, self.v0])
        with open(os.path.join(outdir, cfg.fnames['pars']), 'a') as f:
            np.savetxt(f, np.atleast_2d(pars), fmt='\t%d' * 2 + '\t%f' * 7)

        ## save z
        # path = os.path.join(outdir, cfg.fnames['z'])
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # with open(os.path.join(path, str(self.t)), 'w') as f:
        #     np.savetxt(f, self.z, fmt='%d', delimiter='\t')

    ##
    def plot_fitted_model(self, sample, data, fig=None, xmin=-1, xmax=12, npoints=1000, nbins=100, epsilon=0.5):
        """Computes the fitted model"""

        ## fetch group
        group = [i for i, item in enumerate(data.groups.items()) if sample in item[1]][0]

        ## fetch clusters
        beta = self.beta[:, [group]][self.c]
        phi = self.phi
        mu = self.mu

        ## fetch data
        counts = data.counts[sample].values.astype('float')
        counts[counts < 1] = epsilon
        counts = np.log(counts)

        lib_size = data.lib_sizes[sample].values.ravel()

        ## compute fitted model
        x = np.reshape(np.linspace(xmin, xmax, npoints), (-1, 1))
        xx = np.exp(x)
        loglik = type(self).compute_loglik((xx[:, :, np.newaxis], lib_size, 1), phi, mu, beta).squeeze()
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

        self.t += 1

        ## update mu phi
        self._update_phi(data)
        self._update_mu(data)

        ## update c
        self._update_c(data)
        self.occ[:] = np.bincount(self.c, minlength=self.ntrunc)
        self.iact[:] = self.occ > 0
        self.nact = np.sum(self.iact)

        ## update beta
        self._update_beta(data)

        ## update occupancies, eta and log-weights
        self.eta = st.sample_eta_ishwaran(self.lw)
        # self.eta = st.sample_eta_west(self.eta, self.nact, self.occ.sum())
        self.lw[:], _ = st.sample_stick(self.occ, self.eta)

        ## update hyper-parameters
        self._update_hpars()

    ##
    def _update_phi(self, data):

        ## propose
        phi_ = rn.normal(self.m1, np.sqrt(self.v1), self.nfeatures)

        ## compute log-likelihoods
        loglik = self.compute_loglik(data, self.phi, self.mu, self.beta[self.c]).sum(-1)
        loglik_ = self.compute_loglik(data, phi_, self.mu, self.beta[self.c]).sum(-1)

        ## do Metropolis step
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.phi[idxs] = phi_[idxs]

    ##
    def _update_mu(self, data):

        ## propose
        mu_ = rn.normal(self.m2, np.sqrt(self.v2), self.nfeatures)

        ## compute log-likelihoods
        loglik = self.compute_loglik(data, self.phi, self.mu, self.beta[self.c]).sum(-1)
        loglik_ = self.compute_loglik(data, self.phi, mu_, self.beta[self.c]).sum(-1)

        ## do Metropolis step
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.mu[idxs] = mu_[idxs]

    # ##
    # def _update_mu(self, data):
    #
    #     ##
    #     counts, lib_sizes, nreplicas = data
    #     beta = np.repeat(np.exp(self.beta)[self.c], nreplicas, axis=1)
    #
    #     ##
    #     c1 = self.nsamples / self.phi
    #     c2 = (counts / lib_sizes / beta).sum(-1)
    #
    #     ##
    #     p = rn.beta(1 + c1, 1 + c2)
    #     self.mu[:] = np.log(1 - p) - np.log(p) - np.log(self.phi)

    ##
    def _update_c(self, data):
        """Propose matrix of indicators c and corresponding delta"""

        ##
        c_ = rn.choice(self.ntrunc, self.nfeatures, p=np.exp(self.lw))

        ##
        loglik = self.compute_loglik(data, self.phi, self.mu, self.beta[self.c]).sum(-1)
        loglik_ = self.compute_loglik(data, self.phi, self.mu, self.beta[c_]).sum(-1)

        ##
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.c[idxs] = c_[idxs]

    ##
    def _update_beta(self, data):

        ##
        beta_ = np.c_[
            np.zeros((self.ntrunc, 1)),
            rn.normal(self.m0, np.sqrt(self.v0), (self.ntrunc, self.ngroups-1))   # matrix of fold-changes
        ]

        ##
        loglik = self.compute_loglik(data, self.phi, self.mu, self.beta[self.c]).sum(-1)       # shouldn't sum
        loglik_ = self.compute_loglik(data, self.phi, self.mu, beta_[self.c]).sum(-1)

        loglik = np.bincount(self.c, loglik, minlength=self.ntrunc)[self.iact]
        loglik_ = np.bincount(self.c, loglik_, minlength=self.ntrunc)[self.iact]

        ##
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.beta[self.iact][idxs] = beta_[self.iact][idxs]

        ##
        self.beta[~self.iact] = beta_[~self.iact]

    ##
    def _update_hpars(self):
        """Samples the mean and var of the log-normal from the posterior, given phi"""

        ## sample first group of hyper-parameters
        self.m1, self.v1 = st.sample_normal_mean_var_jeffreys(np.sum(self.phi), np.sum(self.phi**2), self.phi.size)
        self.m2, self.v2 = st.sample_normal_mean_var_jeffreys(np.sum(self.mu), np.sum(self.mu**2), self.mu.size)

        ## sample second group of hyper-parameters
        beta = self.beta[self.c]
        beta = beta[beta != 0]
        self.m0, self.v0 = st.sample_normal_mean_var_jeffreys(np.sum(beta), np.sum(beta**2), beta.size)

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
    def compute_loglik(data, phi, mu, beta):
        """Computes the log-likelihood of each element of counts for each element of theta"""

        ##
        counts, lib_sizes, nreplicas = data

        ##
        phi = phi.reshape(-1, 1)
        mu = mu.reshape(-1, 1)
        beta = np.repeat(beta, nreplicas, axis=1)

        ##
        alpha = 1 / np.exp(phi)
        p = alpha / (alpha + lib_sizes * np.exp(mu + beta))

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
        m1 = pars[:, [3]]
        v1 = pars[:, [4]]
        m2 = pars[:, [5]]
        v2 = pars[:, [6]]
        m0 = pars[:, [7]]
        v0 = pars[:, [8]]

        ## plot
        fig = pl.figure() if fig is None else fig
        pl.figure(fig.number)

        pl.subplot(2, 2, 1)
        pl.plot(t, nact)
        pl.grid()

        pl.subplot(2, 2, 2)
        pl.plot(t, eta)
        pl.grid()

        pl.subplot(2, 2, 3)
        pl.plot(t, np.c_[m1, v1, m2, v2])
        pl.grid()

        pl.subplot(2, 2, 4)
        pl.plot(t, np.c_[m0, v0])
        pl.grid()

########################################################################################################################

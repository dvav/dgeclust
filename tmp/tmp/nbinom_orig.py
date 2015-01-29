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
    def __init__(self, data, hpars=(-4, 1, 4, 10, 0, 1)):
        """Initializes model from raw data"""

        ## various parameters
        self.ngroups = len(data.groups)
        self.nfeatures, self.nsamples = data.counts.shape

        ## basic model state
        self.t = 0

        self.m1, self.v1, self.m2, self.v2, self.m0, self.v0 = hpars                         # hyper-parameters

        self.phi = rn.normal(self.m1, np.sqrt(self.v1), self.nfeatures)                 # cluster centers
        self.mu = rn.normal(self.m2, np.sqrt(self.v2), self.nfeatures)

        self.eta = 1
        self.z = np.zeros((self.nfeatures, self.ngroups), dtype='int')
        self.beta = np.zeros((self.nfeatures, self.ngroups))                 # matrix of fold-changes

        self.occ = np.bincount(self.z.ravel(), minlength=self.ngroups)
        self.lw = np.log(self.occ / self.occ.sum())

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
        pars = np.hstack([self.t, self.eta, self.m1, self.v1, self.m2, self.v2, self.m0, self.v0, self.lw])
        with open(os.path.join(outdir, cfg.fnames['pars']), 'a') as f:
            np.savetxt(f, np.atleast_2d(pars), fmt='\t%d' + '\t%f' * (7 + self.lw.size))

        ## save z
        path = os.path.join(outdir, cfg.fnames['z'])
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, str(self.t)), 'w') as f:
            np.savetxt(f, self.z, fmt='%d', delimiter='\t')

    ##
    def plot_fitted_model(self, sample, data, fig=None, xmin=-1, xmax=12, npoints=1000, nbins=100, epsilon=0.5):
        """Computes the fitted model"""

        ## fetch group
        group = [i for i, item in enumerate(data.groups.items()) if sample in item[1]][0]

        ## fetch clusters
        beta = self.beta[:, [group]]
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
        loglik = self._compute_loglik((xx[:, :, np.newaxis], lib_size, 1), phi, mu, beta).squeeze()
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
        self._update_z_beta(data)

        ##
        self.occ = np.bincount(self.z.ravel(), minlength=self.ngroups)
        self.lw = np.log(self.occ / self.occ.sum())
        self.eta = st.sample_eta_west(self.eta, np.sum(self.occ > 0), self.occ.sum())

        ## update hyper-parameters
        self._update_hpars()

    ##
    def _update_phi(self, data):

        ## propose
        phi_ = rn.normal(self.m1, np.sqrt(self.v1), self.nfeatures)

        ## compute log-likelihoods
        loglik = self._compute_loglik(data, self.phi, self.mu, self.beta).sum(-1)
        loglik_ = self._compute_loglik(data, phi_, self.mu, self.beta).sum(-1)

        ## do Metropolis step
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.phi[idxs] = phi_[idxs]

    ##
    def _update_mu(self, data):

        ## propose
        mu_ = rn.normal(self.m2, np.sqrt(self.v2), self.nfeatures)

        ## compute log-likelihoods
        loglik = self._compute_loglik(data, self.phi, self.mu, self.beta).sum(-1)
        loglik_ = self._compute_loglik(data, self.phi, mu_, self.beta).sum(-1)

        ## do Metropolis step
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.mu[idxs] = mu_[idxs]

    ##
    def _update_z_beta(self, data):
        """Propose matrix of indicators c and corresponding delta"""

        ##
        w = np.r_[1, self.eta]
        z_ = np.c_[
            np.zeros((self.nfeatures, 1), dtype='int'),
            rn.choice(w.size, self.nfeatures, p=w / np.sum(w))
        ]

        ##
        rnds = np.c_[
            np.zeros((self.nfeatures, 1)),
            rn.normal(self.m0, np.sqrt(self.v0), (self.nfeatures, self.ngroups-1))
        ]
        beta_ = rnds[np.repeat(range(self.nfeatures), self.ngroups), z_.ravel()].reshape(self.nfeatures, self.ngroups)

        ##
        loglik = self._compute_loglik(data, self.phi, self.mu, self.beta).sum(-1)
        loglik_ = self._compute_loglik(data, self.phi, self.mu, beta_).sum(-1)

        ##
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.z[idxs] = z_[idxs]
        self.beta[idxs] = beta_[idxs]

    ##
    def _update_hpars(self):
        """Samples the mean and var of the log-normal from the posterior, given phi"""

        ## sample first group of hyper-parameters
        self.m1, self.v1 = st.sample_normal_mean_var_jeffreys(np.sum(self.phi), np.sum(self.phi**2), self.phi.size)
        self.m2, self.v2 = st.sample_normal_mean_var_jeffreys(np.sum(self.mu), np.sum(self.mu**2), self.mu.size)

        ## sample second group of hyper-parameters
        beta = self.beta[self.z > 0]
        self.m0, self.v0 = st.sample_normal_mean_var_jeffreys(np.sum(beta), np.sum(beta**2), beta.size)
        # self.v0 = st.sample_normal_var_jeffreys(np.sum(beta), np.sum(beta**2), beta.size)

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
        p = alpha / (alpha + np.exp(np.log(lib_sizes) + mu + beta))

        ##
        return st.nbinomln(counts, alpha, p)

    ##
    @staticmethod
    def plot_progress(indir, fig=None):
        """Plot simulation progress"""

        ## load data
        pars = np.loadtxt(os.path.join(indir, cfg.fnames['pars']))

        t = pars[:, [0]]
        eta = pars[:, [1]]
        m1 = pars[:, [2]]
        v1 = pars[:, [3]]
        m2 = pars[:, [4]]
        v2 = pars[:, [5]]
        m0 = pars[:, [6]]
        v0 = pars[:, [7]]
        lw = pars[:, 8:]

        ## plot
        fig = pl.figure() if fig is None else fig
        pl.figure(fig.number)

        pl.subplot(2, 2, 1)
        pl.plot(t, eta)
        pl.grid()

        pl.subplot(2, 2, 2)
        pl.plot(t, np.c_[m1, v1, m2, v2])
        pl.grid()

        pl.subplot(2, 2, 3)
        pl.plot(t, np.c_[m0, v0])
        pl.grid()

        pl.subplot(2, 2, 4)
        pl.plot(t, np.exp(lw[:, 1:]))
        pl.grid()

########################################################################################################################

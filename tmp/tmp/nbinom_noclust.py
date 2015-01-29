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
    def __init__(self, data, hpars=(0, 1)):
        """Initializes model from raw data"""

        ## various parameters
        self.ngroups = len(data.groups)
        self.nfeatures, self.nsamples = data.counts.shape

        ## basic model state
        self.iter = 0

        dmean = np.mean(np.log(data.counts.values+1))     # add 1 to avoid infinities
        dvar = np.var(np.log(data.counts.values+1))
        self.m1, self.t1 = np.log(np.abs(dvar - dmean) / dmean**2), 1        # hyper-parameters
        self.m2, self.t2 = dmean, 1 / dvar
        self.m0, self.t0 = hpars

        self.log_phi = rn.normal(self.m1, 1/np.sqrt(self.t1), self.nfeatures)
        self.log_mu = rn.normal(self.m2, 1/np.sqrt(self.t2), self.nfeatures)
        self.beta = np.c_[
            np.zeros((self.nfeatures, 1)),
            rn.normal(self.m0, 1/np.sqrt(self.t0), (self.nfeatures, self.ngroups-1))      # matrix of fold-changes
        ]

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
        pars = np.hstack([self.iter, self.m1, self.t1, self.m2, self.t2, self.m0, self.t0])
        with open(os.path.join(outdir, cfg.fnames['pars']), 'a') as f:
            np.savetxt(f, np.atleast_2d(pars), fmt='\t%d' + '\t%f' * 6)

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
        loglik = type(self).compute_loglik((xx[:, :, np.newaxis], lib_size, 1), log_phi, log_mu, beta).squeeze()
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

        ## update mu, phi, beta
        self._update_phi(data)
        self._update_mu(data)
        self._update_beta(data)

        ## update hyper-parameters
        self._update_hpars()

    ##
    def _update_phi(self, data):

        ## propose
        log_phi_ = rn.normal(self.m1, 1/np.sqrt(self.t1), self.nfeatures)

        ## compute log-likelihoods
        loglik = self.compute_loglik(data, self.log_phi, self.log_mu, self.beta).sum(-1)
        loglik_ = self.compute_loglik(data, log_phi_, self.log_mu, self.beta).sum(-1)

        ## do Metropolis step
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.log_phi[idxs] = log_phi_[idxs]

    ##
    def _update_mu(self, data):

        ## propose
        log_mu_ = rn.normal(self.m2, 1/np.sqrt(self.t2), self.nfeatures)

        ## compute log-likelihoods
        loglik = self.compute_loglik(data, self.log_phi, self.log_mu, self.beta).sum(-1)
        loglik_ = self.compute_loglik(data, self.log_phi, log_mu_, self.beta).sum(-1)

        ## do Metropolis step
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.log_mu[idxs] = log_mu_[idxs]

    ##
    def _update_beta(self, data):
        """Propose matrix of indicators c and corresponding delta"""

        ##
        beta_ = np.c_[
            np.zeros((self.nfeatures, 1)),
            rn.normal(self.m0, 1/np.sqrt(self.t0), (self.nfeatures, self.ngroups-1))
        ]

        ##
        loglik = self.compute_loglik(data, self.log_phi, self.log_mu, self.beta)
        loglik_ = self.compute_loglik(data, self.log_phi, self.log_mu, beta_)

        _, _, nreplicas = data
        idxs = np.cumsum(nreplicas)[:-1]
        loglik = np.asarray([item.sum(-1) for item in np.hsplit(loglik, idxs)]).T
        loglik_ = np.asarray([item.sum(-1) for item in np.hsplit(loglik_, idxs)]).T

        ##
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.beta[idxs] = beta_[idxs]

    ##
    def _update_hpars(self):
        """Samples the mean and var of the log-normal from the posterior, given phi"""

        ##
        s1 = np.sum(self.log_phi)
        s2 = np.sum(self.log_phi**2)
        n = self.log_phi.size
        self.m1, self.t1 = st.sample_normal_mean_prec_jeffreys(s1, s2, n)

        s1 = np.sum(self.log_mu)
        s2 = np.sum(self.log_mu**2)
        n = self.log_mu.size
        self.m2, self.t2 = st.sample_normal_mean_prec_jeffreys(s1, s2, n)

        ##
        beta = self.beta[:, 1:]
        self.m0, self.t0 = st.sample_normal_mean_prec_jeffreys(np.sum(beta), np.sum(beta**2), beta.size)

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
    def compute_loglik(data, log_phi, log_mu, beta):
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
        m1 = pars[:, [1]]
        t1 = pars[:, [2]]
        m2 = pars[:, [3]]
        t2 = pars[:, [4]]
        m0 = pars[:, [5]]
        t0 = pars[:, [6]]

        ## plot
        fig = pl.figure() if fig is None else fig
        pl.figure(fig.number)

        pl.subplot(2, 2, 1)
        pl.plot(t, np.c_[m1, 1/t1])
        pl.grid()

        pl.subplot(2, 2, 2)
        pl.plot(t, np.c_[m2, 1/t2])
        pl.grid()

        pl.subplot(2, 2, 3)
        pl.plot(t, np.c_[m0, 1/t0])
        pl.grid()


########################################################################################################################

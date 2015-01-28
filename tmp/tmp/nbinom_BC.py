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
    def __init__(self, data, ntrunc=(1000, 100), hpars=(1, 4, 10, 0, 1)):
        """Initializes model from raw data"""

        ## various parameters
        self.ngroups = len(data.groups)
        self.nfeatures, self.nsamples = data.counts.shape
        self.ntrunc = ntrunc

        ## basic model state
        self.t = 0

        self.s, self.m, self.v, self.m0, self.v0 = hpars                                     # hyper-parameters

        self.phi = rn.exponential(self.s, self.nfeatures)                                       # cluster centers
        self.mu = rn.lognormal(self.m, np.sqrt(self.v), self.nfeatures)
        self.beta = rn.lognormal(self.m0, np.sqrt(self.v0), ntrunc)               # matrix of fold-changes

        self.eta1 = 1
        self.eta2 = 1
        self.lw1, _ = st.sample_stick(np.zeros(ntrunc[0]), self.eta1)
        self.lw2, _ = st.sample_stick(np.zeros(ntrunc[1]), self.eta2)

        self.c = rn.choice(self.lw1.size, self.nfeatures, p=np.exp(self.lw1))
        self.d = rn.choice(self.lw2.size, self.ngroups, p=np.exp(self.lw2))

        self.occ1 = np.bincount(self.c, minlength=self.lw1.size)
        self.occ2 = np.bincount(self.d, minlength=self.lw2.size)

        self.iact1 = self.occ1 > 0
        self.nact1 = np.sum(self.iact1)

        self.iact2 = self.occ2 > 0
        self.nact2 = np.sum(self.iact2)

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
        pars = np.hstack([self.t, self.nact, self.eta, self.T, self.s, self.m, self.v, self.m0, self.v0])
        with open(os.path.join(outdir, cfg.fnames['pars']), 'a') as f:
            np.savetxt(f, np.atleast_2d(pars), fmt='\t%d' * 2 + '\t%f' * 7)

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
        beta = self.beta[self.z[:, [group]]]
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

        ## update mu, phi
        self._update_phi(data)
        self._update_mu(data)

        ## update z
        self._update_c(data)
        self._update_d(data)

        self.occ1[:] = np.bincount(self.c, minlength=self.lw1.size)
        self.occ2[:] = np.bincount(self.d, minlength=self.lw2.size)

        self.iact1[:] = self.occ1 > 0
        self.iact2[:] = self.occ2 > 0

        self.nact1 = np.sum(self.iact1)
        self.nact2 = np.sum(self.iact2)

        ## update occupancies, eta and log-weights
        self.eta1 = st.sample_eta_ishwaran(np.sort(self.lw1)[::-1])
        self.eta2 = st.sample_eta_ishwaran(np.sort(self.lw2)[::-1])

        self.lw1[:], _ = st.sample_stick(self.occ1, self.eta1)
        self.lw2[:], _ = st.sample_stick(self.occ2, self.eta2)

        ## update beta
        self._update_beta(data)

        ## update hyper-parameters
        self._update_hpars()

    ##
    def _update_phi(self, data):

        ## propose
        phi_ = rn.exponential(self.s, self.nfeatures)

        ## compute log-likelihoods
        z = np.repeat(self.c, self.ngroups), np.tile(self.d, self.nfeatures)
        loglik = self._compute_loglik(data, self.phi, self.mu, self.beta[z].reshape(self.nfeatures, self.ngroups)).\
            sum(-1)
        loglik_ = self._compute_loglik(data, phi_, self.mu, self.beta[z].reshape(self.nfeatures, self.ngroups))\
            .sum(-1)

        ## do Metropolis step
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.phi[idxs] = phi_[idxs]

    # ##
    # def _update_mu(self, data):
    #
    #     ## propose
    #     mu_ = rn.lognormal(self.m, np.sqrt(self.v), self.nfeatures)
    #
    #     ## compute log-likelihoods
    #     loglik = self._compute_loglik(data, self.phi, self.mu, self.beta[self.z]).sum(-1)
    #     loglik_ = self._compute_loglik(data, self.phi, mu_, self.beta[self.z]).sum(-1)
    #
    #     ## do Metropolis step
    #     idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
    #     self.mu[idxs] = mu_[idxs]

    ##
    def _update_mu(self, data):

        ##
        counts, lib_sizes, nreplicas = data
        z = np.repeat(self.c, self.ngroups), np.tile(self.d, self.nfeatures)
        beta = np.repeat(self.beta[z].reshape(self.nfeatures, self.ngroups), nreplicas, axis=1)

        ##
        c1 = self.nsamples / self.phi
        c2 = (counts / lib_sizes / beta).sum(-1)

        ##
        p = rn.beta(1 + c1, 1 + c2)
        self.mu[:] = (1 - p) / p / self.phi

    ##
    def _update_c(self, data):
        """Propose matrix of indicators c and corresponding delta"""

        ##
        c_ = rn.choice(self.lw1.size, self.nfeatures, p=np.exp(self.lw1))


        ##
        z = np.repeat(self.c, self.ngroups), np.tile(self.d, self.nfeatures)
        loglik = self._compute_loglik(data, self.phi, self.mu, self.beta[z].reshape(self.nfeatures, self.ngroups))\
            .sum(-1)

        z_ = np.repeat(c_, self.ngroups), np.tile(self.d, self.nfeatures)
        loglik_ = self._compute_loglik(data, self.phi, self.mu, self.beta[z_].reshape(self.nfeatures, self.ngroups))\
            .sum(-1)

        ##
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.c[idxs] = c_[idxs]

    ##
    def _update_d(self, data):
        """Propose matrix of indicators c and corresponding delta"""

        ##
        d_ = rn.choice(self.lw2.size, self.ngroups, p=np.exp(self.lw2))

        ##
        z = np.repeat(self.c, self.ngroups), np.tile(self.d, self.nfeatures)
        loglik = self._compute_loglik(data, self.phi, self.mu, self.beta[z].reshape(self.nfeatures, self.ngroups))\
            .sum(0)

        z_ = np.repeat(self.c, self.ngroups), np.tile(d_, self.nfeatures)
        loglik_ = self._compute_loglik(data, self.phi, self.mu, self.beta[z_].reshape(self.nfeatures, self.ngroups))\
            .sum(0)

        ##
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.d[idxs] = d_[idxs]

    ##
    def _update_beta(self, data):

        ##
        beta_ = rn.lognormal(self.m0, np.sqrt(self.v0), self.ntrunc)

        ##
        z = np.repeat(self.c, self.ngroups), np.tile(self.d, self.nfeatures)
        loglik = self._compute_loglik(data, self.phi, self.mu, self.beta[z].reshape(self.nfeatures, self.ngroups))
        loglik_ = self._compute_loglik(data, self.phi, self.mu, beta_[z].reshape(self.nfeatures, self.ngroups))

        _, _, nreplicas = data
        nreplicas = np.repeat(range(self.ngroups), nreplicas)
        loglik = np.asarray([loglik[:, nreplicas == i].sum(-1) for i in range(self.ngroups)]).T
        loglik_ = np.asarray([loglik_[:, nreplicas == i].sum(-1) for i in range(self.ngroups)]).T

        loglik = np.bincount(self.z.ravel(), loglik.ravel(), minlength=self.ntrunc)[self.iact]
        loglik_ = np.bincount(self.z.ravel(), loglik_.ravel(), minlength=self.ntrunc)[self.iact]

        ##
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.beta[self.iact][idxs] = beta_[self.iact][idxs]

        ##
        self.beta[~self.iact] = beta_[~self.iact]

    ##
    def _update_hpars(self):
        """Samples the mean and var of the log-normal from the posterior, given phi"""

        ## sample first group of hyper-parameters
        mu = np.log(self.mu)
        self.s = st.sample_exponential_scale(np.sum(self.phi), self.phi.size)
        self.m, self.v = st.sample_normal_mean_var_jeffreys(np.sum(mu), np.sum(mu**2), mu.size)

        ## sample second group of hyper-parameters
        if self.nact > 2:
            beta = np.log(self.beta[self.iact])
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
    def _compute_loglik(data, phi, mu, beta):
        """Computes the log-likelihood of each element of counts for each element of theta"""

        ##
        counts, lib_sizes, nreplicas = data

        ##
        phi = phi.reshape(-1, 1)
        mu = mu.reshape(-1, 1)
        beta = np.repeat(beta, nreplicas, axis=1)

        ##
        alpha = 1 / phi
        p = alpha / (alpha + lib_sizes * mu * beta)

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
        T = pars[:, [3]]
        s = pars[:, [4]]
        m = pars[:, [5]]
        v = pars[:, [6]]
        m0 = pars[:, [7]]
        v0 = pars[:, [8]]

        ## plot
        fig = pl.figure() if fig is None else fig
        pl.figure(fig.number)

        pl.subplot(2, 2, 1)
        pl.plot(t, nact)
        pl.grid()

        pl.subplot(2, 2, 2)
        pl.semilogy(t, np.c_[eta, T])
        pl.grid()

        pl.subplot(2, 2, 3)
        pl.plot(t, np.c_[s, m, v])
        pl.grid()

        pl.subplot(2, 2, 4)
        pl.plot(t, np.c_[m0, v0])
        pl.grid()

########################################################################################################################

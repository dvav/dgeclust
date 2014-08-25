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
    def __init__(self, data, ntrunc=1000, hpars=(-4, 10, -8, 10, 0, 1), lr=0.01):
        """Initializes model from raw data"""

        ## various parameters
        self.lr = lr     # learning rate for eta

        self.ngroups = len(data.groups)
        self.nfeatures = data.counts.shape[0]
        self.ntrunc = ntrunc

        ## basic model state
        self.t = 0

        self.m1, self.v1, self.m2, self.v2, self.m0, self.v0 = hpars                          # hyper-parameters

        self.pars = type(self)._sample_pars_prior(ntrunc, self.m1, self.v1, self.m2, self.v2)  # cluster centers
        self.eta = np.sqrt(self.nfeatures)                                                    # concentration parameter
        self.lw, _ = st.sample_stick(np.zeros(ntrunc), self.eta)                              # vector of log-weights
        self.c = rn.choice(ntrunc, self.nfeatures, p=np.exp(self.lw))                         # gene-specific indicators

        self.beta = np.ones((self.nfeatures, self.ngroups))                                  # matrix of fold-changes
        self.z = np.zeros((self.nfeatures, self.ngroups), dtype='int')                       # matrix of indicators
        self.zeta = np.sqrt(self.ngroups)

        ## derivative model state
        self.cocc = np.bincount(self.c, minlength=self.ntrunc)
        self.ciact = self.cocc > 0
        self.nact = np.sum(self.ciact)
        self.ntot = self.lw.size

        self.zocc = np.bincount(self.z.ravel(), minlength=self.ngroups)
        self.lu = np.log(self.zocc / np.sum(self.zocc))

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
        pars = np.hstack([self.t, self.ntot, self.nact, self.zeta, self.eta,
                          self.m1, self.v1, self.m2, self.v2, self.m0, self.v0, self.lu])
        with open(os.path.join(outdir, cfg.fnames['pars']), 'a') as f:
            np.savetxt(f, np.atleast_2d(pars), fmt='\t%d' * 3 + '\t%f' * (self.lu.size + 8),
                       delimiter='\t')

        ## save z
        path = os.path.join(outdir, cfg.fnames['z'])
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, str(self.t)), 'w') as f:
            np.savetxt(f, self.z, fmt='%d', delimiter='\t')

    ##
    def plot_fitted_model(self, sample, data, fig=None, xmin=-1, xmax=12, npoints=1000, nbins=100, epsilon=0.5,
                          plot_components=False):
        """Computes the fitted model"""

        ## fetch group
        group = [i for i, item in enumerate(data.groups.items()) if sample in item[1]][0]

        ## fetch clusters
        beta = self.beta[:, [group]]
        pars = self.pars[self.c]

        occ = np.bincount(self.c, minlength=self.ntrunc)

        ## fetch data
        counts = data.counts[sample].values.astype('float')
        counts[counts < 1] = epsilon
        counts = np.log(counts)

        lib_size = data.lib_sizes[sample].values.ravel()

        ## compute fitted model
        x = np.reshape(np.linspace(xmin, xmax, npoints), (-1, 1))
        xx = np.exp(x)
        loglik = type(self).compute_loglik((xx[:, :, np.newaxis], lib_size, 1, None), pars, beta).sum(-1)
        y = xx * np.exp(loglik)

        ## groups
        idxs = np.nonzero(occ)[0]
        yg = [np.sum(y[:, self.c == idx], 1) / self.nfeatures for idx in idxs]
        yg = np.asarray(yg).T

        ## plot
        fig = pl.figure() if fig is None else fig
        pl.figure(fig.number)

        pl.hist(counts, nbins, histtype='stepfilled', linewidth=0, normed=True, color='gray')
        if plot_components is True:
            pl.plot(x, yg, 'k')
        pl.plot(x, yg.sum(1), 'r')

        ## return
        return x, y

    ##
    def update(self, data):
        """Implements a single step of the blocked Gibbs sampler"""

        self.t += 1                     # update time

        self._update_lw()               # update log-weights
        self._update_pars(data)         # update phi and mu
        self._update_c(data)            # update c and derivative state

        self._update_z_beta(data)       # update z and beta simultaneously
        self._update_lu()

        self._update_eta()              # update eta
        self._update_zeta()             # update zeta

        self._update_hpars()            # update hyper-parameters

    ##
    def _update_c(self, data):
        """Samples feature-specific indicator variables"""

        ## propose c
        c_ = rn.choice(self.ntrunc, self.nfeatures, p=np.exp(self.lw))

        ## sample currently inactive theta, if necessary
        # iact = self.ciact
        # iact_ = np.bincount(c_, minlength=self.ntrunc) > 0
        # idxs = np.logical_and(iact != iact_, iact_)
        # self.phi[idxs] = type(self)._sample_phi_prior(np.sum(idxs), self.m1, self.v1)
        # self.mu[idxs] = type(self)._sample_mu_prior(np.sum(idxs), self.m2, self.v2)

        ## compute log-likelihoods
        loglik = type(self).compute_loglik(data, self.pars[self.c], self.beta).sum(-1)
        loglik_ = type(self).compute_loglik(data, self.pars[c_], self.beta).sum(-1)

        ## do Metropolis step
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.c[idxs] = c_[idxs]

        ## update derivative state
        self.cocc = np.bincount(self.c, minlength=self.ntrunc)
        self.ciact = self.cocc > 0                             # active clusters
        self.nact = np.sum(self.ciact)

    ##
    def _update_lw(self):
        ## update log-weights
        self.lw[:] = st.sample_stick(self.cocc, self.eta)[0]

        ## derivative state
        u = rn.rand(self.nfeatures) * np.exp(self.lw)[self.c]               # total sufficient clusters
        self.ntot = np.sum(np.any(np.exp(self.lw) > u.reshape(-1, 1), 0))

    ##
    def _update_pars(self, data):
        """Sample phi and mu from their posterior, using Metropolis"""

        iact = self.ciact
        idxs = np.nonzero(iact)[0]

        ## sample phi
        self.pars = type(self)._update_pars_aux(0, idxs, data, self.pars, self.c, self.beta, self.m1, self.v1, self.m2, self.v2)

        ## sample mu
        self.pars = type(self)._update_pars_aux(1, idxs, data, self.pars, self.c, self.beta, self.m1, self.v1, self.m2, self.v2)

        ## update inactive clusters
        self.pars[~iact] = type(self)._sample_pars_prior(np.sum(~iact), self.m1, self.v1, self.m2, self.v2)


        # ## propose
        # pars_ = np.copy(self.pars)
        # pars_[iact] = self.pars[iact] * np.exp(0.01 * rn.randn(self.nact, 2))
        #
        # ## compute log-likelihoods
        # loglik = type(self).compute_loglik(data, self.pars[self.c], self.beta).sum(-1)
        # loglik = np.bincount(self.c, loglik, self.ntrunc)[iact]
        #
        # loglik_ = type(self).compute_loglik(data, pars_[self.c], self.beta).sum(-1)
        # loglik_ = np.bincount(self.c, loglik_, self.ntrunc)[iact]
        #
        # ## compute log-priors
        # logprior = type(self)._compute_pars_logprior(self.pars[iact], self.m1, self.v1, self.m2, self.v2)
        # logprior_ = type(self)._compute_pars_logprior(pars_[iact], self.m1, self.v1, self.m2, self.v2)
        #
        # ## compute log-posteriors
        # logpost = loglik + logprior
        # logpost_ = loglik_ + logprior_
        #
        # ## do Metropolis step
        # idxs = np.logical_or(logpost_ > logpost, rn.rand(*logpost.shape) < np.exp(logpost_ - logpost))
        # self.pars[iact][idxs] = pars_[iact][idxs]

        ## finally, sample inactive clusters
        # self.pars[~iact] = type(self)._sample_pars_prior(np.sum(~iact), self.m1, self.v1, self.m2, self.v2)

    ##
    @staticmethod
    def _update_pars_aux(i, idxs, data, pars, c, beta, m1, v1, m2, v2):

        ## propose
        pars_ = np.copy(pars)
        pars_[idxs, i] = pars[idxs, i] * np.exp(0.01 * rn.randn(idxs.size))

        ## compute log-likelihoods
        loglik = NBinomModel.compute_loglik(data, pars[c], beta).sum(-1)
        loglik = np.bincount(c, loglik)[idxs]

        loglik_ = NBinomModel.compute_loglik(data, pars_[c], beta).sum(-1)
        loglik_ = np.bincount(c, loglik_)[idxs]

        ## compute log-priors
        logprior = NBinomModel._compute_pars_logprior(pars[idxs], m1, v1, m2, v2)
        logprior_ = NBinomModel._compute_pars_logprior(pars_[idxs], m1, v1, m2, v2)

        ## compute log-posteriors
        logpost = loglik + logprior
        logpost_ = loglik_ + logprior_

        ## do Metropolis step
        ii = np.logical_or(logpost_ > logpost, rn.rand(*logpost.shape) < np.exp(logpost_ - logpost))
        pars[idxs, i][ii] = pars_[idxs, i][ii]

        ##
        return pars

    ##
    def _update_z_beta(self, data):
        """Propose matrix of indicators c and corresponding delta"""

        ##
        z_ = type(self)._sample_z_prior(self.zeta, self.nfeatures, self.ngroups)
        beta_ = type(self)._sample_beta_prior(z_, self.m0, self.v0)

        ##
        loglik = type(self).compute_loglik(data, self.pars[self.c], self.beta).sum(-1)
        loglik_ = type(self).compute_loglik(data, self.pars[self.c], beta_).sum(-1)

        # loglik = type(self).compute_loglik(data, self.phi[self.c], self.mu[self.c], self.beta)
        # loglik_ = type(self).compute_loglik(data, self.phi[self.c], self.mu[self.c], beta_)
        #
        # groups = np.repeat(range(self.ngroups), data[2])
        # groups = [np.nonzero(i == groups)[0] for i in range(self.ngroups)]
        #
        # loglik = np.hstack([loglik[:, group].sum(-1).reshape(-1, 1) for group in groups])
        # loglik_ = np.hstack([loglik_[:, group].sum(-1).reshape(-1, 1) for group in groups])

        ##
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.z[idxs] = z_[idxs]
        self.beta[idxs] = beta_[idxs]

        ##
        self.zocc = np.bincount(self.z.ravel(), minlength=self.ngroups)

    ##
    def _update_lu(self):
        self.lu = np.log(self.zocc / np.sum(self.zocc))

    ##
    def _update_eta(self):
        # st.sample_eta_ishwaran(np.sort(self.lw)[::-1])
        # self.eta = st.sample_eta_west(self.eta, self.nact, self.nfeatures)
        # self.eta = st.sample_eta(self.eta, self.nact, self.nfeatures)
        self.eta = self.lr / self.nfeatures + (1 - self.lr) * self.eta
        # lr = self.t**-self.lr
        # self.eta = lr * self.eta + (1 - lr) / self.nfeatures

    ##
    def _update_zeta(self):
        self.zeta = st.sample_eta_west(self.zeta, np.sum(self.zocc > 0), np.sum(self.zocc))
        # self.zeta = st.sample_eta_west(self.zeta, np.max(self.z) + 1, self.ngroups)
        # self.zeta = st.sample_eta(self.zeta, np.sum(self.zocc > 0), np.sum(self.zocc))
        # lr = self.t**-self.lr
        # self.zeta = lr * self.zeta + (1 - lr) / self.ngroups

    ##
    def _update_hpars(self):
        """Samples the mean and var of the log-normal from the posterior, given phi"""

        ## sample first group of hyper-parameters
        pars = self.pars[self.ciact]
        phi = np.log(pars[:, 0])
        mu = np.log(pars[:, 1])

        self.m1, self.v1 = st.sample_normal_mean_var_jeffreys(np.sum(phi), np.sum(phi**2), phi.size)
        self.m2, self.v2 = st.sample_normal_mean_var_jeffreys(np.sum(mu), np.sum(mu**2), mu.size)

        ## sample second group of hyper-parameters
        beta = self.beta[self.z > 0]
        dde = np.log(np.unique(beta))
        if dde.size > 1:
            self.m0, self.v0 = st.sample_normal_mean_var_jeffreys(np.sum(dde), np.sum(dde**2), dde.size)

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
    def _sample_z_prior(zeta, nfeatures, ngroups):
        """Propose c using Polya's urn scheme"""

        ##
        z = np.zeros((nfeatures, ngroups), dtype='int')

        ##
        w = np.r_[1, zeta]
        z[:, 1] = rn.choice(w.size, nfeatures, p=w / np.sum(w))

        ##  !!! DOUBLE  CHECK THIS !!!!
        if ngroups > 2:
            for i in range(2, ngroups):
                occ = [z[:, :i] == j for j in range(i+1)]
                occ = np.sum(occ, 2, dtype=np.float).T
                idxs = (range(nfeatures), np.max(z, 1) + 1)
                occ[idxs] = zeta
                w = occ / np.sum(occ, 1).reshape(-1, 1)
                z[:, i] = st.sample_categorical(w.T)

        ## return
        return z

    ##
    @staticmethod
    def _sample_beta_prior(z, m0, v0):
        """Sample delta, given c"""

        ##
        nfeatures, ngroups = z.shape
        rnds = np.c_[np.ones((nfeatures, 1)), rn.lognormal(m0, np.sqrt(v0), (nfeatures, ngroups-1))]
        beta = rnds[np.repeat(range(nfeatures), ngroups), z.ravel()].reshape(nfeatures, ngroups)
        # beta = np.transpose(np.choose(z.T, rnds.T))    # limited to 32 choices

        ##
        return beta

    ##
    @staticmethod
    def _sample_pars_prior(size, m1, v1, m2, v2):
        """Samples phi and mu from their priors, log-normal in both cases"""

        phi = rn.lognormal(m1, np.sqrt(v1), (size, 1))
        mu = rn.lognormal(m2, np.sqrt(v2), (size, 1))

        ##
        return np.c_[phi, mu]

    ##
    @staticmethod
    def compute_loglik(data, pars, beta):
        """Computes the log-likelihood of each element of counts for each element of theta"""

        ## read input
        counts, lib_sizes, nreplicas, _ = data

        ## fix delta, counts and lib_sizes
        beta = np.repeat(beta, nreplicas, axis=1)

        ## fix pars
        phi = pars[:, [0]]
        mu = pars[:, [1]]

        ## compute p
        alpha = 1 / phi
        p = alpha / (alpha + lib_sizes * mu * beta)

        ## return
        return st.nbinomln(counts, alpha, p)

    ##
    @staticmethod
    def _compute_pars_logprior(pars, m1, v1, m2, v2):
        """Computes the log-density of the prior of theta"""

        phi = pars[:, 0]
        mu = pars[:, 1]

        ## compute log-priors for phi and mu
        logprior_phi = st.lognormalln(phi, m1, v1)
        logprior_mu = st.lognormalln(mu, m2, v2)

        ## return
        return logprior_phi + logprior_mu

    ##
    @staticmethod
    def plot_progress(indir, fig=None):
        """Plot simulation progress"""

        ## load data
        pars = np.loadtxt(os.path.join(indir, cfg.fnames['pars']))
        t = pars[:, [0]]
        _ = pars[:, [1]]
        nact = pars[:, [2]]
        zeta = pars[:, [3]]
        eta = pars[:, [4]]
        m1 = pars[:, [5]]
        v1 = pars[:, [6]]
        m2 = pars[:, [7]]
        v2 = pars[:, [8]]
        m0 = pars[:, [9]]
        v0 = pars[:, [10]]
        lu = pars[:, 11:]

        ## plot
        fig = pl.figure() if fig is None else fig
        pl.figure(fig.number)

        pl.subplot(3, 2, 1)
        pl.plot(t, nact)
        pl.grid()

        pl.subplot(3, 2, 2)
        pl.plot(t, zeta)
        pl.grid()

        pl.subplot(3, 2, 3)
        pl.semilogy(t, eta)
        pl.grid()

        pl.subplot(3, 2, 4)
        pl.plot(t, np.c_[m1, v1, m2, v2])
        pl.grid()

        pl.subplot(3, 2, 5)
        pl.plot(t, np.c_[m0, v0])
        pl.grid()

        pl.subplot(3, 2, 6)
        pl.plot(t, np.exp(lu[:, 1:]))
        pl.grid()

########################################################################################################################

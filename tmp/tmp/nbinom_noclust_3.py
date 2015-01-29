from __future__ import division

import os
import pickle as pkl

import numpy as np
import numpy.random as rn
import matplotlib.pylab as pl

import dgeclust.utils as ut
import dgeclust.stats as st
import dgeclust.config as cfg

########################################################################################################################

_W_SIZE = 2


class NBinomModel(object):
    """Class representing a negative binomial model"""

    ## constructor
    def __init__(self, data, ntrunc=100, thr=0.):
        """Initializes model from raw data"""

        ## various parameters
        self.ngroups = len(data.groups)
        self.nfeatures, self.nsamples = data.counts.shape
        self.thr = thr

        dmean = np.mean(np.log(data.counts.values+1))     # add 1 to avoid infinities
        dvar = np.var(np.log(data.counts.values+1))
        self.mu1, self.tau1 = np.log(np.abs(dvar - dmean) / dmean**2), 1        # hyper-parameters
        self.mu2, self.tau2 = dmean, 1 / dvar
        self.t0 = 1
        self.m, self.t = 0, 10
        self.a, self.b = 1, 0.1

        ## time
        self.iter = 0

        ## phi, mu
        self.log_phi = rn.normal(self.mu1, 1/np.sqrt(self.tau1), self.nfeatures)
        self.log_mu = rn.normal(self.mu2, 1/np.sqrt(self.tau2), self.nfeatures)

        ## u
        self.u = np.ones((_W_SIZE, self.ngroups)) / _W_SIZE
        self.u[:, 0] = [1, 0]

        ## eta and lw
        self.eta = 1   # np.log(self.ntrunc)
        self.lw, _ = st.sample_stick(np.zeros(ntrunc), self.eta)

        ## means and precisions for beta
        self.beta_means = rn.normal(self.m, 1/np.sqrt(self.t), self.lw.size)
        self.beta_precs = rn.gamma(self.a, 1 / self.b, self.lw.size)

        ## c, z and beta
        self.c = st.sample_categorical(self.u, self.nfeatures)
        self.c[:, 0] = 0

        group0 = self.c == 0
        group1 = self.c == 1

        self.z = np.zeros((self.nfeatures, self.ngroups), dtype='int')
        self.z[group1] = rn.choice(self.lw.size, np.sum(group1), p=np.exp(self.lw))
        self.z[:, 0] = 0    # not necessary

        self.beta = np.zeros((self.nfeatures, self.ngroups))
        self.beta[group0] = rn.normal(0, 1 / np.sqrt(self.t0), np.sum(group0))
        self.beta[group1] = rn.normal(self.beta_means[self.z[group1]], 1/np.sqrt(self.beta_precs[self.z[group1]]))
        self.beta[:, 0] = 0

        ## threshold correction
        idxs = np.abs(self.beta) < self.thr
        self.c[idxs] = 0
        self.z[idxs] = 0

        ## occupancies
        self.cocc = ut.compute_occupancies(_W_SIZE, self.c)
        self.occ = np.bincount(self.z[self.c == 1], minlength=self.lw.size)
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
        u = np.sum(self.u[:, 1:], 1)
        u /= np.sum(u)
        pars = np.hstack([self.iter, self.nact, self.eta, self.mu1, self.tau1, self.mu2, self.tau2,
                          self.t0, self.m, self.t, self.a, self.b, u])
        with open(os.path.join(outdir, cfg.fnames['pars']), 'a') as f:
            np.savetxt(f, np.atleast_2d(pars), fmt='\t%d' * 2 + '\t%f' * 12)

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

        self._update_phi(data)
        self._update_mu(data)
        self._update_c_z_beta(data)
        self._update_pars()
        self._update_hpars()

        self.eta = st.sample_eta_west(self.eta, self.nact, self.occ.sum()) if self.nact > 0 else self.eta
        self.lw[:], _ = st.sample_stick(self.occ, self.eta)
        self.u[:] = st.sample_dirichlet(1 + self.cocc)

    ##
    def _update_phi(self, data):

        ## propose
        log_phi_ = rn.normal(self.mu1, 1/np.sqrt(self.tau1), self.nfeatures)

        ## compute log-likelihoods
        loglik = self.compute_loglik(data, self.log_phi, self.log_mu, self.beta).sum(-1)
        loglik_ = self.compute_loglik(data, log_phi_, self.log_mu, self.beta).sum(-1)

        ## do Metropolis step
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.log_phi[idxs] = log_phi_[idxs]

    ##
    def _update_mu(self, data):

        ## propose
        log_mu_ = rn.normal(self.mu2, 1/np.sqrt(self.tau2), self.nfeatures)

        ## compute log-likelihoods
        loglik = self.compute_loglik(data, self.log_phi, self.log_mu, self.beta).sum(-1)
        loglik_ = self.compute_loglik(data, self.log_phi, log_mu_, self.beta).sum(-1)

        ## do Metropolis step
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.log_mu[idxs] = log_mu_[idxs]

    ##
    def _update_c_z_beta(self, data):
        """Propose matrix of indicators c and corresponding delta"""

        ##
        c_ = st.sample_categorical(self.u, self.nfeatures)
        c_[:, 0] = 0   # not necessary, given IC

        group0 = self.c == 0
        group1 = self.c == 1

        ##
        z_ = np.zeros((self.nfeatures, self.ngroups), dtype='int')
        z_[group1] = rn.choice(self.lw.size, np.sum(group1), p=np.exp(self.lw))
        z_[:, 0] = 0     # not necessary

        ##
        beta_ = np.zeros((self.nfeatures, self.ngroups))
        beta_[group0] = rn.normal(0, 1 / np.sqrt(self.t0), np.sum(group0))
        beta_[group1] = rn.normal(self.beta_means[z_[group1]], 1/np.sqrt(self.beta_precs[z_[group1]]))
        beta_[:, 0] = 0

        ## threshold correction
        idxs = np.abs(beta_) < self.thr
        c_[idxs] = 0
        z_[idxs] = 0

        ##
        loglik = self.compute_loglik(data, self.log_phi, self.log_mu, self.beta)
        loglik_ = self.compute_loglik(data, self.log_phi, self.log_mu, beta_)

        _, _, nreplicas = data
        idxs = np.cumsum(nreplicas)[:-1]
        loglik = np.asarray([item.sum(-1) for item in np.hsplit(loglik, idxs)]).T
        loglik_ = np.asarray([item.sum(-1) for item in np.hsplit(loglik_, idxs)]).T

        ##
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.c[idxs] = c_[idxs]
        self.z[idxs] = z_[idxs]
        self.beta[idxs] = beta_[idxs]

        ## occupancies
        self.cocc = ut.compute_occupancies(_W_SIZE, self.c)
        self.occ = np.bincount(self.z[self.c == 1], minlength=self.lw.size)
        self.iact = self.occ > 0
        self.nact = np.sum(self.iact)

    ##
    def _update_pars(self):

        ##
        c = self.c[:, 1:]
        z = self.z[:, 1:]
        beta = self.beta[:, 1:]

        ##
        group0 = c == 0

        s1 = np.sum(beta[group0])
        s2 = np.sum(beta[group0]**2)
        n = np.sum(group0)
        self.t0 = st.sample_normal_prec_jeffreys(s1, s2, n)

        ##
        group1 = c == 1

        s1 = np.bincount(z[group1], beta[group1], minlength=self.lw.size)[self.iact]
        s2 = np.bincount(z[group1], beta[group1]**2, minlength=self.lw.size)[self.iact]
        n = self.occ[self.iact]

        self.beta_means[self.iact] = st.sample_normal_mean(s1, n, self.beta_precs[self.iact], self.m, self.t)
        self.beta_precs[self.iact] = st.sample_normal_prec(s1, s2, n, self.beta_means[self.iact], self.a, self.b)

        ##
        ninact = self.lw.size-self.nact
        self.beta_means[~self.iact] = rn.normal(self.m, 1/np.sqrt(self.t) + 1e-12, ninact)
        self.beta_precs[~self.iact] = rn.gamma(self.a, 1 / self.b + 1e-12, ninact)

    ##
    def _update_hpars(self):
        """Samples the mean and var of the log-normal from the posterior, given phi"""

        ##
        s1 = np.sum(self.log_phi)
        s2 = np.sum(self.log_phi**2)
        n = self.log_phi.size
        self.mu1, self.tau1 = st.sample_normal_mean_prec_jeffreys(s1, s2, n)

        s1 = np.sum(self.log_mu)
        s2 = np.sum(self.log_mu**2)
        n = self.log_mu.size
        self.mu2, self.tau2 = st.sample_normal_mean_prec_jeffreys(s1, s2, n)

        ## sample second group of hyper-parameters
        beta = self.beta[:, 1:][self.c[:, 1:] == 1]
        beta_mean = np.mean(beta)
        beta_var = np.var(beta)
        beta_prec = 1 / beta_var

        # self.m0 = beta_mean
        # self.t0 = beta_prec
        #
        # bprec = self.beta_prec[self.iact][1:]
        # sl = np.sum(np.log(bprec))
        # n = self.nact - 1
        # self.a0 = st.sample_gamma_shape(sl, n, self.a0, self.b0)
        # self.b0 = self.a0 * beta_var

        beta_means = self.beta_means[self.iact]
        beta_precs = self.beta_precs[self.iact]

        s1 = np.sum(beta_means)
        s2 = np.sum(beta_means**2)
        n = self.nact
        self.m = st.sample_normal_mean(s1, n, self.t, beta_mean, beta_prec)
        self.t = st.sample_normal_prec(s1, s2, n, self.m, 1, beta_var)

        s1 = np.sum(beta_precs)
        sl = np.sum(np.log(beta_precs))
        n = self.nact
        self.a = st.sample_gamma_shape(sl, n, self.a, self.b + 1e-12)
        w = rn.gamma(self.a * n + 1, 1 / (beta_prec + self.a * s1) + 1e-12)
        self.b = self.a * w

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

        itr = pars[:, [0]]
        nact = pars[:, [1]]
        eta = pars[:, [2]]
        mu1 = pars[:, [3]]
        tau1 = pars[:, [4]]
        mu2 = pars[:, [5]]
        tau2 = pars[:, [6]]
        t0 = pars[:, [7]]
        m = pars[:, [8]]
        t = pars[:, [9]]
        a = pars[:, [10]]
        b = pars[:, [11]]
        w = pars[:, 12:]

        ## plot
        fig = pl.figure() if fig is None else fig
        pl.figure(fig.number)

        pl.subplot(3, 2, 1)
        pl.plot(itr, nact)
        pl.grid()

        pl.subplot(3, 2, 2)
        pl.plot(itr, eta)
        pl.grid()

        pl.subplot(3, 2, 3)
        pl.plot(itr, np.c_[mu1, 1/tau1, mu2, 1/tau2])
        pl.grid()

        pl.subplot(3, 2, 4)
        pl.plot(itr, np.c_[1/t0, m, 1/t])
        pl.grid()

        pl.subplot(3, 2, 5)
        pl.plot(itr, np.c_[a, b])
        pl.grid()

        pl.subplot(3, 2, 6)
        pl.plot(itr, w)
        pl.grid()

########################################################################################################################

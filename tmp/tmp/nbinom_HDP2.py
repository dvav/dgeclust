from __future__ import division

import os
import pickle as pkl
import itertools as it
import multiprocessing as mp

import numpy as np
import numpy.random as rn
import matplotlib.pylab as pl

import dgeclust.stats as st
import dgeclust.config as cfg

########################################################################################################################


class NBinomModel(object):
    """Class representing a negative binomial model"""

    ## constructor
    def __init__(self, data, ntrunc=(100, 50), init_hpars=(1, 1), eta=None, thr=0.3, nthreads=None):
        """Initializes model from raw data"""

        ## various parameters
        self.ngroups = len(data.groups)
        self.nfeatures, self.nsamples = data.counts.shape
        self.thr = thr

        ## iterations
        self.iter = 0

        ## initial hyper-parameter values
        dmean = np.mean(np.log(data.counts.values+1))
        dvar = np.var(np.log(data.counts.values+1))

        self.mu1, self.tau1 = np.log(np.abs(dvar - dmean) / dmean**2), 1        # hyper-parameters
        self.mu2, self.tau2 = dmean, 1 / dvar

        self.r1, self.r2 = init_hpars

        ## initial log-values for phi and mu
        self.log_phi = rn.normal(self.mu1, 1/np.sqrt(self.tau1), self.nfeatures)
        self.log_mu = rn.normal(self.mu2, 1/np.sqrt(self.tau2), self.nfeatures)

        ## concentration parameters
        self.eta = np.log(ntrunc[0]) if eta is None else eta
        self.zeta = np.ones(self.ngroups)

        ## weights
        self.lw = np.tile(-np.log(ntrunc[0]), ntrunc[0])
        self.lu = np.tile(-np.log(ntrunc[1]), (self.ngroups, ntrunc[1]))

        ## initial cluster centers
        self.pb = np.r_[0.5, 0.5]
        self.zb = np.r_[0, rn.choice([1, 2], self.lw.size-1, p=self.pb)]
        up = self.zb == 1
        down = self.zb == 2
        self.beta = np.zeros(self.lw.size)
        self.beta[up] = self.thr + rn.exponential(1/self.r1, np.sum(up))
        self.beta[down] = - (self.thr + rn.exponential(1/self.r2, np.sum(down)))

        ## indicators
        self.c = rn.choice(self.lw.size, (self.ngroups, ntrunc[1]), p=np.exp(self.lw))
        self.d = np.asarray([rn.choice(lu.size, self.nfeatures, p=np.exp(lu)) for lu in self.lu])
        self.c[0, :] = 0
        self.d[0, :] = 0

        self.c[:, 0] = 0

        # self.c = np.zeros((self.ngroups, ntrunc[1]), dtype='int')
        # self.d = np.zeros((self.ngroups, self.nfeatures), dtype='int')

        self.z = np.asarray([c[d] for c, d in zip(self.c, self.d)])

        ## cluster info
        self.occ = np.bincount(self.z[1:].ravel(), minlength=self.lw.size)
        self.iact = self.occ > 0
        self.nact = np.sum(self.iact)

        ## multi-processing
        nthreads = mp.cpu_count() if nthreads is None else nthreads
        # self.pool = mp.Pool(nthreads)

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
        pars = np.hstack([self.iter, self.nact, self.mu1, self.tau1, self.mu2, self.tau2, self.r1, self.r2,
                          self.eta, self.pb])
        with open(os.path.join(outdir, cfg.fnames['pars']), 'a') as f:
            np.savetxt(f, np.atleast_2d(pars), fmt='\t%d' * 2 + '\t%f' * 9)

        ## save z
        fout = os.path.join(outdir, cfg.fnames['z'], str(self.iter))
        with open(fout, 'w') as f:
            np.savetxt(f, self.z.T, fmt='%d', delimiter='\t')

    ##
    def plot_fitted_model(self, sample, data, plot_components=True, fig=None,
                          xmin=-1, xmax=12, npoints=1000, nbins=100, epsilon=0.5):
        """Computes the fitted model"""

        ## fetch group
        group = [i for i, item in enumerate(data.groups.items()) if sample in item[1]][0]

        ## fetch clusters
        z = self.z[group]
        beta = self.beta[z].reshape(-1, 1)

        ## fetch data
        counts = data.counts[sample].values.astype('float')
        counts[counts < 1] = epsilon
        counts = np.log(counts)

        lib_size = data.lib_sizes[sample].values.ravel()

        ## compute fitted model
        x = np.reshape(np.linspace(xmin, xmax, npoints), (-1, 1))
        xx = np.exp(x)
        loglik = _compute_loglik((xx[:, :, np.newaxis], lib_size, 1), self.log_phi, self.log_mu, beta).squeeze()
        y = xx * np.exp(loglik) / self.nfeatures

        ## group
        # idxs = np.nonzero(self.iact)[0]
        # y = np.asarray([y[:, z == idx].sum(-1) for idx in idxs]).T

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

        ##
        self.iter += 1

        ##
        self._update_phi(data)
        self._update_mu(data)

        ##
        self._update_pb_zb_beta(data)

        ## update group-specific variables
        common_args = it.repeat((self.log_phi, self.log_mu, self.beta, self.lw))
        counts, lib_sizes, nreplicas = data
        idxs = np.cumsum(nreplicas)[:-1]
        counts = np.hsplit(counts, idxs)
        lib_sizes = np.hsplit(lib_sizes, idxs)
        args = zip(self.c[1:], self.d[1:], self.lu[1:], self.zeta[1:], counts[1:], lib_sizes[1:], common_args)

        self.c[1:], self.d[1:], self.z[1:], self.lu[1:], self.zeta[1:] = zip(*map(_update_group_vars, args))
        # self.c[1:], self.d[1:], self.z[1:], self.lu[1:], self.zeta[1:] = zip(*self.pool.map(_update_group_vars, args))

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
    def _update_phi(self, data):

        ## update phi
        log_phi_ = rn.normal(self.mu1, 1/np.sqrt(self.tau1), self.nfeatures)

        loglik = _compute_loglik(data, self.log_phi, self.log_mu, self.beta[self.z.T]).sum(-1)
        loglik_ = _compute_loglik(data, log_phi_, self.log_mu, self.beta[self.z.T]).sum(-1)

        idxs = np.logical_or(loglik_ >= loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.log_phi[idxs] = log_phi_[idxs]

    ##
    def _update_mu(self, data):

        ## update mu
        log_mu_ = rn.normal(self.mu2, 1/np.sqrt(self.tau2), self.nfeatures)

        loglik = _compute_loglik(data, self.log_phi, self.log_mu, self.beta[self.z.T]).sum(-1)
        loglik_ = _compute_loglik(data, self.log_phi, log_mu_, self.beta[self.z.T]).sum(-1)

        idxs = np.logical_or(loglik_ >= loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.log_mu[idxs] = log_mu_[idxs]

    # def _update_mu(self, data):
    #
    #     ##
    #     counts, lib_sizes, nreplicas = data
    #     beta = np.repeat(self.beta[self.z], nreplicas, axis=1)
    #
    #     ##
    #     c1 = self.nsamples / self.phi
    #     c2 = (counts / lib_sizes / beta).sum(-1)
    #
    #     ##
    #     p = rn.beta(1 + c1, 1 + c2)
    #     self.mu[:] = (1 - p) / p / self.phi

    ##
    def _update_pb_zb_beta(self, data):
        """Propose matrix of indicators c and corresponding delta"""

        ##
        zb_ = np.r_[0, rn.choice([1, 2], self.lw.size-1, p=self.pb)]
        up_ = zb_ == 1
        down_ = zb_ == 2
        beta_ = np.zeros(self.lw.size)
        beta_[up_] = self.thr + rn.exponential(1/self.r1, np.sum(up_))
        beta_[down_] = - (self.thr + rn.exponential(1/self.r2, np.sum(down_)))

        ##
        loglik = _compute_loglik(data, self.log_phi, self.log_mu, self.beta[self.z.T])
        loglik_ = _compute_loglik(data, self.log_phi, self.log_mu, beta_[self.z.T])

        _, _, nreplicas = data
        z = np.repeat(self.z.T, nreplicas, axis=1)
        loglik = np.bincount(z.ravel(), loglik.ravel(), minlength=self.lw.size)
        loglik_ = np.bincount(z.ravel(), loglik_.ravel(), minlength=self.lw.size)

        ##
        idxs = np.logical_or(loglik_ >= loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.zb[idxs] = zb_[idxs]
        self.beta[idxs] = beta_[idxs]

        ##
        self.zb[~self.iact] = zb_[~self.iact]
        self.beta[~self.iact] = beta_[~self.iact]

        ##
        up = self.zb[self.iact] == 1
        down = self.zb[self.iact] == 2
        occ = np.r_[np.sum(up), np.sum(down)]
        self.pb[:] = rn.dirichlet(1 + occ)

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
        zb = self.zb[self.iact]
        beta = self.beta[self.iact]

        up = zb == 1
        down = zb == 2
        dup = beta[up] - self.thr
        ddown = - beta[down] - self.thr
        nup = np.size(dup)
        ndown = np.size(ddown)

        self.r1 = st.sample_gamma_rate(np.sum(dup), nup, shape=1, a0=0, b0=0) if nup > 2 else self.r1
        self.r2 = st.sample_gamma_rate(np.sum(ddown), ndown, shape=1, a0=0, b0=0) if ndown > 2 else self.r2

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
    def plot_progress(indir, fig=None):
        """Plot simulation progress"""

        ## load data
        pars = np.loadtxt(os.path.join(indir, cfg.fnames['pars']))

        t = pars[:, [0]]
        nact = pars[:, [1]]
        mu1 = pars[:, [2]]
        tau1 = pars[:, [3]]
        mu2 = pars[:, [4]]
        tau2 = pars[:, [5]]
        r1 = pars[:, [6]]
        r2 = pars[:, [7]]
        eta = pars[:, [8]]
        pb = pars[:, [9, 10]]

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
        pl.plot(t, np.c_[r1, r2])
        pl.grid()

        pl.subplot(3, 2, 6)
        pl.plot(t, pb)
        pl.grid()

########################################################################################################################


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
    p = alpha / (alpha + lib_sizes*np.exp(log_mu + beta))

    ##
    return st.nbinomln(counts, alpha, p)

########################################################################################################################


def _update_group_vars(args):
    c, d, lu, zeta, counts, lib_sizes, (log_phi, log_mu, beta, lw) = args

    ##
    nfeatures, _ = counts.shape
    beta = beta.reshape(-1, 1)

    ## update d
    d_ = rn.choice(lu.size, nfeatures, p=np.exp(lu))

    loglik = _compute_loglik((counts, lib_sizes, 1), log_phi, log_mu, beta[c[d]]).sum(-1)
    loglik_ = _compute_loglik((counts, lib_sizes, 1), log_phi, log_mu, beta[c[d_]]).sum(-1)

    idxs = np.logical_or(loglik_ >= loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
    d[idxs] = d_[idxs]

    occ = np.bincount(d, minlength=lu.size)
    iact = occ > 0
    kact = np.sum(iact)

    ## update c
    c_ = rn.choice(lw.size, c.size, p=np.exp(lw))
    c_[0] = 0

    loglik = _compute_loglik((counts, lib_sizes, 1), log_phi, log_mu, beta[c[d]]).sum(-1)
    loglik_ = _compute_loglik((counts, lib_sizes, 1), log_phi, log_mu, beta[c_[d]]).sum(-1)

    loglik = np.bincount(d, loglik, minlength=lu.size)
    loglik_ = np.bincount(d, loglik_, minlength=lu.size)

    idxs = np.logical_or(loglik_ >= loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
    c[idxs] = c_[idxs]
    c[~iact] = c_[~iact]

    ## update zeta
    zeta = st.sample_eta_west(zeta, kact, occ.sum())

    ## update lu
    lu, _ = st.sample_stick(occ, zeta)

    ##
    return c, d, c[d], lu, zeta

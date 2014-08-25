from __future__ import division

import os

import numpy as np
import numpy.random as rn
import matplotlib.pylab as pl

import dgeclust.stats as st
import dgeclust.config as cfg
from dgeclust.models.nbinom import NBinomModel

########################################################################################################################


class NBinomModel4(NBinomModel):
    """Class representing a negative binomial model"""

    ## constructor
    def __init__(self, data, ntrunc=1000, phi=None, mu=None, lw=None, z=None, beta=None, c=None,
                 eta=None, zeta=1, hpars=(1, 0.1, 10, 0.1, 10, 0.1), lr=0.01, cr=0.01, t0=1, te=1):
        """Initializes model from raw data"""

        super(NBinomModel4, self).__init__(data, ntrunc, phi, mu, lw, z, beta, c, eta, zeta,
                                           (hpars[0], hpars[2], hpars[3]), lr, cr, t0, te)
        self.s, self.v0, self.m1, self.v1, self.m2, self.v2 = hpars       # vector of model hyper-parameters
        self.m, self.v = None, None
        self.x = rn.dirichlet([1, 1])

    ##
    def save(self, outdir):
        """Saves the state of the Gibbs sampler"""

        ## save state
        self.dump(os.path.join(outdir, cfg.fnames['state']))

        ## save chains
        pars = np.hstack([self.t, self.ntot, self.nact, self.zeta, self.eta, self.T, self.s, self.v0, self.m1, self.v1,
                          self.m2, self.v2, self.x, self.p])
        with open(os.path.join(outdir, cfg.fnames['pars']), 'a') as f:
            np.savetxt(f, np.atleast_2d(pars),
                       fmt='%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f' + '\t%f' * self.p.size,
                       delimiter='\t')

        ## save z
        path = os.path.join(outdir, cfg.fnames['z'])
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, str(self.t)), 'w') as f:
            np.savetxt(f, self.z, fmt='%d', delimiter='\t')

    ##
    def _update_hpars(self):
        """Samples the mean and var of the log-normal from the posterior, given phi"""

        ## sample first group of hyper-parameters
        phi = self.phi[self.ciact]
        self.s = st.sample_exponential_scale(np.sum(phi), self.nact)

        ## sample second group of hyper-parameters
        ee = np.logical_and(self.z == 0, self.beta != 1)
        dee = np.log(self.beta[ee])
        self.v0 = st.sample_normal_var_jeffreys(np.sum(dee), np.sum(dee**2), dee.size)

        up = np.logical_and(self.z > 0, self.beta > 1)
        dup = np.log(np.unique(self.beta[up]))
        if dup.size > 1:
            self.m1 = st.sample_gamma_shape(np.sum(np.log(dup)), dup.size, self.m1, self.v1)
            self.v1 = st.sample_gamma_scale(np.sum(dup), dup.size, self.m1)

        down = np.logical_and(self.z > 0, self.beta < 1)
        ddown = -np.log(np.unique(self.beta[down]))
        if ddown.size > 1:
            self.m2 = st.sample_gamma_shape(np.sum(np.log(ddown)), ddown.size, self.m2, self.v2)
            self.v2 = st.sample_gamma_scale(np.sum(ddown), ddown.size, self.m2)

    ##
    def _update_z_beta(self, data):
        """Propose matrix of indicators c and corresponding delta"""

        ##
        z_ = type(self)._sample_z_prior(self.zeta, self.nfeatures, self.ngroups)
        beta_ = type(self)._sample_beta_prior2(z_, self.x, self.v0, self.m1, self.v1, self.m2, self.v2)

        ##
        loglik = type(self).compute_loglik(data, self.phi[self.c], self.mu[self.c], self.beta).sum(-1)
        loglik_ = type(self).compute_loglik(data, self.phi[self.c], self.mu[self.c], beta_).sum(-1)

        ##
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.z[idxs] = z_[idxs]
        self.beta[idxs] = beta_[idxs]

        ##
        self.zocc = np.bincount(self.z.ravel(), minlength=self.ngroups)
        self.p = self.zocc / np.sum(self.zocc)

        up = np.logical_and(self.z > 0, self.beta > 1).sum()
        down = np.logical_and(self.z > 0, self.beta < 1).sum()
        self.x = rn.dirichlet(1 + np.r_[up, down])

    ##
    @staticmethod
    def _sample_beta_prior2(z, x, v0, m1, v1, m2, v2):
        """Sample delta, given c"""

        nfeatures, ngroups = z.shape

        ##
        beta = np.ones((nfeatures, ngroups))
        ee = z == 0
        beta[ee] = rn.lognormal(0, np.sqrt(v0), np.sum(ee))
        beta[:, 0] = 1

        ##
        for i in range(1, np.max(z)+1):
            ii = rn.choice([1, -1], (nfeatures, 1), p=x)
            up = np.logical_and(z == i, ii == 1)
            down = np.logical_and(z == i, ii == -1)

            ##
            rup = np.exp(rn.gamma(m1, v1, (nfeatures, 1)))
            rdown = np.exp(-rn.gamma(m2, v2, (nfeatures, 1)))
            beta[up] = np.tile(rup, (1, ngroups))[up]
            beta[down] = np.tile(rdown, (1, ngroups))[down]

        ##
        return beta

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
        _ = pars[:, [5]]
        s = pars[:, [6]]
        v0 = pars[:, [7]]
        m1 = pars[:, [8]]
        v1 = pars[:, [9]]
        m2 = pars[:, [10]]
        v2 = pars[:, [11]]
        x = pars[:, [12, 13]]
        p = pars[:, 14:]

        ## plot
        fig = pl.figure() if fig is None else fig
        pl.figure(fig.number)

        pl.subplot(3, 2, 1)
        pl.plot(t, nact)

        pl.subplot(3, 2, 2)
        pl.semilogy(t, np.c_[eta, zeta])

        pl.subplot(3, 2, 3)
        pl.plot(t, s)

        pl.subplot(3, 2, 4)
        pl.plot(t, np.c_[v0, m1 * v1, m2 * v2])

        pl.subplot(3, 2, 5)
        pl.plot(t, x[:, 0])

        pl.subplot(3, 2, 6)
        pl.plot(t, p[:, 1:])

########################################################################################################################

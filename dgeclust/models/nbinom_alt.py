from __future__ import division

import os

import numpy as np
import numpy.random as rn
import matplotlib.pylab as pl

import dgeclust.stats as st
import dgeclust.config as cfg

from dgeclust.models.nbinom import NBinomModel

########################################################################################################################


class NBinomModelAlt(NBinomModel):
    """Class representing a negative binomial model"""

    ## constructor
    def __init__(self, data, ntrunc=1000, hpars=(0, 1, 0, 1, 0, 1), lr=0.01, cr=0.01, t0=1, te=1):
        """Initializes model from raw data"""

        super(NBinomModelAlt, self).__init__(data, ntrunc, (hpars[01], hpars[3], hpars[4]), lr, cr, t0, te)
        self.m1, self.v1, self.m2, self.v2, self.m, self.v = hpars      # vector of model hyper-parameters (other than concentration parameters)

        self.phi = type(self)._sample_phi_prior2(ntrunc, self.m1, self.v1)                     # cluster centers
        self.mu = type(self)._sample_mu_prior2(ntrunc, self.m2, self.v2)            # cluster centers

    ##
    def save(self, outdir):
        """Saves the state of the Gibbs sampler"""

        ## save state
        self.dump(os.path.join(outdir, cfg.fnames['state']))

        ## save chains
        pars = np.hstack([self.t, self.ntot, self.nact, self.zeta, self.eta, self.T, self.m1, self.v1, self.m2, self.v2, self.m,
                          self.v, self.p])
        with open(os.path.join(outdir, cfg.fnames['pars']), 'a') as f:
            np.savetxt(f, np.atleast_2d(pars),
                       fmt='\t%d' * 3 + '\t%f' * (self.p.size + 9),
                       delimiter='\t')

        ## save z
        path = os.path.join(outdir, cfg.fnames['z'])
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.join(path, str(self.t)), 'w') as f:
            np.savetxt(f, self.z, fmt='%d', delimiter='\t')

    ##
    def update(self, data):
        """Implements a single step of the blocked Gibbs sampler"""

        self.t += 1                     # update time
        self._update_temperature()      # update temperature
        self._update_c(data)            # update c and cluster occupancy
        self._update_lw()               # update log-weights and ntot
        self._update_phi_global(data)  # update active and inactive theta
        # self._update_phi_local(data)    # update active and inactive theta
        self._update_mu(data)  # update active and inactive theta
        self._update_z_beta(data)       # update z and beta simultaneously; also the derivative state p
        self._update_eta()              # update eta
        self._update_zeta()             # update zeta
        self._update_hpars()            # update hyper-parameters

    ##
    def _update_pars_global(self, data):
        """Sample phi and mu from their posterior, using Metropolis"""

        iact = self.ciact

        ## propose
        phi_ = np.copy(self.phi)
        mu_ = np.copy(self.mu)
        phi_[iact] = type(self)._sample_phi_prior2(self.nact, self.m1, self.v1)
        mu_[iact] = type(self)._sample_mu_prior2(self.nact, self.m2, self.v2)

        ## compute log-likelihoods
        loglik = type(self).compute_loglik(data, self.phi[self.c], self.mu[self.c], self.beta).sum(-1)
        loglik = np.bincount(self.c, loglik, self.ntrunc)[iact]

        loglik_ = type(self).compute_loglik(data, phi_[self.c], mu_[self.c], self.beta).sum(-1)
        loglik_ = np.bincount(self.c, loglik_, self.ntrunc)[iact]

        ## do Metropolis step
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.phi[iact][idxs] = phi_[iact][idxs]
        self.mu[iact][idxs] = mu_[iact][idxs]

    ##
    def _update_hpars(self):
        """Samples the mean and var of the log-normal from the posterior, given phi"""

        ## sample first group of hyper-parameters
        phi = np.log(self.phi[self.ciact])
        self.m1, self.v1 = st.sample_normal_mean_var_jeffreys(np.sum(phi), np.sum(phi**2), self.nact)
        # self.s = np.sum(phi) / self.nact

        mu = np.log(self.mu[self.ciact])
        self.m2, self.v2 = st.sample_normal_mean_var_jeffreys(np.sum(mu), np.sum(mu**2), self.nact)

        ## sample second group of hyper-parameters
        dde = np.log(np.unique(self.beta[self.z > 0]))
        if dde.size > 1:
            self.m, self.v = st.sample_normal_mean_var_jeffreys(np.sum(dde), np.sum(dde**2), dde.size)
            # self.m = np.sum(dde) / dde.size
            # self.v = np.sum((dde - self.m)**2) / dde.size

    ##
    @staticmethod
    def _sample_phi_prior2(size, m, v):
        """Samples phi and mu from their priors, log-normal in both cases"""

        ## return
        return rn.lognormal(m, np.sqrt(v), size)

    ##
    @staticmethod
    def _sample_mu_prior2(size, m, v):
        """Samples phi and mu from their priors, log-normal in both cases"""

        ## return
        return rn.lognormal(m, np.sqrt(v), size)

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
        m1 = pars[:, [6]]
        v1 = pars[:, [7]]
        m2 = pars[:, [8]]
        v2 = pars[:, [9]]
        m = pars[:, [10]]
        v = pars[:, [11]]
        p = pars[:, 12:]

        ## plot
        fig = pl.figure() if fig is None else fig
        pl.figure(fig.number)

        pl.subplot(3, 2, 1)
        pl.plot(t, nact)

        pl.subplot(3, 2, 2)
        pl.plot(t, zeta)

        pl.subplot(3, 2, 3)
        pl.semilogy(t, eta)

        pl.subplot(3, 2, 4)
        pl.plot(t, np.c_[m1, v1, m2, v2])

        pl.subplot(3, 2, 5)
        pl.plot(t, np.c_[m, v])

        pl.subplot(3, 2, 6)
        pl.plot(t, p[:, 1:])

########################################################################################################################

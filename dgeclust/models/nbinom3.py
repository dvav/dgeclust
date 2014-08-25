from __future__ import division

import os

import numpy as np
import numpy.random as rn
import matplotlib.pylab as pl

import dgeclust.stats as st
import dgeclust.config as cfg

from dgeclust.models.nbinom import NBinomModel

########################################################################################################################


class NBinomModel3(NBinomModel):
    """Class representing a negative binomial model"""

    ## constructor
    def __init__(self, data, ntrunc=1000, hpars=(-4, 1, 0, 1), lr=0.01, cr=0.001, t0=1, te=1):
        """Initializes model from raw data"""

        super(NBinomModel3, self).__init__(data, ntrunc, hpars, lr, cr, t0, te)
        self.lu, _ = st.sample_stick(np.zeros(self.ngroups), self.zeta)

  ##
    def save(self, outdir):
        """Saves the state of the Gibbs sampler"""

        ## save state
        self.dump(os.path.join(outdir, cfg.fnames['state']))

        ## save chains
        pars = np.hstack([self.t, self.ntot, self.nact, self.zeta, self.eta, self.T, self.m1, self.v1,
                          self.m0, self.v0, self.lu])
        with open(os.path.join(outdir, cfg.fnames['pars']), 'a') as f:
            np.savetxt(f, np.atleast_2d(pars), fmt='\t%d' * 3 + '\t%f' * (self.lu.size + 7), delimiter='\t')

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
        self._update_phi_global(data)   # update active and inactive theta
        # self._update_phi_local(data)    # update active and inactive theta
        self._update_mu(data)           # update active and inactive theta
        self._update_z_beta(data)       # update z and beta simultaneously; also the derivative state zocc
        self._update_lu()               # update log-weights
        self._update_eta()              # update eta
        self._update_zeta()             # update zeta
        self._update_hpars()            # update hyper-parameters

    ##
    def _update_z_beta(self, data):
        """Propose matrix of indicators c and corresponding delta"""

        ##
        z_ = type(self)._sample_z_prior2(self.lu, self.nfeatures, self.ngroups)
        beta_ = type(self)._sample_beta_prior(z_, self.m0, self.v0)

        ##
        # loglik = type(self).compute_loglik(data, self.phi[self.c], self.mu[self.c], self.beta).sum(-1)
        # loglik_ = type(self).compute_loglik(data, self.phi[self.c], self.mu[self.c], beta_).sum(-1)

        loglik = type(self).compute_loglik(data, self.phi[self.c], self.mu[self.c], self.beta)
        loglik_ = type(self).compute_loglik(data, self.phi[self.c], self.mu[self.c], beta_)

        groups = np.repeat(range(self.ngroups), data[2])
        groups = [np.nonzero(i == groups)[0] for i in range(self.ngroups)]

        loglik = np.hstack([loglik[:, group].sum(-1).reshape(-1, 1) for group in groups])
        loglik_ = np.hstack([loglik_[:, group].sum(-1).reshape(-1, 1) for group in groups])

        ##
        idxs = np.logical_or(loglik_ > loglik, rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))
        self.z[idxs] = z_[idxs]
        self.beta[idxs] = beta_[idxs]

    ##
    def _update_lu(self):
        self.zocc = np.bincount(self.z.ravel(), minlength=self.ngroups)
        self.lu, _ = st.sample_stick(self.zocc, self.zeta)

    ##
    def _update_zeta(self):
        self.zeta = st.sample_eta_ishwaran(self.lu)

    ##
    @staticmethod
    def _sample_z_prior2(lu, nfeatures, ngroups):
        """Propose c using Polya's urn scheme"""

        ##
        z = rn.choice(lu.size, (nfeatures, ngroups), p=np.exp(lu))
        z[:, 0] = 0

        ## return
        return z

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
        m0 = pars[:, [8]]
        v0 = pars[:, [9]]
        lu = pars[:, 10:]

        ## plot
        fig = pl.figure() if fig is None else fig
        pl.figure(fig.number)

        pl.subplot(3, 2, 1)
        pl.plot(t, nact)

        pl.subplot(3, 2, 2)
        pl.semilogy(t, zeta)

        pl.subplot(3, 2, 3)
        pl.semilogy(t, eta)

        pl.subplot(3, 2, 4)
        pl.plot(t, np.c_[m1, v1])

        pl.subplot(3, 2, 5)
        pl.plot(t, np.c_[m0, v0])

        pl.subplot(3, 2, 6)
        pl.plot(t, np.exp(lu[:, 1:]))


########################################################################################################################

from __future__ import division

import os

import numpy as np
import numpy.random as rn
import matplotlib.pylab as pl

from dgeclust.models.nbinom import NBinomModel

########################################################################################################################


class NBinomModel2(NBinomModel):
    """Class representing a negative binomial model"""

    ## constructor
    def __init__(self, data, ntrunc=1000, hpars=(1, 0, 1), lr=0.01, cr=0.001, t0=1, te=1):
        """Initializes model from raw data"""

        super(NBinomModel2, self).__init__(data, ntrunc, hpars, lr, cr, t0, te)
        self.p = rn.dirichlet([1] * self.ngroups)

    ##
    def save(self, outdir):
        """Saves the state of the Gibbs sampler"""

        ## save state
        self.dump(os.path.join(outdir, 'state.pkl'))

        ## save chains
        pars = np.hstack([self.t, self.ntot, self.nact, self.eta, self.T, self.s, self.m, self.v, self.p])
        with open(os.path.join(outdir, 'pars.txt'), 'a') as f:
            np.savetxt(f, np.atleast_2d(pars),
                       fmt='%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f' + '\t%f' * self.p.size,
                       delimiter='\t')

        ## save z
        path = os.path.join(outdir, 'z')
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
        self._update_z_beta(data)       # update z and beta simultaneously; also the derivative state p
        self._update_eta()              # update eta
        self._update_p()                # update p
        self._update_hpars()            # update hyper-parameters

    ##
    def _update_z_beta(self, data):
        """Propose matrix of indicators c and corresponding delta"""

        ##
        z_ = type(self)._sample_z_prior(self.p, self.nfeatures, self.ngroups)
        beta_ = type(self)._sample_beta_prior(z_, self.m, self.v)

        ##
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
    def _update_p(self):
        occ = np.bincount(self.z[:, 1:].ravel(), minlength=self.ngroups)
        self.p[:] = rn.dirichlet(1 + occ)

    ##
    @staticmethod
    def _sample_z_prior(p, nfeatures, ngroups):
        """Propose c using Polya's urn scheme"""

        ##
        z = np.zeros((nfeatures, ngroups), dtype='int')
        z[:, 1:] = rn.choice(p.size, (nfeatures, ngroups-1), p=p)
        if ngroups > 2:
            idxs = np.all(z[:, [1]] == z[:, 2:])
            z[idxs, 1:] = 1

        ##
        return z

    ##
    @staticmethod
    def plot_progress(indir, fig=None):
        """Plot simulation progress"""

        ## load data
        pars = np.loadtxt(os.path.join(indir, 'pars.txt'))
        t = pars[:, [0]]
        _ = pars[:, [1]]
        nact = pars[:, [2]]
        eta = pars[:, [3]]
        _ = pars[:, [4]]
        s = pars[:, [5]]
        m = pars[:, [6]]
        v = pars[:, [7]]
        p = pars[:, 8:]

        ## plot
        fig = pl.figure() if fig is None else fig
        pl.figure(fig.number)

        pl.subplot(3, 2, 1)
        pl.plot(t, nact)

        pl.subplot(3, 2, 2)
        pl.semilogy(t, eta)

        pl.subplot(3, 2, 3)
        pl.plot(t, s)

        pl.subplot(3, 2, 4)
        pl.plot(t, np.c_[m, v])

        pl.subplot(3, 2, 5)
        pl.plot(t, p[:, 1:])

########################################################################################################################

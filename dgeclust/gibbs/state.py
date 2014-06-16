from __future__ import division

import numpy as np
import numpy.random as rn

import dgeclust.utils as ut

########################################################################################################################


class GibbsState(object):
    """Represents the state of the Gibbs sampler"""

    def __init__(self, pars, lw, p, z, d, eta, delta, hpars, t0):
        """Initializes state from raw data"""

        ## basic sampler state
        self.pars = pars        # model parameters
        self.lw = lw            # vector of global log-weights
        self.p = p
        self.z = z              # matrix of level 1 cluster indicators
        self.d = d
        self.delta = delta;
        self.eta = eta
        self.hpars = hpars      # vector of hyper-parameters
        self.t = t0             # the current iteration

        _, self.iact, self.nact, _ = ut.get_cluster_info(self.lw.size, self.d)

    ####################################################################################################################

    @classmethod
    def random(cls, nfeatures, ngroups, sample_pars_prior, hpars, nglobal):
        """Initialises state randomly"""

        pars = sample_pars_prior(nglobal, *hpars)
        lw = np.tile(-np.log(nglobal), nglobal)
        d = rn.randint(0, nglobal, nfeatures)
        eta = 1

        p = np.tile(1/2, 2)
        z = rn.choice(2, size=(nfeatures, ngroups), p = p); z[:, 0] = 0
        delta = np.ones((nfeatures, ngroups)); delta[:, 0] = 1
        t0 = 0

        ## return
        return cls(pars, lw, p, z, d, eta, delta, hpars, t0)

    ####################################################################################################################

    @classmethod
    def load(cls, fnames):
        """Initializes state from file"""

        pars = np.loadtxt(fnames['pars'])
        lw = np.loadtxt(fnames['lw'])
        lu = np.loadtxt(fnames['lu'])
        c = np.loadtxt(fnames['c'], dtype='uint32')
        z = np.loadtxt(fnames['z'], dtype='uint32')
        tmp = np.loadtxt(fnames['eta'])
        eta0 = tmp[-1, 1]
        eta = tmp[-1, 2:]
        tmp = np.loadtxt(fnames['hpars'])
        hpars = tmp[-1, 1:]
        t0 = int(tmp[-1, 0])             # the last iteration of the previous simulation
        loglik, logprior = np.loadtxt(fnames['lp'])[-1, [1, 2]]

        ## return
        return cls(pars, lw, lu, c, z, eta0, eta, hpars, loglik, logprior, t0)

    ####################################################################################################################

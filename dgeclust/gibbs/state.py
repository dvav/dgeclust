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
        self.delta = delta
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

        p = np.tile(1/ngroups, ngroups)
        z = rn.choice(ngroups, size=(nfeatures, ngroups), p = p)
        delta = np.ones((nfeatures, ngroups))
        t0 = 0

        ## return
        return cls(pars, lw, p, z, d, eta, delta, hpars, t0)

    ####################################################################################################################

    @classmethod
    def load(cls, fnames):
        """Initializes state from file"""

        pars = np.loadtxt(fnames['pars'])
        lw = np.loadtxt(fnames['lw'])
        delta = np.loadtxt(fnames['delta'])
        p = np.loadtxt(fnames['p'])
        d = np.loadtxt(fnames['d'], dtype='uint32')
        z = np.loadtxt(fnames['z'], dtype='uint32')
        eta = np.loadtxt(fnames['eta'])
        eta = eta[-1, 1]
        hpars = np.loadtxt(fnames['hpars'])
        hpars = hpars[-1, 1:]
        t0 = int(hpars[-1, 0])             # the last iteration of the previous simulation

        ## return
        return cls(pars, lw, p, z, d, eta, delta, hpars, t0)

    ####################################################################################################################

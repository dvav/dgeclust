from __future__ import division

import numpy as np
import numpy.random as rn

import dgeclust.utils as ut

########################################################################################################################


class GibbsState(object):
    """Represents the state of the Gibbs sampler"""

    def __init__(self, pars, lw, lu, c, z, eta0, eta, hpars, t0):
        """Initializes state from raw data"""

        ## basic sampler state
        self.pars = pars        # model parameters
        self.lw = lw            # vector of global log-weights
        self.lu = lu            # matrix of local (i.e. class-specific) log-weights
        self.c = c              # matrix of level 0 cluster indicators
        self.z = z              # matrix of level 1 cluster indicators
        self.eta0 = eta0        # global density parameter
        self.eta = eta          # vector of local density parameters
        self.hpars = hpars      # vector of hyper-parameters
        self.t = t0             # the current iteration

        ## direct cluster indicators and number of active clusters
        self.zz = [c[z] for c, z in zip(self.c, self.z)]
        _, _, self.nact0, _ = ut.get_cluster_info(self.lw.size, np.asarray(self.zz).ravel())

        ## local number of active clusters
        _, _, self.nact, _ = zip(*[ut.get_cluster_info(lu.size, z) for z, lu in zip(self.z, self.lu)])

    ####################################################################################################################

    @classmethod
    def random(cls, ngroups, nfeatures, sample_prior, hpars, nglobal, nlocal):
        """Initialises state randomly"""

        pars = sample_prior(nglobal, *hpars)
        lw = np.tile(-np.log(nglobal), nglobal)
        lu = np.tile(-np.log(nlocal), (ngroups, nlocal))
        c = rn.randint(0, nglobal, (ngroups, nlocal))
        z = rn.randint(0, nlocal, (ngroups, nfeatures))
        eta0 = 1
        eta = np.ones(ngroups)
        t0 = 0

        ## return
        return cls(pars, lw, lu, c, z, eta0, eta, hpars, t0)

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

        ## return
        return cls(pars, lw, lu, c, z, eta0, eta, hpars, t0)

    ####################################################################################################################

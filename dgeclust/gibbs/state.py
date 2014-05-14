from __future__ import division

import numpy as np
import numpy.random as rn

import dgeclust.utils as ut

########################################################################################################################


class GibbsState:
    """Represents the state of the Gibbs sampler"""

    def __init__(self, theta, lw, lu, c, z, eta0, eta, pars, t0):
        """Initializes state from raw data"""

        ## basic sampler state
        self.theta = theta      # parameters phi and p
        self.lw = lw            # vector of global log-weights
        self.lu = lu            # matrix of local (i.e. class-specific) log-weights
        self.c = c              # matrix of level 0 cluster indicators
        self.z = z              # matrix of level 1 cluster indicators
        self.eta0 = eta0        # global density parameter
        self.eta = eta          # vector of local density parameters
        self.pars = pars        # vector of hyper-parameters
        self.t = t0             # the current iteration

        ## direct cluster indicators and cluster occupancies
        self.zz = [c[z] for c, z in zip(self.c, self.z)]
        self.cluster_occupancies, self.iactive, self.nactive, _ = ut.get_cluster_info(
            self.lw.size, np.asarray(self.zz).ravel())

    ####################################################################################################################

    @classmethod
    def random(cls, ngroups, nfeatures, sample_prior, pars, nglobal, nlocal):
        """Initialises state randomly"""

        theta = sample_prior(nglobal, *pars)
        lw = np.tile(-np.log(nglobal), nglobal)
        lu = np.tile(-np.log(nlocal), (ngroups, nlocal))
        c = rn.randint(0, nglobal, (ngroups, nlocal))
        z = rn.randint(0, nlocal, (ngroups, nfeatures))
        eta0 = 1
        eta = np.ones(ngroups)
        t0 = 0

        ## return
        return cls(theta, lw, lu, c, z, eta0, eta, pars, t0)

    ####################################################################################################################

    @classmethod
    def from_file(cls, fnames):
        """Initializes state from file"""

        theta = np.loadtxt(fnames['theta'])
        lw = np.loadtxt(fnames['lw'])
        lu = np.loadtxt(fnames['lu'])
        c = np.loadtxt(fnames['c'], dtype='uint32')
        z = np.loadtxt(fnames['z'], dtype='uint32')
        tmp = np.loadtxt(fnames['eta'])
        eta0 = tmp[-1, 1]
        eta = tmp[-1, 2:]
        tmp = np.loadtxt(fnames['pars'])
        pars = tmp[-1, 2:]
        t0 = int(tmp[-1, 0])             # the last iteration of the previous simulation

        ## return
        return cls(theta, lw, lu, c, z, eta0, eta, pars, t0)

    ####################################################################################################################

from __future__ import division

import os
import numpy as np

import dgeclust.config as cfg

########################################################################################################################


class GibbsOutput(object):
    """Represents the output of the Gibbs sampler"""

    def __init__(self, t, pars, eta0, eta, nactive0, nactive, theta, c, z, zz):
        """Initialise from raw data"""
        self.t = t
        self.pars = pars
        self.eta0 = eta0
        self.eta = eta
        self.nactive0 = nactive0
        self.nactive = nactive
        self.theta = theta
        self.c = c
        self.z = z
        self.zz = zz

    ####################################################################################################################

    @classmethod
    def load(cls, indir):
        """Reads the results of a previously executed simulation from the disk"""

        ## read parameters
        pars = np.loadtxt(os.path.join(indir, cfg.fnames['pars']))
        eta = np.loadtxt(os.path.join(indir, cfg.fnames['eta']))
        nactive = np.loadtxt(os.path.join(indir, cfg.fnames['nactive']), dtype='uint32')

        t = pars[:, 0]
        pars = pars[:, 1:]
        eta0 = eta[:, 1]
        eta = eta[:, 2:]
        nactive0 = nactive[:, 1]
        nactive = nactive[:, 2:]

        ## read cluster centers and cluster indicators
        theta = np.loadtxt(os.path.join(indir, cfg.fnames['theta']))
        c = np.loadtxt(os.path.join(indir, cfg.fnames['c']), dtype='uint32')
        z = np.loadtxt(os.path.join(indir, cfg.fnames['z']), dtype='uint32')
        zz = np.asarray([ci[zi] for ci, zi in zip(c, z)])

        ## return
        return cls(t, pars, eta0, eta, nactive0, nactive, theta, c, z, zz)

########################################################################################################################

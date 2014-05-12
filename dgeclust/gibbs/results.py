from __future__ import division

import os
import numpy as np

import dgeclust.config as cfg

########################################################################################################################


class GibbsOutput:
    """Represents the output of the Gibbs sampler"""

    def __init__(self, pars, eta, theta, c, z, zz):
        """Initialise from raw data"""
        self.pars = pars
        self.eta = eta
        self.theta = theta
        self.c = c
        self.z = z
        self.zz = zz

    ####################################################################################################################

    @classmethod
    def from_file(cls, indir):
        """Reads the results of a previously executed simulation from the disk"""

        ## read parameters
        pars = np.loadtxt(os.path.join(indir, cfg.fnames['pars']))
        eta = np.loadtxt(os.path.join(indir, cfg.fnames['eta']))

        ## read cluster centers and cluster indicators
        theta = np.loadtxt(os.path.join(indir, cfg.fnames['theta']))
        c = np.loadtxt(os.path.join(indir, cfg.fnames['c']), dtype='uint32')
        z = np.loadtxt(os.path.join(indir, cfg.fnames['z']), dtype='uint32')
        zz = np.asarray([ci[zi] for ci, zi in zip(c, z)])

        ## return
        return cls(pars, eta, theta, c, z, zz)

########################################################################################################################

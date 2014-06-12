from __future__ import division

import os
import json
import collections as cl

import numpy as np
import pandas as pd

import dgeclust.config as cfg

########################################################################################################################


class GibbsOutput(object):
    """Represents the output of the Gibbs sampler"""

    def __init__(self, hpars, eta, nact, pars, lw, c, z, zz, lu, lp):
        """Initialise from raw data"""
        self.hpars = hpars
        self.eta = eta
        self.nact = nact
        self.pars = pars
        self.lw = lw
        self.c = c
        self.z = z
        self.zz = zz
        self.lu = lu
        self.lp = lp


    ####################################################################################################################

    @classmethod
    def load(cls, indir):
        """Reads the results of a previously executed simulation from the disk"""

        ## read output
        hpars = np.loadtxt(os.path.join(indir, cfg.fnames['hpars']))
        eta = np.loadtxt(os.path.join(indir, cfg.fnames['eta']))
        nact = np.loadtxt(os.path.join(indir, cfg.fnames['nact']), dtype='uint32')
        pars = np.loadtxt(os.path.join(indir, cfg.fnames['pars']))
        c = np.loadtxt(os.path.join(indir, cfg.fnames['c']), dtype='uint32')
        z = np.loadtxt(os.path.join(indir, cfg.fnames['z']), dtype='uint32')
        lw = np.loadtxt(os.path.join(indir, cfg.fnames['lw']))
        lu = np.loadtxt(os.path.join(indir, cfg.fnames['lu']))
        lp = np.loadtxt(os.path.join(indir, cfg.fnames['lp']))

        zz = np.asarray([ci[zi] for ci, zi in zip(c, z)])

        ## read config file
        with open(os.path.join(indir, cfg.fnames['config'])) as f:
            config = json.load(f, object_pairs_hook=cl.OrderedDict)

        pars_names = config['pars']
        hpar_names = config['hpars'].keys()
        group_names = config['groups'].keys()
        feature_names = config['featureNames']

        ## create data frames
        hpars = pd.DataFrame(hpars[:, 1:], index=hpars[:, 0], columns=hpar_names)
        eta = pd.DataFrame(eta[:, 1:], index=eta[:, 0], columns=['global']+group_names)
        nact = pd.DataFrame(nact[:, 1:], index=nact[:, 0], columns=['global']+group_names)
        pars = pd.DataFrame(pars, columns=pars_names)
        lw = pd.DataFrame(lw, columns=['global'])
        c = pd.DataFrame(c.T, columns=group_names)
        lu = pd.DataFrame(lu.T, columns=group_names)
        z = pd.DataFrame(z.T, index=feature_names, columns=group_names)
        zz = pd.DataFrame(zz.T, index=feature_names, columns=group_names)

        lp = pd.DataFrame(lp[:, 1:], index=lp[:, 0], columns=['loglik', 'logprior'])

        ## return
        return cls(hpars, eta, nact, pars, lw, c, z, zz, lu, lp)

########################################################################################################################

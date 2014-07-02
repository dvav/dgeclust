from __future__ import division

import os
import json
import collections as cl

import numpy as np
import pandas as pd

from dgeclust.gibbs.state import GibbsState
import dgeclust.config as cfg

########################################################################################################################


class GibbsOutput(object):
    """Represents the output of the Gibbs sampler"""

    def __init__(self, state, nclust, conc, x, p, hpars):
        """Initialise from raw data"""

        self.state = state
        self.nclust = nclust
        self.conc = conc
        self.x = x
        self.p = p
        self.hpars = hpars

    ####################################################################################################################

    @classmethod
    def load(cls, indir):
        """Reads the results of a previously executed simulation from the disk"""

        ## read state and pars
        state = GibbsState.load(os.path.join(indir, cfg.fnames['state']))
        pars = np.loadtxt(os.path.join(indir, cfg.fnames['pars']))

        ## read config file
        with open(os.path.join(indir, cfg.fnames['config'])) as f:
            config = json.load(f, object_pairs_hook=cl.OrderedDict)

        hpar_names = config['hpars'].keys()
        group_names = config['groups'].keys()

        ## create data frames
        nclust = pd.DataFrame(pars[:, [1, 2]], index=pars[:, 0], columns=['total', 'active'])
        conc = pd.DataFrame(pars[:, [3, 4]], index=pars[:, 0], columns=['zeta', 'eta'])
        x = pd.DataFrame(pars[:, [5, 6]], index=pars[:, 0], columns=['down-regulated', 'up-regulated'])
        p = pd.DataFrame(pars[:, 7:7+state.p.size], index=pars[:, 0], columns=group_names)
        hpars = pd.DataFrame(pars[:, 7+state.p.size:], index=pars[:, 0], columns=hpar_names)

        ## return
        return cls(state, nclust, conc, x, p, hpars)

########################################################################################################################

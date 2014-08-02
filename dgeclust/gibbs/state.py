from __future__ import division

import pickle as pkl

import numpy as np
import numpy.random as rn

import dgeclust.config as cfg
import dgeclust.utils as ut
import dgeclust.stats as st

########################################################################################################################


class GibbsState(object):
    """Represents the state of the Gibbs sampler"""

    def __init__(self, pars, lw, eta, z, delta, c, zeta, p, hpars, ntries, ltries, t0):
        """Initializes state from raw data"""

        ## basic sampler state
        self.pars = pars        # cluster centers
        self.lw = lw            # vector of log-weights
        self.eta = eta          # concentration parameter
        self.z = z              # vector of gene-specific indicator variables

        self.occ, self.iact, self.nact, _ = ut.get_cluster_info(self.lw.size, self.z)  # gene-wise cluster info
        self.ntot = lw.size

        self.delta = delta      # matrix of fold-changes
        self.c = c              # matrix of gene- and group-specific indicator variables
        self.zeta = zeta        # concentration parameter
        self.p = p              # gene- and group-specific relative occupancies
        self.hpars = hpars      # vector of model hyper-parameters (other than concentration parameters)
        self.ntries = ntries    # number of trials for the multipoint Metropolis
        self.ltries = ltries    # apply ntries for ltries iterations
        self.t = t0             # the current iteration

    ####################################################################################################################

    def save(self, fname=None):
        """Save the state of the Gibbs sampler"""

        fname = cfg.fnames['state'] if fname is None else fname

        with open(fname, 'wb') as f:
            pkl.dump(self, f)

    ####################################################################################################################

    @classmethod
    def random(cls, nfeatures, ngroups, sample_pars_prior, hpars, nclusters_max, ntries, ltries):
        """Initialises state randomly"""

        t0 = 0

        ##
        pars = sample_pars_prior(nclusters_max, hpars)
        eta = np.sqrt(nfeatures)
        lw, _ = st.sample_stick(np.zeros(nclusters_max), eta)       #np.tile(-np.log(nclusters_max), nclusters_max)
        z = rn.choice(nclusters_max, nfeatures, p=np.exp(lw))

        ##
        delta = np.ones((nfeatures, ngroups))
        c = np.zeros((nfeatures, ngroups), dtype='int')
        zeta = 1
        p = np.tile(1 / ngroups, ngroups)

        ## return
        return cls(pars, lw, eta, z, delta, c, zeta, p, hpars, ntries, ltries, t0)

    ####################################################################################################################

    @classmethod
    def load(cls, fname=None):
        """Initializes state from file"""

        fname = cfg.fnames['state'] if fname is None else fname

        with open(fname, 'rb') as f:
            state = pkl.load(f)

        ## return
        return state

    ####################################################################################################################

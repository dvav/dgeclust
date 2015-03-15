# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals
#
# from builtins import *

##

import os
import pickle as pkl
import abc

##

import dgeclust.config as cfg

########################################################################################################################


class Model(metaclass=abc.ABCMeta):
    """Abstract class representing a model"""

    # constructor
    def __init__(self, data, outdir=cfg.fnames['outdir']):
        # file names
        if os.path.exists(outdir):
            raise Exception("Directory '{}' already exists!".format(outdir))
        else:
            outdir = os.path.abspath(outdir)
            self.fnames = {
                'outdir': outdir,
                'state': os.path.join(outdir, cfg.fnames['state']),
                'pars': os.path.join(outdir, cfg.fnames['pars']),
                'z': os.path.join(outdir, cfg.fnames['z'])
            }
            os.makedirs(self.fnames['z'])

        # various parameters
        self.nfeatures, self.nsamples = data.counts.shape
        self.ngroups = len(data.groups)

        # current iteration
        self.iter = 0

    ##
    def dump(self, fname):
        """Save current model state"""

        with open(fname, 'wb') as f:
            pkl.dump(self, f)

    ##
    @staticmethod
    def load(indir):
        """Initializes model state from file"""

        # sanity check
        if not os.path.exists(indir):
            raise Exception("Directory '{}' does not exist!".format(indir))

        # load state
        indir = os.path.abspath(indir)
        with open(os.path.join(indir, cfg.fnames['state']), 'rb') as f:
            state = pkl.load(f)

        # correct, in case the original output dir was moved
        if indir != state.fnames['outdir']:
            print('Original output directory has been moved! Updating model state...')
            state.fnames = {
                'outdir': indir,
                'state': os.path.join(indir, cfg.fnames['state']),
                'pars': os.path.join(indir, cfg.fnames['pars']),
                'z': os.path.join(indir, cfg.fnames['z'])
            }

        # return
        return state

    ##
    @abc.abstractmethod
    def save(self):
        """Save current model state and state traces"""
        pass

    ##
    @abc.abstractmethod
    def plot_fitted_model(self, sample, data, fig=None, xmin=-1, xmax=12, npoints=1000, nbins=100, epsilon=0.25):
        """Plot fitted model"""
        pass

    ##
    @abc.abstractmethod
    def plot_clusters(self, fig=None, npoints=100):
        """Plot LFC clusters"""
        pass

    ##
    @abc.abstractmethod
    def plot_progress(self, fig=None):
        """Plot simulation progress"""
        pass

    ##
    @abc.abstractmethod
    def update(self, data, pool):
        """Implements a single step of the blocked Gibbs sampler"""
        pass

from __future__ import division

import os
import sys
import json
import collections as cl

import numpy as np
import IPython.lib.backgroundjobs as bgjobs

import dgeclust.config as cfg

########################################################################################################################


class SimulationManager(object):
    """Represents a blocked Gibbs sampler for HDPMMs"""

    def __init__(self):
        """Initialise sampler from raw data"""

        self.simulations = bgjobs.BackgroundJobManager()

    ##

    def new(self, data, model, niters=10000, outdir=cfg.fnames['clust'], extend=False, bg=False):
        """Spawn a new simulation"""

        ## prepare output directory
        if os.path.exists(outdir):
            if extend is False:
                raise Exception("Directory '{}' already exists!".format(outdir))
            else:
                print >> sys.stderr, "Extending previous simulation in directory {}".format(outdir)
        else:
            print >> sys.stderr, "Output directory is {}".format(outdir)
            os.makedirs(outdir)

        ## save groups, feature and sample names
        with open(os.path.join(outdir, cfg.fnames['config']), 'w') as f:
            json.dump(cl.OrderedDict([
                # ("data", args.data),
                # ("norm", args.norm),
                # ("ntrunc", model.ntrunc),
                # ("lr", model.lr),
                # ("cr", model.crate),
                # ("T0", args.T0),
                # ("Te", args.Te),
                # ("model", args.model),
                # ("pars", cfg.models['options'][args.model]['pars']),
                # ("hpars", cl.OrderedDict(zip(cfg.models['options'][args.model]['hpars'].keys(), args.hpars))),
                ("groups", data.groups),
                ("featureNames", data.counts.index.tolist())
            ]), f, indent=4, separators=(',', ':'))

        ## reformat data
        counts = np.hstack([data.counts[samples].values for samples in data.groups.values()])
        lib_sizes = np.hstack([data.lib_sizes[samples].values.ravel() for samples in data.groups.values()])
        nreplicas = data.nreplicas.values()
        norm_counts = counts / lib_sizes
        data = (counts, lib_sizes, nreplicas, norm_counts)

        ## start new job

        if bg is True:
            self.simulations.new(SimulationManager._run, data, model, niters, outdir)
        else:
            SimulationManager._run(data, model, niters, outdir)

    ##
    @staticmethod
    def _run(data, model, niters, outdir):
        """Executes simulation"""

        if model.t == 0:
            model.save(outdir)

        ## loop
        for t in range(model.t, model.t+niters):
            model.update(data)                      # update model state
            model.save(outdir)                      # save state

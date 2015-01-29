from __future__ import division

import os
import sys
import json
import collections as cl
import multiprocessing as mp

import IPython.lib.backgroundjobs as bgjobs

import config as cfg

########################################################################################################################


class SimulationManager(object):

    def __init__(self):

        self.simulations = bgjobs.BackgroundJobManager()

    ##

    def new(self, data, model, niters=10000, outdir=cfg.fnames['clust'], extend=False, bg=False, nthreads=None):

        ## prepare output directory
        if os.path.exists(outdir):
            if extend is False:
                raise Exception("Directory '{}' already exists!".format(outdir))
            else:
                print >> sys.stderr, "Extending previous simulation in directory {}".format(outdir)
        else:
            os.makedirs(os.path.join(outdir, cfg.fnames['z']))
            print >> sys.stderr, "Output directory is {}".format(outdir)

        ## save groups, feature and sample names
        with open(os.path.join(outdir, cfg.fnames['config']), 'w') as f:
            json.dump(cl.OrderedDict([
                ("groups", data.groups),
                ("geneNames", data.counts.index.tolist())
            ]), f, indent=4, separators=(',', ':'))

        ## multi-processing
        nthreads = mp.cpu_count() if nthreads is None or nthreads <= 0 else nthreads
        pool = None if nthreads == 1 else mp.Pool(nthreads)

        ## reformat data
        counts_norm = [data.counts_norm[samples].values for samples in data.groups.values()]
        nreplicas = data.nreplicas.values()
        data = counts_norm, nreplicas

        ## start new job
        if bg is True:
            self.simulations.new(_run, data, model, niters, outdir, pool)
        else:
            _run(data, model, niters, outdir, pool)


########################################################################################################################

def _run(data, model, niters, outdir, pool):

    ##
    if model.iter == 0:
        model.save(outdir)

    ## loop
    for t in range(model.iter, model.iter+niters):
        model.update(data, pool)                # update model state
        model.save(outdir)                      # save state




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
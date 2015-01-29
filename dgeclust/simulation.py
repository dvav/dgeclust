from __future__ import division

import models as mdls
import manager as mgr
import config as cfg

########################################################################################################################

_sim = mgr.SimulationManager()


def DGEclust(data, model='NegativeBinomial', niters=10000, outdir=cfg.fnames['clust'], extend=False, bg=False,
             nthreads=None):

    mdl = {
        'NegativeBinomial': mdls.NBinomModel
    }[model](data)
    _sim.new(data, mdl, niters, outdir, extend, bg, nthreads)

    ##
    return mdl, _sim

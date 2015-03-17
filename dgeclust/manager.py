# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals
#
# from builtins import *

##

import multiprocessing as mp
import IPython.lib.backgroundjobs as bgjobs

##

from dgeclust.models import NBinomModel
import dgeclust.config as cfg

########################################################################################################################

_jobs = bgjobs.BackgroundJobManager()


def run(data, model='NegativeBinomial', outdir=cfg.fnames['outdir'], extend=False, bg=True, niters=10000,
        nthreads=1, **args):

    # which model type to use
    model_type = {
        'NegativeBinomial': NBinomModel
    }[model]

    # where to save model state (create output dir, if necessary)
    if extend:
        state = model_type.load(outdir)
    else:
        state = model_type(data, outdir, **args)

    # multi-processing
    nthreads = mp.cpu_count() if nthreads is None or nthreads <= 0 else nthreads
    pool = None if nthreads == 1 else mp.Pool(nthreads)

    # reformat data
    counts_norm = [data.counts_norm[samples].values for samples in data.groups.values()]
    nreplicas = data.nreplicas.values()
    data = {'norm_counts': counts_norm, 'nreplicas': nreplicas}

    # start new job
    if bg is True:
        _jobs.new(_run, data, state, niters, pool)
    else:
        _run(data, model, niters, pool)

    ##
    return state


########################################################################################################################

def _run(data, state, niters, pool):

    # save initial state
    if state.iter == 0:
        state.save()

    # loop
    for t in range(state.iter, state.iter+niters):
        state.update(data, pool)                # update model state
        state.save()                            # save state

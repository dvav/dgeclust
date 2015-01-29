from __future__ import division

import multiprocessing as mp
import IPython.lib.backgroundjobs as bgjobs

########################################################################################################################


class SimulationManager(object):

    def __init__(self):

        self.simulations = bgjobs.BackgroundJobManager()

    ##

    def new(self, data, model, niters=10000, bg=True, nthreads=None):

        ## multi-processing
        nthreads = mp.cpu_count() if nthreads is None or nthreads <= 0 else nthreads
        pool = None if nthreads == 1 else mp.Pool(nthreads)

        ## reformat data
        counts_norm = [data.counts_norm[samples].values for samples in data.groups.values()]
        nreplicas = data.nreplicas.values()
        data = counts_norm, nreplicas

        ## start new job
        if bg is True:
            self.simulations.new(_run, data, model, niters, pool)
        else:
            _run(data, model, niters, pool)


########################################################################################################################

def _run(data, model, niters, pool):

    ##
    if model.iter == 0:
        model.save()

    ## loop
    for t in range(model.iter, model.iter+niters):
        model.update(data, pool)                # update model state
        model.save()                            # save state

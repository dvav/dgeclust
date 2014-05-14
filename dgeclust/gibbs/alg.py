from __future__ import division

import os
import itertools as it
import numpy as np

import dgeclust.utils as ut
import dgeclust.stats as st

########################################################################################################################


class GibbsSampler(object):
    """Represents a blocked Gibbs sampler for HDPMMs"""

    def __init__(self, data, model, state, niters, burnin, nlog, fnames, pool):
        """Initialise sampler from raw data"""

        self.data = data
        self.model = model
        self.state = state
        self.niters = niters
        self.burnin = burnin
        self.nlog = nlog
        self.fnames = fnames
        self.pool = pool

    ####################################################################################################################

    def run(self):
        """Executes simulation"""

        ## save initial conditions, if necessary
        if self.state.t == 0:
            self.to_disk()

        ## loop
        for t in range(self.state.t, self.niters):
            self.step()       # update state
            self.to_disk()    # save state

    ####################################################################################################################

    def step(self):
        """Implements a single step of the blocked Gibbs sampler"""

        data = self.data
        state = self.state
        model = self.model
        pool = self.pool

        ## update simulation time
        state.t += 1

        ## do local (i.e. sample-specific) updates
        args = zip(range(data.ngroups), it.repeat((data, state, model.compute_loglik)))
        state.lu, state.c, state.z, state.eta = zip(*pool.map(do_local_sampling, args))

        ## get top-level cluster info
        nglobal = state.lw.size
        state.zz = [c[z] for c, z in zip(state.c, state.z)]
        state.cluster_occupancies, state.iactive, state.nactive, _ = ut.get_cluster_info(
            nglobal, np.asarray(state.zz).ravel())
        idxs = state.iactive.nonzero()[0]

        ## sample lw and eta0
        state.lw, _ = st.sample_stick(state.cluster_occupancies, state.eta0)

        ## sample theta
        args = zip(idxs, it.repeat((data, state, model.sample_posterior)))
        state.theta[state.iactive] = pool.map(do_global_sampling, args)                         # active clusters
        state.theta[~state.iactive] = model.sample_prior(nglobal - state.nactive, *state.pars)  # inactive clusters

        ## update hyper-parameters
        state.eta0 = np.random.gamma(2, 1)   # st.sample_eta(state.lw)
        state.pars = model.sample_params(state.theta[state.cluster_occupancies > 0], *state.pars)

    ####################################################################################################################

    def to_disk(self):
        """Saves the state of the Gibbs sampler to disk"""

        state = self.state
        fnames = self.fnames

        ## write theta
        with open(fnames['theta'], 'w') as f:
            np.savetxt(f, state.theta, fmt='%f', delimiter='\t')

        ## write lw
        with open(fnames['lw'], 'w') as f:
            np.savetxt(f, state.lw, fmt='%f', delimiter='\t')

        ## write c
        with open(fnames['c'], 'w') as f:
            np.savetxt(f, state.c, fmt='%d', delimiter='\t')

        ## write z
        with open(fnames['z'], 'w') as f:
            np.savetxt(f, state.z, fmt='%d', delimiter='\t')

        ## write lu
        with open(fnames['lu'], 'w') as f:
            np.savetxt(f, state.lu, fmt='%f', delimiter='\t')

        ## write eta's
        with open(fnames['eta'], 'a') as f:
            np.savetxt(f, np.atleast_2d(np.r_[state.t, state.eta0, state.eta]),
                       fmt='%d\t%f' + '\t%f ' * np.size(state.eta))

        ## write pars
        with open(fnames['pars'], 'a') as f:
            np.savetxt(f, np.atleast_2d(np.r_[state.t, state.nactive, state.pars]),
                       fmt='%d\t%d' + '\t%f' * np.size(state.pars))

        ## write zz
        if (state.t > self.burnin) and (self.nlog > 0) and not (state.t % self.nlog):
            with open(os.path.join(fnames['zz'], str(state.t)), 'w') as f:
                np.savetxt(f, state.zz, fmt='%d', delimiter='\t')

########################################################################################################################


def do_global_sampling(args):
    """Samples the global cluster centers of the HDPMM"""

    ## read arguments
    idx, (data, state, sample_posterior) = args

    ## sample from the posterior
    theta = sample_posterior(idx, data, state)

    ## return
    return theta

########################################################################################################################


def do_local_sampling(args):
    """Samples the local variables of the HDPMM"""

    ## read arguments
    j, (data, state, compute_loglik) = args

    lu = state.lu[j]
    c = state.c[j]
    eta = state.eta[j]

    ## compute log-likelihood
    loglik = compute_loglik(j, data, state).sum(0)

    ## sample z
    logw = lu + loglik[:, c]
    logw = ut.normalize_log_weights(logw.T)
    z = st.sample_categorical(np.exp(logw))

    ## get local cluster info
    local_occupancies, iactive, nactive, occupancy_matrix = ut.get_cluster_info(lu.size, z)

    ## sample lu and eta
    lu, _ = st.sample_stick(local_occupancies, eta)
    eta = np.random.gamma(2, 1)  # st.sample_eta(lu)
    
    ## sample c
    logw = [(state.lw + loglik[zi].sum(0) if ia > 0 else state.lw) for zi, ia in zip(occupancy_matrix, iactive)]
    logw = np.asarray(logw)
    logw = ut.normalize_log_weights(logw.T)
    c = st.sample_categorical(np.exp(logw))
                                    
    ## return
    return lu, c, z, eta

########################################################################################################################

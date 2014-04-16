from __future__ import division

import os
import itertools as it
import numpy as np

import utils as ut
import stats as st

########################################################################################################################


def do_gibbs_sampling(data, model, state, pool, fnames, niters, burnin, nlog):
    """Implements a blocked Gibbs sampler for Hierarchical Dirichlet Process Mixture Models"""

    ## normalise and reformat data
    counts = (data.counts / data.norm_factors).T
    counts = [counts[group] for group in data.groups]

    ## save initial conditions, if necessary
    if state.t == 0:
        save_gibbs_state(state, fnames, burnin, nlog)

    ## loop
    for t in range(state.t+1, niters+1):
        do_gibbs_step(t, counts, model, state, pool)       # update state
        save_gibbs_state(state, fnames, burnin, nlog)      # save state

    ## return
    return state
    
########################################################################################################################


def do_gibbs_step(t, counts, model, state, pool):
    """Implements a single step of the blocked Gibbs sampler"""

    state.t = t

    ## do local (i.e. sample-specific) updates
    args = zip(state.lu, state.c, state.z, state.eta, counts, it.repeat((state, model.compute_loglik)))
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
    args = zip(state.theta[state.iactive], idxs, it.repeat((state, counts, model.sample_posterior)))
    state.theta[state.iactive] = pool.map(do_global_sampling, args)                          # sample active clusters
    state.theta[~state.iactive] = model.sample_prior(nglobal - state.nactive, *state.pars)   # sample inactive clusters

    ## update hyper-parameters
    state.eta0 = st.sample_eta(state.lw)
    state.pars = model.sample_params(state.theta[state.cluster_occupancies > 0], *state.pars)
            
    ## return
    return state
    
########################################################################################################################


def do_global_sampling(args):
    """Samples the global cluster centers of the HDPMM"""

    ## read arguments
    theta, idx, (state, counts, sample_posterior) = args

    ## sample from the posterior
    theta = sample_posterior(theta, idx, state.c, state.z, counts, *state.pars)

    ## return
    return theta
    
########################################################################################################################


def do_local_sampling(args):
    """Samples the local variables of the HDPMM"""

    lu, c, z, eta, counts, (state, compute_loglik) = args
     
    ## compute log-likelihood
    loglik = compute_loglik(counts, state.theta).sum(0)

    ## sample z
    logw = lu + loglik[:, c]
    logw = ut.normalize_log_weights(logw.T)
    z = st.sample_categorical(np.exp(logw))

    ## get local cluster info
    local_occupancies, iactive, nactive, occupancy_matrix = ut.get_cluster_info(lu.size, z)

    ## sample lu and eta
    lu, _ = st.sample_stick(local_occupancies, eta)
    eta = st.sample_eta(lu)
    
    ## sample c
    logw = [(state.lw + loglik[zi].sum(0) if ia > 0 else state.lw) for zi, ia in zip(occupancy_matrix, iactive)]
    logw = np.asarray(logw)
    logw = ut.normalize_log_weights(logw.T)
    c = st.sample_categorical(np.exp(logw))
                                    
    ## return
    return lu, c, z, eta

########################################################################################################################


def save_gibbs_state(state, fnames, burnin, nlog):
    """Saves the state of the Gibbs sampler to disk"""

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
    if (state.t > burnin) and (nlog > 0) and not (state.t % nlog):
        with open(os.path.join(fnames['zz'], str(state.t)), 'w') as f:
            np.savetxt(f, state.zz, fmt='%d', delimiter='\t')

########################################################################################################################

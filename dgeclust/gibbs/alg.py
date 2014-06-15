from __future__ import division

import os
import itertools as it
import numpy as np
import numpy.random as rn

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
            self.save()

        ## loop
        for t in range(self.state.t, self.niters):
            self.step()    # update state
            self.save()    # save state

    ####################################################################################################################

    def step(self):
        """Implements a single step of the blocked Gibbs sampler"""

        data = self.data
        state = self.state
        model = self.model
        pool = self.pool

        ## update simulation time
        state.t += 1

        ## sample d
        loglik = model.compute_loglik1(data, state)
        logw = state.lw + loglik
        logw = ut.normalize_log_weights(logw.T)
        state.d = st.sample_categorical(np.exp(logw))

        ## get cluster info
        occ, iact, state.nact, _ = ut.get_cluster_info(state.lw.size, state.d)
        idxs = iact.nonzero()[0]

        ## sample lw and eta
        state.lw, _ = st.sample_stick(occ, state.eta)
        # state.eta = st.sample_eta(state.lw)
        state.eta = st.sample_eta2(state.eta, state.nact, state.lw.size)

        ## sample pars
        args = zip(idxs, it.repeat((data, state, model.sample_posterior)))
        state.pars[iact] = zip(*pool.map(do_global_sampling, args))           # active clusters
        state.pars[~iact] = model.sample_pars_prior(state.lw.size - state.nact, *state.hpars)       # inactive clusters

        ## sample delta and z

        ## sample p
        occ, _, _, _ = ut.get_cluster_info(3, np.asarray(state.z).ravel())
        state.p = rn.dirichlet(1 + occ)

        ## update hyper-parameters
        state.hpars = model.sample_hpars(state.pars[iact], *state.hpars)

    ####################################################################################################################

    def save(self):
        """Saves the state of the Gibbs sampler to disk"""

        state = self.state
        fnames = self.fnames

        ## write pars
        with open(fnames['pars'], 'w') as f:
            np.savetxt(f, state.pars, fmt='%f', delimiter='\t')

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
                       fmt='%d\t%f' + '\t%f' * np.size(state.eta))

        ## write nact's
        with open(fnames['nact'], 'a') as f:
            np.savetxt(f, np.atleast_2d(np.r_[state.t, state.nact0, state.nact]),
                       fmt='%d\t%d' + '\t%d' * np.size(state.nact))

        ## write hpars
        with open(fnames['hpars'], 'a') as f:
            np.savetxt(f, np.atleast_2d(np.r_[state.t, state.hpars]),
                       fmt='%d' + '\t%f' * np.size(state.hpars))

        ## write log-likelihood and log-prior density
        with open(fnames['lp'], 'a') as f:
            np.savetxt(f, np.atleast_2d(np.r_[state.t, state.loglik, state.logprior]), fmt='%d\t%f\t%f')

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
    pars, loglik, logprior = sample_posterior(idx, data, state)

    ## return
    return pars, loglik, logprior

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

    ## apply correction
    groups = [np.nonzero(c == idx)[0] for idx in range(state.lw.size)]
    mats = [z == np.reshape(group, (-1, 1)) for group in groups]
    vecs = [np.any(mat, 0) for mat in mats]
    for vec, group in zip(vecs, groups):
        if np.any(group):
            z[vec] = group[0]

    ## get local cluster info
    occ, _, nact, occ_mat = ut.get_cluster_info(lu.size, z)

    ## sample lu and eta
    lu, _ = st.sample_stick(occ, eta)
    # eta = st.sample_eta(lu)
    eta = st.sample_eta2(eta, nact, lu.size)

    ## sample c
    logw = [state.lw + loglik[occ_vec].sum(0) for occ_vec in occ_mat]
    logw = np.asarray(logw)
    logw = ut.normalize_log_weights(logw.T)
    c = st.sample_categorical(np.exp(logw))

    ## return
    return lu, c, z, eta, nact

########################################################################################################################

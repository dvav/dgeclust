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
        occ, state.iact, state.nact, _ = ut.get_cluster_info(state.lw.size, state.d)
        idxs = state.iact.nonzero()[0]

        ## sample lw and eta
        state.lw, _ = st.sample_stick(occ, state.eta)
        # state.eta = st.sample_eta(state.lw)
        state.eta = st.sample_eta2(state.eta, state.nact, state.lw.size)

        ## sample pars
        args = zip(idxs, it.repeat((data, state, model.sample_posterior)))
        state.pars[state.iact] = pool.map(do_global_sampling, args)           # active clusters
        state.pars[~state.iact] = model.sample_pars_prior(state.lw.size - state.nact, *state.hpars)       # inactive clusters

        ## sample delta and z
        z_ = rn.choice(2, state.z.shape, p=state.p); z_[:, 0] = 0  # propose z
        delta_ = np.zeros(state.z.shape)    # propose delta
        ee = z_ == 0
        de = z_ == 1
        delta_[ee] = 1 #np.exp(rn.randn(np.sum(ee)) * state.hpars[4])   # no DE
        delta_[de] = np.exp(state.hpars[4] + rn.randn(np.sum(de)) * np.sqrt(state.hpars[5]))   # up-regulation
        delta_[:, 0] = 1
        loglik = model.compute_loglik2(data, state.delta, state)
        loglik_ = model.compute_loglik2(data, delta_, state)
        idxs = np.any(((loglik_ > loglik), (rn.rand(*state.z.shape) < np.exp(loglik_ - loglik))), 0)
        state.z[idxs] = z_[idxs]; state.z[:, 0] = 0
        state.delta[idxs] = delta_[idxs]; state.delta[:, 0] = 1

        ## sample p
        occ, _, _, _ = ut.get_cluster_info(2, np.asarray(state.z).ravel())
        state.p = rn.dirichlet(1 + occ)

        ## update hyper-parameters
        state.hpars = model.sample_hpars(state, *state.hpars)

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
        with open(fnames['p'], 'a') as f:
            np.savetxt(f, np.atleast_2d(np.r_[state.t, state.p]), fmt='%d' + '\t%f' * np.size(state.p), delimiter='\t')

        ## write z
        with open(fnames['z'], 'w') as f:
            np.savetxt(f, state.z, fmt='%d', delimiter='\t')

        ## write d
        with open(fnames['d'], 'w') as f:
            np.savetxt(f, state.d, fmt='%d', delimiter='\t')

        ## write log-likelihood and log-prior density
        with open(fnames['delta'], 'w') as f:
            np.savetxt(f, state.delta, fmt='%f')

        ## write eta's
        with open(fnames['eta'], 'a') as f:
            np.savetxt(f, np.atleast_2d(np.r_[state.t, state.eta]), fmt='%d\t%f')

        ## write nact's
        with open(fnames['nact'], 'a') as f:
            np.savetxt(f, np.atleast_2d(np.r_[state.t, state.nact]), fmt='%d\t%d')

        ## write hpars
        with open(fnames['hpars'], 'a') as f:
            np.savetxt(f, np.atleast_2d(np.r_[state.t, state.hpars]),
                       fmt='%d' + '\t%f' * np.size(state.hpars))

        ## write zz
        if (state.t > self.burnin) and (self.nlog > 0) and not (state.t % self.nlog):
            with open(os.path.join(fnames['zz'], str(state.t)), 'w') as f:
                np.savetxt(f, state.z, fmt='%d', delimiter='\t')

########################################################################################################################


def do_global_sampling(args):
    """Samples the global cluster centers of the HDPMM"""

    ## read arguments
    idx, (data, state, sample_posterior) = args

    ## sample from the posterior
    pars = sample_posterior(idx, data, state)

    ## return
    return pars

########################################################################################################################

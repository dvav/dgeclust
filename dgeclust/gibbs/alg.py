from __future__ import division

import os
import sys
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

        ## sample lw and u
        state.lw, _ = st.sample_stick(state.occ, state.eta)
        u = rn.rand(state.z.size) * np.exp(state.lw[state.z])

        ## sample pars
        idxs = state.iact.nonzero()[0]
        args = zip(idxs, it.repeat((data, state, model.sample_posterior)))
        state.pars[state.iact] = pool.map(_do_global_sampling, args)                                # active clusters
        state.pars[~state.iact] = model.sample_pars_prior(state.lw.size - state.nact, state.hpars)  # inactive clusters

        ## sample z
        state.z, ids = _sample_z(data, state, model, u)

        ## get cluster info
        state.occ, state.iact, state.nact, _ = ut.get_cluster_info(state.lw.size, state.z)
        state.ntot = np.sum(ids)

        ## update eta
        state.eta = max(1e-6, 0.986**state.t)
        # state.eta = st.sample_eta_west(state.eta, state.nact, state.z.size, b=state.z.size)
        # state.eta = st.sample_eta_ishwaran(state.lw[ids], b=state.z.size)

        ## sample c and delta
        state.c, state.delta = _sample_c_delta(data, state, model)

        ## compute p
        occ, _, _, _ = ut.get_cluster_info(state.c.shape[1], state.c.ravel())
        state.p = occ / np.sum(occ)

        ## update zeta
        # state.zeta = st.sample_eta_west(state.zeta, np.sum(occ > 0), np.sum(occ))

        ## compute x
        ndown = np.sum(np.logical_and(state.c > 0, state.delta < 1))
        nup = np.sum(np.logical_and(state.c > 0, state.delta > 1))
        occ = np.asarray([ndown, nup])

        # state.x = occ / np.sum(occ)
        state.x = rn.dirichlet(1 / occ.size + occ)

        ## sample hyper-parameters
        state.hpars = model.sample_hpars(state.pars[state.iact], state.c, state.delta, state.hpars)

    ####################################################################################################################

    def save(self):
        """Saves the state of the Gibbs sampler"""

        state = self.state
        fnames = self.fnames

        ## save state
        state.save(fnames['state'])

        ## save chains
        pars = np.hstack([state.t, state.ntot, state.nact, state.zeta, state.eta, state.x, state.p, state.hpars])
        with open(fnames['pars'], 'a') as f:
            np.savetxt(f, np.atleast_2d(pars),
                       fmt='%d\t%d\t%d\t%f\t%f\t%f\t%f' + '\t%f' * (state.p.size+state.hpars.size),
                       delimiter='\t')

        ## write cc
        if (state.t > self.burnin) and (self.nlog > 0) and not (state.t % self.nlog):
            with open(os.path.join(fnames['cc'], str(state.t)), 'w') as f:
                np.savetxt(f, state.c, fmt='%d', delimiter='\t')

########################################################################################################################


def _do_global_sampling(args):
    """Samples the gene-wise cluster centers of the DP"""

    ## read arguments
    idx, (data, state, sample_posterior) = args

    ## sample from the posterior
    pars = sample_posterior(idx, data, state)

    ## return
    return pars

########################################################################################################################


def _sample_z(data, state, model, u):
    """Samples gene-specific indicator variables"""

    ## fetch indices of sufficient clusters
    idxs = np.exp(state.lw) > u.reshape(-1, 1)
    idxs2 = np.any(idxs, 0)

    if state.t > 1 and np.sum(idxs2) == state.lw.size:
        print >> sys.stderr, 'Maximum number of clusters ({0}) is too low. If this message persists, try ' \
                             'increasing the value of parameter -k at the command line ...'.format(state.lw.size)

    ## compute log-likelihoods
    loglik = -np.ones((state.z.size, state.lw.size)) * np.inf
    loglik[:, idxs2] = model.compute_loglik(data, state.pars[idxs2][:, np.newaxis, :], state.delta).sum(-1).T

    ## compute log-weights
    logw = -np.ones(loglik.shape) * np.inf
    logw[idxs] = loglik[idxs]
    logw = ut.normalize_log_weights(logw.T)

    ## return z
    return st.sample_categorical(np.exp(logw)), idxs2


########################################################################################################################


def _sample_c_delta(data, state, model):
    """Propose matrix of indicators c and corresponding delta"""

    ##
    c, delta = state.c, state.delta

    ##
    c_ = _propose_c(state.zeta, *state.c.shape)
    delta_ = model.sample_delta_prior(c_, state.hpars)

    ##
    loglik = model.compute_loglik(data, state.pars[state.z], state.delta).sum(-1)
    loglik_ = model.compute_loglik(data, state.pars[state.z], delta_).sum(-1)

    ##
    idxs = np.any(((loglik_ > loglik), (rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))), 0)
    c[idxs] = c_[idxs]
    delta[idxs] = delta_[idxs]

    ## return
    return c, delta

########################################################################################################################


def _propose_c(zeta, nfeatures, ngroups):
    """Propose c using Polya urn scheme"""

    ##
    c = np.zeros((nfeatures, ngroups), dtype='int')

    ##
    w = np.asarray([1, zeta])
    c[:, 1] = rn.choice(w.size, nfeatures, p=w / np.sum(w))

    ##  !!! DOUBLE  CHECK THIS !!!!
    if ngroups > 2:
        for i in range(2, ngroups):
            occ = [c[:, :i] == j for j in range(i+1)]
            occ = np.sum(occ, 2, dtype='float').T
            idxs = (range(nfeatures), np.max(c, 1) + 1)
            occ[idxs] = zeta
            w = occ / np.sum(occ, 1).reshape(-1, 1)
            c[:, i] = st.sample_categorical(w.T)

    ## return
    return c

########################################################################################################################

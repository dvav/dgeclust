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
        u = rn.rand(state.d.size) * np.exp(state.lw[state.d])

        ## sample pars
        idxs = state.iact.nonzero()[0]
        args = zip(idxs, it.repeat((data, state, model.sample_posterior)))
        state.pars[state.iact] = pool.map(do_global_sampling, args)           # active clusters
        state.pars[~state.iact] = model.sample_pars_prior(state.lw.size - state.nact, *state.hpars)  # inactive clusters

        ## sample d
        idxs = np.exp(state.lw) > u.reshape(-1, 1)
        ids = np.any(idxs, 0)

        if np.sum(ids) == state.lw.size and state.t > 1:
            print >> sys.stderr, 'Truncation level too low. If this message starts appearing repeatedly, try ' \
                                 'increasing the value of parameter -k at the command line ...'


        tmp = state.pars
        state.pars = state.pars[ids]
        loglik = -np.ones((state.d.size, state.lw.size)) * np.inf
        loglik[:, ids] = model.compute_loglik1(data, state)
        state.pars = tmp

        logw = -np.ones((state.d.size, state.lw.size)) * np.inf
        logw[idxs] = loglik[idxs]
        logw = ut.normalize_log_weights(logw.T)
        state.d = st.sample_categorical(np.exp(logw))

        ## get cluster info
        state.occ, state.iact, state.nact, _ = ut.get_cluster_info(state.lw.size, state.d)

        ## update eta
        # state.eta = rn.gamma(1000, 1/1000)
        # state.eta = st.sample_eta(state.lw[ids], a=1, b=np.sum(ids))
        state.eta = 1 / np.sum(ids)
        # state.eta = st.sample_eta2(state.eta, state.nact, state.d.size, a=0, b=0)

        ## sample delta and z
        patt = np.asarray([[0, 0], [0, 1]])
        # patt = np.asarray([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 2]])
        # patt = np.asarray([
        #     [0, 0, 0, 0],
        #     [0, 0, 0, 1],
        #     [0, 0, 1, 0],
        #     [0, 0, 1, 1],
        #     [0, 0, 1, 2],
        #     [0, 1, 0, 0],
        #     [0, 1, 0, 1],
        #     [0, 1, 0, 2],
        #     [0, 1, 1, 0],
        #     [0, 1, 1, 1],
        #     [0, 1, 1, 2],
        #     [0, 1, 2, 0],
        #     [0, 1, 2, 1],
        #     [0, 1, 2, 2],
        #     [0, 1, 2, 3]
        # ])
        p = np.exp([np.log(state.p[el]).sum() for el in patt])
        zz = rn.choice(len(p), state.d.shape, p=p/np.sum(p))
        z_ = patt[zz]
        # z_ = rn.choice(state.p.size, state.z.shape, p=state.p)  # propose z
        # for i in range(state.p.size):    # correct nulls
        #     z_[np.all(z_ == i, 1), :] = 0
        # # z_[np.all(z_ == [1, 0], 1), :] = [0, 1]
        de = [z_ == i for i in range(state.p.size)]
        # null = np.ones((state.d.size, 1))
        delta_space = np.exp(rn.randn(*state.z.shape) * np.sqrt(state.hpars[4]))
        # delta_space = np.hstack(rnds)
        delta_ = np.zeros(state.delta.shape)      # propose delta
        for i, el in enumerate(de):
            if i == 0:
                delta_[el] = 1
            else:
                delta_[el] = np.tile(delta_space[:, [i]], (1, state.p.size))[el]
                # delta_[el] = np.exp(rn.randn(el.sum()) * np.sqrt(state.hpars[4]))
        loglik = model.compute_loglik2(data, state.delta, state)
        loglik_ = model.compute_loglik2(data, delta_, state)
        idxs = np.any(((loglik_ > loglik), (rn.rand(*loglik.shape) < np.exp(loglik_ - loglik))), 0)
        state.z[idxs] = z_[idxs]
        state.delta[idxs] = delta_[idxs]

        ## sample p
        occ, _, _, _ = ut.get_cluster_info(state.p.size, state.z.ravel())
        state.p = rn.dirichlet(1 / len(occ) + occ)

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
            np.savetxt(f, np.atleast_2d(np.r_[state.t, state.p]), fmt='%d' + '\t%f' * state.p.size, delimiter='\t')

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

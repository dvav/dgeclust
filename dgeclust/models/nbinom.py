from __future__ import division

import numpy as np
import numpy.random as rn
import itertools as it

import dgeclust.stats as st

########################################################################################################################


def _compute_loglik(counts, lib_sizes, phi, mu, delta):
    """Computes the log-likelihood of each element of counts for each element of phi and mu"""

    ## compute p
    alpha = 1 / phi
    p = alpha / (alpha + lib_sizes * mu * delta)

    ## return
    return st.nbinomln(counts, alpha, p)


########################################################################################################################

def compute_loglik(data, pars, delta):
    """Computes the log-likelihood of each element of counts for each element of theta"""

    ## read input
    counts, lib_sizes, nreplicas = data

    ## fix delta, counts and lib_sizes
    delta = np.repeat(delta, nreplicas, axis=1)

    ## fix pars
    phi = pars[..., [0]]
    mu = pars[..., [1]]

    ## return
    return _compute_loglik(counts, lib_sizes, phi, mu, delta)

########################################################################################################################


def compute_logprior(phi, mu, hpars):
    """Computes the log-density of the prior of theta"""

    m1, v1, m2, v2 = hpars[:4]

    ## compute log-priors for phi and mu
    logprior_phi = st.lognormalln(phi, m1, v1)
    logprior_mu = st.lognormalln(mu, m2, v2)

    ## return
    return logprior_phi + logprior_mu

########################################################################################################################


def sample_pars_prior(size, hpars):
    """Samples phi and mu from their priors, log-normal in both cases"""

    m1, v1, m2, v2 = hpars[:4]

    ## sample phi and mu
    phi = np.exp(rn.randn(size, 1) * np.sqrt(v1) + m1)
    mu = np.exp(rn.randn(size, 1) * np.sqrt(v2) + m2)

    ## return
    return np.hstack((phi, mu))
    
########################################################################################################################


def sample_delta_prior(c, hpars):
    """Sample delta, given c"""

    ##
    a0, s0 = hpars[4:]

    ##
    nfeatures, ngroups = c.shape

    ##
    delta = np.ones(c.shape)

    ##
    for i in range(1, np.max(c) + 1):
        de = c == i
        rnds = np.exp(a0 + rn.randn(nfeatures, 1) * np.sqrt(s0))
        delta[de] = np.tile(rnds, (1, ngroups))[de]

    ## return
    return delta

########################################################################################################################


def sample_hpars(pars, c, delta, hpars):
    """Samples the mean and var of the log-normal from the posterior, given phi"""

    ## read parameters
    phi = np.log(pars[:, 0])
    mu = np.log(pars[:, 1])

    ## sample first group of hyper-parameters
    ndata = len(pars)

    m1, v1 = st.sample_normal_mean_var_jeffreys(np.sum(phi), np.sum(phi**2), ndata)
    m2, v2 = st.sample_normal_mean_var_jeffreys(np.sum(mu), np.sum(mu**2), ndata)

    ## sample second group of hyper-parameters
    a0, s0 = hpars[4:]

    de = c > 0
    dde = np.log(delta[de])

    a0, s0 = st.sample_normal_mean_var_jeffreys(np.sum(dde), np.sum(dde**2), dde.size) if dde.size > 0 else (a0, s0)
    # s0 = st.sample_normal_var_jeffreys(np.sum(dde), np.sum(dde**2), dde.size) if dde.size > 0 else s0

    ## return
    return np.asarray([m1, v1, m2, v2, a0, s0])

########################################################################################################################


def sample_posterior(idxs, data, state, pool):
    """Sample phi and mu from their posterior, using Metropolis"""

    ## read input
    counts, lib_sizes, nreplicas = data

    ## fix delta
    delta = np.repeat(state.delta, nreplicas, axis=1)

    ## read and propose pars
    pars = state.pars[idxs]
    pars_ = pars * np.exp(0.01 * rn.randn(*pars.shape))

    ## compute log-likelihoods
    args = zip(idxs, pars, pars_, it.repeat((counts, lib_sizes, delta, state)))
    loglik, loglik_ = zip(*pool.map(_aux, args))
    # loglik = [_compute_loglik(counts[state.z == idx], lib_sizes, par[0], par[1], delta[state.z == idx]).sum()
    #           for par, idx in zip(pars, idxs)]
    # loglik_ = [_compute_loglik(counts[state.z == idx], lib_sizes, par[0], par[1], delta[state.z == idx]).sum()
    #            for par, idx in zip(pars_, idxs)]

    ## compute log-priors
    logprior = compute_logprior(pars[:, 0], pars[:, 1], state.hpars)
    logprior_ = compute_logprior(pars_[:, 0], pars_[:, 1], state.hpars)

    ## compute log-posteriors
    logpost = np.asarray(loglik) + logprior
    logpost_ = np.asarray(loglik_) + logprior_

    ## do Metropolis step
    idxs = np.any((logpost_ > logpost, rn.rand(*logpost.shape) < np.exp(logpost_ - logpost)))    # do Metropolis step
    pars[idxs] = pars_[idxs]

    ## return
    return pars

########################################################################################################################


def _aux(args):
    """Auxiliary function"""

    ## read args
    idx, par, par_, (counts, lib_sizes, delta, state) = args

    ## fetch counts and delta
    counts = counts[state.z == idx]
    delta = delta[state.z == idx]

    ## compute logliks
    loglik = _compute_loglik(counts, lib_sizes, par[0], par[1], delta).sum()
    loglik_ = _compute_loglik(counts, lib_sizes, par_[0], par_[1], delta).sum()

    ##
    return loglik, loglik_

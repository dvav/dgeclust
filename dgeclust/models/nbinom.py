from __future__ import division

import numpy as np
import numpy.random as rn

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
    phi = rn.lognormal(m1, np.sqrt(v1), (size, 1))
    mu = rn.lognormal(m2, np.sqrt(v2), (size, 1))

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
    rnds = np.c_[np.ones((nfeatures, 1)), rn.lognormal(a0, np.sqrt(s0), (nfeatures, ngroups-1))]
    delta = np.choose(c.T, rnds.T).T

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

    dde = np.log(np.unique(delta[c > 0]))

    a0, s0 = st.sample_normal_mean_var_jeffreys(np.sum(dde), np.sum(dde**2), dde.size) if dde.size > 0 else (a0, s0)
    # s0 = st.sample_normal_var_jeffreys(np.sum(dde), np.sum(dde**2), dde.size) if dde.size > 0 else s0

    ## return
    return np.asarray([m1, v1, m2, v2, a0, s0])

########################################################################################################################


def sample_posterior(idxs, data, state):
    """Sample phi and mu from their posterior, using Metropolis"""

    ## read input
    counts, lib_sizes, nreplicas = data

    ## fix delta
    delta = np.repeat(state.delta, nreplicas, axis=1)

    ## read and propose pars
    pars, z, hpars = state.pars, state.z, state.hpars

    ## sample phi
    pars = _sample_posterior_aux(0, idxs, counts, lib_sizes, pars, z, delta, hpars)

    ## sample mu
    pars = _sample_posterior_aux(1, idxs, counts, lib_sizes, pars, z, delta, hpars)

    ## return
    return pars[idxs]

########################################################################################################################


def _sample_posterior_aux(i, idxs, counts, lib_sizes, pars, z, delta, hpars):
    """Sample phi from its posterior, using Metropolis"""

    ## propose
    pars_ = np.copy(pars)
    pars_[idxs, i] = pars[idxs, i] * np.exp(0.01 * rn.randn(*pars[idxs, i].shape))

    ## compute log-likelihoods
    loglik = _compute_loglik(counts, lib_sizes, pars[z][:, [0]], pars[z][:, [1]], delta).sum(-1)
    loglik = np.bincount(z, loglik)[idxs]

    loglik_ = _compute_loglik(counts, lib_sizes, pars_[z][:, [0]], pars_[z][:, [1]], delta).sum(-1)
    loglik_ = np.bincount(z, loglik_)[idxs]

    ## compute log-priors
    logprior = compute_logprior(pars[idxs, 0], pars[idxs, 1], hpars)
    logprior_ = compute_logprior(pars_[idxs, 0], pars_[idxs, 1], hpars)

    ## compute log-posteriors
    logpost = loglik + logprior
    logpost_ = loglik_ + logprior_

    ## do Metropolis step
    ii = np.logical_or(logpost_ > logpost, rn.rand(*logpost.shape) < np.exp(logpost_ - logpost))
    pars[idxs, i][ii] = pars_[idxs, i][ii]

    ##
    return pars

from __future__ import division

import numpy as np
import numpy.random as rn

import dgeclust.stats as st

########################################################################################################################


def _compute_loglik(phi, mu, delta, counts, lib_sizes):
    """Computes the log-likelihood of each element of counts for each element of phi and mu"""

    ## prepare data
    counts = counts.T
    counts = counts[:, :, np.newaxis]

    lib_sizes = lib_sizes.T
    lib_sizes = lib_sizes[:, :, np.newaxis]

    delta = delta[:, np.newaxis]

    ## compute p
    alpha = 1 / phi
    p = alpha / (alpha + lib_sizes * mu * delta)

    ## return
    return st.nbinomln(counts, alpha, p)


########################################################################################################################

def compute_loglik1(data, state):
    """Computes the log-likelihood of each element of counts for each element of theta"""

    ## read data
    groups = data.groups.values()
    counts = [data.counts[group].values for group in groups]
    lib_sizes = [data.lib_sizes[group].values for group in groups]

    ## read state
    phi = state.pars[:, 0]
    mu = state.pars[:, 1]
    delta = state.delta.T

    loglik = [_compute_loglik(phi, mu, d, c, l).sum(0) for c, l, d in zip(counts, lib_sizes, delta)]

    ## return
    return np.sum(loglik, 0)

########################################################################################################################

def compute_loglik(j, data, state):
    """Computes the log-likelihood of each element of counts for each element of theta"""

    ## read data
    group = data.groups.values()[j]
    counts = data.counts[group].values
    lib_sizes = data.lib_sizes[group].values

    ## read theta
    phi = state.pars[:, 0]
    mu = state.pars[:, 1]

    ## return
    return _compute_loglik(phi, mu, counts, lib_sizes)

########################################################################################################################


def compute_logprior(phi, mu, m1, v1, m2, v2, *_):
    """Computes the log-density of the prior of theta"""

    ## compute log-priors for phi and mu
    logprior_phi = st.lognormalln(phi, m1, v1)
    logprior_mu = st.lognormalln(mu, m2, v2)

    ## return
    return logprior_phi + logprior_mu

########################################################################################################################


def sample_pars_prior(size, m1, v1, m2, v2, *_):
    """Samples phi and mu from their priors, log-normal in both cases"""

    ## sample phi and mu
    phi = np.exp(rn.randn(size, 1) * np.sqrt(v1) + m1)
    mu = np.exp(rn.randn(size, 1) * np.sqrt(v2) + m2)

    ## return
    return np.hstack((phi, mu))
    
########################################################################################################################


def sample_hpars(pars, *_):
    """Samples the mean and var of the log-normal from the posterior, given phi"""

    ## read parameters
    phi = np.log(pars[:, 0])
    mu = np.log(pars[:, 1])

    ## sample hyper-parameters
    ndata = len(pars)

    m1, v1 = st.sample_normal_mean_var_jeffreys(np.sum(phi), np.sum(phi**2), ndata)
    m2, v2 = st.sample_normal_mean_var_jeffreys(np.sum(mu), np.sum(mu**2), ndata)

    ## return
    return m1, v1, m2, v2, 0, 2, 0, 2

########################################################################################################################


def sample_posterior(idx, data, state):
    """Sample phi and mu from their posterior, using Metropolis"""

    ## fetch all data points that belong to cluster idx
    idxs = state.d == idx
    groups = data.groups.values()
    counts = data.counts[idxs]
    counts = [counts[group].values for group in groups]
    lib_sizes = [data.lib_sizes[group].values for group in groups]
    delta = state.delta[idxs].T

    ## read theta
    phi, mu = state.pars[idx]

    ## propose theta
    phi_, mu_ = (phi, mu) * np.exp(0.01 * rn.randn(2))

    ## compute log-likelihoods
    loglik = np.sum([_compute_loglik(phi, mu, d, cnts, lbsz).sum() for cnts, lbsz, d in zip(counts, lib_sizes, delta)])
    loglik_ = np.sum([_compute_loglik(phi_, mu_, d, cnts, lbsz).sum() for cnts, lbsz, d in zip(counts, lib_sizes, delta)])

    ## compute log-priors
    logprior = compute_logprior(phi, mu, *state.hpars)
    logprior_ = compute_logprior(phi_, mu_, *state.hpars)

    ## compute log-posteriors
    logpost = loglik + logprior
    logpost_ = loglik_ + logprior_

    ## do Metropolis step
    if (logpost_ > logpost) or (rn.rand() < np.exp(logpost_ - logpost)):    # do Metropolis step
        phi = phi_
        mu = mu_

    ## return
    return phi, mu

########################################################################################################################

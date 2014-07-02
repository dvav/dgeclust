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
    counts, lib_sizes = data

    ## fix delta, counts and lib_sizes
    rep = [el.size for el in lib_sizes]
    delta = np.repeat(delta, rep, axis=1)
    counts = np.hstack(counts)
    lib_sizes = np.hstack(lib_sizes)

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
    de = [c == i for i in range(1, np.max(c) + 1)]
    rnds = np.exp(a0 + rn.randn(nfeatures, len(de)) * np.sqrt(s0))
    for i, el in enumerate(de):
        delta[el] = np.tile(rnds[:, [i]], (1, ngroups))[el]

    ## return
    return delta

########################################################################################################################


def sample_hpars(pars, c, delta, _):
    """Samples the mean and var of the log-normal from the posterior, given phi"""

    ## read parameters
    phi = np.log(pars[:, 0])
    mu = np.log(pars[:, 1])

    ## sample first group of hyper-parameters
    ndata = len(pars)

    m1, v1 = st.sample_normal_mean_var_jeffreys(np.sum(phi), np.sum(phi**2), ndata)
    m2, v2 = st.sample_normal_mean_var_jeffreys(np.sum(mu), np.sum(mu**2), ndata)

    ## sample second group of hyper-parameters
    de = c > 0
    dde = np.log(delta[de])

    a0, s0 = st.sample_normal_mean_var_jeffreys(np.sum(dde), np.sum(dde**2), dde.size)

    ## return
    return np.asarray([m1, v1, m2, v2, a0, s0])

########################################################################################################################


def sample_posterior(idx, data, state):
    """Sample phi and mu from their posterior, using Metropolis"""

    ## read input
    counts, lib_sizes = data

    ## fix delta, counts and lib_sizes
    rep = [len(el) for el in lib_sizes]
    delta = np.repeat(state.delta, rep, axis=1)
    counts = np.hstack(counts)
    lib_sizes = np.hstack(lib_sizes)

    ## fetch all data points that belong to cluster idx
    idxs = state.z == idx
    counts = counts[idxs]
    delta = delta[idxs]

    ## read theta
    phi, mu = state.pars[idx]

    ## propose theta
    phi_, mu_ = (phi, mu) * np.exp(0.01 * rn.randn(2))

    ## compute log-likelihoods
    loglik = _compute_loglik(counts, lib_sizes, phi, mu, delta).sum()
    loglik_ = _compute_loglik(counts, lib_sizes, phi_, mu_, delta).sum()

    ## compute log-priors
    logprior = compute_logprior(phi, mu, state.hpars)
    logprior_ = compute_logprior(phi_, mu_, state.hpars)

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

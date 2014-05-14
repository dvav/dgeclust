from __future__ import division

import numpy as np
import numpy.random as rn

import dgeclust.stats as st

########################################################################################################################


def _compute_loglik(phi, mu, counts, norm_factors):
    """Computes the log-likelihood of each element of counts for each element of phi and mu"""

    ## prepare data
    counts = counts.T
    counts = np.atleast_2d(counts)
    counts = counts[:, :, np.newaxis]

    norm_factors = np.atleast_2d(norm_factors).T
    norm_factors = norm_factors[:, :, np.newaxis]

    ## compute p
    alpha = 1 / phi
    p = alpha / (alpha + norm_factors * mu)

    ## return
    return st.nbinomln(counts, alpha, p)


########################################################################################################################


def compute_loglik(j, data, state):
    """Computes the log-likelihood of each element of counts for each element of theta"""

    ## read data
    group = data.groups[j]
    counts = data.counts[:, group]
    norm_factors = data.norm_factors[group]

    ## read theta
    phi = state.theta[:, 0]
    mu = state.theta[:, 1]

    ## return
    return _compute_loglik(phi, mu, counts, norm_factors)

########################################################################################################################


def compute_logprior(phi, mu, mu_phi, s2_phi, mu_mu, s2_mu):
    """Computes the log-density of the prior of theta"""

    ## compute log-priors for phi and mu
    logprior_phi = st.lognormalln(phi, mu_phi, s2_phi)
    logprior_mu = st.lognormalln(mu, mu_mu, s2_mu)

    ## return
    return logprior_phi + logprior_mu

########################################################################################################################


def sample_prior(size, mu_phi, s2_phi, mu_mu, s2_mu):
    """Samples phi and mu from their priors, log-normal in both cases"""

    ## sample phi and mu
    phi = np.exp(rn.randn(size, 1) * np.sqrt(s2_phi) + mu_phi)
    mu = np.exp(rn.randn(size, 1) * np.sqrt(s2_mu) + mu_mu)

    ## return
    return np.hstack((phi, mu))
    
########################################################################################################################


def sample_params(theta, *args, **kargs):
    """Samples the mean and var of the log-normal from the posterior, given phi"""

    phi = np.log(theta[:, 0])
    mu = np.log(theta[:, 1])

    ## compute S1_phi, S2_phi, S1_mu, S2_mu and n
    n = phi.size

    s1_phi = phi.sum()
    s1_mu = mu.sum()

    s2_phi = np.sum(phi**2)
    s2_mu = np.sum(mu**2)

    mu_phi, s2_phi = st.sample_normal_meanvar(s1_phi, s2_phi, n)
    mu_mu, s2_mu = st.sample_normal_meanvar(s1_mu, s2_mu, n)

    ## return
    return mu_phi, s2_phi, mu_mu, s2_mu

########################################################################################################################


def sample_posterior(idx, data, state):
    """Sample phi and mu from their posterior, using Metropolis"""

    ## fetch all data points that belong to cluster idx
    counts = [data.counts[:, group][zz == idx] for group, zz in zip(data.groups, state.zz)]
    norm_factors = [data.norm_factors[group] for group in data.groups]

    ## read theta
    phi, mu = state.theta[idx]

    ## propose theta
    phi_, mu_ = (phi, mu) * np.exp(0.01 * rn.randn(2))

    ## compute log-likelihoods
    loglik = np.sum([_compute_loglik(phi, mu, cnts, fac).sum() for cnts, fac in zip(counts, norm_factors)])
    loglik_ = np.sum([_compute_loglik(phi_, mu_, cnts, fac).sum() for cnts, fac in zip(counts, norm_factors)])

    ## compute log-priors
    logprior = compute_logprior(phi, mu, *state.pars)
    logprior_ = compute_logprior(phi_, mu_, *state.pars)

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

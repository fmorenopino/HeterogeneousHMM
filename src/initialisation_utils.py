"""
Created on Feb 05, 2020

@author: esukei

For theoretical bases see:
 - K.P. Murphy, "Machine Learning: A Probabilistic Perspective", The MIT Press
   ©2012, ISBN:0262018020 9780262018029
"""

import numpy as np
from scipy.stats import norm
from scipy.stats import invgamma
from scipy.special import psi, polygamma

# ---------------------------------------------------------------------------- #
#       Scripts to initialise the parameters for the Gaussian HMM.             #
#                              Univariate case                                 #
# ---------------------------------------------------------------------------- #
def concatenate_observation_sequences(observation_sequences, gidx=None):
    """
        Function to concatenate the observation sequences and remove the
        partially or completely missing observations to create a proper
        input for the KMeans.
        Args:
            observation_sequences (list) - each element is an array of observations
        Returns:
            concatenated (list) - concatenated observations without missing values
    """
    concatenated = []
    for obs_seq in observation_sequences:
        for obs in obs_seq:
            if gidx is not None:
                gidx = int(gidx)
                if not np.any(obs[:gidx] is np.nan or obs[:gidx] != obs[:gidx]):
                    concatenated.append(obs[:gidx])
            else:
                if not np.any(obs is np.nan or obs != obs):
                    concatenated.append(obs)
    return np.asarray(concatenated, dtype=float)


def calc_posterior_distribution_params(observation_sequences):
    """
        Function  to compute the parameters of the posterior distribution of
        the overvations.
        Args:
            observation_sequences (array) - array in which the columns are assumed
                to be independent univariate Gaussian variables
    """
    dist_params = {}
    for idx in range(observation_sequences.shape[1]):
        column = observation_sequences[:, idx]
        dist_params[idx] = calc_posterior_nix_params(
            column[~(column is np.nan or column != column)]
        )
    return dist_params


def calc_posterior_nix_params(x, NIX_prior=(0, 0, -1, 0)):
    """
        Given a sequence of observations of a univariate Gaussian variable, the
        posterior NIX parameters are computed.
        Args:
            x (list) - observations of a Gaussian variable
            NIX_prior (tuple) - the prior parameters of the NIX distribution;
                default is uninformative, hence
                    mu_0 = 0, kappa_0 = 0, nu_0 = -1, Sigma_0 = 0
        Returns:
            (kappa_n, mu_n, nu_n, Sigma_n) - the parameters of a posterior NIX
    """
    # compute data related parameters
    n = x.shape[0]
    x_mean = np.mean(x)
    ssd = np.sum((x - x_mean) ** 2)

    # set the prior parameters -uninformative
    (mu_0, kappa_0, nu_0, Sigma_0) = NIX_prior

    # compute parameters of the posterior distribution
    kappa_n = kappa_0 + n
    mu_n = (kappa_0 * mu_0 + n * x_mean) / kappa_n
    nu_n = nu_0 + n
    Sigma_n = (1.0 / nu_n) * (
        (nu_0 * Sigma_0) + ssd + (((n * kappa_0) / kappa_n) * ((mu_0 - x_mean) ** 2))
    )

    return (kappa_n, mu_n, nu_n, Sigma_n)


def scaled_inverse_chi_squared_pdf(dof, scale, size):
    """
    Function to draw samples from a scaled inverse Chi-squared distribution.
    Source: https://github.com/probml/pyprobml/blob/master/scripts/nix_plots.py
    Args:
        dof (float) - degrees of freedom of the scaled inverse Chi-squared
            distribution
        scale (float) - scale parameter of the scaled inverse Chi-squared
            distribution
        size (int/tuple) - number of samples to draw
    Returns:
        An array of requested size, containing samples drawn from the scaled
        inverse Chi-squared distribution.
    """
    # The scaled inverse Chi-squared distribution with the provided params
    # is equal to an inverse-gamma distribution with the following parameters:
    ig_shape = dof / 2
    ig_scale = dof * scale / 2
    return invgamma.rvs(ig_shape, loc=0, scale=ig_scale, size=size)


def draw_nix_samples(kappa, mu, nu, Sigma, size=1):
    """
        Function to draw samples for the mean and variance from a NIX
        distribution.
    """
    # Source: https://github.com/probml/pyprobml/blob/master/scripts/nix_plots.py
    sigma2 = scaled_inverse_chi_squared_pdf(nu, Sigma, size)
    mu = norm.rvs(loc=mu, scale=np.sqrt(sigma2 / kappa), size=size)
    return (mu, sigma2)


def init_gaussian_hmm(n_states, n_features, X=None, dist_params=None):
    """
        Function to initialise the mean and covariance of the emission probabilities,
        using the NIX distribution for determining the conjugate prior.
        Args:
            n_states (int) - number of hidden states
            n_features (int) - number of features
            X (list, optional) - a list of observation sequences; if the distribution
                parameters were already computed on the observation sequences,
                it doesn't have to be given
            dist_params (dictionary, optional) - the computed distribution parameters
                computed from the observations
        Returns:
            mu_ (array) - [n_states x n_features] array of initial means of the
                different variables
            sigma_ (array) - covars of the different variables (n_states array of
                [n_features x n_features] covar matrices of type 'covariance_type')
            dist_params (dictionary) - the computed distribution parameters
                computed from the observations; only returned if X is not None
    """
    if X is not None:
        # compute distribution parameters for the
        dist_params = calc_posterior_distribution_params(X)

    mu_ = np.zeros((n_states, n_features))
    sigma_ = np.zeros((n_features, n_features))

    for feature_idx in range(n_features):
        for i in range(n_states):
            # sample mean and variance from an NIX distribution
            mu_[i][feature_idx], sigma_[feature_idx][feature_idx] = draw_nix_samples(
                *dist_params[feature_idx]
            )
    if X is not None:
        return mu_, sigma_, dist_params
    return mu_, sigma_

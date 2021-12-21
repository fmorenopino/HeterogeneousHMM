"""
Created on Nov 26, 2019
@author: semese

This code is based on:
 - HMM implementation by guyz- https://github.com/guyz/HMM
 - HMM implementation by fmorenopino - https://github.com/fmorenopino/HMM_eb2
 - hmmlearn by anntzer - https://github.com/hmmlearn/
 
For theoretical bases see:
 - L. R. Rabiner, A tutorial on hidden Markov models and selected applications
   in speech recognition, in Proceedings of the IEEE, vol. 77, no. 2,
   pp. 257-286, Feb. 1989.
 - K.P. Murphy, Machine Learning: A Probabilistic Perspective, The MIT Press
   ©2012, ISBN:0262018020 9780262018029
"""

import numpy as np
from scipy.stats import multivariate_normal
from sklearn import cluster

from .base import BaseHMM
from .utils import (
    init_covars, fill_covars, validate_covars, concatenate_observation_sequences
)

COVARIANCE_TYPES = frozenset(('spherical', 'tied', 'diagonal', 'full'))


class GaussianHMM(BaseHMM):
    """Hidden Markov Model with Gaussian emissions.

    :param n_states: number of hidden states in the model
    :type n_states: int, optional
    :param n_emissions: number of Gaussian features
    :type n_emissions: int, optional
    :param tr_params: controls which parameters are updated in the training process. Can contain any combination of 's' for starting probabilities (pi), 't' for transition matrix, 'm' for state means and 'c' for covariances. Defaults to all parameters.
    :type params: str, optional
    :param init_params: controls which parameters are initialised prior to training. Can contain any combination of 's' for starting probabilities (pi), 't' for transition matrix, 'm' for state means and 'c' for covariances. Defaults to all parameters.
    :type init_params: str, optional
    :param covariance_type: string describing the type of covariance parameters to use, defaults to 'full'.
    :type covariance_type: str, optional
    :param pi_prior: array of shape (n_states, ) setting the parameters of the Dirichlet prior distribution for the starting probabilities. Defaults to 1.
    :type pi_prior: np.array, optional 
    :param pi: array of shape (n_states, ) giving the initial state occupation distribution 'pi'
    :type pi: np.array 
    :param A_prior: array of shape (n_states, ), giving the parameters of the Dirichlet  prior distribution for each row of the transition probabilities 'A'. Defaults to 1.
    :type A_prior: np.array, optional 
    :param A: array of shape (n_states, n_states) giving the matrix of transition probabilities between states
    :type A: np.array
    :param means_prior: array of shape (n_states, 1), the mean of the Normal prior distribution for the means. Defaults to 0.
    :type means_prior: np.array, optional 
    :param means_weight: array of shape (n_states, 1), the precision of the Normal prior distribution for the means. Defaults to 0.
    :type means_weight: np.array, optional 
    :param means: array of shape (n_states, n_emissions) containing the mean parameters for each state
    :type means: np.array 
    :param covars_prior: array of shape (n_states, 1), the mean of the Normal prior distribution for the covariance matrix. Defaults to 0.
    :type covars_prior: np.array, optional 
    :param covars_weight: array of shape (n_states, 1), the precision of the Normal prior distribution for the covariance. Defaults to 0.
    :type covars_weight: np.array, optional 
    :param min_covar: floor on the diagonal of the covariance matrix to prevent overfitting. Defaults to 1e-3.
    :type min_covar: float, optional
    :param covars: covariance parameters for each state arranged in an array of shape depends `covariance_type`: (n_states, ) if 'spherical', (n_states, n_emissions) if 'diagonal', (n_states, n_emissions, n_emissions) if 'full', (n_emissions, n_emissions) if 'tied'
    :type covars: np.array 
    :param learning_rate: a value from [0,1), controlling how much the past values of the model parameters count when computing the new model parameters during training; defaults to 0.
    :type learning_rate: float, optional
    :param verbose: flag to be set to True if per-iteration convergence reports should be printed. Defaults to True.
    :type verbose: bool, optional 
    """

    def __init__(
        self,
        n_states=1,
        n_emissions=1,
        tr_params='stmc',
        init_params='stmc',
        covariance_type='diagonal',
        pi_prior=1.0,
        A_prior=1.0,
        means_prior=0,
        means_weight=0,
        covars_prior=1e-2,
        covars_weight=1,
        min_covar=1e-3,
        learning_rate=0.,
        verbose=False,
    ):
        if covariance_type not in COVARIANCE_TYPES:
            raise ValueError(
                'covariance_type must be one of {}'.format(COVARIANCE_TYPES)
            )

        super().__init__(
            n_states,
            tr_params=tr_params,
            init_params=init_params,
            init_type='kmeans',
            pi_prior=pi_prior,
            A_prior=A_prior,
            learning_rate=learning_rate,
            verbose=verbose,
        )

        self.n_emissions = n_emissions
        self.covariance_type = covariance_type
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight
        self.min_covar = min_covar

    def __str__(self):
        """Function to allow directly printing the object."""
        temp = super().__str__()
        return (
            temp
            + '\nMeans:\n'
            + str(self.means)
            + '\nCovariances:\n'
            + str(self.covars)
        )

    # ----------------------------------------------------------------------- #
    #        Public methods. These are callable when using the class.         #
    # ----------------------------------------------------------------------- #
    @property
    def covars(self):
        """Return covariances as a full matrix."""
        return fill_covars(
            self._covars, self.covariance_type, self.n_states, self.n_emissions
        )

    @covars.setter
    def covars(self, new_covars):
        """Setter function for the covariance matrices."""
        covars = np.array(new_covars, copy=True)
        validate_covars(covars, self.covariance_type, self.n_states)
        self._covars = covars

    def get_n_fit_scalars_per_param(self):
        """Return the number of trainable model parameters."""
        ns = self.n_states
        ne = self.n_emissions

        return {
            's': ns,
            't': ns * ns,
            'm': ns * ne,
            'c': {
                'spherical': ns,
                'diagonal': ns * ne,
                'full': ns * ne * (ne + 1) // 2,
                'tied': ne * (ne + 1) // 2,
            }[self.covariance_type],
        }

    # ----------------------------------------------------------------------- #
    #             Private methods. These are used internally only.            #
    # ----------------------------------------------------------------------- #
    def _init_model_params(self, X):
        """Initialises model parameters prior to fitting. Extends the base classes method by adding the emission probability matrix. See base.py for more information.

        :param X: list of observation sequences
        :type X: list
        """
        super()._init_model_params()

        X_concat = concatenate_observation_sequences(X)

        if 'm' in self.init_params:
            kmeans = cluster.KMeans(n_clusters=self.n_states)
            kmeans.fit(X_concat)
            self.means = kmeans.cluster_centers_
        if 'c' in self.init_params:
            cv = np.cov(X_concat.T) + self.min_covar * np.eye(self.n_emissions)
            self._covars = init_covars(cv, self.covariance_type, self.n_states)

    def _initialise_sufficient_statistics(self):
        """Initialises sufficient statistics required for M-step. Extends the base classes method by adding the emission probability matrix.See base.py for more information.
        """
        stats = super()._initialise_sufficient_statistics()

        stats['post'] = np.zeros(self.n_states)
        stats['obs'] = np.zeros((self.n_states, self.n_emissions))
        stats['obs**2'] = np.zeros((self.n_states, self.n_emissions))
        if self.covariance_type in ('tied', 'full'):
            stats['obs*obs.T'] = np.zeros(
                (self.n_states, self.n_emissions, self.n_emissions)
            )

        return stats

    def _accumulate_sufficient_statistics(
        self, stats, obs_stats, obs_seq
    ):
        """Updates sufficient statistics from a given sample.Extends the base classes method. See base.py for more information.
        """
        super()._accumulate_sufficient_statistics(
            stats, obs_stats
        )

        if 'm' in self.tr_params:
            stats['post'] += obs_stats['gamma'].sum(axis=0)
            stats['obs'] += self._reestimate_stat_obs(
                obs_stats['gamma'], obs_seq
            )

        if 'c' in self.tr_params:
            if self.covariance_type in ('spherical', 'diagonal'):
                stats['obs**2'] += self._reestimate_stat_obs2(
                    obs_stats['gamma'], obs_seq
                )
            elif self.covariance_type in ('tied', 'full'):
                stats['obs*obs.T'] += self._reestimate_stat_obs2(
                    obs_stats['gamma'], obs_seq
                )

    def _reestimate_stat_obs(self, gamma, obs_seq):
        """Helper method for the statistics accumulation. Computes the sum of
        the posteriors times the observations for the update of the means.

        :param gamma: array of shape (n_samples, n_states), the posteriors
        :type gamma: array_like
        :param obs_seq: an observation sequence
        :type obs_seq: array_like
        """
        stat_obs = np.zeros_like(self.means)
        for j in range(self.n_states):
            for t, obs in enumerate(obs_seq):
                if np.any(np.isnan(obs)):
                    # If there are missing observation we infer from the conditional posterior
                    stat_obs[j] += gamma[t][j] * self._infer_missing(obs, j)
                else:
                    # If there are no missing values we take this observation into account
                    stat_obs[j] += gamma[t][j] * obs
        return stat_obs

    def _reestimate_stat_obs2(self, gamma, obs_seq):
        """Helper method for the statistics accumulation. Computes the sum of
        the posteriors times the square of the observations for the update
        of the covariances.

        :param gamma: array of shape (n_samples, n_states), the posteriors
        :type gamma: array_like
        :param obs_seq: an observation sequence
        :type obs_seq: array_like
        """
        if self.covariance_type in ('tied', 'full'):
            stat_obs2 = np.zeros(
                (self.n_states, self.n_emissions, self.n_emissions))
        else:
            stat_obs2 = np.zeros((self.n_states, self.n_emissions))

        for j in range(self.n_states):
            for t, obs in enumerate(obs_seq):
                if np.any(np.isnan(obs)):
                    # If there are missing observation we infer from the conditional posterior
                    obs = self._infer_missing(obs, j)
                if self.covariance_type in ('tied', 'full'):
                    stat_obs2[j] += gamma[t][j] * np.outer(obs, obs)
                else:
                    stat_obs2[j] += np.dot(gamma[t][j], obs ** 2)
        return stat_obs2

    def _infer_missing(self, obs, state):
        """Helper method for the statistics accumulation. It infers the missing
        observation from the conditional posterior for a given state.

        :param obs: a single observation
        :type obs: array_like
        :param state: the index of the hidden state to consider
        :type state: int
        """
        # If all the features of the observations are missed, it is not necessary
        # to use this observation to update the means and covars. So we set them to 0.
        if np.all(np.isnan(obs)):
            return np.zeros_like(obs)
        # For partial observations we compute the conditional posterior.
        elif np.any(np.isnan(obs)) and not np.all(np.isnan(obs)):
            _, _, obs_vector = self._calc_conditional_posterior(obs, state)
            return obs_vector

    def _M_step(self, stats):
        """Required extension of M_step. Adds a re-estimation of the mixture parameters 'means', 'covars'. # Based on Huang, Acero, Hon, 'Spoken Language Processing', p. 443 - 445
        """
        new_model = super()._M_step(stats)

        denom = stats['post'][:, np.newaxis]
        if 'm' in self.tr_params:
            new_model['means'] = (
                self.means_weight * self.means_prior + stats['obs']
            ) / (self.means_weight + denom)

        if 'c' in self.tr_params:
            meandiff = new_model['means'] - self.means_prior
            if self.covariance_type in ('spherical', 'diagonal'):
                cv_num = (
                    self.means_weight * meandiff ** 2
                    + stats['obs**2']
                    - 2 * new_model['means'] * stats['obs']
                    + new_model['means'] ** 2 * denom
                )
                cv_den = np.amax(self.covars_weight - 1, 0) + denom
                covars_new = (self.covars_prior + cv_num) / \
                    np.maximum(cv_den, 1e-5)
                if self.covariance_type == 'spherical':
                    covars_new = covars_new.mean(1)
            elif self.covariance_type in ('tied', 'full'):
                cv_num = np.empty(
                    (self.n_states, self.n_emissions, self.n_emissions))
                for c in range(self.n_states):
                    obs_mean = np.outer(stats['obs'][c], new_model['means'][c])
                    cv_num[c] = (
                        self.means_weight * np.outer(meandiff[c], meandiff[c])
                        + stats['obs*obs.T'][c]
                        - obs_mean
                        - obs_mean.T
                        + np.outer(new_model['means'][c],
                                   new_model['means'][c])
                        * stats['post'][c]
                    )
                cvweight = np.amax(self.covars_weight - self.n_emissions, 0)
                if self.covariance_type == 'tied':
                    covars_new = (self.covars_prior + cv_num.sum(axis=0)) / (
                        cvweight + stats['post'].sum()
                    )
                elif self.covariance_type == 'full':
                    covars_new = (self.covars_prior + cv_num) / (
                        cvweight + stats['post'][:, None, None]
                    )
            new_model['covars'] = covars_new

        return new_model

    def _update_model(self, new_model):
        """Required extension of _updatemodel. Adds 'means' and 'covars',
        which holds the in-state information.
        """
        super()._update_model(new_model)
        if 'm' in self.tr_params:
            self.means = (1 - self.learning_rate) * new_model[
                'means'
            ] + self.learning_rate * self.means

        if 'c' in self.tr_params:
            self._covars = (1 - self.learning_rate) * new_model[
                'covars'
            ] + self.learning_rate * self._covars

    def _map_B(self, obs_seq):
        """Required implementation for _map_B. Refer to _BaseHMM for more details.
        """
        B_map = np.zeros((self.n_states, len(obs_seq)))

        for j in range(self.n_states):
            for t in range(len(obs_seq)):
                if np.all(np.isnan(obs_seq[t])):
                    # 1st Case: Complete Missed Observation
                    # If all the features at time 't' is nan, bjt = 1
                    # When we have a missing observation, the value of p(y_t | s_t = i)
                    # is equal to p(means[j]), which is 1 (because we are integrating
                    # over p(y_t | s_t = i).
                    obs = self.means[j]
                elif np.any(np.isnan(obs_seq[t])):
                    # 2nd Case: Partial Missing Data
                    # Some features of the observations are missed, but others are present
                    _, _, obs = self._calc_conditional_posterior(obs_seq[t], j)
                else:
                    obs = obs_seq[t]
                B_map[j][t] = self._pdf(obs, self.means[j], self.covars[j])
        return B_map

    def _calc_conditional_posterior(self, obs, state):
        """Helper function to compute the posterior conditional probability of themissing features given the not missing ones:
        p(missing|not_missing) = Gaussian(missing | mean(missing|not_missing),
        covariance(missing|not_missing). For extra information regarding the mathematical development of this case, you can consult the 4.3.1 section (Inference in jointly Gaussian distributions) of Kevin Murphy's book: Machine Learning, a probabilistic perspective.
        On the code, we use the '1' to refer to the missing features of the
        observation and the '2' to refer to the not missing features when
        naming the variables.

        :param obs: a single observation
        :type obs: array_like
        :param state: the index of the hidden state to consider
        :type state: int
        """

        nan_index = np.asarray(
            [True if (ot is np.nan or ot != ot) else False for ot in obs]
        )

        mu_1 = self.means[state][nan_index]
        mu_2 = self.means[state][~nan_index]
        sigma_11 = self._calc_sigma(state, nan_index, sigma_flat='11')
        sigma_12 = self._calc_sigma(state, nan_index, sigma_flat='12')
        sigma_21 = self._calc_sigma(state, nan_index, sigma_flat='21')
        sigma_22 = self._calc_sigma(state, nan_index, sigma_flat='22')
        sigma_22 += self.min_covar * np.eye(sigma_22.shape[0])

        sigma_1_given_2 = sigma_11 - np.matmul(
            np.matmul(sigma_12, np.linalg.inv(sigma_22)), sigma_21
        )
        mu_1_given_2 = mu_1 + np.matmul(
            np.matmul(sigma_12, np.linalg.inv(
                sigma_22)), (obs[~nan_index] - mu_2)
        )

        obs_vector = np.zeros_like(obs)
        obs_vector[~nan_index] = obs[~nan_index]
        obs_vector[nan_index] = mu_1_given_2

        return mu_1_given_2, sigma_1_given_2, obs_vector

    def _calc_sigma(self, state, nan_index, sigma_flat):
        """Helper function for the _calc_conditional_posterior function.

        :param state: the index of the hidden state to consider
        :type state: int
        :param nan_index: contains the indices of NaN elements in the observation
        :type nan_index: array_like
        :param sigma_flat: indicator of which calculation to use
        :type sigma_flat: str
        :return: the computed covariance values
        :rtype: array_like
        """
        number_missing_features = np.sum(nan_index)
        number_non_missing_features = np.sum(~nan_index)

        if sigma_flat == '11':
            cond_1 = True
            cond_2 = True
            shape_1 = number_missing_features
            shape_2 = number_missing_features
        elif sigma_flat == '12':
            cond_1 = True
            cond_2 = False
            shape_1 = number_missing_features
            shape_2 = number_non_missing_features
        elif sigma_flat == '21':
            cond_1 = False
            cond_2 = True
            shape_1 = number_non_missing_features
            shape_2 = number_missing_features
        elif sigma_flat == '22':
            cond_1 = False
            cond_2 = False
            shape_1 = number_non_missing_features
            shape_2 = number_non_missing_features

        tmp = []
        for i in range(self.n_emissions):
            for j in range(self.n_emissions):
                if nan_index[i] == cond_1 and nan_index[j] == cond_2:
                    tmp.append(self.covars[state][i][j])

        res = np.array(tmp).reshape((shape_1, shape_2))

        return res

    def _pdf(self, x, mean, covar):
        """Multivariate Gaussian PDF function. 

        :param x: a multivariate sample 
        :type x: array_like
        :param mean: mean of the distribution 
        :type mean: array_like
        :param covar: covariance matrix of the distribution
        :type covar: array_like
        :return: the PDF of the sample
        :rtype: float
        """
        if not np.all(np.linalg.eigvals(covar) > 0):
            covar = covar + self.min_covar * np.eye(self.n_emissions)
        return multivariate_normal.pdf(x, mean=mean, cov=covar, allow_singular=True)

    def _generate_sample_from_state(self, state):
        """Generates a random sample from a given component.

        :param state: index of the component to condition on
        :type state: int
        :return: array of shape (n_emissions, ) containing a random sample
            from the emission distribution corresponding to a given state
        :rtype: array_like
        """
        return np.random.multivariate_normal(self.means[state], self.covars[state])

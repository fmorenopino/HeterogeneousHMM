"""
Created on Jan 15, 2020
@authors: semese, fmorenopino
This code is based on:
  - HMM implementation by guyz- https://github.com/guyz/HMM
  - hmmlearn by anntzer - https://github.com/hmmlearn/
  - HMM with labels implementation by fmorenopino: https://github.com/fmorenopino/HMM_eb2
For theoretical bases see:
 - L. R. Rabiner, 'A tutorial on hidden Markov models and selected applications
   in speech recognition,' in Proceedings of the IEEE, vol. 77, no. 2,
   pp. 257-286, Feb. 1989.
 - K.P. Murphy, 'Machine Learning: A Probabilistic Perspective', The MIT Press
   ©2012, ISBN:0262018020 9780262018029
"""

import numpy as np
from sklearn import cluster
from scipy.stats import multivariate_normal

from .base import BaseHMM
from .utils import (
    init_covars, fill_covars, validate_covars, normalise,
    concatenate_observation_sequences, check_if_attributes_set
)

COVARIANCE_TYPES = frozenset(('diagonal', 'full', 'tied', 'spherical'))


class HeterogeneousHMM(BaseHMM):
    """Implementation of HMM with labels. It can manage Gaussian and categorical features.  

    :param n_states: number of hidden states in the model
    :type n_states: int
    :param n_g_emissions: number of Gaussian features
    :type n_g_emissions: int
    :param n_d_emissions: number of categorical features
    :type n_d_emissions: int
    :param n_d_features: number of distinct observation symbols per state
    :type n_d_features: list
    :param tr_params: controls which parameters are updated in thetraining process; can contain any combination of 's' for startingprobabilities (pi), 't' for transition matrix, and other charactersfor subclass-specific emission parameters, defaults to 'stmce'
    :type tr_params: str, optional
    :param init_params: controls which parameters are initialised prior to training.  Can contain any combination of 's' for starting probabilities (pi), 't' for transition matrix, and other characters for subclass-specific emission parameters, defaults to 'stmce'
    :type init_params: str, optional
    :param nr_no_train_de: this number indicates the number of discrete emissions whose Matrix Emission Probabilities are fixed and are not trained; it is important to to order the observed variables such that the ones whose emissions aren't trained are the last ones, defaults to 0
    :type nr_no_train_de: int, optional
    :param state_no_train_de: a state index for nr_no_train_de which shouldn't be updated; defaults to None, which means that the entire emission probability matrix for that discrete emission will be kept unchanged during training, otherwise the last state_no_train_de states won't be updated, defaults to None
    :type state_no_train_de: int, optional
    :param covariance_type: string describing the type of covariance parameters to use. Defaults to 'full'.
    :type covariance_type: str, optional
    :param pi_prior: array of shape (n_states, ) setting the parameters of the Dirichlet prior distribution for the starting probabilities. Defaults to 1.
    :type pi_prior: array_like, optional 
    :param pi: array of shape (n_states, ) giving the initial state occupation distribution 'pi'
    :type pi: array_like
    :param A_prior: array of shape (n_states, ), giving the parameters of the Dirichlet prior distribution for each row of the transition probabilities 'A'. Defaults to 1.
    :type A_prior: array_like, optional 
    :param A: array of shape (n_states, n_states) giving the matrix of transition probabilities between states
    :type A: array_like
    :param B: the probabilities of emitting a given discrete symbol when in each state
    :type B: list
    :param means_prior: array of shape (n_states, 1), the mean of the Normal prior distribution for the means. Defaults to 0.
    :type means_prior: array_like, optional 
    :param means_weight: array of shape (n_states, 1), the precision of the Normal prior distribution for the means. Defaults to 0.
    :type means_weight: array_like, optional 
    :param means: array of shape (n_states, n_emissions) containing the mean parameters for each state
    :type means: array_like 
    :param covars_prior: array of shape (n_states, 1), the mean of the Normal prior distribution for the covariance matrix. Defaults to 0.
    :type covars_prior: array_like, optional 
    :param covars_weight: array of shape (n_states, 1), the precision of the Normal prior distribution for the covariance. Defaults to 0.
    :type covars_weight: array_like, optional 
    :param min_covar: floor on the diagonal of the covariance matrix to prevent overfitting. Defaults to 1e-3.
    :type min_covar: float, optional
    :param covars: covariance parameters for each state arranged in an array
        of shape depends `covariance_type`.
    :type covars: array_like 
    :param learning_rate: a value from [0,1), controlling how much the past values of the model parameters count when computing the new model parameters during training; defaults to 0.
    :type learning_rate: float, optional
    :param verbose: flag to be set to True if per-iteration convergence reports should be printed, defaults to True
    :type verbose: bool, optional
    """

    def __init__(
        self,
        n_states,
        n_g_emissions,
        n_d_emissions,
        n_d_features,
        tr_params='stmce',
        init_params='stmce',
        nr_no_train_de=0,
        state_no_train_de=None,
        covariance_type='diagonal',
        pi_prior=1.0,
        A_prior=1.0,
        means_prior=0,
        means_weight=0,
        covars_prior=1e-2,
        covars_weight=1,
        min_covar=1e-3,
        learning_rate=0,
        verbose=False,
    ):
        """Constructor method.

        :raises ValueError: if covariance_type is not one of ('diagonal', 'full', 'tied', 'spherical')
        :raises ValueError: if init_type is not one of ('uniform', 'random')
        :raises TypeError: if n_d_features is not a list of length n_d_emissions
        """
        if covariance_type not in COVARIANCE_TYPES:
            raise ValueError(
                'covariance_type must be one of {}'.format(COVARIANCE_TYPES)
            )

        if len(n_d_features) != n_d_emissions:
            raise TypeError(
                'n_d_features must be one of length {}'.format(n_d_emissions)
            )

        BaseHMM.__init__(
            self,
            n_states,
            tr_params=tr_params,
            init_params=init_params,
            init_type='kmeans',
            pi_prior=pi_prior,
            A_prior=A_prior,
            learning_rate=learning_rate,
            verbose=verbose,
        )

        self.n_g_emissions = n_g_emissions
        self.n_d_emissions = n_d_emissions
        self.n_d_features = n_d_features
        self.nr_no_train_de = nr_no_train_de
        self.state_no_train_de = state_no_train_de
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
            + '\nB:\n'
            + str(self.B)
        )

    # ----------------------------------------------------------------------- #
    #        Public methods. These are callable when using the class.         #
    # ----------------------------------------------------------------------- #
    @property
    def covars(self):
        """Return covariances as a full matrix."""
        return fill_covars(
            self._covars, self.covariance_type, self.n_states, self.n_g_emissions
        )

    @covars.setter
    def covars(self, new_covars):
        """Setter for covariances. It expects the input to be of a corresponding shape to the 'covariance_type'.
        """
        covars = np.array(new_covars, copy=True)
        validate_covars(covars, self.covariance_type, self.n_states)
        self._covars = covars

    def get_n_fit_scalars_per_param(self):
        """ Return a dictionary containing the number of trainable variables
        for each model parameter.
        """
        ns = self.n_states
        neg = self.n_g_emissions
        ned = self.n_d_emissions
        nf = self.n_d_features
        return {
            's': ns,
            't': ns * ns,
            'm': ns * neg,
            'c': {
                'spherical': ns,
                'diagonal': ns * neg,
                'full': ns * neg * (neg + 1) // 2,
                'tied': neg * (neg + 1) // 2,
            }[self.covariance_type],
            'e': sum(ns * (nf[i] - 1) for i in range(ned)),
        }

    # ----------------------------------------------------------------------- #
    #             Private methods. These are used internally only.            #
    # ----------------------------------------------------------------------- #
    def _init_model_params(self, X):
        """Initialises model parameters prior to fitting. Extends the base classes method. See _BaseHMM.py for more information.
        """
        super()._init_model_params()

        X_concat = concatenate_observation_sequences(
            X, gidx=self.n_g_emissions)

        if 'm' in self.init_params:
            kmeans = cluster.KMeans(n_clusters=self.n_states, random_state=0)
            kmeans.fit(X_concat)
            self.means = kmeans.cluster_centers_
        if 'c' in self.init_params:
            cv = np.cov(X_concat.T) + self.min_covar * \
                np.eye(self.n_g_emissions)
            self._covars = init_covars(cv, self.covariance_type, self.n_states)
        if 'e' in self.init_params:
            if self.nr_no_train_de == 0:
                self.B = [
                    np.full(
                        (self.n_states, self.n_d_features[i]),
                        (1.0 / self.n_d_features[i]),
                    )
                    for i in range(self.n_d_emissions)
                ]
            else:
                check_if_attributes_set(self, attr='e')

    def _initialise_sufficient_statistics(self):
        """Initialises sufficient statistics required for M-step. Extends the base classes method by adding the emission probability matrix. See _BaseHMM.py for more information.
        """
        stats = super()._initialise_sufficient_statistics()

        stats['post'] = np.zeros(self.n_states)
        stats['obs'] = np.zeros((self.n_states, self.n_g_emissions))
        stats['obs**2'] = np.zeros((self.n_states, self.n_g_emissions))
        if self.covariance_type in ('tied', 'full'):
            stats['obs*obs.T'] = np.zeros(
                (self.n_states, self.n_g_emissions, self.n_g_emissions)
            )

        stats['B'] = {
            'numer': [
                np.zeros((self.n_states, self.n_d_features[i]))
                for i in range(self.n_d_emissions)
            ],
            'denom': [
                np.zeros((self.n_states, self.n_d_features[i]))
                for i in range(self.n_d_emissions)
            ],
        }
        return stats

    def _accumulate_sufficient_statistics(
        self, stats, obs_stats, obs_seq
    ):
        """Updates sufficient statistics from a given sample. Extends the base classes method. See _BaseHMM.py for more information.
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

        if 'e' in self.tr_params:
            B_new = self._reestimate_B(
                [obs[self.n_g_emissions:] for obs in obs_seq],
                obs_stats['gamma'],
            )
            for i in range(self.n_d_emissions):
                stats['B']['numer'][i] += B_new['numer'][i]
                stats['B']['denom'][i] += B_new['denom'][i]

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
            for t in range(len(obs_seq)):
                obs = obs_seq[t][: self.n_g_emissions]
                if np.any(obs is np.nan) or np.any(obs != obs):
                    # If there are missing observation we infer from the conditional posterior
                    stat_obs[j] = stat_obs[j] + gamma[t][j] * self._infer_missing(
                        obs, j
                    )
                else:
                    # If there are no missing values we take this observation into account
                    stat_obs[j] = stat_obs[j] + gamma[t][j] * obs
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
                (self.n_states, self.n_g_emissions, self.n_g_emissions)
            )
        else:
            stat_obs2 = np.zeros((self.n_states, self.n_g_emissions))

        for j in range(self.n_states):
            for t in range(len(obs_seq)):
                obs = obs_seq[t][: self.n_g_emissions]
                if np.any(obs is np.nan) or np.any(obs != obs):
                    # If there are missing observation we infer from the conditional posterior
                    obs = self._infer_missing(obs, j)
                if self.covariance_type in ('tied', 'full'):
                    stat_obs2[j] = stat_obs2[j] + \
                        gamma[t][j] * np.outer(obs, obs)
                else:
                    stat_obs2[j] = stat_obs2[j] + np.dot(gamma[t][j], obs ** 2)
        return stat_obs2

    def _infer_missing(self, obs, state):
        """Helper method for the statistics accumulation. It infers the missing
        observation from the conditional posterior for a given state.

        :param obs: a single observation
        :type obs: array_like
        :param state: the index of the hidden state to consider
        :type state: int
        """
        # If all the features of the obs_seq are missed, it is not necessary
        # to use this observation to update the means and covars. So we set them to 0.
        if np.all(obs is np.nan) or np.all(obs != obs):
            return np.zeros_like(obs)
        # For partial obs_seq we compute the conditional posterior.
        elif (np.any(obs is np.nan) or np.any(obs != obs)) and not (
            np.all(obs is np.nan) or np.all(obs != obs)
        ):
            _, _, obs_vector = self._calc_conditional_posterior(obs, state)
            return obs_vector

    def _reestimate_B(self, obs_seq, gamma):
        """Re-estimation of the emission matrix (part of the 'M' step of Baum-Welch). Computes B_new = expected # times in state s_j with symbol v_k /expected # times in state s_j

        :param obs_seq: array of shape (n_samples, n_d_features)
                containing the observation samples
        :type obs_seq: array_like
        :param gamma: posteriors, array of shape (n_samples, n_states)
        :type gamma: array_like
        :return: the modified parts of the emission matrix
        :rtype: dict
        """
        B_new = {
            'numer': [
                np.zeros((self.n_states, self.n_d_features[i]))
                for i in range(self.n_d_emissions)
            ],
            'denom': [
                np.zeros((self.n_states, self.n_d_features[i]))
                for i in range(self.n_d_emissions)
            ],
        }

        for e in range(self.n_d_emissions):
            for j in range(self.n_states):
                for k in range(self.n_d_features[e]):
                    numer = 0.0
                    denom = 0.0
                    for t, obs in enumerate(obs_seq):
                        if obs[e] == k:
                            numer += gamma[t][j]
                        denom += gamma[t][j]
                    B_new['numer'][e][j][k] = numer
                    B_new['denom'][e][j][k] = denom

        return B_new

    def _M_step(self, stats):
        """Required extension of M_step. Adds a re-estimation of the parameters 'means', 'covars' and 'B'.
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
                    (self.n_states, self.n_g_emissions, self.n_g_emissions)
                )
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
                cvweight = np.amax(self.covars_weight - self.n_g_emissions, 0)
                if self.covariance_type == 'tied':
                    covars_new = (self.covars_prior + cv_num.sum(axis=0)) / (
                        cvweight + stats['post'].sum()
                    )
                elif self.covariance_type == 'full':
                    covars_new = (self.covars_prior + cv_num) / (
                        cvweight + stats['post'][:, None, None]
                    )
            new_model['covars'] = covars_new

        if 'e' in self.tr_params:
            new_model['B'] = [
                stats['B']['numer'][i] / stats['B']['denom'][i]
                for i in range(self.n_d_emissions)
            ]

        return new_model

    def _update_model(self, new_model):
        """ Required extension of _updatemodel. Adds 'B', 'means' and 'covars',
        which holds the in-state information.
        """
        super()._update_model(new_model)

        if 'm' in self.tr_params:
            self.means = (1 - self.learning_rate) * new_model[
                'means'
            ] + self.learning_rate * new_model['means']

        if 'c' in self.tr_params:
            self._covars = (1 - self.learning_rate) * new_model[
                'covars'
            ] + self.learning_rate * self._covars

        if 'e' in self.tr_params:
            if self.state_no_train_de is None:
                for i in range(self.n_d_emissions - self.nr_no_train_de):
                    self.B[i] = (1 - self.learning_rate) * new_model['B'][
                        i
                    ] + self.learning_rate * self.B[i]
            else:
                for i in range(self.n_d_emissions):
                    if i < self.n_d_emissions - self.nr_no_train_de:
                        self.B[i] = (1 - self.learning_rate) * new_model['B'][
                            i
                        ] + self.learning_rate * self.B[i]
                    else:
                        self.B[i][: -self.state_no_train_de, :] = (
                            (1 - self.learning_rate)
                            * new_model['B'][i][: -self.state_no_train_de, :]
                            + self.learning_rate *
                            self.B[i][: -self.state_no_train_de, :]
                        )
            for i in range(self.n_d_emissions):
                normalise(self.B[i], axis=1)

    def _map_B(self, obs_seq):
        """Required implementation for _map_B. Refer to _BaseHMM for more details.
        """

        def _map_gB(y_t, j):
            """
            Implementation of _map_B for the Gaussian emissions.
            """
            if np.all(y_t is np.nan or y_t != y_t):
                # 1st Case: Complete Missed Observation. For more details see GaussianHMM
                obs = self.means[j]
            elif np.any(y_t is np.nan or y_t != y_t):
                # 2nd Case: Partially Missing Observation
                _, _, obs = self._calc_conditional_posterior(y_t, j)
            else:
                obs = y_t
            return self._pdf(obs, self.means[j], self.covars[j])

        def _map_dB(y_t, j):
            """Implementation of _map_B for the multinomial emissions.
            """
            bjt = 1.0
            for e, symbol in enumerate(y_t):
                if symbol is np.nan or symbol != symbol:
                    bjt *= self.B[e][j][np.argmax(self.B[e][j])]
                else:
                    bjt *= self.B[e][j][int(symbol)]
            return bjt

        B_map = np.zeros((self.n_states, len(obs_seq)))

        for j in range(self.n_states):
            for t in range(len(obs_seq)):
                bjt_gauss = _map_gB(obs_seq[t][: self.n_g_emissions], j)
                bjt_disc = _map_dB(obs_seq[t][self.n_g_emissions:], j)
                B_map[j][t] = bjt_gauss * bjt_disc

        return B_map

    def _calc_conditional_posterior(self, obs, state):
        """Helper function to compute the posterior conditional probability of the missing features given the not missing ones:  p(missing|not_missing) = Gaussian(missing | mean(missing|not_missing), covariance(missing|not_missing). For extra information regarding the mathematical development of this case, you can consult the 4.3.1 section (Inference in jointly Gaussian distributions)
        of Kevin Murphy's book: Machine Learning, a probabilistic perspective.
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
        for i in range(self.n_g_emissions):
            for j in range(self.n_g_emissions):
                if nan_index[i] == cond_1 and nan_index[j] == cond_2:
                    tmp.append(self.covars[state][i, j])

        res = np.array(tmp).reshape((shape_1, shape_2))

        return res

    def _pdf(self, x, mean, covar):
        """ Multivariate Gaussian PDF function. 

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
            covar = covar + self.min_covar * np.eye(self.n_g_emissions)
        return multivariate_normal.pdf(x, mean=mean, cov=covar, allow_singular=True)

    def _generate_sample_from_state(self, state):
        """ Generates a random sample from a given component.
        :param state: index of the component to condition on
        :type state: int
        :return: array of shape (n_g_features+n_d_features, ) containing a random sample
            from the emission distribution corresponding to a given state
        :rtype: array_like
        """
        gauss_sample = np.random.multivariate_normal(
            self.means[state], self.covars[state]
        )

        cat_sample = []
        for e in range(self.n_d_emissions):
            cdf = np.cumsum(self.B[e][state, :])
            cat_sample.append((cdf > np.random.rand()).argmax())

        return np.concatenate([gauss_sample, cat_sample])

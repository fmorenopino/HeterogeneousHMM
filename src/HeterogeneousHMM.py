"""
Created on Jan 15, 2020

@authors: esukei, fmorenopino

This code is based on:
  - HMM implementation by guyz- https://github.com/guyz/HMM
  - hmmlearn by anntzer - https://github.com/hmmlearn/
  - HMM with labels implementation by fmorenopino: https://github.com/fmorenopino/HMM_eb2

For theoretical bases see:
 - L. R. Rabiner, "A tutorial on hidden Markov models and selected applications
   in speech recognition," in Proceedings of the IEEE, vol. 77, no. 2,
   pp. 257-286, Feb. 1989.
 - K.P. Murphy, "Machine Learning: A Probabilistic Perspective", The MIT Press
   ©2012, ISBN:0262018020 9780262018029
"""

import numpy as np
from sklearn import cluster
from scipy.stats import multivariate_normal

from ._BaseHMM import _BaseHMM
from .utils import (
    fill_covars,
    normalise,
    validate_covars,
    init_covars,
    check_if_attributes_set,
)
from .initialisation_utils import concatenate_observation_sequences, init_gaussian_hmm

COVARIANCE_TYPES = frozenset(("diagonal", "full", "tied", "spherical"))
INIT_TYPES = frozenset(("random", "kmeans"))


class HeterogeneousHMM(_BaseHMM):
    """
        Implementation of HMM with labels. It can manage Gaussian and categorical
        features.
        Parameters:
            n_states (int) - number of hidden states in the model
            n_g_emissions (int) - dimensionality of the Gaussian emissions
            n_d_emissions (int) - dimensionality of the categorical features
            n_d_features (list of ints) - list of the number of possible observable
                symbols for each discrete emission
            params (string, optional) - controls which parameters are updated in the
                training process.  Can contain any combination of 's' for starting
                probabilities (pi), 't' for transition matrix, and other characters
                for subclass-specific emission parameters. Defaults to all parameters.
            init_params (string, optional) - controls which parameters are
                initialised prior to training.  Can contain any combination of 's'
                 for starting probabilities (pi), 't' for transition matrix, and
                 other characters for subclass-specific emission parameters.
                 Defaults to all parameters.
            nr_no_train_de (int) - this number indicates the number of discrete
                emissions whose Matrix Emission Probabilities are fixed and
                are not trained; it is important to to order the observed variables
                such that the ones whose emissions aren't trained are the last ones
            state_no_train_de (int) - a state index for nr_no_train_de
                which shouldn't be updated; defaults to None, which means that the
                entire emission probability matrix for that discrete emission will
                be kept unchanged during training, otherwise the last state_no_train_de states won't be updated
            init_type (string, optional) - name of the initialisation
                    method to use for initialising the start and transition matrices;
            covariance_type (string, optional) - string describing the type of
                covariance parameters to use.  Must be one of:
                * "diag" --- each state uses a diagonal covariance matrix.
                * "full" --- each state uses a full (i.e. unrestricted)
                  covariance matrix.
                Defaults to "diagonal".
            pi_prior (array, optional) - array of shape (n_states, ) setting the
                parameters of the Dirichlet prior distribution for 'pi'
            A_prior (array, optional) - array of shape (n_states, n_states),
                giving the parameters of the Dirichlet prior distribution for each
                row of the transition probabilities 'A'
            means_prior, means_weight (array,optional) - arrays of shape (n_states, 1)
                providing the mean and precision of the Normal prior distribution for
                the means
            covars_prior, covars_weight (array, optional) - shape (n_states, 1), provides
                the parameters of the prior distribution for the covariance matrix
            min_covar (float, optional)- floor on the diagonal of the covariance
                matrix to prevent overfitting. Defaults to 1e-3.
            learn_rate (float, optional) - a value from [0,1), controlling how much
                the past values of the model parameters count when computing the new
                model parameters during training. By default it's 0.
            verbose (bool, optional) - flag to be set to True if per-iteration
                convergence reports should be printed
        Attributes:
            pi (array) - array of shape (n_states, ) giving the initial state
                occupation distribution 'pi'
            A (array) - array of shape (n_states, n_states) giving the matrix of
                transition probabilities between states
            B (list) - list of n_d_emissions arrays of shape (n_states, n_d_features[i])
                for i = 0 ... n_d_emissions, containing the probability of emitting
                a given symbol when in each state
            means (array) - array of shape (n_states, n_g_emissions) containing the
                mean parameters for each state
            covars (array) - covariance parameters for each state arranged in an array
                of shape depends `covariance_type`:
                (n_states, )                          if "spherical",
                (n_states, n_g_emissions)              if "diagonal",
                (n_states, n_g_emissions, n_g_emissions)  if "full",
                (n_g_emissions, n_g_emissions)            if "tied"
    """

    def __init__(
        self,
        n_states=1,
        n_g_emissions=1,
        n_d_emissions=1,
        n_d_features=[1],
        params="stmce",
        init_params="stmce",
        nr_no_train_de=0,
        state_no_train_de=None,
        init_type="random",
        covariance_type="diagonal",
        pi_prior=1.0,
        A_prior=1.0,
        means_prior=0,
        means_weight=0,
        covars_prior=1e-2,
        covars_weight=1,
        min_covar=1e-3,
        learn_rate=0,
        verbose=False,
    ):
        if covariance_type not in COVARIANCE_TYPES:
            raise ValueError(
                "covariance_type must be one of {}".format(COVARIANCE_TYPES)
            )

        if init_type not in INIT_TYPES:
            raise ValueError("init_type must be one of {}".format(INIT_TYPES))

        if len(n_d_features) != n_d_emissions:
            raise TypeError(
                "n_d_features must be one of length {}".format(n_d_emissions)
            )

        _BaseHMM.__init__(
            self,
            n_states,
            params=params,
            init_params=init_params,
            init_type=init_type,
            pi_prior=pi_prior,
            A_prior=A_prior,
            learn_rate=learn_rate,
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
        self.dist_params = None

    def __str__(self):
        """
            Function to allow directly printing the object.
        """
        temp = super(HeterogeneousHMM, self).__str__()
        return (
            temp
            + "\nMeans:\n"
            + str(self.means)
            + "\nCovariances:\n"
            + str(self.covars)
            + "\nB:\n"
            + str(self.B)
        )

    # ----------------------------------------------------------------------- #
    #        Public methods. These are callable when using the class.         #
    # ----------------------------------------------------------------------- #
    @property
    def covars(self):
        """
            Return covariances as a full matrix.
        """
        return fill_covars(
            self._covars, self.covariance_type, self.n_states, self.n_g_emissions
        )

    @covars.setter
    def covars(self, new_covars):
        """
            Setter for covariances. It expects the input to be of a corresponding
            shape to the 'covariance_type'.
        """
        covars = np.array(new_covars, copy=True)
        validate_covars(covars, self.covariance_type, self.n_states)
        self._covars = covars

    def get_n_fit_scalars_per_param(self):
        """
            Return a dictionary containing the number of trainable variables
            for each model parameter.
        """
        ns = self.n_states
        neg = self.n_g_emissions
        ned = self.n_d_emissions
        nf = self.n_d_features
        return {
            "s": ns,
            "t": ns * ns,
            "m": ns * neg,
            "c": {
                "spherical": ns,
                "diagonal": ns * neg,
                "full": ns * neg * (neg + 1) // 2,
                "tied": neg * (neg + 1) // 2,
            }[self.covariance_type],
            "e": sum(ns * (nf[i] - 1) for i in range(ned)),
        }

    # ----------------------------------------------------------------------- #
    #             Private methods. These are used internally only.            #
    # ----------------------------------------------------------------------- #
    def _init(self, X):
        super(HeterogeneousHMM, self)._init(X=X)

        X_concat = concatenate_observation_sequences(X, gidx=self.n_g_emissions)

        if self.init_type == "random":
            if self.dist_params is not None:
                mu_, sigma_ = init_gaussian_hmm(
                    self.n_states, self.n_g_emissions, dist_params=self.dist_params
                )
            else:
                mu_, sigma_, self.dist_params = init_gaussian_hmm(
                    self.n_states, self.n_g_emissions, X=X_concat
                )
            if "m" in self.init_params:
                self.means = mu_
            if "c" in self.init_params:
                cv = sigma_ + self.min_covar * np.eye(self.n_g_emissions)
                self._covars = init_covars(cv, self.covariance_type, self.n_states)
            if "e" in self.init_params:
                if self.nr_no_train_de == 0:
                    self.B = [
                        normalise(
                            np.random.rand(self.n_states, self.n_d_features[i]), axis=1
                        )
                        for i in range(self.n_d_emissions)
                    ]
                else:
                    check_if_attributes_set(self, attr="e")

        elif self.init_type == "kmeans":
            if "m" in self.init_params:
                kmeans = cluster.KMeans(n_clusters=self.n_states, random_state=0)
                kmeans.fit(X_concat)
                self.means = kmeans.cluster_centers_
            if "c" in self.init_params:
                cv = np.cov(X_concat.T) + self.min_covar * np.eye(self.n_g_emissions)
                self._covars = init_covars(cv, self.covariance_type, self.n_states)
            if "e" in self.init_params:
                if self.nr_no_train_de == 0:
                    self.B = [
                        np.full(
                            (self.n_states, self.n_d_features[i]),
                            (1.0 / self.n_d_features[i]),
                        )
                        for i in range(self.n_d_emissions)
                    ]
                else:
                    check_if_attributes_set(self, attr="e")

    def _initialise_sufficient_statistics(self):
        """
        Initialises sufficient statistics required for M-step.
        Extends the base classes method by adding the emission probability matrix.
        See _BaseHMM.py for more information.
        """
        stats = super(HeterogeneousHMM, self)._initialise_sufficient_statistics()

        stats["post"] = np.zeros(self.n_states)
        stats["obs"] = np.zeros((self.n_states, self.n_g_emissions))
        stats["obs**2"] = np.zeros((self.n_states, self.n_g_emissions))
        if self.covariance_type in ("tied", "full"):
            stats["obs*obs.T"] = np.zeros(
                (self.n_states, self.n_g_emissions, self.n_g_emissions)
            )

        stats["B"] = {
            "numer": [
                np.zeros((self.n_states, self.n_d_features[i]))
                for i in range(self.n_d_emissions)
            ],
            "denom": [
                np.zeros((self.n_states, self.n_d_features[i]))
                for i in range(self.n_d_emissions)
            ],
        }
        return stats

    def _accumulate_sufficient_statistics(
        self, stats, observations_stats, observations
    ):
        """
        Updates sufficient statistics from a given sample.
        Extends the base classes method. See _BaseHMM.py for more information.
        """
        super(HeterogeneousHMM, self)._accumulate_sufficient_statistics(
            stats, observations_stats, observations
        )

        if "m" in self.params:
            stats["post"] += observations_stats["gamma"].sum(axis=0)
            stats["obs"] += self._reestimate_stat_obs(
                observations_stats["gamma"], observations
            )

        if "c" in self.params:
            if self.covariance_type in ("spherical", "diagonal"):
                stats["obs**2"] += self._reestimate_stat_obs2(
                    observations_stats["gamma"], observations
                )
            elif self.covariance_type in ("tied", "full"):
                stats["obs*obs.T"] += self._reestimate_stat_obs2(
                    observations_stats["gamma"], observations
                )

        if "e" in self.params:
            B_new = self._reestimate_B(
                [obs[self.n_g_emissions :] for obs in observations],
                observations_stats["gamma"],
            )
            for i in range(self.n_d_emissions):
                stats["B"]["numer"][i] += B_new["numer"][i]
                stats["B"]["denom"][i] += B_new["denom"][i]

    def _reestimate_stat_obs(self, gamma, observations):
        """
            Helper method for the statistics accumulation. Computes the sum of
            the posteriors times the observations for the update of the means.
        """
        stat_obs = np.zeros_like(self.means)
        for j in range(self.n_states):
            for t in range(len(observations)):
                obs = observations[t][: self.n_g_emissions]
                if np.any(obs is np.nan) or np.any(obs != obs):
                    # If there are missing observation we infer from the conditional posterior
                    stat_obs[j] = stat_obs[j] + gamma[t][j] * self._infer_missing(
                        obs, j
                    )
                else:
                    # If there are no missing values we take this observation into account
                    stat_obs[j] = stat_obs[j] + gamma[t][j] * obs
        return stat_obs

    def _reestimate_stat_obs2(self, gamma, observations):
        """
            Helper method for the statistics accumulation. Computes the sum of
            the posteriors times the square of the observations for the update
            of the covariances.
        """
        if self.covariance_type in ("tied", "full"):
            stat_obs2 = np.zeros(
                (self.n_states, self.n_g_emissions, self.n_g_emissions)
            )
        else:
            stat_obs2 = np.zeros((self.n_states, self.n_g_emissions))

        for j in range(self.n_states):
            for t in range(len(observations)):
                obs = observations[t][: self.n_g_emissions]
                if np.any(obs is np.nan) or np.any(obs != obs):
                    # If there are missing observation we infer from the conditional posterior
                    obs = self._infer_missing(obs, j)
                if self.covariance_type in ("tied", "full"):
                    stat_obs2[j] = stat_obs2[j] + gamma[t][j] * np.outer(obs, obs)
                else:
                    stat_obs2[j] = stat_obs2[j] + np.dot(gamma[t][j], obs ** 2)
        return stat_obs2

    def _infer_missing(self, obs, state):
        """
            Helper method for the statistics accumulation. It infers the missing
            observation from the conditional posterior for a given state.
        """
        # If all the features of the observations are missed, it is not necessary
        # to use this observation to update the means and covars. So we set them to 0.
        if np.all(obs is np.nan) or np.all(obs != obs):
            return np.zeros_like(obs)
        # For partial observations we compute the conditional posterior.
        elif (np.any(obs is np.nan) or np.any(obs != obs)) and not (
            np.all(obs is np.nan) or np.all(obs != obs)
        ):
            _, _, obs_vector = self._calc_conditional_posterior(obs, state)
            return obs_vector

    def _reestimate_B(self, observations, gamma):
        """
        Reestimation of the emission matrix (part of the 'M' step of Baum-Welch).
        Computes B_new = expected # times in state s_j with symbol v_k /expected
         # times in state s_j
        Args:
            observations (array) - array of shape (n_samples, n_d_features)
                containing the observation samples
            gamma (array) - array of shape (n_samples, n_states)
        Returns:
            B_new (dictionary) - the modified parts of the emission matrix
        """
        B_new = {
            "numer": [
                np.zeros((self.n_states, self.n_d_features[i]))
                for i in range(self.n_d_emissions)
            ],
            "denom": [
                np.zeros((self.n_states, self.n_d_features[i]))
                for i in range(self.n_d_emissions)
            ],
        }

        for e in range(self.n_d_emissions):
            for j in range(self.n_states):
                for k in range(self.n_d_features[e]):
                    numer = 0.0
                    denom = 0.0
                    for t, obs in enumerate(observations):
                        if obs[e] == k:
                            numer += gamma[t][j]
                        denom += gamma[t][j]
                    B_new["numer"][e][j][k] = numer
                    B_new["denom"][e][j][k] = denom

        return B_new

    def _M_step(self, stats):
        """
        Required extension of M_step.
        Adds a re-estimation of the parameters 'means', 'covars' and 'B'.
        """
        new_model = super(HeterogeneousHMM, self)._M_step(stats)

        denom = stats["post"][:, np.newaxis]
        if "m" in self.params:
            new_model["means"] = (
                self.means_weight * self.means_prior + stats["obs"]
            ) / (self.means_weight + denom)

        if "c" in self.params:
            meandiff = new_model["means"] - self.means_prior
            if self.covariance_type in ("spherical", "diagonal"):
                cv_num = (
                    self.means_weight * meandiff ** 2
                    + stats["obs**2"]
                    - 2 * new_model["means"] * stats["obs"]
                    + new_model["means"] ** 2 * denom
                )
                cv_den = np.amax(self.covars_weight - 1, 0) + denom
                covars_new = (self.covars_prior + cv_num) / np.maximum(cv_den, 1e-5)
                if self.covariance_type == "spherical":
                    covars_new = covars_new.mean(1)
            elif self.covariance_type in ("tied", "full"):
                cv_num = np.empty(
                    (self.n_states, self.n_g_emissions, self.n_g_emissions)
                )
                for c in range(self.n_states):
                    obs_mean = np.outer(stats["obs"][c], new_model["means"][c])
                    cv_num[c] = (
                        self.means_weight * np.outer(meandiff[c], meandiff[c])
                        + stats["obs*obs.T"][c]
                        - obs_mean
                        - obs_mean.T
                        + np.outer(new_model["means"][c], new_model["means"][c])
                        * stats["post"][c]
                    )
                cvweight = np.amax(self.covars_weight - self.n_g_emissions, 0)
                if self.covariance_type == "tied":
                    covars_new = (self.covars_prior + cv_num.sum(axis=0)) / (
                        cvweight + stats["post"].sum()
                    )
                elif self.covariance_type == "full":
                    covars_new = (self.covars_prior + cv_num) / (
                        cvweight + stats["post"][:, None, None]
                    )
            new_model["covars"] = covars_new

        if "e" in self.params:
            new_model["B"] = [
                stats["B"]["numer"][i] / stats["B"]["denom"][i]
                for i in range(self.n_d_emissions)
            ]
            for i in range(self.n_d_emissions):
                normalise(new_model["B"][i], axis=1)

        return new_model

    def _update_model(self, new_model):
        """
        Required extension of _updatemodel. Adds 'B', 'means' and 'covars',
        which holds the in-state information.
        """
        super(HeterogeneousHMM, self)._update_model(new_model)

        if "m" in self.params:
            self.means = (1 - self.learn_rate) * new_model[
                "means"
            ] + self.learn_rate * new_model["means"]

        if "c" in self.params:
            self._covars = (1 - self.learn_rate) * new_model[
                "covars"
            ] + self.learn_rate * self._covars

        if "e" in self.params:
            if self.state_no_train_de is None:
                for i in range(self.n_d_emissions - self.nr_no_train_de):
                    self.B[i] = (1 - self.learn_rate) * new_model["B"][
                        i
                    ] + self.learn_rate * self.B[i]
            else:
                for i in range(self.n_d_emissions):
                    if i < self.n_d_emissions - self.nr_no_train_de:
                        self.B[i] = (1 - self.learn_rate) * new_model["B"][
                            i
                        ] + self.learn_rate * self.B[i]
                    else:
                        self.B[i][: -self.state_no_train_de, :] = (
                            (1 - self.learn_rate)
                            * new_model["B"][i][: -self.state_no_train_de, :]
                            + self.learn_rate * self.B[i][: -self.state_no_train_de, :]
                        )

    def _map_B(self, observations):
        """
        Required implementation for _map_B. Refer to _BaseHMM for more details.
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
            """
            Implementation of _map_B for the multinomial emissions.
            """
            bjt = 1.0
            for e, symbol in enumerate(y_t):
                if symbol is np.nan or symbol != symbol:
                    bjt *= self.B[e][j][np.argmax(self.B[e][j])]
                else:
                    bjt *= self.B[e][j][int(symbol)]
            return bjt

        self.B_map = np.zeros((self.n_states, len(observations)))

        for j in range(self.n_states):
            for t in range(len(observations)):
                bjt_gauss = _map_gB(observations[t][: self.n_g_emissions], j)
                bjt_disc = _map_dB(observations[t][self.n_g_emissions :], j)
                self.B_map[j][t] = bjt_gauss * bjt_disc

    def _calc_conditional_posterior(self, obs, state):

        """
        Helper function to compute the posterior conditional probability of the
        missing features given the not missing ones:
        p(missing|not_missing) = Gaussian(missing | mean(missing|not_missing),
            covariance(missing|not_missing).

        For extra information regarding the mathematical development of this case,
        you can consult the 4.3.1 section (Inference in jointly Gaussian distributions)
        of Kevin Murphy´s book: Machine Learning, a probabilistic perspective.

        On the code, we use the "1" to refer to the missing features of the
        observation and the "2" to refer to the not missing features when
        naming the variables.
        """

        nan_index = np.asarray(
            [True if (ot is np.nan or ot != ot) else False for ot in obs]
        )

        mu_1 = self.means[state][nan_index]
        mu_2 = self.means[state][~nan_index]
        sigma_11 = self._calc_sigma(state, nan_index, sigma_flat="11")
        sigma_12 = self._calc_sigma(state, nan_index, sigma_flat="12")
        sigma_21 = self._calc_sigma(state, nan_index, sigma_flat="21")
        sigma_22 = self._calc_sigma(state, nan_index, sigma_flat="22")
        sigma_22 += self.min_covar * np.eye(sigma_22.shape[0])

        sigma_1_given_2 = sigma_11 - np.matmul(
            np.matmul(sigma_12, np.linalg.inv(sigma_22)), sigma_21
        )
        mu_1_given_2 = mu_1 + np.matmul(
            np.matmul(sigma_12, np.linalg.inv(sigma_22)), (obs[~nan_index] - mu_2)
        )

        obs_vector = np.zeros_like(obs)
        obs_vector[~nan_index] = obs[~nan_index]
        obs_vector[nan_index] = mu_1_given_2

        return mu_1_given_2, sigma_1_given_2, obs_vector

    def _calc_sigma(self, state, nan_index, sigma_flat):
        """
        Helper function for the _calc_conditional_posterior function.
        Args:
            state (int) - the index of the hidden state to consider
            nan_index (array) - contains the indices of NaN elements in the observation
            sigma_flat (str) - indicator of which calculation to use.
        Returns:
            res (array) - the computed covariance values
        """

        number_missing_features = np.sum(nan_index)
        number_non_missing_features = np.sum(~nan_index)

        if sigma_flat == "11":
            cond_1 = True
            cond_2 = True
            shape_1 = number_missing_features
            shape_2 = number_missing_features
        elif sigma_flat == "12":
            cond_1 = True
            cond_2 = False
            shape_1 = number_missing_features
            shape_2 = number_non_missing_features
        elif sigma_flat == "21":
            cond_1 = False
            cond_2 = True
            shape_1 = number_non_missing_features
            shape_2 = number_missing_features
        elif sigma_flat == "22":
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

    def _pdf(self, x, mean, covar, min_covar=1e-03):
        """
        Gaussian PDF function
        """
        if not np.all(np.linalg.eigvals(covar) > 0):
            covar = covar + self.min_covar * np.eye(self.n_g_emissions)
        return multivariate_normal.pdf(x, mean=mean, cov=covar, allow_singular=True)

    def _generate_sample_from_state(self, state):
        """
        Generates a random sample from a given component.
        Args:
            state (int) - index of the component to condition on
        Returns:
            X (array) - array of shape (n_gfeatures+n_dfeatures, ) containing a
            random sample from the emission distribution corresponding to a given
             state.
        """
        gauss_sample = np.random.multivariate_normal(
            self.means[state], self.covars[state]
        )

        cat_sample = []
        for e in range(self.n_d_emissions):
            cdf = np.cumsum(self.B[e][state, :])
            cat_sample.append((cdf > np.random.rand()).argmax())

        return np.concatenate([gauss_sample, cat_sample])

"""
Created on Nov 22, 2019

@author: semese

This code is based on:
  - HMM implementation by guyz - https://github.com/guyz/HMM
  - hmmlearn by anntzer - https://github.com/hmmlearn/

For theoretical bases see:
 - L. R. Rabiner, "A tutorial on hidden Markov models and selected applications
   in speech recognition," in Proceedings of the IEEE, vol. 77, no. 2,
   pp. 257-286, Feb. 1989.
 - K.P. Murphy, "Machine Learning: A Probabilistic Perspective", The MIT Press
   ©2012, ISBN:0262018020 9780262018029
"""

from ._BaseHMM import _BaseHMM
from .utils import normalise, check_if_attributes_set
import numpy as np

INIT_TYPES = frozenset(("uniform", "random"))


class MultinomialHMM(_BaseHMM):
    """
    Hidden Markov Model with multiple multinomial (discrete) emissions.
    Parameters:
        n_states (int) - number of hidden states in the model
        n_emissions (int) - number of distinct observation symbols per state
        n_features (list of ints) - list of the number of possible observable
            symbols for each emission
        params (string, optional) - controls which parameters are updated in the
            training process.  Can contain any combination of 's' for starting
            probabilities (pi), 't' for transition matrix, and other characters
            for subclass-specific emission parameters. Defaults to all parameters.
        init_params (string, optional) - controls which parameters are initialised
            prior to training.  Can contain any combination of 's' for starting
            probabilities (pi), 't' for transition matrix, and other characters
            for subclass-specific emission parameters. Defaults to all parameters.
        init_type (string, optional) - name of the initialisation
            method to use for initialising the start,transition and emission
            matrices
        pi_prior (array, optional) - array of shape (n_states, ) setting the
            parameters of the Dirichlet prior distribution for 'pi'
        A_prior (array, optional) - array of shape (n_states, n_states),
            giving the parameters of the Dirichlet prior distribution for each
            row of the transition probabilities 'A'
        learn_rate (float, optional) - a value from [0,1), controlling how much
            the past values of the model parameters count when computing the new
            model parameters during training. By default it's 0.
        missing (int or NaN) - a value indicating what character indicates a missed
            observation in the observation sequences
        verbose (bool, optional) - flag to be set to True if per-iteration
            convergence reports should be printed
    Attributes:
        pi (array) - array of shape (n_states, ) giving the initial
            state occupation distribution 'pi'
        A (array) - array of shape (n_states, n_states) giving
            the matrix of transition probabilities between states
        B (array) - array of shape (n_emissions, n_states, n_features) containing the
            probability of emitting a given symbol when in each state
    """

    def __init__(
        self,
        n_states,
        n_emissions,
        n_features,
        params="ste",
        init_params="ste",
        init_type="random",
        pi_prior=1.0,
        A_prior=1.0,
        missing=np.nan,
        nr_no_train_de=0,
        learn_rate=0,
        verbose=False,
    ):
        """
        Class initialiser.
        Args:
            n_states (int) - number of hidden states in the model
            n_emissions (int) - number of discrete emissions
            n_features (list) - number of observable symbols for each emission
            params (string, optional) - controls which parameters are updated in the
                training process.  Can contain any combination of 's' for starting
                probabilities (pi), 't' for transition matrix, and other characters
                for subclass-specific emission parameters. Defaults to all parameters.
            init_params (string, optional) - controls which parameters are initialised
                prior to training.  Can contain any combination of 's' for starting
                probabilities (pi), 't' for transition matrix, and other characters
                for subclass-specific emission parameters. Defaults to all parameters.
            init_type (string, optional) - name of the initialisation
                method to use for initialising the start,transition and emission
                matrices
            pi_prior (array, optional) - an array of shape (n_states, ) which
                gives the parameters of the Dirichlet prior distribution for 'pi'
            A_prior (array, optional) - array of shape (n_states, n_states)
                providing the parameters of the Dirichlet prior distribution for
                each row of the transition probabilities 'A'
            missing (optional) - the character that marks the missing data in the
                observation sequences; by default it's set to np.nan but it can
                be any integer set by the user.
            nr_no_train_de (int) - this number indicates the number of discrete
                emissions whose Matrix Emission Probabilities are fixed and
                are not trained; it is important to to order the observed variables
                such that the ones whose emissions aren't trained are the last ones
            verbose (bool, optional) - flag to be set to True if per-iteration
                convergence reports should be printed
        """
        if init_type not in INIT_TYPES:
            raise ValueError("init_type must be one of {}".format(INIT_TYPES))

        if len(n_features) != n_emissions:
            raise TypeError("n_features must be one of length {}".format(n_emissions))

        super(MultinomialHMM, self).__init__(
            n_states,
            params=params,
            init_params=init_params,
            init_type=init_type,
            pi_prior=pi_prior,
            A_prior=A_prior,
            learn_rate=learn_rate,
            verbose=verbose,
        )
        self.n_emissions = n_emissions
        self.n_features = n_features
        self.MISSING = missing
        self.nr_no_train_de = nr_no_train_de

    def __str__(self):
        """
            Function to allow directly printing the object.
        """
        temp = super(MultinomialHMM, self).__str__()
        return temp + "\nB:\n" + str(self.B)

    # ----------------------------------------------------------------------- #
    #        Public methods. These are callable when using the class.         #
    # ----------------------------------------------------------------------- #
    @property
    def missing(self):
        return self.MISSING

    @missing.setter
    def missing(self, value):
        """
            Setter for the missing value character in the data. By default it's
            -1, but it can be set to an unused character based on the dataset
            at hand.
        """
        self.MISSING = value

    def get_n_fit_scalars_per_param(self):
        """
            Return the number of trainable model parameters.
        """
        ns = self.n_states
        ne = self.n_emissions
        nf = self.n_features
        nnt = self.nr_no_train_de
        return {
            "s": ns,
            "t": ns * ns,
            "e": sum(ns * (nf[i] - 1) for i in range(ne - nnt)),
        }

    # ----------------------------------------------------------------------- #
    #             Private methods. These are used internally only.            #
    # ----------------------------------------------------------------------- #
    def _init(self, X=None):
        """
        Initialises model parameters prior to fitting.
        Extends the base classes method. See _BaseHMM.py for more information.
        """
        super(MultinomialHMM, self)._init()

        if "e" in self.init_params:
            if self.init_type == "uniform":
                if self.nr_no_train_de == 0:
                    self.B = [
                        np.full(
                            (self.n_states, self.n_features[i]),
                            (1.0 / self.n_features[i]),
                        )
                        for i in range(self.n_emissions)
                    ]
                else:
                    check_if_attributes_set(self, attr="e")
            elif self.init_type == "random":
                if self.nr_no_train_de == 0:
                    self.B = [
                        normalise(
                            np.random.rand(self.n_states, self.n_features[i]), axis=1
                        )
                        for i in range(self.n_emissions)
                    ]
                else:
                    check_if_attributes_set(self, attr="e")
            else:
                raise ValueError("Unknown initialiser type {!r}".format(self.init_type))

    def _initialise_sufficient_statistics(self):
        """
        Initialises sufficient statistics required for M-step.
        Extends the base classes method by adding the emission probability matrix.
        See _BaseHMM.py for more information.
        """
        stats = super(MultinomialHMM, self)._initialise_sufficient_statistics()

        stats["B"] = {
            "numer": [
                np.zeros((self.n_states, self.n_features[i]))
                for i in range(self.n_emissions)
            ],
            "denom": [
                np.zeros((self.n_states, self.n_features[i]))
                for i in range(self.n_emissions)
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
        super(MultinomialHMM, self)._accumulate_sufficient_statistics(
            stats, observations_stats, observations
        )

        if "e" in self.params:
            B_new = self._reestimate_B(observations, observations_stats["gamma"])
            for i in range(self.n_emissions):
                stats["B"]["numer"][i] += B_new["numer"][i]
                stats["B"]["denom"][i] += B_new["denom"][i]

    def _reestimate_B(self, observations, gamma):
        """
        Reestimation of the emission matrix (part of the 'M' step of Baum-Welch).
        Computes B_new = expected # times in state s_j with symbol v_k /expected
         # times in state s_j
        Args:
            observations (array) - array of shape (n_samples, n_features)
                containing the observation samples
            gamma (array) - array of shape (n_samples, n_states)
        Returns:
            B_new (dictionary) - the modified parts of the emission matrix
        """
        B_new = {
            "numer": [
                np.zeros((self.n_states, self.n_features[i]))
                for i in range(self.n_emissions)
            ],
            "denom": [
                np.zeros((self.n_states, self.n_features[i]))
                for i in range(self.n_emissions)
            ],
        }

        for e in range(self.n_emissions):
            for j in range(self.n_states):
                for k in range(self.n_features[e]):
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
        Performs the 'M' step of the Baum-Welch algorithm.
        Extends the base classes method. See _BaseHMM.py for more information.
        """
        new_model = super(MultinomialHMM, self)._M_step(stats)

        if "e" in self.params:
            new_model["B"] = [
                (stats["B"]["numer"][i] / stats["B"]["denom"][i])
                for i in range(self.n_emissions)
            ]

        return new_model

    def _update_model(self, new_model):
        """
        Updates the emission probability matrix.
        Extends the base classes method. See _BaseHMM.py for more information.
        """
        super(MultinomialHMM, self)._update_model(new_model)

        if "e" in self.params:
            for i in range(self.n_emissions - self.nr_no_train_de):
                self.B[i] = (1 - self.learn_rate) * new_model["B"][
                    i
                ] + self.learn_rate * self.B[i]

            for i in range(self.n_emissions):
                normalise(new_model["B"][i], axis=1)

    def _map_B(self, observations):
        """
        Required implementation for _map_B. Refer to _BaseHMM for more details.
        """
        self.B_map = np.ones((self.n_states, len(observations)))

        for j in range(self.n_states):
            for t, obs in enumerate(observations):
                for i, symbol in enumerate(obs):
                    if symbol == self.MISSING or (symbol is np.nan or symbol != symbol):
                        temp_symbol = np.argmax(
                            self.B[i][j]
                        )  # the maximum likelihood symbol for that state
                        self.B_map[j][t] *= self.B[i][j][temp_symbol]
                    else:
                        self.B_map[j][t] *= self.B[i][j][symbol]

    def _generate_sample_from_state(self, state):
        """
        Generates a random sample from a given component.
        Args:
            state (int) - index of the component to condition on
        Returns:
            X (array) - array of shape (n_emissions, ) containing a random sample
            from the emission distribution corresponding to a given state.
        """
        res = []
        for e in range(self.n_emissions):
            cdf = np.cumsum(self.B[e][state, :])
            res.append((cdf > np.random.rand()).argmax())
        return np.asarray(res)

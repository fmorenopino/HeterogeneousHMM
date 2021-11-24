"""
Created on Nov 22, 2019
@author: semese

This code is based on:
  - HMM implementation by guyz - https://github.com/guyz/HMM
  - hmmlearn by anntzer - https://github.com/hmmlearn/
  
For theoretical bases see:
 - L. R. Rabiner, 'A tutorial on hidden Markov models and selected applications
   in speech recognition,' in Proceedings of the IEEE, vol. 77, no. 2,
   pp. 257-286, Feb. 1989.
 - K.P. Murphy, 'Machine Learning: A Probabilistic Perspective', The MIT Press
   ©2012, ISBN:0262018020 9780262018029
"""

import numpy as np
from .base import BaseHMM
from .utils import check_if_attributes_set, normalise


class MultinomialHMM(BaseHMM):
    """Hidden Markov Model with multiple multinomial (discrete) emissions.

    :param n_states: number of hidden states in the model
    :type n_states: int
    :param n_emissions: number of distinct observation symbols per state
    :type n_emissions: int
    :param n_features: list of the number of possible observable symbols for each emission
    :type n_features: list
    :param tr_params: controls which parameters are updated in the training process; can contain any combination of 's' for starting probabilities (pi), 't' for transition matrix, and other characters for subclass-specific emission parameters, defaults to 'ste'
    :type tr_params: str, optional
    :param init_params: controls which parameters are initialised prior to training.  Can contain any combination of 's' for starting probabilities (pi), 't' for transition matrix, and other characters for subclass-specific emission parameters, defaults to 'ste'
    :type init_params: str, optional
    :param init_type: name of the initialisation  method to use for initialising the start, transition and emission matrices, defaults to 'random'
    :type init_type: str, optional
    :param pi_prior: float or array of shape (n_states, ) setting the parameters of the Dirichlet prior distribution for 'pi', defaults to 1.0
    :type pi_prior: float or array_like, optional
    :param pi: array of shape (n_states, ) giving the initial state occupation distribution 'pi'
    :type pi: array_like
    :param A_prior: float or array of shape (n_states, n_states), giving the parameters of the Dirichlet prior distribution for each row of the transition probabilities 'A', defaults to 1.0
    :type A_prior: float or array_like, optional
    :param A: array of shape (n_states, n_states) giving the matrix of transition probabilities between states
    :type A: array_like
    :param B: the probabilities of emitting a given symbol when in each state
    :type B: list
    :param missing: a value indicating what character indicates a missed observation in the observation sequences, defaults to np.nan
    :type missing: int or np.nan, optional
    :param nr_no_train_de: this number indicates the number of discrete  emissions whose Matrix Emission Probabilities are fixed and are not trained; it is important to to order the observed variables such that the ones whose emissions aren't trained are the last ones, defaults to 0
    :type nr_no_train_de: int, optional
    :param state_no_train_de: a state index for nr_no_train_de which shouldn't be updated; defaults to None, which means that the entire emission probability matrix for that discrete emission will be kept unchanged during training, otherwise the last state_no_train_de states won't be updated, defaults to None
    :type state_no_train_de: int, optional
    :param learning_rate: a value from [0,1), controlling how much the past values of the model parameters count when computing the new model parameters during training; defaults to 0.
    :type learning_rate: float, optional
    :param verbose: flag to be set to True if per-iteration convergence reports should be printed, defaults to True
    :type verbose: bool, optional
    """

    def __init__(
        self,
        n_states,
        n_emissions,
        n_features,
        tr_params='ste',
        init_params='ste',
        init_type='uniform',
        pi_prior=1.0,
        A_prior=1.0,
        missing=np.nan,
        nr_no_train_de=0,
        state_no_train_de=None,
        learning_rate=0.1,
        verbose=True,
    ):
        """Constructor method

        :raises ValueError: if init_type is not one of ('uniform', 'random')
        :raises TypeError: if n_features is not a list of length n_emissions
        """

        if len(n_features) != n_emissions:
            raise TypeError(
                'n_features must be one of length {}'.format(n_emissions))

        super().__init__(
            n_states,
            tr_params=tr_params,
            init_params=init_params,
            init_type=init_type,
            pi_prior=pi_prior,
            A_prior=A_prior,
            verbose=verbose,
            learning_rate=learning_rate,
        )
        self.n_emissions = n_emissions
        self.n_features = n_features
        self.MISSING = missing
        self.nr_no_train_de = nr_no_train_de
        self.state_no_train_de = state_no_train_de

    def __str__(self):
        """Function to allow directly printing the object.
        """
        temp = super().__str__()
        return temp + '\nB:\n' + str(self.B)

    # ----------------------------------------------------------------------- #
    #        Public methods. These are callable when using the class.         #
    # ----------------------------------------------------------------------- #
    @property
    def missing(self):
        """Getter for the missing value character in the data. 
        """
        return self.MISSING

    @missing.setter
    def missing(self, value):
        """Setter for the missing value character in the data. 

        :param value: a value indicating what character indicates a missed
            observation in the observation sequences
        :type value: int or np.nan
        """
        self.MISSING = value

    def get_n_fit_scalars_per_param(self):
        """Return a mapping of trainable model parameters names (as in ``self.tr_params``) to the number of corresponding scalar parameters that will actually be fitted.
        """
        ns = self.n_states
        ne = self.n_emissions
        nf = self.n_features
        nnt = self.nr_no_train_de
        return {
            's': ns,
            't': ns * ns,
            'e': sum(ns * (nf[i] - 1) for i in range(ne - nnt)) if self.state_no_train_de is None else sum(ns * (nf[i] - 1) for i in range(ne - nnt) if i != self.state_no_train_de),
        }

    # ----------------------------------------------------------------------- #
    #             Private methods. These are used internally only.            #
    # ----------------------------------------------------------------------- #
    def _init_model_params(self):
        """Initialises model parameters prior to fitting. Extends the base classes method. See _BaseHMM.py for more information.
        """
        super()._init_model_params()

        if 'e' in self.init_params:
            if self.init_type == 'uniform':
                if self.nr_no_train_de == 0:
                    self.B = [
                        np.full(
                            (self.n_states, self.n_features[i]), 1.0 / self.n_features[i])
                        for i in range(self.n_emissions)
                    ]
                else:
                    check_if_attributes_set(self, attr='e')
            else:
                if self.nr_no_train_de == 0:
                    self.B = [
                        np.random.rand(self.n_states, self.n_features[i])
                        for i in range(self.n_emissions)
                    ]
                    for i in range(self.n_emissions):
                        normalise(self.B[i], axis=1)

                else:
                    check_if_attributes_set(self, attr='e')

    def _initialise_sufficient_statistics(self):
        """Initialises sufficient statistics required for M-step. Extends the base classes method by adding the emission probability matrix. See _BaseHMM.py for more information.
        """
        stats = super()._initialise_sufficient_statistics()

        stats['B'] = {
            'numer': [
                np.zeros((self.n_states, self.n_features[i]))
                for i in range(self.n_emissions)
            ],
            'denom': [
                np.zeros((self.n_states, self.n_features[i]))
                for i in range(self.n_emissions)
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

        if 'e' in self.tr_params:
            B_new = self._reestimate_B(
                obs_seq, obs_stats['gamma'])
            for i in range(self.n_emissions):
                stats['B']['numer'][i] += B_new['numer'][i]
                stats['B']['denom'][i] += B_new['denom'][i]

    def _reestimate_B(self, obs_seq, gamma):
        """Re-estimation of the emission matrix (part of the 'M' step of Baum-Welch). Computes B_new = expected # times in state s_j with symbol v_k /expected # times in state s_j

        :param obs_seq: array of shape (n_samples, n_features)
                containing the observation samples
        :type obs_seq: np.array
        :param gamma: array of shape (n_samples, n_states)
        :return: the modified parts of the emission matrix
        :rtype: dict
        """
        B_new = {
            'numer': [
                np.zeros((self.n_states, self.n_features[i]))
                for i in range(self.n_emissions)
            ],
            'denom': [
                np.zeros((self.n_states, self.n_features[i]))
                for i in range(self.n_emissions)
            ],
        }

        for e in range(self.n_emissions):
            for j in range(self.n_states):
                for k in range(self.n_features[e]):
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
        """Performs the 'M' step of the Baum-Welch algorithm. Extends the base classes method. See _BaseHMM.py for more information.
        """
        new_model = super()._M_step(stats)

        if 'e' in self.tr_params:
            new_model['B'] = [
                (stats['B']['numer'][i] / stats['B']['denom'][i])
                for i in range(self.n_emissions)
            ]

        return new_model

    def _update_model(self, new_model):
        """ Updates the emission probability matrix. Extends the base classes method. See _BaseHMM.py for more information.
        """
        super()._update_model(new_model)

        if 'e' in self.tr_params:
            if self.state_no_train_de is None:
                for i in range(self.n_emissions - self.nr_no_train_de):
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

            for i in range(self.n_emissions):
                normalise(new_model['B'][i], axis=1)

    def _map_B(self, obs_seq):
        """Required implementation for _map_B. Refer to _BaseHMM for more details.
        """
        B_map = np.ones((self.n_states, len(obs_seq)))

        for j in range(self.n_states):
            for t, obs in enumerate(obs_seq):
                for i, symbol in enumerate(obs):
                    if symbol == self.MISSING or (symbol is np.nan or symbol != symbol):
                        # if the symbol is missing, use the maximum likelihood symbol for that state
                        temp_symbol = np.argmax(
                            self.B[i][j]
                        )
                        B_map[j][t] *= self.B[i][j][temp_symbol]
                    else:
                        B_map[j][t] *= self.B[i][j][symbol]
        return B_map

    def _generate_sample_from_state(self, state):
        """Required implementation for _generate_sample_from_state. Refer to _BaseHMM for more details.
        """

        res = []
        for e in range(self.n_emissions):
            cdf = np.cumsum(self.B[e][state, :])
            res.append((cdf > np.random.rand()).argmax())
        return np.asarray(res)

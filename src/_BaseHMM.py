# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Nov 20, 2019

@authors: semese, fmorenopino

This code is based on:
 - HMM implementation by guyz- https://github.com/guyz/HMM
 - HMM implementation by fmorenopino - https://github.com/fmorenopino/HMM_eb2
 - HMM implementation by anntzer - https://github.com/hmmlearn/

For theoretical bases see:
 - L. R. Rabiner, "A tutorial on hidden Markov models and selected applications
   in speech recognition," in Proceedings of the IEEE, vol. 77, no. 2,
   pp. 257-286, Feb. 1989.
 - K.P. Murphy, "Machine Learning: A Probabilistic Perspective", The MIT Press
   ©2012, ISBN:0262018020 9780262018029
"""

import numpy as np
from scipy.special import logsumexp
from multiprocessing import Pool
from .utils import (
    plot_log_likelihood_evolution,
    log_normalise,
    normalise,
    log_mask_zero,
    check_if_attributes_set,
)

#: Supported decoder algorithms.
DECODER_ALGORITHMS = frozenset(("viterbi", "map"))


class _BaseHMM(object):

    """
    Base class for Hidden Markov Models.
    Parameters:
        n_states (int) - number of hidden states in the model
        params (string, optional) - controls which parameters are updated in the
            training process.  Can contain any combination of 's' for starting
            probabilities (pi), 't' for transition matrix, and other characters
            for subclass-specific emission parameters. Defaults to all parameters.
        init_params (string, optional) - controls which parameters are initialised
            prior to training.  Can contain any combination of 's' for starting
            probabilities (pi), 't' for transition matrix, and other characters
            for subclass-specific emission parameters. Defaults to all parameters.
        init_type (string, optional) - name of the initialisation
                method to use for initialising the start and transition matrices;
        pi_prior (array, optional) - array of shape (n_states, ) setting the
            parameters of the Dirichlet prior distribution for 'pi'
        A_prior (array, optional) - array of shape (n_states, ),
            giving the parameters of the Dirichlet prior distribution for each
            row of the transition probabilities 'A'
        verbose (bool, optional) - flag to be set to True if per-iteration
            convergence reports should be printed
    Attributes:
        pi (array) - array of shape (n_states, ) giving the initial
            state occupation distribution 'pi'
        A (array) - array of shape (n_states, n_states) giving
            the matrix of transition probabilities between states A
    """

    def __init__(
        self,
        n_states,
        params="st",
        init_params="st",
        init_type="random",
        pi_prior=1.0,
        A_prior=1.0,
        learn_rate=0,
        verbose=False,
    ):
        """
        Class initialiser.
        Args:
        n_states (int) - number of hidden states in the model
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
        learn_rate (float, optional) - a value from [0,1), controlling how much
            the past values of the model parameters count when computing the new
            model parameters during training. By default it's 0.
        verbose (bool, optional) - flag to be set to True if per-iteration
            convergence reports should be printed
        """
        self.n_states = n_states
        self.params = params
        self.init_params = init_params
        self.init_type = init_type
        self.pi_prior = pi_prior
        self.A_prior = A_prior
        self.learn_rate = learn_rate
        self.verbose = verbose

    def __str__(self):
        """
            Function to allow directly printing the object.
        """
        return "Pi: " + str(self.pi) + "\nA:\n" + str(self.A)

    # ----------------------------------------------------------------------- #
    #        Public methods. These are callable when using the class.         #
    # ----------------------------------------------------------------------- #
    # Solution to Problem 1 - compute P(O|model)
    def forward(self, observations, B_map=None):
        """
        Forward-Backward procedure is used to efficiently calculate the probability
        of the observations, given the model - P(O|model)

        alpha_t(x) = P(O1...Ot,qt=Sx|model) - The probability of state x and the
         observation up to time t, given the model.

        The returned value is the log of the probability, i.e: the log likehood
        model, give the observation - logL(model|O).
        """
        if B_map is None:
            B_map = self._map_B(observations)
        alpha = self._calc_alpha(observations, B_map)
        return logsumexp(alpha[-1])

    def score(self, observation_sequences):
        """
        Compute the log probability under the model.
        Args:
            observation_sequences (list) - a list of ndarrays containing the
                observation sequences of different lengths
        Returns:
            log_likelihood (float) - log-likelihood of all the observation sequences
        """
        log_likelihood = 0.0
        for observations in observation_sequences:
            log_likelihood += self.forward(observations)
        return log_likelihood

    def score_samples(self, observation_sequences):
        """
        Compute the posterior probability for each state in the model.
        Args:
            observation_sequences (list) - a list of ndarrays containing the
                observation sequences of different lengths
        Returns:
            posteriors (list) - list of arrays of shape (n_samples, n_states)
                containing the state-membership probabilities for each
                sample in the observation sequences
        """
        posteriors = []
        for observations in observation_sequences:
            B_map = self._map_B(observations)
            posteriors.append(
                self._calc_gamma(
                    self._calc_alpha(observations, B_map),
                    self._calc_beta(observations, B_map),
                )
            )
        return posteriors

    # Solution to Problem 2 - finding the optimal state sequence associated with
    # the given observation sequence -> Viterbi, MAP
    def decode(self, observation_sequences, algorithm="viterbi"):
        """
        Find the best state sequence (path), given the model and an observation.
         i.e: max(P(Q|O,model)).
        This method is usually used to predict the next state after training.
        Args:
            observation_sequences (list) - a list of ndarrays containing the observation
                sequences of different lengths
            algorithm (string, optional) - name of the decoder algorithm to use;
                Must be one of "viterbi" or "map". Defaults to "viterbi".
        Returns:
            log_likelihood (float) - log probability of the produced state sequence
            state_sequences (list) - list of arrays containing labels for each
                observation from observation_sequences obtained via the given
                decoder algorithm
        """
        if algorithm not in DECODER_ALGORITHMS:
            raise ValueError("Unknown decoder {!r}".format(algorithm))

        decoder = {"viterbi": self._decode_viterbi, "map": self._decode_map}[algorithm]

        log_likelihood = 0.0
        state_sequences = []
        for observations in observation_sequences:
            log_likelihood_, state_sequence_ = decoder(observations)
            log_likelihood += log_likelihood_
            state_sequences.append(state_sequence_)

        return log_likelihood, state_sequences

    # Solution to Problem 3 - adjust the model parameters to maximise P(O,model)
    def train(
        self,
        observation_sequences,
        n_init=5,
        n_iter=100,
        thres=0.1,
        conv_iter=5,
        plot_log_likelihood=False,
        ignore_conv_crit=False,
        no_init=False,
        n_processes=None,
        print_every=1,
    ):
        """
        Updates the HMMs parameters given a new set of observed sequences.
        The observations can either be a single (1D) array of observed symbols, or when using
        a continuous HMM, a 2D array (matrix), where each row denotes a multivariate
        time sample (multiple features).
        The model parameters are reinitialised 'n_init' times. For each initialisation the
        updated model parameters and the log likelihood is stored and the best model is selected
        at the end.
        Args:
            observation_sequences (list) - a list of ndarrays containing the observation
                sequences of different lengths
            n_init (int) - number of initialisations to perform
            n_iter (int) - max number of iterations to run for each initialisation
            thres (float) - the threshold for the likelihood increase (convergence)
            plot_log_likelihood (boolean) - parameter to activate plotting the evolution
                of the log-likelihood after each initialisation
            ignore_conv_crit (boolean) - flag to indicate whether to iterate until
                n_iter is reached or perform early stopping
            no_init (boolean) - flag to indicate wheather to initialise the Parameters
                before training (it can only be True if the parameters are manually
                set before or they were already trained)
            n_processes (int) - number of processes to use if the training should
                be performed using parallelisation
            print_every (int) - if verbose is True, print progress info every
                'print_every' iterations
        Returns:
                self (object) - the updated model
                max(log_likelihoods) (float) - the log_likelihood of the best model
        """

        # lists to temporarily save the new model parameters and corresponding log likelihoods
        new_models = []
        log_likelihoods = []

        for init in range(n_init):
            if self.verbose:
                print("Initialisation " + str(init + 1))

            new_model, log_likelihood = self._train(
                observation_sequences,
                n_processes=n_processes,
                n_iter=n_iter,
                thres=thres,
                conv_iter=conv_iter,
                plot_log_likelihood=plot_log_likelihood,
                ignore_conv_crit=ignore_conv_crit,
                print_every=print_every,
                no_init=no_init
            )
            new_models.append(new_model)
            log_likelihoods.append(log_likelihood)

        # select best model (the one that had the largest log_likelihood) and update the model
        best_index = log_likelihoods.index(max(log_likelihoods))
        self._update_model(new_models[best_index])

        return self, max(log_likelihoods)

    def sample(self, n_sequences=1, n_samples=1):
        """
        Generate random samples from the model.
        Args:
            n_sequences (int,optional) - number of sequences to generate; by
                default it generates one sequence
            n_samples (int or list) - number of samples per sequence; if multiple
                sequences have to be generated, it is a list of the individual
                sequence lengths
        Returns:
            samples (list) - a list containing one or n_sequences sample sequences
            state_sequences (list) - a list containing the state sequences that
                generated each sample sequence
        """
        samples = []
        state_sequences = []

        startprob_cdf = np.cumsum(self.pi)
        transmat_cdf = np.cumsum(self.A, axis=1)

        for ns in range(n_sequences):
            currstate = (startprob_cdf > np.random.rand()).argmax()
            state_sequence = [currstate]
            X = [self._generate_sample_from_state(currstate)]

            for t in range(n_samples - 1):
                currstate = (transmat_cdf[currstate] > np.random.rand()).argmax()
                state_sequence.append(currstate)
                X.append(self._generate_sample_from_state(currstate))
            samples.append(X)
            state_sequences.append(state_sequence)

        return samples, state_sequences

    def get_stationary_distribution(self):
        """
            Compute the stationary distribution of states.
        """
        # The stationary distribution is proportional to the left-eigenvector
        # associated with the largest eigenvalue (i.e., 1) of the transition
        # matrix.
        eigvals, eigvecs = np.linalg.eig(self.A.T)
        eigvec = np.real_if_close(eigvecs[:, np.argmax(eigvals)])
        return eigvec / eigvec.sum()

    # ----------------------------------------------------------------------- #
    #             Private methods. These are used internally only.            #
    # ----------------------------------------------------------------------- #
    def _init(self, X=None):
        """
        Initialises model parameters prior to fitting. If init_type if random,
        it samples from a Dirichlet distribution according to the given priors.
        Otherwise it initialises the starting probabilities and transition
        probabilities uniformly.
        Args:
            X (list, optional) - list of concatenated observations, only needed
                for the Gaussian model when random or kmeans is used to initialise
                the means and covars.
        """
        if self.init_type == "uniform":
            init = 1.0 / self.n_states
            if "s" in self.init_params:
                self.pi = np.full(self.n_states, init)

            if "t" in self.init_params:
                self.A = np.full((self.n_states, self.n_states), init)
        else:
            if "s" in self.init_params:
                self.pi = np.random.dirichlet(
                    alpha=self.pi_prior * np.ones(self.n_states), size=1
                )[0]

            if "t" in self.init_params:
                self.A = np.random.dirichlet(
                    alpha=self.A_prior * np.ones(self.n_states), size=self.n_states
                )

    def _decode_map(self, observations):
        """
        Find the best state sequence (path) using MAP.
        Args:
             observations (array) - array of shape (n_samples, n_features)
                containing the observation sequence
        Returns:
             state_sequence (array) - the optimal path for the observation sequence
             log_likelihood (float) - the maximum probability for the entire sequence
        """
        posteriors = self.score_samples([observations])[0]
        log_likelihood = np.max(posteriors, axis=1).sum()
        state_sequence = np.argmax(posteriors, axis=1)
        return log_likelihood, state_sequence

    def _decode_viterbi(self, observations):
        """
        Find the best state sequence (path) using viterbi algorithm - a method
        of dynamic programming, very similar to the forward-backward algorithm,
        with the added step of maximisation and eventual backtracing.

        Args:
             observations (array) - array of shape (n_samples, n_features)
                containing the observation sequence
        Returns:
             state_sequence (array) - the optimal path for the observation sequence
             log_likelihood (float) - the maximum log-probability for the entire sequence
        """
        n_samples = len(observations)

        # similar to the forward-backward algorithm, we need to make sure that
        # we're using fresh data for the given observations
        B_map = self._map_B(observations)

        log_pi = log_mask_zero(self.pi)
        log_A = log_mask_zero(self.A)
        log_B_map = log_mask_zero(B_map)

        # delta[t][i] = max(P[q1..qt=i,O1...Ot|model] - the path ending in Si and
        # until time t, that generates the highest probability.
        delta = np.zeros((n_samples, self.n_states))

        # init
        for x in range(self.n_states):
            delta[0][x] = log_pi[x] + log_B_map[x][0]

        # induction
        work_buffer = np.empty(self.n_states)
        for t in range(1, n_samples):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    work_buffer[j] = log_A[j][i] + delta[t - 1][j]
                delta[t][i] = np.amax(work_buffer) + log_B_map[i][t]

        # Observation traceback
        state_sequence = np.empty(n_samples, dtype=np.int32)
        state_sequence[n_samples - 1] = where_from = np.argmax(delta[n_samples - 1])
        log_likelihood = delta[n_samples - 1, where_from]

        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                work_buffer[i] = delta[t, i] + log_A[i, where_from]
            state_sequence[t] = where_from = np.argmax(work_buffer)

        return log_likelihood, state_sequence

    def _calc_alpha(self, observations, B_map):
        """
        Calculates 'alpha' the forward variable given an observation sequence.

        Args:
            observations (array) - array of shape (n_samples, n_features)
                containing the observation samples
            B_map (array) - the observations' mass/density Bj(Ot) to Bj(t)
        Returns:
            alpha (array) - array of shape (n_samples, n_states) containing
                the forward variables
        """
        n_samples = len(observations)

        # The alpha variable is a np array indexed by time, then state (TxN).
        # alpha[t][i] = the probability of being in state 'i' after observing the
        # first t symbols.
        alpha = np.zeros((n_samples, self.n_states))
        log_pi = log_mask_zero(self.pi)
        log_A = log_mask_zero(self.A)
        log_B_map = log_mask_zero(B_map)

        # init stage - alpha_1(i) = pi(i)b_i(o_1)
        for i in range(self.n_states):
            alpha[0][i] = log_pi[i] + log_B_map[i][0]

        # induction
        work_buffer = np.zeros(self.n_states)
        for t in range(1, n_samples):
            for j in range(self.n_states):
                for i in range(self.n_states):
                    work_buffer[i] = alpha[t - 1][i] + log_A[i][j]
                alpha[t][j] = logsumexp(work_buffer) + log_B_map[j][t]

        return alpha

    def _calc_beta(self, observations, B_map):
        """
        Calculates 'beta' the backward variable for each observation sequence.
        Args:
            observations (array) - array of shape (n_samples, n_features)
                containing the observation samples
            B_map (array) - the observations' mass/density Bj(Ot) to Bj(t)
        Returns:
            beta (array) - array of shape (n_samples, n_states) containing
                the backward variables
        """
        n_samples = len(observations)

        # The beta variable is a ndarray indexed by time, then state (TxN).
        # beta[t][i] = the probability of being in state 'i' and then observing the
        # symbols from t+1 to the end (T).
        beta = np.zeros((n_samples, self.n_states))

        log_A = log_mask_zero(self.A)
        log_B_map = log_mask_zero(B_map)

        # init stage
        for i in range(self.n_states):
            beta[len(observations) - 1][i] = 0.0

        # induction
        work_buffer = np.zeros(self.n_states)
        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    work_buffer[j] = log_A[i][j] + log_B_map[j][t + 1] + beta[t + 1][j]
                    beta[t][i] = logsumexp(work_buffer)

        return beta

    def _calc_xi(
        self, observations, B_map=None, alpha=None, norm_coeffs=None, beta=None
    ):
        """
        Calculates 'xi', a joint probability from the 'alpha' and 'beta' variables.

        Args:
            observations (array) - array of shape (n_samples, n_features)
                containing the observation samples
            B_map (array, optional) - the observations' mass/density Bj(Ot) to Bj(t)
            alpha (array, optional) - array of shape (n_samples, n_states)
                containing the forward variables
            beta (array, optional) - array of shape (n_samples, n_states)
                containing the backward variables
        Returns:
            xi (array) - array of shape (n_samples, n_states, n_states) conatining
                the a joint probability from the 'alpha' and 'beta' variables.

        """
        if B_map is None:
            B_map = self._map_B(observations)
        if alpha is None:
            alpha = self._calc_alpha(observations, B_map)
        if beta is None:
            beta = self._calc_beta(observations, B_map)

        n_samples = len(observations)

        # The xi variable is a np array indexed by time, state, and state (TxNxN).
        # xi[t][i][j] = the probability of being in state 'i' at time 't', and 'j' at
        # time 't+1' given the entire observation sequence.
        log_xi_sum = np.zeros((self.n_states, self.n_states))
        work_buffer = np.full((self.n_states, self.n_states), -np.inf)

        # compute the logarithm of the parameters
        log_A = log_mask_zero(self.A)
        log_B_map = log_mask_zero(B_map)
        logprob = logsumexp(alpha[n_samples - 1])

        for t in range(n_samples - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    work_buffer[i, j] = (
                        alpha[t][i]
                        + log_A[i][j]
                        + log_B_map[j][t + 1]
                        + beta[t + 1][j]
                        - logprob
                    )

            for i in range(self.n_states):
                for j in range(self.n_states):
                    log_xi_sum[i][j] = np.logaddexp(log_xi_sum[i][j], work_buffer[i][j])

        return log_xi_sum

    def _calc_gamma(self, alpha, beta):
        """
        Calculates 'gamma' from 'alpha' and 'beta'.

        Args:
            alpha (array) - array of shape (n_samples, n_states) containing the forward variables
            beta (array) - array of shape (n_samples, n_states) containing the backward variables
        Returns:
            gamma (array) - array of shape (n_samples, n_states), the posteriors
        """
        log_gamma = alpha + beta
        log_normalise(log_gamma, axis=1)
        with np.errstate(under="ignore"):
            return np.exp(log_gamma)

    # Methods used by self.train()
    def _train(
        self,
        observation_sequences,
        n_iter=100,
        thres=0.1,
        conv_iter=5,
        plot_log_likelihood=False,
        ignore_conv_crit=False,
        return_log_likelihoods=False,
        no_init=False,
        print_every=1,
        n_processes=None,
    ):
        """
        Training is repeated 'n_iter' times, or until log likelihood of the model
        increases by less than a threshold.

        Args:
            observation_sequences (list) - a list of ndarrays/lists containing
                the observation sequences. Each sequence can be the same or of
                different lengths.
            n_iter (int, optional) - max number of iterations to perform.
                Deafult n_iter = 100.
            thres (float, optional) - the convergence threshold for the likelihood
                increase. Default: thres = 0.1
            plot_log_likelihood (boolean, optional) - flag to indicate whether
                to plot the evolution of the log-likelihood over the iterations.
                Default is False (no plotting).
            ignore_conv_crit (boolean, optional) - flag to indicate whether to
                iterate until n_iter is reached or stop when the convergence criterium
                is met. Default is False (consider convergence criteria).
            return_log_likelihoods (boolean, optional) - indicates if the log-
                likelihoods over the iterations should be returned or not. Default
                is False (don't return). Only used for testing!!!
            no_init (boolean, optional) - indicates whether to re-initialise the
                model parameters or not. It can only be set to True, if the
                parameters have already been trained or manually set before.
            n_processes (int, optional) - if not None, multiprocessing is used during
                training with n_processes parallel processes
        Returns:
            new_model (dictionary) - containing the updated model parameters
            if return_log_likelihoods is True then
                log_likelihood_iter (list, optional) -  the log-likelihood values
                    from each iteration. Only returned if return_log_likelihoods = True.
            else
                curr_log_likelihood (float) - the accumulated log-likelihood for all
                    the observations
        """
        if not no_init:
            if self.init_type in ("kmeans", "random"):
                self._init(X=observation_sequences)
            else:
                self._init()
        else:
            check_if_attributes_set(self)

        log_likelihood_iter = []
        old_log_likelihood = np.nan
        for it in range(n_iter):

            # if train without multiprocessing
            if n_processes is None:
                stats, curr_log_likelihood = self._compute_intermediate_values(
                    observation_sequences
                )
            else:
                # split up observation sequences between the processes
                n_splits = int(np.ceil(len(observation_sequences) / n_processes))
                split_list = [
                    sl
                    for sl in list(
                        (
                            observation_sequences[
                                i * n_splits : i * n_splits + n_splits
                            ]
                            for i in range(n_processes)
                        )
                    )
                    if sl
                ]
                # create pool of processes
                p = Pool(processes=n_processes)
                stats_list = p.map(
                    self._compute_intermediate_values,
                    [split_i for split_i in split_list],
                )
                p.close()
                stats, curr_log_likelihood = self._sum_up_suffcient_statistics(
                    stats_list
                )

            # perform the M-step to update the model parameters
            new_model = self._M_step(stats)
            self._update_model(new_model)

            if self.verbose and it % print_every == 0:
                print(
                    "iter: {}, log_likelihood = {}, delta = {}".format(
                        it,
                        curr_log_likelihood,
                        (curr_log_likelihood - old_log_likelihood),
                    )
                )

            if not ignore_conv_crit:
                if (
                    abs(curr_log_likelihood - old_log_likelihood)
                    / abs(old_log_likelihood)
                    <= thres
                ):
                    counter += 1
                    if counter == conv_iter:
                        # converged
                        if self.verbose:
                            print(
                                "Converged -> iter: {}, log_likelihood = {}".format(
                                    it, curr_log_likelihood
                                )
                            )
                        break
                else:
                    counter = 0

            log_likelihood_iter.append(curr_log_likelihood)
            old_log_likelihood = curr_log_likelihood

        if counter < conv_iter and self.verbose:
            # max_iter reached
            print(
                "Maximum number of iterations reached. log_likelihood = {}".format(
                    curr_log_likelihood
                )
            )

        if plot_log_likelihood:
            plot_log_likelihood_evolution(log_likelihood_iter)

        if return_log_likelihoods:  # this is just for testing
            return new_model, log_likelihood_iter
        else:
            return new_model, curr_log_likelihood

    def _compute_intermediate_values(self, observation_sequences):
        """
        Calculates the various intermediate values for the Baum-Welch on a list
        of observation sequences.
        Args:
            observation_sequences (list) - a list of ndarrays/lists containing
                the observation sequences. Each sequence can be the same or of
                different lengths.
        Returns:
            stats (dictionary) - dictionary of sufficient statistics required
                for the M-step
        """
        stats = self._initialise_sufficient_statistics()
        curr_log_likelihood = 0

        for observations in observation_sequences:
            B_map = self._map_B(observations)

            # calculate the log likelihood of the previous model
            # we compute the P(O|model) for the set of old parameters
            log_likelihood = self.forward(observations, B_map)
            curr_log_likelihood += log_likelihood

            # do the E-step of the Baum-Welch algorithm
            observations_stats = self._E_step(observations, B_map)

            # accumulate stats
            self._accumulate_sufficient_statistics(
                stats, observations_stats, observations
            )

        return stats, curr_log_likelihood

    def _E_step(self, observations, B_map):
        """
        Calculates required statistics of the current model, as part
        of the Baum-Welch 'E' step.

        Deriving classes should override (extend) this method to include
        any additional computations their model requires.

        Returns:
            observations_stats (dictionary) - containing required statistics
        """

        # compute the parameters for the observation
        # compute the parameters for the observation
        observations_stats = {
            "alpha": self._calc_alpha(observations, B_map),
            "beta": self._calc_beta(observations, B_map),
        }

        observations_stats["xi"] = self._calc_xi(
            observations,
            B_map=B_map,
            alpha=observations_stats["alpha"],
            beta=observations_stats["beta"],
        )
        observations_stats["gamma"] = self._calc_gamma(
            observations_stats["alpha"], observations_stats["beta"]
        )
        return observations_stats

    def _M_step(self, stats):
        """
        Performs the 'M' step of the Baum-Welch algorithm.
        Deriving classes should override (extend) this method to include
        any additional computations their model requires.
        Args:
            stats (dictionary) - containing the accumulated statistics
        Returns:
            new_model (dictionary) - containing the updated model parameters
        """
        new_model = {}

        if "s" in self.params:
            pi_ = np.maximum(self.pi_prior - 1 + stats["pi"], 0)
            new_model["pi"] = np.where(self.pi == 0, 0, pi_)
            normalise(new_model["pi"])

        if "t" in self.params:
            A_ = np.maximum(self.A_prior - 1 + stats["A"], 0)
            new_model["A"] = np.where(self.A == 0, 0, A_)
            normalise(new_model["A"], axis=1)

        return new_model

    def _update_model(self, new_model):
        """
        Replaces the current model parameters with the new ones.
        Args:
            new_model (dictionary) - contains the new model parameters
        """
        if "s" in self.params:
            self.pi = (1 - self.learn_rate) * new_model[
                "pi"
            ] + self.learn_rate * self.pi

        if "t" in self.params:
            self.A = (1 - self.learn_rate) * new_model["A"] + self.learn_rate * self.A

    def _initialise_sufficient_statistics(self):
        """
        Initialises sufficient statistics required for M-step.
        Returns:
            stats (dictionary) - having as key-value pairs:
                nobs (int) - number of samples in the data
                start (array) - array of shape (n_components, ) where the i-th
                    element corresponds to the posterior probability of the first
                    sample being generated by the i-th state
                trans (dictionary) - containing the numerator and denominator
                    parts of the new A matrix as arrays of shape (n_components, n_components)
        """
        stats = {
            "nobs": 0,
            "pi": np.zeros(self.n_states),
            "A": np.zeros((self.n_states, self.n_states)),
        }
        return stats

    def _accumulate_sufficient_statistics(
        self, stats, observations_stats, observations
    ):
        """
        Updates sufficient statistics from a given sample.
        Args:
            stats (dictionary) - containing the sufficient statistics for all
                observation sequences
            observations_stats (dictionary) - containing the sufficient statistic for one sample
        """
        stats["nobs"] += 1
        if "s" in self.params:
            stats["pi"] += observations_stats["gamma"][0]

        if "t" in self.params:
            with np.errstate(under="ignore"):
                stats["A"] += np.exp(observations_stats["xi"])

    def _sum_up_suffcient_statistics(self, stats_list):
        """
        Sums sufficient statistics from a given sub-set of observation sequences.
        Args:
            stats_list (list) - list containing the sufficient statistics from the
                different processes
        Returns:
            stats_all (dictionary) - dictionary of sufficient statistics
        """
        stats_all = self._initialise_sufficient_statistics()
        logL_all = 0
        for (stat_i, logL_i) in stats_list:
            logL_all += logL_i
            for stat in stat_i.keys():
                if isinstance(stat_i[stat], dict):
                    for i in range(len(stats_all[stat]["numer"])):
                        stats_all[stat]["numer"][i] += stat_i[stat]["numer"][i]
                        stats_all[stat]["denom"][i] += stat_i[stat]["denom"][i]
                else:
                    stats_all[stat] += stat_i[stat]
        return stats_all, logL_all

    # Methods that have to be implemented in the deriving classes
    def _map_B(self, observations):
        """
        Deriving classes should implement this method, so that it maps the
        observations' mass/density Bj(Ot) to Bj(t).
        This method has no explicit return value, but it expects that 'B_map'
         is internally computed as mentioned above.
        'B_map' is an (TxN) numpy array.
        The purpose of this method is to create a common parameter that will
        conform both to the discrete case where PMFs are used, and the continuous
        case where PDFs are used.
        For the continuous case, since PDFs of vectors could be computationally
        expensive (Matrix multiplications), this method also serves as a caching
         mechanism to significantly increase performance.
        """
        raise NotImplementedError(
            "a mapping function for B(observable probabilities) must be implemented"
        )

    def _generate_sample_from_state(self, state):
        """
        Generates a random sample from a given component.
        Args:
            state (int) - index of the component to condition on
        Returns:
            X (array) - array of shape (n_features, ) containing a random sample
            from the emission distribution corresponding to a given state.
        """

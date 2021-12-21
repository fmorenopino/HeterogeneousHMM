"""
Created on Nov 20, 2019
@authors: semese, fmorenopino

This code is based on:
 - HMM implementation by guyz- https://github.com/guyz/HMM
 - HMM implementation by fmorenopino - https://github.com/fmorenopino/HMM_eb2
 - HMM implementation by anntzer - https://github.com/hmmlearn/
 
For theoretical bases see:
 - L. R. Rabiner, A tutorial on hidden Markov models and selected applications
   in speech recognition, in Proceedings of the IEEE, vol. 77, no. 2,
   pp. 257-286, Feb. 1989.
 - K.P. Murphy, Machine Learning: A Probabilistic Perspective, The MIT Press
   ©2012, ISBN:0262018020 9780262018029
"""

import numpy as np
from scipy.special import logsumexp
from multiprocessing import Pool

from .utils import (
    plot_log_likelihood_evolution,
    check_if_attributes_set,
    log_normalise,
    normalise,
    log_mask_zero,
)


# Supported decoder algorithms.
DECODER_ALGORITHMS = frozenset(('viterbi', 'map'))
# Supported initialisation methods.
INIT_TYPES = frozenset(('uniform', 'random', 'kmeans'))


class BaseHMM(object):
    """Base class for the Hidden Markov Models. It allows for training evaluation and sampling from the HMM. 

    :param n_states: number of hidden states in the model
    :type n_states: int, optional
    :param tr_params: controls which parameters are updated in the
            training process. Can contain any combination of 's' for starting
            probabilities (pi), 't' for transition matrix, and other characters
            for subclass-specific emission parameters. Defaults to all parameters.
    :type tr_params: str, optional
    :param init_params: controls which parameters are initialised
            prior to training.  Can contain any combination of 's' for starting
            probabilities (pi), 't' for transition matrix, and other characters
            for subclass-specific emission parameters. Defaults to all parameters.
    :type init_params: str, optional
    :param init_type: controls whether pi and A are initialised from a Dirichlet or a 
        uniform distribution. Defaults to 'uniform'.
    :type init_type: str, optional
    :param pi_prior: array of shape (n_states, ) setting the parameters of the 
        Dirichlet prior distribution for the starting probabilities. Defaults to 1.
    :type pi_prior: array_like, optional 
    :param A_prior: array of shape (n_states, ), giving the parameters of the Dirichlet 
        prior distribution for each row of the transition probabilities 'A'. Defaults to 1.
    :param pi: array of shape (n_states, ) giving the initial state
        occupation distribution 'pi'
    :type pi: array_like 
    :type A_prior: array_like, optional 
    :param A: array of shape (n_states, n_states) giving the matrix of
            transition probabilities between states
    :type A: array_like
    :param learning_rate: a value from [0,1), controlling how much
        the past values of the model parameters count when computing the new
        model parameters during training; defaults to 0.
    :type learning_rate: float, optional
    :param verbose: flag to be set to True if per-iteration convergence reports should 
        be printed. Defaults to True.
    :type verbose: bool, optional 
    """

    def __init__(
        self,
        n_states,
        tr_params='st',
        init_params='st',
        init_type='uniform',
        pi_prior=1.0,
        A_prior=1.0,
        learning_rate=0.,
        verbose=True,
    ):
        """Constructor method."""

        if init_type not in INIT_TYPES:
            raise ValueError('init_type must be one of {}'.format(INIT_TYPES))

        self.n_states = n_states
        self.tr_params = tr_params
        self.init_params = init_params
        self.init_type = init_type
        self.pi_prior = pi_prior
        self.A_prior = A_prior
        self.learning_rate = learning_rate
        self.verbose = verbose

    def __str__(self):
        """Function to allow directly printing the object."""
        return 'Pi: ' + str(self.pi) + '\nA:\n' + str(self.A)

    # ----------------------------------------------------------------------- #
    #        Public methods. These are callable when using the class.         #
    # ----------------------------------------------------------------------- #
    # Solution to Problem 1 - compute P(O|model)
    def forward(self, obs_seq, B_map=None):
        """Forward-Backward procedure is used to efficiently calculate the probability of the observations, given the model - P(O|model).

        :param obs_seq: an observation sequence 
        :type obs_seq: array_like
        :param B_map: mapping of the observations' mass/density Bj(Ot) to Bj(t)
        :type B_map: array_like, optional
        :return: the log of the probability, i.e. the log likehood model, give the 
            observation - logL(model|O).
        :rtype: float
        """
        if B_map is None:
            # if the emission probabilies not given, compute
            B_map = self._map_B(obs_seq)

        # alpha_t(x) = P(O1...Ot,qt=Sx|model) - The probability of state x and the
        # observation up to time t, given the model.
        alpha = self._calc_alpha(obs_seq, B_map)

        return logsumexp(alpha[-1])

    def score(self, obs_sequences):
        """Compute the per-sample average log-likelihood of the given data.

        :param obs_sequences: a list of ndarrays containing the
                observation sequences of different lengths
        :type obs_sequences: list
        :return: the average of log-likelihoods over all the observation sequences
        :rtype: float
        """
        return np.mean(self.score_samples(obs_sequences))

    def score_samples(self, obs_sequences):
        """Compute the log-likelihood of each sample.

        :param obs_sequences: a list of ndarrays containing the
                observation sequences of different lengths
        :type obs_sequences: list
        :return: list of log-likelihoods over all the observation sequences
        :rtype: list
        """
        return [self.forward(obs_seq) for obs_seq in obs_sequences]

    def predict(self, X, algorithm='viterbi'):
        """Find the most probable state sequence corresponding to X.

        :param X: feature matrix of individual samples, shape (n_samples, n_features)
        :type X: array-like
        :param algorithm: name of the decoder algorithm to use;
            must be one of 'viterbi' or 'map'. Defaults to 'viterbi'.
        :type algorithm: string, optional
        :return: labels for each sample from X
        :rtype: array-like
        """
        _, state_sequence = self.decode(X, algorithm)
        return state_sequence

    def predict_proba(self, obs_sequences):
        """Compute the posterior probability for each state in the model.

        :param obs_sequences: a list of ndarrays containing the
                observation sequences of different lengths
        :type obs_sequences: list
        :return: list of arrays of shape (n_samples, n_states)
                containing the state-membership probabilities for each
                sample in the observation sequences
        :rtype: list
        """
        posteriors = []
        for obs_seq in obs_sequences:
            B_map = self._map_B(obs_seq)
            posteriors.append(
                self._calc_gamma(
                    self._calc_alpha(obs_seq, B_map),
                    self._calc_beta(obs_seq, B_map),
                )
            )
        return posteriors

    # Solution to Problem 2 - finding the optimal state sequence associated with
    # the given observation sequence -> Viterbi, MAP
    def decode(self, obs_sequences, algorithm='viterbi'):
        """Find the best state sequence (path), given the model and an observation.
         i.e: max(P(Q|O,model)).

        :param obs_sequences: a list of ndarrays containing the
            observation sequences of different lengths
        :type obs_sequences: list
        :param algorithm: name of the decoder algorithm to use;
            must be one of 'viterbi' or 'map'. Defaults to 'viterbi'.
        :type algorithm: string, optional
        :return: log-probability of the produced state sequence
        :rtype: float
        :return: list of arrays of shape (n_samples, n_states) containing labels for each
            observation from obs_sequences obtained via the given
            decoder algorithm
        :rtype: list
        """
        if algorithm not in DECODER_ALGORITHMS:
            raise ValueError('Unknown decoder {!r}'.format(algorithm))

        decoder = {'viterbi': self._decode_viterbi,
                   'map': self._decode_map}[algorithm]

        log_likelihood = 0.0
        state_sequences = []
        for obs_seq in obs_sequences:
            logL, state_seq = decoder(obs_seq)
            log_likelihood += logL
            state_sequences.append(state_seq)

        return log_likelihood, state_sequences

    # Solution to Problem 3 - adjust the model parameters to maximise P(O,model)
    def train(
        self,
        obs_sequences,
        n_init=1,
        n_iter=100,
        conv_thresh=0.1,
        conv_iter=5,
        ignore_conv_crit=False,
        plot_log_likelihood=False,
        no_init=False,
        n_processes=None,
        print_every=1,
    ):
        """Updates the HMMs parameters given a new set of observed sequences.
        The observations can either be a single (1D) array of observed symbols, or a 2D array (matrix), where each row denotes a multivariate time sample (multiple features). The model parameters are reinitialised 'n_init' times. For each initialisation the updated model parameters and the log-likelihood is stored and the best model is selected at the end.

        :param obs_sequences: a list of arrays containing the observation
                sequences of different lengths
        :type obs_sequences: list
        :param n_init: number of initialisations to perform; defaults to 1
        :type n_init: int, optional
        :param n_iter: max number of iterations to run for each initialisation; defaults to 100
        :type n_iter: int, optional
        :param conv_thresh: the threshold for the likelihood increase (convergence); defaults to 0.1
        :type conv_thresh: float, optional
        :param conv_iter: number of iterations for which the convergence criteria has to hold before 
            early-stopping; defaults to 5
        :type conv_iter: int, optional
        :param ignore_conv_crit: flag to indicate whether to iterate until
                n_iter is reached or perform early-stopping; defaults to False
        :type ignore_conv_crit: bool, optional    
        :param plot_log_likelihood: parameter to activate plotting the evolution
                of the log-likelihood after each initialisation; defaults to False
        :type plot_log_likelihood: bool, optional  
        :param no_init: flag to indicate wheather to initialise the Parameters
                before training (it can only be True if the parameters are manually
                set before or they were already trained); defaults to False
        :type no_init: bool, optional    
        :param n_processes: number of processes to use if the training should
                be performed using parallelisation; defaults to None
        :type n_processes: int, optional
        :param print_every: if verbose is True, print progress info every
                'print_every' iterations; defaults to 1
        :type print_every: int, optional
        :return: the updated model
        :rtype: object
        :return: the log_likelihood of the best model
        :rtype: float
        """

        # lists to temporarily save the new model parameters and corresponding log likelihoods
        new_models = []
        log_likelihoods = []
        keyboard_interrupted = False

        try:
            for init in range(n_init):
                if self.verbose:
                    print('Initialisation ' + str(init + 1))

                n_model, logL = self._train(
                    obs_sequences,
                    n_processes=n_processes,
                    n_iter=n_iter,
                    conv_thresh=conv_thresh,
                    conv_iter=conv_iter,
                    plot_log_likelihood=plot_log_likelihood,
                    ignore_conv_crit=ignore_conv_crit,
                    print_every=print_every,
                    no_init=no_init
                )
                new_models.append(n_model)
                log_likelihoods.append(logL)
        except KeyboardInterrupt:
            print('[Ctrl + C] Interrupted')
            keyboard_interrupted = True

        # select best model (the one that had the largest log_likelihood) and update the model
        if not keyboard_interrupted:
            best_index = log_likelihoods.index(max(log_likelihoods))
            self._update_model(new_models[best_index])

        return self, max(log_likelihoods)

    def sample(self, n_sequences=1, n_samples=1, return_states=False):
        """Generate random samples from the model.

        :param n_sequences: number of sequences to generate, defeaults to 1.
        :type n_sequences: int, optional
        :param n_samples: number of samples to generate per sequence; defeaults to 1. If multiple
                sequences have to be generated, it is a list of the individual
                sequence lengths
        :type n_samples: int, optional
        :param return_states: if True, the method returns the state sequence from which the samples
            were generated, defeaults to False.
        :type return_states: bool, optional
        :return: a list containing one or n_sequences sample sequences
        :rtype: list
        :return: a list containing the state sequences that
                generated each sample sequence
        :rtype: list
        """
        samples = []
        state_sequences = []

        startprob_cdf = np.cumsum(self.pi)
        transmat_cdf = np.cumsum(self.A, axis=1)

        for _ in range(n_sequences):
            currstate = (startprob_cdf > np.random.rand()).argmax()
            state_sequence = [currstate]
            X = [self._generate_sample_from_state(currstate)]

            for _ in range(n_samples - 1):
                currstate = (transmat_cdf[currstate]
                             > np.random.rand()).argmax()
                state_sequence.append(currstate)
                X.append(self._generate_sample_from_state(currstate))
            samples.append(np.vstack(X))
            state_sequences.append(state_sequence)

        if return_states:
            return samples, state_sequences
        return samples

    def get_stationary_distribution(self):
        """Compute the stationary distribution of states. The stationary distribution is proportional to the left-eigenvector associated with the largest eigenvalue (i.e., 1) of the transition matrix.

        :return: the stationary distribution of states
        :rtype: array_like
        """
        eigvals, eigvecs = np.linalg.eig(self.A.T)
        eigvec = np.real_if_close(eigvecs[:, np.argmax(eigvals)])
        return eigvec / eigvec.sum()

    # ----------------------------------------------------------------------- #
    #             Private methods. These are used internally only.            #
    # ----------------------------------------------------------------------- #
    def _init_model_params(self):
        """Initialises model parameters prior to fitting. If init_type if random, it samples from a Dirichlet distribution according to the given priors. Otherwise it initialises the starting probabilities and transition probabilities uniformly.

        :param X: list of observation sequences used to find the initial state means and covariances for the Gaussian and Heterogeneous models
        :type X: list, optional
        """
        if self.init_type == 'uniform':
            init = 1.0 / self.n_states
            if 's' in self.init_params:
                self.pi = np.full(self.n_states, init)

            if 't' in self.init_params:
                self.A = np.full((self.n_states, self.n_states), init)
        else:
            if 's' in self.init_params:
                self.pi = np.random.dirichlet(
                    alpha=self.pi_prior * np.ones(self.n_states), size=1
                )[0]

            if 't' in self.init_params:
                self.A = np.random.dirichlet(
                    alpha=self.A_prior * np.ones(self.n_states), size=self.n_states
                )

    def _decode_map(self, obs_seq):
        """Find the best state sequence (path) using MAP.

        :param obs_seq: an observation sequence 
        :type obs_seq: array_like
        :return: state_sequence - the optimal path for the observation sequence
        :rtype:  array_like
        :return: log_likelihood - the maximum log-probability for the entire sequence
        :rtype: float
        """
        posteriors = self.predict_proba([obs_seq])[0]
        log_likelihood = np.max(posteriors, axis=1).sum()
        state_sequence = np.argmax(posteriors, axis=1)

        return log_likelihood, state_sequence

    def _decode_viterbi(self, obs_seq):
        """Find the best state sequence (path) using viterbi algorithm - a method
        of dynamic programming, very similar to the forward-backward algorithm,
        with the added step of maximisation and eventual backtracing.

        :param obs_seq: an observation sequence 
        :type obs_seq: array_like
        :return: state_sequence - the optimal path for the observation sequence
        :rtype:  array_like
        :return: log_likelihood - the maximum log-probability for the entire sequence
        :rtype: float
        """
        n_samples = len(obs_seq)

        # similar to the forward-backward algorithm, we need to make sure that
        # we're using fresh data for the given obs_seq
        B_map = self._map_B(obs_seq)

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
        state_sequence[n_samples -
                       1] = where_from = np.argmax(delta[n_samples - 1])
        log_likelihood = delta[n_samples - 1, where_from]

        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                work_buffer[i] = delta[t, i] + log_A[i, where_from]
            state_sequence[t] = where_from = np.argmax(work_buffer)

        return log_likelihood, state_sequence

    def _calc_alpha(self, obs_seq, B_map):
        """Calculates 'alpha' the forward variable given an observation sequence.

        :param obs_seq: an observation sequence 
        :type obs_seq: array_like
        :param B_map: mapping of the observations' mass/density Bj(Ot) to Bj(t)
        :type B_map: array_like, optional
        :return: array of shape (n_samples, n_states) containing the forward variables
        :rtype: array_like
        """
        n_samples = len(obs_seq)

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

    def _calc_beta(self, obs_seq, B_map):
        """Calculates 'beta', the backward variable for each observation sequence.

        :param obs_seq: an observation sequence 
        :type obs_seq: array_like
        :param B_map: mapping of the observations' mass/density Bj(Ot) to Bj(t)
        :type B_map: array_like, optional
        :return: array of shape (n_samples, n_states) containing the backward variables
        :rtype: array_like
        """
        n_samples = len(obs_seq)

        # The beta variable is a ndarray indexed by time, then state (TxN).
        # beta[t][i] = the probability of being in state 'i' and then observing the
        # symbols from t+1 to the end (T).
        beta = np.zeros((n_samples, self.n_states))

        log_A = log_mask_zero(self.A)
        log_B_map = log_mask_zero(B_map)

        # init stage
        for i in range(self.n_states):
            beta[len(obs_seq) - 1][i] = 0.0

        # induction
        work_buffer = np.zeros(self.n_states)
        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    work_buffer[j] = log_A[i][j] + \
                        log_B_map[j][t + 1] + beta[t + 1][j]
                    beta[t][i] = logsumexp(work_buffer)

        return beta

    def _calc_xi(
        self, obs_seq, B_map=None, alpha=None, beta=None
    ):
        """Calculates 'xi', a joint probability from the 'alpha' and 'beta' variables.

        :param obs_seq: an observation sequence 
        :type obs_seq: array_like
        :param B_map: mapping of the observations' mass/density Bj(Ot) to Bj(t)
        :type B_map: array_like, optional
        :param alpha: array of shape (n_samples, n_states) containing the forward variables
        :type alpha: array_like, optional
        :param beta: array of shape (n_samples, n_states) containing the backward variables
        :type beta: array_like, optional
        :return: array of shape (n_samples, n_states, n_states) containing the a joint probability from the 'alpha' and 'beta' variables
        :rtype: array_like
        """
        if B_map is None:
            B_map = self._map_B(obs_seq)
        if alpha is None:
            alpha = self._calc_alpha(obs_seq, B_map)
        if beta is None:
            beta = self._calc_beta(obs_seq, B_map)

        n_samples = len(obs_seq)

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
                    log_xi_sum[i][j] = np.logaddexp(
                        log_xi_sum[i][j], work_buffer[i][j])

        return log_xi_sum

    def _calc_gamma(self, alpha, beta):
        """Calculates 'gamma' from 'alpha' and 'beta'.

        :param alpha: array of shape (n_samples, n_states) containing the forward variables
        :type alpha: array_like
        :param beta: array of shape (n_samples, n_states) containing the backward variables
        :type beta: array_like
        :return:  array of shape (n_samples, n_states), the posteriors
        :rtype: array_like
        """
        log_gamma = alpha + beta
        log_normalise(log_gamma, axis=1)
        with np.errstate(under='ignore'):
            return np.exp(log_gamma)

    # Methods used by self.train()
    def _train(
        self,
        obs_sequences,
        n_iter=100,
        conv_thresh=0.1,
        conv_iter=5,
        ignore_conv_crit=False,
        plot_log_likelihood=False,
        no_init=False,
        print_every=1,
        n_processes=None,
        return_log_likelihoods=False,
    ):
        """Training is repeated 'n_iter' times, or until log-likelihood of the model increases by less than a threshold.

        :param obs_sequences: a list of arrays containing the observation
                sequences of different lengths
        :type obs_sequences: list
        :param n_iter: max number of iterations to run for each initialisation; defaults to 100
        :type n_iter: int, optional
        :param conv_thresh: the threshold for the likelihood increase (convergence); defaults to 0.1
        :type conv_thresh: float, optional
        :param conv_iter: number of iterations for which the convergence criteria has to hold before early-stopping; defaults to 5
        :type conv_iter: int, optional
        :param ignore_conv_crit: flag to indicate whether to iterate until n_iter is reached or perform early-stopping; defaults to False
        :type ignore_conv_crit: bool, optional    
        :param plot_log_likelihood: parameter to activate plotting the evolution of the log-likelihood after each initialisation; defaults to False
        :type plot_log_likelihood: bool, optional  
        :param no_init: flag to indicate wheather to initialise the  parameters before training (it can only be True if the parameters are manually set before or they were already trained); defaults to False
        :type no_init: bool, optional    
        :param n_processes: number of processes to use if the training should be performed using parallelisation; defaults to None
        :type n_processes: int, optional
        :param return_log_likelihoods: if True it returns the evolution of the log-likelihoods; used for testing purposes;
        :type return_log_likelihoods: bool, optional
        :return: dictionary containing the updated model parameters
        :rtype: dict
        :return: the accumulated log-likelihood for all the observations. (if  return_log_likelihoods is True then the list of log-likelihood values from each iteration)
        :rtype: float

        """
        if not no_init:
            if self.init_type == 'kmeans':
                self._init_model_params(X=obs_sequences)
            else:
                self._init_model_params()
        else:
            check_if_attributes_set(self)

        log_likelihood_iter = []
        old_log_likelihood = np.nan
        for it in range(n_iter):

            # if train without multiprocessing
            if n_processes is None:
                stats, curr_log_likelihood = self._compute_intermediate_values(
                    obs_sequences
                )
            else:
                # split up observation sequences between the processes
                n_splits = int(np.ceil(len(obs_sequences) / n_processes))
                split_list = [
                    sl
                    for sl in list(
                        (
                            obs_sequences[
                                i * n_splits: i * n_splits + n_splits
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
                stats, curr_log_likelihood = self._sum_up_sufficient_statistics(
                    stats_list
                )

            # perform the M-step to update the model parameters
            new_model = self._M_step(stats)
            self._update_model(new_model)

            if self.verbose and it % print_every == 0:
                print(
                    'iter: {}, log_likelihood = {}, delta = {}'.format(
                        it,
                        curr_log_likelihood,
                        (curr_log_likelihood - old_log_likelihood),
                    )
                )

            if not ignore_conv_crit:
                if (
                    abs(curr_log_likelihood - old_log_likelihood)
                    / abs(old_log_likelihood)
                    <= conv_thresh
                ):
                    counter += 1
                    if counter == conv_iter:
                        # converged
                        if self.verbose:
                            print(
                                'Converged -> iter: {}, log_likelihood = {}'.format(
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
                'Maximum number of iterations reached. log_likelihood = {}'.format(
                    curr_log_likelihood
                )
            )

        if plot_log_likelihood:
            plot_log_likelihood_evolution(log_likelihood_iter)

        if return_log_likelihoods:  # this is just for testing
            return new_model, log_likelihood_iter
        else:
            return new_model, curr_log_likelihood

    def _compute_intermediate_values(self, obs_sequences):
        """Calculates the various intermediate values for the Baum-Welch on a list of observation sequences.

        :param obs_sequences: a list of ndarrays/lists containing
                the observation sequences. Each sequence can be the same or of
                different lengths.
        :type ob_sequences: list
        :return: a dictionary of sufficient statistics required for the M-step
        :rtype: dict
        """
        stats = self._initialise_sufficient_statistics()
        curr_log_likelihood = 0

        for obs_seq in obs_sequences:
            B_map = self._map_B(obs_seq)

            # calculate the log likelihood of the previous model
            # we compute the P(O|model) for the set of old parameters
            log_likelihood = self.forward(obs_seq, B_map)
            curr_log_likelihood += log_likelihood

            # do the E-step of the Baum-Welch algorithm
            obs_stats = self._E_step(obs_seq, B_map)

            # accumulate stats
            self._accumulate_sufficient_statistics(
                stats, obs_stats, obs_seq
            )

        return stats, curr_log_likelihood

    def _E_step(self, obs_seq, B_map):
        """Calculates required statistics of the current model, as part
        of the Baum-Welch 'E' step. Deriving classes should override (extend) 
        this method to include any additional computations their model requires.

        :param obs_seq: an observation sequence 
        :type obs_seq: array_like
        :param B_map: mapping of the observations' mass/density Bj(Ot) to Bj(t)
        :type B_map: array_like, optional
        :return: a dictionary containing the required statistics
        :rtype: dict
        """

        # compute the parameters for the observation
        obs_stats = {
            'alpha': self._calc_alpha(obs_seq, B_map),
            'beta': self._calc_beta(obs_seq, B_map),
        }

        obs_stats['xi'] = self._calc_xi(
            obs_seq,
            B_map=B_map,
            alpha=obs_stats['alpha'],
            beta=obs_stats['beta'],
        )
        obs_stats['gamma'] = self._calc_gamma(
            obs_stats['alpha'], obs_stats['beta']
        )
        return obs_stats

    def _M_step(self, stats):
        """Performs the 'M' step of the Baum-Welch algorithm.
        Deriving classes should override (extend) this method to include
        any additional computations their model requires.

        :param stats: dictionary containing the accumulated statistics
        :type stats: dict
        :return: a dictionary containing the updated model parameters
        :rtype: dict
        """
        new_model = {}

        if 's' in self.tr_params:
            pi_ = np.maximum(self.pi_prior - 1 + stats['pi'], 0)
            new_model['pi'] = np.where(self.pi == 0, 0, pi_)
            normalise(new_model['pi'])

        if 't' in self.tr_params:
            A_ = np.maximum(self.A_prior - 1 + stats['A'], 0)
            new_model['A'] = np.where(self.A == 0, 0, A_)
            normalise(new_model['A'], axis=1)

        return new_model

    def _update_model(self, new_model):
        """Replaces the current model parameters with the new ones.

        :param new_model: contains the new model parameters
        :type new_model: dict
        """
        if 's' in self.tr_params:
            self.pi = (1 - self.learning_rate) * new_model[
                'pi'
            ] + self.learning_rate * self.pi

        if 't' in self.tr_params:
            self.A = (1 - self.learning_rate) * \
                new_model['A'] + self.learning_rate * self.A

    def _initialise_sufficient_statistics(self):
        """Initialises sufficient statistics required for M-step.

        :return: a dictionary having as key-value pairs { nobs - number of samples in the data; start - array of shape (n_states, ) where the i-th element corresponds to the posterior probability of the first sample being generated by the i-th state; trans (dictionary) - containing the numerator and denominator parts of the new A matrix as arrays of shape (n_states, n_states)
        :rtype: dict
        """
        stats = {
            'nobs': 0,
            'pi': np.zeros(self.n_states),
            'A': np.zeros((self.n_states, self.n_states)),
        }
        return stats

    def _accumulate_sufficient_statistics(
        self, stats, obs_stats
    ):
        """Updates sufficient statistics from a given sample.

        :param stats: dictionary containing the sufficient statistics for all observation sequences
        :type stats: dict
        :param obs_stats: dictionary containing the sufficient statistic for one 
            observation sequence
        :type stats: dict
        """
        stats['nobs'] += 1
        if 's' in self.tr_params:
            stats['pi'] += obs_stats['gamma'][0]

        if 't' in self.tr_params:
            with np.errstate(under='ignore'):
                stats['A'] += np.exp(obs_stats['xi'])

    def _sum_up_sufficient_statistics(self, stats_list):
        """Sums sufficient statistics from a given sub-set of observation sequences.

        :param stats_list: list containing the sufficient statistics from the
                different processes
        :type stats_list: list
        :return: a dictionary of sufficient statistics
        :rtype: dict
        """
        stats_all = self._initialise_sufficient_statistics()
        logL_all = 0
        for (stat_i, logL_i) in stats_list:
            logL_all += logL_i
            for stat in stat_i.keys():
                if isinstance(stat_i[stat], dict):
                    for i in range(len(stats_all[stat]['numer'])):
                        stats_all[stat]['numer'][i] += stat_i[stat]['numer'][i]
                        stats_all[stat]['denom'][i] += stat_i[stat]['denom'][i]
                else:
                    stats_all[stat] += stat_i[stat]
        return stats_all, logL_all

    # Methods that have to be implemented in the deriving classes
    def _map_B(self, obs_seq):
        """Deriving classes should implement this method, so that it maps the
        observations' mass/density Bj(Ot) to Bj(t). The purpose of this method is to create a common parameter that will conform both to the discrete case where PMFs are used, and the continuous case where PDFs are used.

        :param obs_seq: an observation sequence of shape (n_samples, n_features)
        :type obs_seq: array_like
        :return: the mass/density mapping of shape (n_states, n_samples)
        :rtype: array_like
        """
        raise NotImplementedError(
            'A mapping function for B(observable probabilities) must be implemented.'
        )

    def _generate_sample_from_state(self, state):
        """Generates a random sample from a given component.

        :param state: index of the component to condition on
        :type state: int
        :return: a random sample from the emission distribution corresponding to the given state.
        :rtype: array_like
        """
        raise NotImplementedError(
            'A sample generator function must be implemented.'
        )

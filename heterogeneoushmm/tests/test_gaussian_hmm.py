"""
Created on Jan 14, 2020
Updated docstrings on Oct 12, 2021

@author: esukei

Parts of the code come from:
    https://github.com/hmmlearn/hmmlearn/blob/master/lib/hmmlearn/tests/test_gaussian_hmm.py
"""
import pytest
import numpy as np
from sklearn.utils import check_random_state
from sklearn.datasets import make_spd_matrix
from heterogeneoushmm.gaussian import GaussianHMM


def make_covar_matrix(covariance_type, n_states, n_features, random_state=None):
    mincv = 0.1
    prng = check_random_state(random_state)
    if covariance_type == "spherical":
        return (mincv + mincv * prng.random_sample((n_states,))) ** 2
    elif covariance_type == "tied":
        return make_spd_matrix(n_features) + mincv * np.eye(n_features)
    elif covariance_type == "diagonal":
        return (mincv + mincv * prng.random_sample((n_states, n_features))) ** 2
    elif covariance_type == "full":
        return np.array(
            [
                (
                    make_spd_matrix(n_features, random_state=prng)
                    + mincv * np.eye(n_features)
                )
                for _ in range(n_states)
            ]
        )


class TestGaussianHMM:
    """
    Test based on the example provided on:
        https://hmmlearn.readthedocs.io/en/latest/tutorial.html#training-hmm-parameters-and-inferring-the-hidden-states
    """

    covariance_type = "diagonal"  # will also be set by subclasses

    @pytest.fixture(autouse=True)
    def setup(self):
        """
        Initialise a Gaussian HMM with some dummy values. 
        """
        self.prng = prng = np.random.RandomState(10)
        self.n_states = n_states = 3
        self.n_emissions = n_emissions = 3
        self.pi = prng.rand(n_states)
        self.pi = self.pi / self.pi.sum()
        self.A = prng.rand(n_states, n_states)
        self.A /= np.tile(self.A.sum(axis=1)[:, np.newaxis], (1, n_states))
        self.means = prng.randint(-20, 20, (n_states, n_emissions))
        self.covars = make_covar_matrix(
            self.covariance_type, self.n_states, self.n_emissions, random_state=prng
        )

    def test_bad_covariance_type(self):
        """
        Test if an error is thrown when a non-existing covariance matrix type is requested.
        """
        with pytest.raises(ValueError):
            _ = GaussianHMM(
                n_states=self.n_states,
                n_emissions=self.n_emissions,
                covariance_type="badcovariance_type",
            )

    def test_score_samples_and_decode(self):
        """
        Tests the score_samples and decode methods, which return a list of arrays of shape
        (n_samples, n_states) containing the state-membership probabilities for each
        sample in the observation sequences, and a list of length n_samples, containing the
        most probable state sequence, respectively.
        """
        h = GaussianHMM(
            self.n_states,
            self.n_emissions,
            covariance_type=self.covariance_type,
            init_params="st",
        )

        h.pi = self.pi
        h.A = self.A

        # Make sure the means are far apart so posteriors.argmax()
        # picks the actual component used to generate the observations.
        h.means = 20 * self.means
        h.covars = self.covars

        gaussidx = np.repeat(np.arange(self.n_states), 5)
        n_samples = len(gaussidx)
        X = [self.prng.randn(n_samples, self.n_emissions) + h.means[gaussidx]]
        posteriors = h.predict_proba(X)[
            0
        ]  # because we only had one observation sequence, but it returns a list anyways

        assert posteriors.shape == (n_samples, self.n_states)
        assert np.allclose(posteriors.sum(axis=1), np.ones(n_samples))

        _, stateseq = h.decode(X)
        assert np.allclose(stateseq, gaussidx)

    def test_sample(self, n_samples=1000, n_sequences=5):
        """
        Test if the sampling method generates the correct number of
        sequences with corrent number of samples.

        :param n_samples: number of samples to generate for each sequence, defaults to 1000
        :type n_samples: int, optional
        :param n_sequences: number of sequences to generate, defaults to 5
        :type n_sequences: int, optional
        """
        h = GaussianHMM(
            n_states=self.n_states,
            n_emissions=self.n_emissions,
            covariance_type=self.covariance_type,
        )
        h.pi = self.pi
        h.A = self.A

        # Make sure the means are far apart so posteriors.argmax()
        # n_emissionscks the actual component used to generate the observations.
        h.means = 20 * self.means
        h.covars = np.maximum(self.covars, 0.1)

        X, state_sequences = h.sample(
            n_sequences=n_sequences, n_samples=n_samples, return_states=True)

        assert np.all(X[i].ndim == 2 for i in range(n_sequences))
        assert np.all(
            len(X[i]) == len(state_sequences[i]) == n_samples
            for i in range(n_sequences)
        )
        assert np.all(
            len(np.unique(X[i])) == self.n_emissions for i in range(n_sequences)
        )

    def test_train(self, n_samples=100, n_sequences=30, tr_params="stmc"):
        """
        Test if the training algorithm works correctly (if the log-likelihood increases).

        :param n_samples: number of samples to generate for each sequence, defaults to 100
        :type n_samples: int, optional
        :param n_sequences: number of sequences to generate, defaults to 30
        :type n_sequences: int, optional
        :param tr_params: which model parameters to train, defaults to "stmc"
        :type tr_params: str, optional
        """
        h = GaussianHMM(
            n_states=self.n_states,
            n_emissions=self.n_emissions,
            covariance_type=self.covariance_type,
            tr_params=tr_params,
            verbose=True,
        )
        h.pi = self.pi
        h.A = self.A
        h.means = 20 * self.means
        h.covars = np.maximum(self.covars, 0.1)

        # Generate observation sequences
        X = h.sample(
            n_sequences=n_sequences, n_samples=n_samples)

        # Mess up the parameters and see if we can re-learn them.
        h, log_likelihoods = h._train(
            X, n_iter=10, conv_thresh=0.01, return_log_likelihoods=True, n_processes=2,
        )

        # we consider learning if the log_likelihood increases
        assert np.all(np.round(np.diff(log_likelihoods), 10) >= 0)

    def test_train_sequences_of_different_length(self, tr_params="stmc"):
        """
        Test if the training algorithm works correctly on sequences with different
        lengths (if the log-likelihood increases).

        :param tr_params: which model parameters to train, defaults to "stmc"
        :type tr_params: str, optional
        """
        h = GaussianHMM(
            n_states=self.n_states,
            n_emissions=self.n_emissions,
            covariance_type=self.covariance_type,
        )
        h.n_emissions = self.n_emissions
        h.A = self.A
        h.pi = self.pi
        h.means = self.means
        h.covars = self.covars
        h.tr_params = tr_params

        # Generate observation sequences
        lengths = [30, 40, 50]
        X = [
            h.sample(n_sequences=1, n_samples=n_samples)[0] for n_samples in lengths
        ]

        h, log_likelihoods = h._train(
            X, n_iter=10, conv_thresh=0.001, return_log_likelihoods=True, n_processes=2,
        )

        # we consider learning if the log_likelihood increases (the first one is discarded, because sometimes it drops
        # after the first iteration and increases for the rest
        assert np.all(np.round(np.diff(log_likelihoods), 10) >= 0)


class TestGaussianHMMWithSphericalCovars(TestGaussianHMM):
    covariance_type = "spherical"


class TestGaussianHMMWithDiagonalCovars(TestGaussianHMM):
    covariance_type = "diagonal"


class TestGaussianHMMWithTiedCovars(TestGaussianHMM):
    covariance_type = "tied"


class TestGaussianHMMWithFullCovars(TestGaussianHMM):
    covariance_type = "full"

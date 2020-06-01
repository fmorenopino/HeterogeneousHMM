"""
Created on Jan 14, 2020

@author: esukei

Parts of the code come from:
    https://github.com/hmmlearn/hmmlearn/blob/master/lib/hmmlearn/tests/test_gaussian_hmm.py
"""
from unittest import TestCase

import numpy as np
import pytest

from src.GaussianHMM import GaussianHMM
from src.utils import make_covar_matrix


class TestGaussianHMM:
    """
    Test based on the example provided on:
        https://hmmlearn.readthedocs.io/en/latest/tutorial.html#training-hmm-parameters-and-inferring-the-hidden-states
    """

    covariance_type = None  # set by subclasses

    def setUp(self):
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

    def test_bad_init_type(self):
        with pytest.raises(ValueError):
            h = h = GaussianHMM(
                n_states=self.n_states,
                n_emissions=self.n_emissions,
                init_type="badinit_type",
            )

    def test_bad_covariance_type(self):
        with pytest.raises(ValueError):
            h = GaussianHMM(
                n_states=self.n_states,
                n_emissions=self.n_emissions,
                covariance_type="badcovariance_type",
            )

    def test_score_samples_and_decode(self):
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
        posteriors = h.score_samples(X)[
            0
        ]  # because we only had one observation sequence, but it returns a list anyways

        self.assertEqual(posteriors.shape, (n_samples, self.n_states))
        assert np.allclose(posteriors.sum(axis=1), np.ones(n_samples))

        viterbi_ll, stateseq = h.decode(X)
        assert np.allclose(stateseq, gaussidx)

    def test_sample(self, n_samples=1000, n_sequences=5):
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

        X, state_sequences = h.sample(n_sequences=n_sequences, n_samples=n_samples)

        assert np.all(X[i].ndim == 2 for i in range(n_sequences))
        assert np.all(
            len(X[i]) == len(state_sequences[i]) == n_samples
            for i in range(n_sequences)
        )
        assert np.all(
            len(np.unique(X[i])) == self.n_emissions for i in range(n_sequences)
        )

    def test_train(self, n_samples=100, n_sequences=30, params="stmc"):
        h = GaussianHMM(
            n_states=self.n_states,
            n_emissions=self.n_emissions,
            covariance_type=self.covariance_type,
            params=params,
            verbose=True,
        )
        h.pi = self.pi
        h.A = self.A
        h.means = 20 * self.means
        h.covars = np.maximum(self.covars, 0.1)

        # Generate observation sequences
        X, state_sequences = h.sample(n_sequences=n_sequences, n_samples=n_samples)

        # Mess up the parameters and see if we can re-learn them.
        h, log_likelihoods = h._train(
            X, n_iter=10, thres=0.01, return_log_likelihoods=True, n_processes=2,
        )

        # we consider learning if the log_likelihood increases
        assert np.all(np.round(np.diff(log_likelihoods), 10) >= 0)

    def test_train_sequences_of_different_length(self, params="stmc"):
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
        h.params = params

        # Generate observation sequences
        lengths = [30, 40, 50]
        X = [
            h.sample(n_sequences=1, n_samples=n_samples)[0][0] for n_samples in lengths
        ]

        h, log_likelihoods = h._train(
            X, n_iter=10, thres=0.001, return_log_likelihoods=True, n_processes=2,
        )

        # we consider learning if the log_likelihood increases (the first one is discarded, because sometimes it drops
        # after the first iteration and increases for the rest
        assert np.all(np.round(np.diff(log_likelihoods), 10) >= 0)

    def test_train_kmeans_init(self, n_samples=100, n_sequences=30, params="stmc"):
        h = GaussianHMM(
            n_states=self.n_states,
            n_emissions=self.n_emissions,
            covariance_type=self.covariance_type,
            init_type="kmeans",
            params=params,
            verbose=True,
        )
        h.pi = self.pi
        h.A = self.A
        h.means = 20 * self.means
        h.covars = np.maximum(self.covars, 0.1)

        # Generate observation sequences
        X, state_sequences = h.sample(n_sequences=n_sequences, n_samples=n_samples)

        # Mess up the parameters and see if we can re-learn them.
        h, log_likelihoods = h._train(
            X, n_iter=10, thres=0.01, return_log_likelihoods=True, n_processes=2
        )

        # we consider learning if the log_likelihood increases
        assert np.all(np.round(np.diff(log_likelihoods), 10) >= 0)

class TestGaussianHMMWithSphericalCovars(TestGaussianHMM, TestCase):
    covariance_type = "spherical"


class TestGaussianHMMWithDiagonalCovars(TestGaussianHMM, TestCase):
    covariance_type = "diagonal"


class TestGaussianHMMWithTiedCovars(TestGaussianHMM, TestCase):
    covariance_type = "tied"


class TestGaussianHMMWithFullCovars(TestGaussianHMM, TestCase):
    covariance_type = "full"

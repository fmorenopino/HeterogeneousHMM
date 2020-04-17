"""
Created on Feb 05, 2020

@author: esukei
"""
from unittest import TestCase

import numpy as np
import pytest

from src.HeterogeneousHMM import HeterogeneousHMM
from src.utils import make_covar_matrix, normalise


class TestHeterogeneousHMM:
    """
    Test based on the example provided on:
        https://hmmlearn.readthedocs.io/en/latest/tutorial.html#training-hmm-parameters-and-inferring-the-hidden-states
    """

    covariance_type = None  # set by subclasses

    def setUp(self):
        self.prng = prng = np.random.RandomState(10)
        self.n_states = n_states = 3
        self.n_g_emissions = n_g_emissions = 2
        self.n_d_emissions = 1
        self.n_d_features = [2]
        self.pi = prng.rand(n_states)
        self.pi = self.pi / self.pi.sum()
        self.A = prng.rand(n_states, n_states)
        self.A /= np.tile(self.A.sum(axis=1)[:, np.newaxis], (1, n_states))
        self.B = np.asarray(
            [
                normalise(
                    np.random.random((self.n_states, self.n_d_features[i])), axis=1
                )
                for i in range(self.n_d_emissions)
            ]
        )
        self.means = prng.randint(-20, 20, (n_states, n_g_emissions))
        self.covars = make_covar_matrix(
            self.covariance_type, self.n_states, self.n_g_emissions, random_state=prng
        )

    def test_bad_init_type(self):
        with pytest.raises(ValueError):
            h =  HeterogeneousHMM(
                n_states=self.n_states,
                n_g_emissions=self.n_g_emissions,
                n_d_emissions=self.n_d_emissions,
                n_d_features=self.n_d_features,
                init_type="badinit_type",
            )

    def test_bad_covariance_type(self):
        with pytest.raises(ValueError):
            h = HeterogeneousHMM(
                n_states=self.n_states,
                n_g_emissions=self.n_g_emissions,
                n_d_emissions=self.n_d_emissions,
                n_d_features=self.n_d_features,
                covariance_type="badcovariance_type",
            )

    def test_score_samples_and_decode(self):
        h = HeterogeneousHMM(
            n_states=self.n_states,
            n_g_emissions=self.n_g_emissions,
            n_d_emissions=self.n_d_emissions,
            n_d_features=self.n_d_features,
            covariance_type=self.covariance_type,
        )

        h.pi = self.pi
        h.A = self.A
        # Make sure the means are far apart so posteriors.argmax()
        # picks the actual component used to generate the observations.
        h.means = 20 * self.means
        h.covars = self.covars
        h.B = self.B

        stateidx = np.repeat(np.arange(self.n_states), 5)
        n_samples = len(stateidx)
        X_gauss = self.prng.randn(n_samples, h.n_g_emissions) + h.means[stateidx]
        X = []
        for idx, state in enumerate(stateidx):
            cat_sample = []
            for e in range(h.n_d_emissions):
                cdf = np.cumsum(h.B[e][state, :])
                cat_sample.append((cdf > np.random.rand()).argmax())
            X.append(np.concatenate([X_gauss[idx], cat_sample]))
        X = [np.asarray(X)]
        posteriors = h.score_samples(X)[
            0
        ]  # because we only had one observation sequence, but it returns a list anyways

        self.assertEqual(posteriors.shape, (n_samples, self.n_states))
        assert np.allclose(posteriors.sum(axis=1), np.ones(n_samples))

        viterbi_ll, stateseq = h.decode(X)
        assert np.allclose(stateseq, stateidx)

    def test_sample(self, n_samples=1000, n_sequences=5):
        h = HeterogeneousHMM(
            n_states=self.n_states,
            n_g_emissions=self.n_g_emissions,
            n_d_emissions=self.n_d_emissions,
            n_d_features=self.n_d_features,
            covariance_type=self.covariance_type,
        )
        h.pi = self.pi
        h.A = self.A
        h.B = self.B
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
            len(np.unique(X[i])) == (h.n_g_emissions + h.n_d_emissions)
            for i in range(n_sequences)
        )

    def test_train(self, n_samples=100, n_sequences=30, params="stmc"):
        h = HeterogeneousHMM(
            n_states=self.n_states,
            n_g_emissions=self.n_g_emissions,
            n_d_emissions=self.n_d_emissions,
            n_d_features=self.n_d_features,
            covariance_type=self.covariance_type,
            params=params,
            verbose=True,
        )
        h.pi = self.pi
        h.A = self.A
        h.B = self.B
        h.means = 20 * self.means
        h.covars = np.maximum(self.covars, 0.1)

        # Generate observation sequences
        X, state_sequences = h.sample(n_sequences=n_sequences, n_samples=n_samples)

        # Mess up the parameters and see if we can re-learn them.
        h, log_likelihoods = h._train(
            X, n_iter=10, thres=0.01, return_log_likelihoods=True
        )

        # we consider learning if the log_likelihood increases
        assert np.all(np.round(np.diff(log_likelihoods), 10) >= 0)

    def test_train_without_init(self, n_samples=100, n_sequences=30, params="ste"):
        h = HeterogeneousHMM(
            n_states=self.n_states,
            n_g_emissions=self.n_g_emissions,
            n_d_emissions=self.n_d_emissions,
            n_d_features=self.n_d_features,
            covariance_type=self.covariance_type,
            params=params,
        )

        h.pi = self.pi
        h.A = self.A
        h.means = 20 * self.means
        h.covars = np.maximum(self.covars, 0.1)
        h.B = self.B

        # Generate observation sequences
        X, state_sequences = h.sample(n_sequences=n_sequences, n_samples=n_samples)

        h_tst = HeterogeneousHMM(
            n_states=self.n_states,
            n_g_emissions=self.n_g_emissions,
            n_d_emissions=self.n_d_emissions,
            n_d_features=self.n_d_features,
            covariance_type=self.covariance_type,
            params=params,
        )
        with pytest.raises(AttributeError):
            h_tst, log_likelihoods = h_tst._train(
                X, n_iter=100, thres=0.01, return_log_likelihoods=True, no_init=True
            )

    def test_train_sequences_of_different_length(self, params="stmc"):
        h = HeterogeneousHMM(
            n_states=self.n_states,
            n_g_emissions=self.n_g_emissions,
            n_d_emissions=self.n_d_emissions,
            n_d_features=self.n_d_features,
            covariance_type=self.covariance_type,
        )
        h.A = self.A
        h.pi = self.pi
        h.B = self.B
        h.means = self.means
        h.covars = self.covars
        h.params = params

        # Generate observation sequences
        lengths = [30, 40, 50]
        X = [
            h.sample(n_sequences=1, n_samples=n_samples)[0][0] for n_samples in lengths
        ]

        h, log_likelihoods = h._train(
            X, n_iter=10, thres=0.01, return_log_likelihoods=True
        )

        # we consider learning if the log_likelihood increases (the first one is discarded, because sometimes it drops
        # after the first iteration and increases for the rest
        assert np.all(np.round(np.diff(log_likelihoods[1:]), 10) >= 0)

    def test_non_trainable_emission(self, n_samples=100, n_sequences=30, params="ste"):
        h = HeterogeneousHMM(
            n_states=self.n_states,
            n_g_emissions=self.n_g_emissions,
            n_d_emissions=self.n_d_emissions,
            n_d_features=self.n_d_features,
            covariance_type=self.covariance_type,
        )
        h.A = self.A
        h.pi = self.pi
        h.B = self.B
        h.means = self.means
        h.covars = self.covars
        h.params = params

        # Generate observation sequences
        X, state_sequences = h.sample(n_sequences=n_sequences, n_samples=n_samples)

        h_tst = HeterogeneousHMM(
            n_states=self.n_states,
            n_g_emissions=self.n_g_emissions,
            n_d_emissions=self.n_d_emissions,
            n_d_features=self.n_d_features,
            covariance_type=self.covariance_type,
            nr_no_train_de=1,
        )

        # Set up the emission probabilities and see if we can re-learn them.
        B_fix = np.eye(self.n_states, self.n_d_features[-1])

        with pytest.raises(AttributeError):
            h_tst, log_likelihoods = h_tst._train(
                X, n_iter=100, thres=0.01, return_log_likelihoods=True, no_init=False
            )

            # we consider learning if the log_likelihood increases
            assert np.all(np.round(np.diff(log_likelihoods), 10) >= 0)

            # we want that the emissions haven't changed
            assert np.allclose(B_fix, h_tst.B[-1])



class TestHeterogeneousHMMWithSphericalCovars(TestHeterogeneousHMM, TestCase):
    covariance_type = "spherical"


class TestHeterogeneousHMMWithDiagonalCovars(TestHeterogeneousHMM, TestCase):
    covariance_type = "diagonal"


class TestHeterogeneousHMMWithTiedCovars(TestHeterogeneousHMM, TestCase):
    covariance_type = "tied"


class TestHeterogeneousHMMWithFullCovars(TestHeterogeneousHMM, TestCase):
    covariance_type = "full"

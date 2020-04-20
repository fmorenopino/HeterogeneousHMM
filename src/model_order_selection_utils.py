"""
Created on Feb 04, 2020

@author: semese

This code is based on:
 - https://rdrr.io/cran/HMMpa/man/AIC_HMM.html

For theoretical bases see:
 - K.P. Murphy, "Machine Learning: A Probabilistic Perspective", The MIT Press
   ©2012, ISBN:0262018020 9780262018029
"""

import numpy as np
import matplotlib.pyplot as plt


def aic_hmm(log_likelihood, dof):
    """
        Function to compute the Aikaike's information criterion for an HMM given
        a time-series of observations..
        Args:
            log_likelihood (float) - logarithmised likelihood of the model
            dof (int) - single numeric value representing the number of trainable
            parameters of the model
        Returns:
            aic (float) - the Aikaike's information criterion
    """
    aic = -2 * log_likelihood + 2 * dof
    return aic


def bic_hmm(log_likelihood, dof, n_samples):
    """
        Function to compute Bayesian information criterion for an HMM given a
        time-series of observations.
        Args:
            log_likelihood (float) - logarithmised likelihood of the model
            dof (int) - single numeric value representing the number of trainable
            parameters of the model
            n_samples (int) - length of the time-series of observations x
        Returns:
            bic (float) - the Bayesian information criterion
    """
    bic = -2 * log_likelihood + dof * np.log(n_samples)
    return bic


def get_n_fit_scalars(hmm):
    """
        Function to compute the degrees of freedom of a HMM based on the parameters
        that are trained in it.
        Args:
            hmm (object) -  a type of HMM (GaussianHMM, MultinomialHMM, etc)
        Returns:
            dof (int) - the number of free parameters of the model
    """
    train_params = hmm.params
    n_fit_scalars_per_param = hmm.get_n_fit_scalars_per_param()
    dof = 0
    for par in n_fit_scalars_per_param:
        if par in train_params:
            dof += n_fit_scalars_per_param[par]

    return dof


def plot_model_selection(n_states, criteria, filename=None):
    """
        Function to plot the different model order selection criteria vs the
        number of states of a HMM.
        Args:
            n_states (list) - list of number of states that were used to compute
                the criteria
            criteria (dictionary) - the keys correspond to the different criteria
                that were computed (AIC, BIC, etc) and the values are a list of
                length len(n_states) containing the criteria values for the
                different number of states.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    for key, value in criteria.items():
        ax.plot(n_states, value, marker="o", label=key)

    ax.set_frame_on(True)
    ax.set_xlabel("# states")
    ax.set_ylabel("Criterion")
    ax.legend()

    if filename is not None:
        fig.savefig(filename)
        plt.close()
    else:
        plt.show()

    return

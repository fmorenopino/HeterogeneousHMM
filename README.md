# Heterogenous-HMM (HMM with labels)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3759439.svg)](https://doi.org/10.5281/zenodo.3759439)

Please, to cite this code, use:

> @article{Moreno2020,
author = {Moreno-Pino, Fernando  and Art\'es-Rodr\'iguez, Antonio and Sukei, Emese},
doi = {10.5281/ZENODO.3759439},
howpublished = "\url{https://github.com/fmorenopino/HeterogeneousHMM}"
month = {apr},
title = {{fmorenopino/HeterogeneousHMM: First stable release of HeterogenousHMM}},
year = {2020}
}

This repository contains different implementations of the Hidden Markov Model with just some basic Python dependencies. The main contributions of this library with respect to other available APIs are:

- **Missing values support**: our implementation supports both partial and complete missing data.

- **Heterogeneous HMM (HMM with labels)**: here we implement a version of the HMM which allow us to use different distributions to manage each the emission probabilities of each of the features (*also, the simpler cases: Gaussian and Multinomial HMMs are implemented*).

- **Semi-Supervised HMM (Fixing discrete emission probabilities)**: in the Heterogenous-HMM model, it is possible to fix the emission probabilities of the discrete features: the model allow us to fix the complete emission probabilities matrix *B* of certain feature or just some states' emission probabilities (while training the other states' emission probabilities).

- **Model selection criteria**: Both "Akaike Information Criterion" (AIC) and Bayesian Information Criterion (BIC) are implemented.

Also, some others aspects like having multiple sequences, several features per observation or sampling from the models are supported. This model is easily extendable with other types of probablistic models.

The information that you can find in this readme file can be seen in the next Table of Contents:

- [Heterogenous-HMM (HMM with labels)](#heterogenous-hmm--hmm-with-labels-)
  * [1. How a HMM, a Heterogeneous HMM and a Semi-Supervised HMM work?](#1-how-a-hmm--a-heterogeneous-hmm-and-a-semi-supervised-hmm-work-)
    + [1.1. Gaussian HMM.](#11-gaussian-hmm)
      - [The three basic inference problems for HMMs.](#the-three-basic-inference-problems-for-hmms)
    + [1.2. Heterogeneous HMM/HMM with labels.](#12-heterogeneous-hmm-hmm-with-labels)
    + [1.3. Semi-Supervised HMM.](#13-semi-supervised-hmm)
  * [2. Available models and How to use them.](#2-available-models-and-how-to-use-them)
    + [2.1. Multinomial HMM.](#21-multinomial-hmm)
    + [2.2. Gaussian HMM.](#22-gaussian-hmm)
    + [2.3. Heterogeneous HMM.](#23-heterogeneous-hmm)
    + [2.4. Semi-supervised HMM.](#24-semi-supervised-hmm)
  * [3. Folder Structure.](#3-folder-structure)
  * [4. Dependencies.](#4-dependencies)
  * [5. Authors.](#5-authors)
  * [6. Contact Information.](#6-contact-information)
  * [7. References.](#7-references)



## 1. How a HMM, a Heterogeneous HMM and a Semi-Supervised HMM work?

In this library, we have developed different implementation of the Hidden Markov Model. In the section 2 of this readme file we explain how to use then but, before that, we would like to briefly explain how our HMM models work.

### 1.1. Gaussian HMM.

The Gaussian HMM manages the emission probabilities with gaussian distributions, its block diagram is represented in the next figure:


 <p align="center">
     <img src="https://raw.githubusercontent.com/fmorenopino/Heterogeneous_HMM/master/notebooks/img/hmm.png">
</p>

The parameters that we have to deal with are:

 <p align="center">
     <img src="https://raw.githubusercontent.com/fmorenopino/Heterogeneous_HMM/master/notebooks/img/parameters.png" width="30%">
</p>
 
 Where:
 
 - *S* is the hidden state sequence, being *I* the number of states of the model.
 - *Y* is the observed continuous sequence (in the discrete case, it would be the observed discrete sequence).
 - ***A*** is the matrix that contains the state transition probabilities.
 - ***B*** are the observation emission probabilities, which for the gaussian case are managed with the means and covariances.
 - ***π*** is the initial state probability distribution. In our model, both random and k-means can be used to initialize it.
 
 Finally, the model's parameters of a HMM would be: *θ*={***A***, ***B***, ***π***}.



#### The three basic inference problems for HMMs.

In order to have a useful HMM model for a real application, there are three basic problems that must be solved:

- Problem 1: given the observed sequence *Y*, which is the probability of that observed sequence for our model's parameters *θ*, that is, which is p(*Y* | *θ*)?

To solve this first problem, the Forward algorithm can be used.

- Problem 2: given the observed sequence *Y* and the model's parameters *θ*, which is the optimal state sequence *S*?

To solve this second problem several algorithms can be used, for example, the Viterbi or the Forward-Backward algorithm. If using Viterbi, we will maximize the p(*S*, *Y* | *θ*). Othercase, with the Forward-Backward algorithm, we optimizes the p(*s<sub>t</sub>*, *Y* | *θ*).
 
- Problem 3: which are the optimal *θ* that maximizes p(*Y* | *θ*)?

To solve this third problem we must consider the joint distribution of *S* and *Y*, that is:

<p align="center">
     <img src="https://raw.githubusercontent.com/fmorenopino/Heterogeneous_HMM/master/notebooks/img/joint.png" width="50%">
</p>

By using the EM algorithm, the model parameters *θ* (that is, the initial state probability ***π***, the state transition probabilities ***A*** and the gaussian emission probabilities {***μ***, ***Σ***}) are updated.

> The solution for these problems is nowadays very well known. If you want to get some extra knowledge about how the α, β, γ, δ... parameters are derived you can check the references below.


### 1.2. Heterogeneous HMM/HMM with labels.


In the Heterogeneous HMM, we can manage some features' emission probabilities with discrete distributions (the labels) and some others' emission probabilities with gaussian distributions. Its block diagram is:

 <p align="center">
     <img src="https://raw.githubusercontent.com/fmorenopino/Heterogeneous_HMM/master/notebooks/img/hhmm.png">
</p>

In addition to the parameters showed for the gaussian case, we must add:

 <p align="center">
     <img src="https://raw.githubusercontent.com/fmorenopino/Heterogeneous_HMM/master/notebooks/img/hhmm_parameters.png" width="40%">
</p>

Where:

- *L* is the labels sequence.
- ***D*** are the labels' emission probabilities.

For the Heterogenous HMM, our joint distribution is:

<p align="center">
     <img src="https://raw.githubusercontent.com/fmorenopino/Heterogeneous_HMM/master/notebooks/img/hhmm_joint.png" width="70%">
</p>

As we can observe in the previous equation, now the joint distribution depends on a new term which is the probability of the observed label given a certain state at an instant *t*.

### 1.3. Semi-Supervised HMM.

The Semi-Supervised HMM is a version of the Heterogenous HMM where the label emission probabilities are set *a priori*. This allows us to asocciate certain states to certain values of the labels, which provides guidance during the learning process.

## 2. Available models and how to use them.

> Several implementations of the HMM have been developed, all these HMM models extend the *_BaseHMM* class. These models are:

- Multinomial HMMs (Discrete HMM): Hidden Markov Model with multinomial (discrete) emission probabilities.
- Gaussian HMMs: Hidden Markov Model with Gaussian emission probabilities.
- Heterogeneous HMM (HMM with labels): Hidden Markov Model with mixed discrete and gaussian emission probabilities.
- Semi-supervised HMM: in the Heterogeneous HMM, it is possible to fix the emission probabilities of the discrete features to guide the learning process of the model. 

> ***In the notebook "hmm_tutorials.ipynb", an example of use of each of the previous models can be found***.

Now, a more detailed explanation of each of them is provided:


### 2.1. Multinomial HMM.

In the multinomial HMM the emission probabilities are discrete, whetever it is binary or categorical.

**Parameters:**

- *n_states* (int) - the number of hidden states
- *n_emissions* (int) - the number of distinct observations
- *n_features* (list) - a list containing the number of different symbols for each emission
- *params* (string, optional) - controls which parameters are updated in the
training process; defaults to all parameters
- *init_params* (string, optional) - controls which parameters are initialised
prior to training; defaults to all parameters
- *init_type* (string, optional) - name of the initialisation
method to use for initialising the model parameters before training
- *pi_prior* (array, optional) - array of shape (n_states, ) setting the
parameters of the Dirichlet prior distribution for 'pi'
- *A_prior* (array, optional) - array of shape (n_states, n_states),
giving the parameters of the Dirichlet prior distribution for each
row of the transition probabilities 'A'
- *learn_rate* (float, optional) - a value from the (0,1) interval, controlling how much
the past values of the model parameters count when computing the new
model parameters during training; defaults to 0
- *missing* (int or NaN, optional) - a value indicating what character indicates a missed
observation in the observation sequences; defaults to NaN
- *verbose* (bool, optional) - flag to be set to True if per-iteration
convergence reports should be printed during training

### 2.2. Gaussian HMM.

In the Gaussian HMM, the emission probabilities are managed with gaussian probabilities distributions.

**Parameters:**

- *n_states* (int) - the number of hidden states
- *n_emissions* (int) - the number of distinct Gaussian observations
- *params* (string, optional) - controls which parameters are updated in the training process; defaults to all parameters
- *init_params* (string, optional) - controls which parameters are initialised prior to training; defaults to all parameters
- *init_type* (string, optional) - name of the initialisation method to use for initialising the model parameters before training; can be "random" or "kmeans"
- *covariance_type* (string, optional) - string describing the type of covariance parameters to use.  Must be one of: "diagonal", "full", "spherical" or "tied"; defaults to "diagonal"
- *pi_prior* (array, optional) - array of shape (n_states, ) setting the parameters of the Dirichlet prior distribution for 'pi'
- *A_prior* (array, optional) - array of shape (n_states, n_states), giving the parameters of the Dirichlet prior distribution for each row of the transition probabilities 'A'
- *means_prior, means_weight* (array, optional) - arrays of shape (n_states, 1) providing the mean and precision of the Normal prior distribution for the means
- *covars_prior, covars_weight* (array, optional) - shape (n_states, 1), provides the parameters of the prior distribution for the covariance matrix
- *min_covar* (float, optional)- floor on the diagonal of the covariance matrix to prevent overfitting. Defaults to 1e-3.
- *learn_rate* (float, optional) - a value from the $[0,1)$ interval, controlling how much the past values of the model parameters count when computing the new model parameters during training; defaults to 0
- *verbose* (bool, optional) - flag to be set to True if per-iteration convergence reports should be printed during training

### 2.3. Heterogeneous HMM.

In the Heterogeneous HMM, we can manage some of the features' emission probabilities with gaussian distributions and others with discrete distributions.

**Parameters:** 

The HeterogeneousHMM class uses the following arguments for initialisation:
- *n_states* (int) - the number of hidden states.
- *n_g_emissions* (int) - the number of distinct Gaussian observations.
- *n_d_emissions* (int) - the number of distinct discrete observations.
- *n_d_features* (list - list of the number of possible observable symbols for each discrete emission.
- *params* (string, optional) - controls which parameters are updated in the training process; defaults to all parameters.
- *init_params* (string, optional) - controls which parameters are initialised prior to training; defaults to all parameters.
- *init_type* (string, optional) - name of the initialisation method to use for initialising the model parameters before training; can be "random" or "kmeans".
- *nr_no_train_de* (int) - this number indicates the number of discrete emissions whose Matrix Emission Probabilities are fixed and are not trained; it is important to to order the observed variables such that the ones whose emissions aren't trained are the last ones. 
- *state_no_train_de* (int) - a state index for nr_no_train_de which shouldn't be updated; defaults to None, which means that the entire emission probability matrix for that discrete emission will be kept unchanged during training, otherwise the last state_no_train_de states won't be updated
- *covariance_type* (string, optional) - string describing the type of covariance parameters to use.  Must be one of: "diagonal", "full", "spherical" or "tied"; defaults to "diagonal".
- *pi_prior* (array, optional) - array of shape (n_states, ) setting the parameters of the Dirichlet prior distribution for 'pi'.
- *A_prior* (array, optional) - array of shape (n_states, n_states), giving the parameters of the Dirichlet prior distribution for each row of the transition probabilities 'A'.
- *means_prior, means_weight* (array, optional) - arrays of shape (n_states, 1) providing the mean and precision of the Normal prior distribution for the means.
- *covars_prior, covars_weight* (array, optional) - shape (n_states, 1), provides the parameters of the prior distribution for the covariance matrix.
- *min_covar* (float, optional)- floor on the diagonal of the covariance matrix to prevent overfitting. Defaults to 1e-3.
- *learn_rate* (float, optional) - a value from the $[0,1)$ interval, controlling how much the past values of the model parameters count when computing the new model parameters during training; defaults to 0.
- *verbose* (bool, optional) - flag to be set to True if per-iteration convergence reports should be printed during training.

### 2.4. Semi-supervised HMM.


Using the HeterogenousHMM it is possible to fix the emission probabilities of the discrete features. To do so, two parameters of the initialization must be taken into account:  

- *'nr_no_train_de'*: indicates the number of discrete features we don´t want to be trainned by the model but the keep fixed to an original value set by the user. 

Two examples to illustrate how to use this variable:

--  If *nr_no_train_de=1* and *n_d_emissions=1*, the model would just have one discrete feature whose emission probabilities would be fixed (not trainned by the EM algorithm).

-- If *nr_no_train_de=1* but *n_d_emissions=3*, the model would train the emission probabilities matrices for the two first discrete features but would keep the value of the last emission probabilities matrix to the values set by the user.

- *'variablestate_no_train_de'*, that can be used to fix **just some of the states** (the last *'variablestate_no_train_de'* are the ones fixed) of the *'nr_no_train_de'* features while training the emission probabilities of the others. By default it is set to None, which means that the entire emission probability matrix for that discrete emission will be kept unchanged during training.

-- For example, if *nr_no_train_de=1*,  *n_d_emissions=2*, *n_states=5* and *variablestate_no_train_de = 2*, the model would train the complete emission probabilities matrix for the first discrete feature. For the second discrete feature, the emission probabilities for the 3 first states would be trainned with the EM algorithm but the emission probabilities for the last 2 states (of the 5 that the model has) would be fixed to the values fixed by the user. 

*This is extremely helpful if we have to have a Semi-Supervised HMM because we can associate certain states to certain labels/discrete values.*

**An example to clarify how to use these variables this can be found on the "hmm_tutorials.ipynb" notebook**.

## 3. Folder Structure.

- /src: it contains all the classes that implement the models.
- /notebooks: it contains:

> "hmm_tutorials.ipynb": an example code to use each of the available models.

>  "model_order_selection.ipynb": an example of how to use the order selection criteria. Both "Akaike Information Criterion" (AIC) and Bayesian Information Criterion (BIC) are implemented.

- /test: it contains the testing files for each of the HMM models.

## 4. Dependencies. 

The required dependencies are specified in *requirements.txt*.

## 5. Authors.

The current project has been developed by:

- [Fernando Moreno-Pino](http://www.tsc.uc3m.es/~fmoreno/).
- [Emese Sukei](https://github.com/semese).
- [Antonio Artés-Rodríguez](http://www.tsc.uc3m.es/~antonio/antonio_artes/Home.html).


## 6. Contact Information.

> fmoreno@tsc.uc3m.es

## 7. References.

- Advanced Signal Processing Course, by Prof. Dr. Antonio Artés-Rodríguez at Universidad Carlos III de Madrid.
- L. R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech recognition," in Proceedings of the IEEE, vol. 77, no. 2, pp. 257-286, Feb. 1989.
- K.P. Murphy, "Machine Learning: A Probabilistic Perspective", The MIT Press ©2012, ISBN:0262018020 9780262018029
- O.Capp, E.Moulines, T.Ryden, "Inference in Hidden Markov Models", Springer Publishing Company, Incorporated, 2010, ISBN:1441923195

This model has been based in previous implementations:

- https://github.com/guyz/HMM
- https://github.com/hmmlearn

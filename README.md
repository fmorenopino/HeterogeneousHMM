# Heterogenous-HMM (HMM with labels)

[![DOI](https://zenodo.org/badge/180347583.svg)](https://zenodo.org/badge/latestdoi/180347583)

NOTE: this README file is still under construction.


This repository contains different implementations of the Hidden Markov Model with just some basic Python dependencies. The main contributions of these libraries with respect to other available APIs are:

- **Missing values support**: our implementation supports both partial and complete missing data.

- **Heterogeneous HMM (HMM with labels)**: here we implement a version of the HMM which allow us to manage each of the features with different probability distributions (*also, the simpler cases: Gaussian and Multinomial HMMs are implemented*).

- **Semi-Supervised HMM (Fixing discrete emission probabilities)**: in the Heterogenous-HMM model, it is possible to fix the emission probabilities of the discrete features: the model allow us to fix the complete B matrix of certain feature or just some states' emission probabilities (while training the other states' emission probabilities).

- **Model selection criteria**: both BIC and AIC are implemented.

Also, some others aspects like having multiple sequences, several features per observation or sampling from the models are supported. This model is easily extendable with other types of probablistic models.


## How a HMM works?

In this library, we have developed different implementation of the Hidden Markov Model. In the next section of this readme file we explain how to use then but, before that, we would like to briefly explain how a HMM works. To do so, we will use the Gaussian HMM, which manages the emission probabilities with a gaussian distribution, its block diagram is represented in the next figure:


 <p align="center">
     <img src="https://raw.githubusercontent.com/fmorenopino/Heterogeneous_HMM/master/notebooks/img/hmm.png">
</p>

The parameters that we have on a HMM are:

 <p align="center">
     <img src="https://raw.githubusercontent.com/fmorenopino/Heterogeneous_HMM/master/notebooks/img/parameters.png" width="30%">
</p>
 
 Where:
 
 - *S* is the hidden state sequence, being *I* the number of states of the model.
 - *Y* is the observed continuous sequence (in the discrete case, it would be the observed discrete sequence).
 - ***A*** is the matrix that contains the state transition probabilities.
 - ***B*** are the observation emission probabilities, which for the gaussian case are managed with the means and covariances.
 - ***π*** is the initial state probability distribution. In our model, both random and k-means can be used to initialize it.
 
 Finally, the model's parameters of a HMM would be:
 
 <p align="center">
     <img src="https://raw.githubusercontent.com/fmorenopino/Heterogeneous_HMM/master/notebooks/img/theta.png" width="15%">
</p>



### The three basic inference problems for HMMs

In order to have a useful HMM model for a real application, there are three basic problems that must be solved:

- Problem 1: given the observed sequence *Y*, which is the probability of that observed sequence for our model's parameters *θ*, that is, which is p(*Y* | *θ*)?

... To solve this first problem, the Forward algorithm can be used.

- Problem 2: given the observed sequence *Y* and the model's parameters *θ*, which is the optimal state sequence *S*?

... To solve this second problem, several algorithms can be used, like the Viterbi or the Forward-Backward algorithm. If using Viterbi, we will maximize the p(*S*, *Y* | *θ*). Othercase, with the Forward-Backward algorithm, we optimizes the p(*s<sub>t</sub>*, *Y* | *θ*).
 
- Problem 3: which are the optimal *θ* that maximizes p(*Y* | *θ*)?

... To solve this third problem we must consider the joint distribution of *S* and *Y*, that is:

<p align="center">
     <img src="https://raw.githubusercontent.com/fmorenopino/Heterogeneous_HMM/master/notebooks/img/joint.png" width="50%">
</p>

By using the EM algorithm, the gaussian emission probabilities can be derived.

> The solution for these problems is nowadays very well known. If you want to get some extra knowledge about how the α, β, γ, δ... are derived, you should check the references for some extra information.


## Available models:

> Several implementations of the HMM have been developed, all these HMM models extend the *_BaseHMM* class.

- Multinomial HMMs (Discrete HMM): Hidden Markov Model with multinomial (discrete) emission probabilities.
- Gaussian HMMs: Hidden Markov Model with Gaussian emission probabilities.
- Heterogeneous HMM (HMM with labels): Hidden Markov Model with mixed discrete and gaussian emission probabilities.
- Semi-supervised HMM: in the Heterogeneous HMM, it is possible to fix the emission probabilities of the discrete features to guide the learning process of the model. *[An example can be found in the "hmm_tutorials.ipynb" notebook]*.

Now, a more detailed explanation of each of them is provided:





### 1. Multinomial HMM.

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

### 2. Gaussian HMM.

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

### 3. Heterogeneous HMM.

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

### 4. Semi-supervised HMM.


Using the HeterogenousHMM it is possible to fix the emission probabilities of the discrete features. To do so, two parameters of its initialization must be taken into account:  

- *'nr_no_train_de'*: indicates the number of discrete features we don´t want to be trainned by the model but the keep fixed to an original value set by the user. 

Two examples to illustrate how to use this variable:

-- First example: if *nr_no_train_de=1* and *n_d_emissions=1*, the model would just have one discrete feature whose emission probabilities would be fixed (not trainned by the EM algorithm).

-- Second example: if *nr_no_train_de=1* but *n_d_emissions=3*, the model would train the emission probabilities matrices for the two first discrete features but would keep the value of the last emission probabilities matrix to the values set by the user.

- *'variablestate_no_train_de'*, that can be used to fix just some of the states of that specific feature while training the emission probabilities of the others. 

-- For example, if *nr_no_train_de=1*,  *n_d_emissions=2*, *n_states=5* and *variablestate_no_train_de = 2*, the model would train the complete emission probabilities matrix for the first discrete feature. For the second discrete feature, the emission probabilities for the 3 first states would be trainned with the EM algorithm but the emission probabilities for the last 2 states (of the 5 that the model has) would be fixed to the values fixed by the user. 

*This is extremely helpful if we have to have a Semi-Supervised HMM because we can associate certain states to certain labels/discrete values.*

**An example to clarify this can be found on the "hmm_tutorials.ipynb" notebook**.

## Folder Structure.

- /src: it contains all the classes that implement the models.
- /notebooks: it contains:
-- "hmm_tutorials.ipynb": notebook that contains an example code to use each of the available models.
-- "model_order_selection.ipynb": notebook that contains an example of how to use the order selection criteria. Both "Akaike Information Criterion" (AIC) and Bayesian Information Criterion (BIC) are implemented.
- /test: it contains the testing files for each of the HMM models.

## Dependencies. 

The required dependencies are specified in *requirements.txt*.

## Authors.

The current project has been developed by:

- Fernando Moreno-Pino (http://www.tsc.uc3m.es/~fmoreno/, https://github.com/fmorenopino).
- Emese Sukei (https://github.com/semese).

## References.

- Advanced Signal Processing Course, by Prof. Dr. Antonio Artés-Rodríguez at Universidad Carlos III de Madrid.
- L. R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech recognition," in Proceedings of the IEEE, vol. 77, no. 2, pp. 257-286, Feb. 1989.
- K.P. Murphy, "Machine Learning: A Probabilistic Perspective", The MIT Press ©2012, ISBN:0262018020 9780262018029
- O.Capp, E.Moulines, T.Ryden, "Inference in Hidden Markov Models", Springer Publishing Company, Incorporated, 2010, ISBN:1441923195

This model has been based in previous implementations:

- https://github.com/guyz/HMM
- https://github.com/hmmlearn

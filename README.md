# Heterogenous-HMM (HMM with labels)

This repository implements a Heterogenous HMM, which is a version of the popular model that is capable to manage the emission probabilities of different observations with different probability distributions. **Also, the simpler cases (Gaussian and Discrete HMM) are implemented** with just Numpy dependencies.


### Available models:

- Multinomial HMMs (Discrete HMM): Hidden Markov Model with multinomial (discrete) emissions.
- Gaussian HMMs: Hidden Markov Model with Gaussian emissions.
- Heterogeneous HMM (HMM with labels): Hidden Markov Model with mixed discrete and gaussian emissions.

For those models, the project supports, among others:

- Multiple features per observation.
- Multiple sequences.
- *Missing data*: partial missing data (just some missing features) or complete missing data (all the features, that is, a complitely missing observation).
- Fixing parameters: for the Heterogeneous-HMM the discrete emission probabilities can be fixed during the training phase, which can help to interpret the results (fixing parameters allow us to, for example, to make the model unable to access a certain state x when the discrete observation have a certain value y by fixing the correspondand B matrix to p(state = x | disc_obs = y) = 0.
- Easily extendable with other types of probablistic models.


### Authors.

The current project has been developed by:

- Fernando Moreno Pino (https://github.com/fmorenopino).
- Emese Sukei (https://github.com/semese).

### Acknowledgments.

- Advanced Signal Processing Course, by Prof. Dr. Antonio Artés-Rodríguez at Universidad Carlos III de Madrid.
-L. R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech recognition," in Proceedings of the IEEE, vol. 77, no. 2, pp. 257-286, Feb. 1989.
- K.P. Murphy, "Machine Learning: A Probabilistic Perspective", The MIT Press ©2012, ISBN:0262018020 9780262018029
- O.Capp, E.Moulines, T.Ryden, "Inference in Hidden Markov Models", Springer Publishing Company, Incorporated, 2010, ISBN:1441923195

This model has been based in previous implementations:

- https://github.com/guyz/HMM
- https://github.com/hmmlearn

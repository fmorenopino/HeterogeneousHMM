# Heterogenous-HMM (HMM with labels)

This repository contains different implementations of the Hidden Markov Model with just some basic Python dependencies. The main contributions of this library with respect to other available APIs are:

- **Missing values support**: our implementation supports both partial and complete missing data.

- **Heterogeneous HMM (HMM with labels)**: here we implement a version of the HMM which allow us to use different distributions to manage the emission probabilities of each of the features (*also, the simpler cases: Gaussian and Multinomial HMMs are implemented*).

- **Semi-Supervised HMM (Fixing discrete emission probabilities)**: in the Heterogenous-HMM model, it is possible to fix the emission probabilities of the discrete features: the model allow us to fix the complete emission probabilities matrix *B* of certain feature or just some states' emission probabilities.

- **Model selection criteria**: Both "Akaike Information Criterion" (AIC) and Bayesian Information Criterion (BIC) are implemented.

- **Several types of covariance matrix implemented** for the gaussian observations: diagonal, full, tied or spherical covariance matrices can be used.

- **The models can be used to sample data**, which means that once you have trainned a model (or you have fixed the parameters according to the distributions you want to use) you can generate sequences of data.

Also, some others aspects like having multiple sequences, several features per observation or sampling from the models are supported. This model is easily extendable with other types of probablistic models. There is also a possibility to run the training using multiprocessing, in order to speed it up when multiple observation sequences are used. 

## Authors

The current project has been developed by:

- [Fernando Moreno-Pino](http://www.tsc.uc3m.es/~fmoreno/). Contact: fmoreno@tsc.uc3m.es
- [Emese Sukei](http://www.tsc.uc3m.es/~esukei/). Contact: esukei@tsc.uc3m.es
- [Antonio Artés-Rodríguez](http://www.tsc.uc3m.es/~antonio/antonio_artes/Home.html). 


## References

- Advanced Signal Processing Course, by Prof. Dr. Antonio Artés-Rodríguez at Universidad Carlos III de Madrid.
- L.R. Rabiner, "A tutorial on hidden Markov models and selected applications in speech recognition," in Proceedings of the IEEE, vol. 77, no. 2, pp. 257-286, Feb. 1989.
- K.P. Murphy, "Machine Learning: A Probabilistic Perspective", The MIT Press ©2012, ISBN:0262018020 9780262018029
- O.Capp, E.Moulines, T.Ryden, "Inference in Hidden Markov Models", Springer Publishing Company, Incorporated, 2010, ISBN:1441923195
- M.V. Anikeev, O.B. Makarevich, "Parallel Implementation of Baum-Welch Algorithm", Workshop on Computer Science and Information Technology CSIT'2006, Karlsruhe, Germany, 2006

This model was based on previous implementations:

- https://github.com/guyz/HMM
- https://github.com/hmmlearn

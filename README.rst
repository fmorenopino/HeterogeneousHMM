PyHHMM
======

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3759439.svg
   :target: https://doi.org/10.5281/zenodo.3759439

[`Read the Docs <https://pyhhmm.readthedocs.io/en/latest/index.html#>`_]
   
This repository contains different implementations of the Hidden Markov Model with just some basic Python dependencies. The main contributions of this library with respect to other available APIs are:

- **Missing values support**: our implementation supports both partial and complete missing data.

- **Heterogeneous HMM (HMM with labels)**: here we implement a version of the HMM which allow us to use different distributions to manage the emission probabilities of each of the features (*also, the simpler cases: Gaussian and Multinomial HMMs are implemented*).

- **Semi-Supervised HMM (Fixing discrete emission probabilities)**: in the HeterogenousHMM model, it is possible to fix the emission probabilities of the discrete features: the model allow us to fix the complete emission probabilities matrix *B* of certain feature or just some states' emission probabilities.

- **Model selection criteria**: Both Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) are implemented.

Also, some others aspects like having multiple sequences, several features per observation or sampling from the models are supported. This model is easily extendable with other types of probabilistic models. There is also a possibility to run the training using multiprocessing, in order to speed it up when multiple observation sequences are used. 

Documentation
-------------
Introductory tutorials, how-to's and API documentation are available on `Read the Docs <https://pyhhmm.readthedocs.io/en/latest/>`_.

Authors
-------
- `Fernando Moreno-Pino <http://www.tsc.uc3m.es/~fmoreno/>`_
- `Emese Sukei <http://www.tsc.uc3m.es/~esukei/>`_
- `Antonio Artés-Rodríguez <http://www.tsc.uc3m.es/~antonio/antonio_artes/Home.html>`_

Contributing
------------
If you like this project and want to help, we would love to have your contribution! Please see `CONTRIBUTING <https://github.com/fmorenopino/HeterogeneousHMM/blob/master/CONTRIBUTING.md>`_ and contact us to get started.

References
----------
- *Advanced Signal Processing Course*, Prof. Dr. Antonio Artés-Rodríguez at Universidad Carlos III de Madrid
- *A tutorial on hidden Markov models and selected applications in speech recognition*, L.R. Rabiner, in Proceedings of the IEEE, vol. 77, no. 2, pp. 257-286, Feb. 1989
- *Machine Learning: A Probabilistic Perspective*, K.P. Murphy, The MIT Press ©2012, ISBN:0262018020 9780262018029
- *Inference in Hidden Markov Models*, O.Capp, E.Moulines, T.Ryden, Springer Publishing Company, Incorporated, 2010, ISBN:1441923195
- *Parallel Implementation of Baum-Welch Algorithm*, M.V. Anikeev, O.B. Makarevich, Workshop on Computer Science and Information Technology CSIT'2006, Karlsruhe, Germany, 2006

**NOTE:** This model was based on previous implementations:

- `https://github.com/guyz/HMM <https://github.com/guyz/HMM>`_
- `https://github.com/hmmlearn <https://github.com/hmmlearn>`_

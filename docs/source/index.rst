.. PyHHMM documentation master file, created by
   sphinx-quickstart on Mon Nov 22 16:53:45 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyHHMM
======

Unsupervised learning and inference of Hidden Markov Models:

* Simple algorithms and models to learn HMMs (`Hidden Markov Models <https://en.wikipedia.org/wiki/Hidden_Markov_model>`_) in Python
* Missing values support: our implementation supports both partial and complete missing data
* Heterogeneous HMM (HMM with labels): here we implement a version of the HMM which allow us to use different distributions to manage the emission probabilities of each of the features (also, the simpler cases: Gaussian and Multinomial HMMs are implemented)
* Semi-Supervised HMM (Fixing discrete emission probabilities): in the HeterogenousHMM model, it is possible to fix the emission probabilities of the discrete features: the model allow us to fix the complete emission probabilities matrix B of certain feature or just some states' emission probabilities,
* Model selection criteria: Both Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) are implemented
* Built on scikit-learn, NumPy, SciPy, and Seaborn
* Open source, commercially usable --- `Apache License 2.0 license <https://opensource.org/licenses/Apache-2.0>`_.
  

User guide: table of contents
-----------------------------

.. toctree::
   :maxdepth: 2

   getting_started
   tutorial
   examples
   api
   
.. topic:: References:

   - *Advanced Signal Processing Course*, by Prof. Dr. Antonio Artés-Rodríguez at Universidad Carlos III de Madrid.
   - *A tutorial on hidden Markov models and selected  applications in speech recognition*, L.R. Rabiner, in Proceedings of the IEEE, vol. 77, no. 2, pp. 257-286, Feb. 1989.
   - *Machine Learning: A Probabilistic Perspective*, K.P. Murphy, The MIT Press ©2012, ISBN:0262018020 9780262018029
   - *Inference in Hidden Markov Models*, O.Capp, E.Moulines, T.Ryden, Springer Publishing Company, Incorporated, 2010, ISBN:1441923195
   - *Parallel Implementation of Baum-Welch Algorithm*, M.V. Anikeev, O.B. Makarevich, Workshop on Computer Science and Information Technology CSIT'2006, Karlsruhe, Germany, 2006
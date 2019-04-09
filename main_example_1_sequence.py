# !/usr/bin/env python3
# -*- coding: utf-8 -*-



"""
This file executes a Gaussian HMM with 1 Gaussian per observation (not GMM), 4 states and 2 features per
observation. It is done with synthetic data sampled from a hmmlearn model (available on the data folder).

Just 1 secuence is used.
"""

from hmm.continuous.GMHMM import GMHMM
from hmm.discrete.DiscreteHMM import DiscreteHMM
import numpy as np
from data.Data import Data
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
import os
import time
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans


def test_main_Synthetic_Data():

    data = Data()
    (obs, labels, A_original) = data.generateSyntheticData(200)
    obs = obs[:,:]
    labels = labels[:]

    obs = preprocessing.scale(obs)

    print (np.mean(obs[:,0]))
    print (np.mean(obs[:,1]))
    print (np.std(obs[:,0]))
    print (np.std(obs[:,1]))

    n = 4
    m = 1
    d = 2

    a = np.array([np.random.dirichlet(alpha=5 * np.ones(n)) for i in range(n)])

    wtmp = np.random.random_sample((n, m))
    row_sums = wtmp.sum(axis=1)
    w = np.array(wtmp / row_sums[:, np.newaxis], dtype=np.double)

    means = np.array((0.6 * np.random.random_sample((n, m, d)) - 0.3), dtype=np.double)
    covars = np.zeros((n, m, d, d))

    #Hago un KMEANS para crear tantos clusters como estados ocultos y obtener sus medias
    #CUIDADO: cuando tenga múltiples secuencias tendré que concatenarlas
    kmeans = KMeans(n_clusters=n, random_state=0).fit(obs)
    kmeans_clusters_centers_ = kmeans.cluster_centers_


    for i in range(0, np.shape(means)[0]):
        for j in xrange(m):
            # Estoy inicializando TODOS LOS GMM con las mismas medias
            means[i][j][0] = kmeans_clusters_centers_[i][0]
            means[i][j][1] = kmeans_clusters_centers_[i][1]


    for i in xrange(n):
        for j in xrange(m):
            for k in xrange(d):
                covars[i][j][k][k] = 1

    pitmp = np.random.random_sample((n))
    pi = np.array(pitmp / sum(pitmp), dtype=np.double)


    """
    pi_prueba = np.array(pitmp / sum(pitmp), dtype=np.double)
    pi_z = np.array([np.random.dirichlet(alpha=5 * np.ones(n))])

    for i in xrange(len(pi_z)):
        pi_prueba[i] = pi_z[0][i]
    """

    gmmhmm = GMHMM(n, m, d, a, means, covars, w, pi, init_type='user', verbose=True)



    #scaler = MinMaxScaler(feature_range=(-1, 1))
    #scaler.fit(obs.reshape(-1, 2))
    #obs = scaler.transform(obs.reshape(-1, 2))


    print "Doing Baum-welch"
    gmmhmm.train(obs, 100)
    print
    print "Pi", gmmhmm.pi
    print "A", gmmhmm.A
    print "weights", gmmhmm.w
    print "means", gmmhmm.means
    print "covars", gmmhmm.covars

    predictions = gmmhmm.decode(obs)



    plt.close('all')
    fig_synthetic_data = plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(labels)
    plt.title("real")
    plt.subplot(2, 1, 2)
    plt.plot(predictions)
    plt.title("predictions")
    plt.show()




test_main_Synthetic_Data()
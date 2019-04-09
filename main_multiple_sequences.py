# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file executes a Gaussian HMM with 1 Gaussian per observation (not GMM), 4 states and 2 features per
observation. It is done with synthetic data sampled from a hmmlearn model (available on the data folder),
that data has 2 different secuences.
"""

from hmm.continuous.GMHMM import GMHMM
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
from scipy.stats import mode

def plotResults(n_sequences, labels_sequences, predictions_sequences):

    for i in range(0, n_sequences):

        plt.close('all')
        plt.figure(i)
        plt.subplot(2, 1, 1)
        plt.plot(labels_sequences[i])
        plt.title("real")
        plt.subplot(2, 1, 2)
        plt.plot(predictions_sequences[i])
        plt.title("predictions")
        plt.show()



def obtainAccuracy(n_sequences, n, predictions, labels_concatenated):


    for i in range(0, n_sequences):
        if (i == 0):
            tmp = predictions[i]
        else:
            tmp = np.concatenate((tmp, predictions[i]))

    preditions_concatenated = tmp


    states_relation = np.empty((n, 1))
    for state in range (0, n):
        labels_indexes = np.where(labels_concatenated == state)
        predicted_labels = preditions_concatenated[labels_indexes]
        states_relation[state] = mode(predicted_labels)[0][0]

    labels_concatenated[0:20] = 0

    error = 0
    for i in range(0, len(labels_concatenated)):
        label = labels_concatenated[i]
        if (preditions_concatenated[i] != states_relation[label][0]):
            error += 1
    error_percentage = (error/float(len(labels_concatenated)))*100

    return  error_percentage




def test_main_Synthetic_Data():


    data = Data()
    (obs_sequences, labels_sequences) = data.generateSyntheticDataSeveralSequences(2)
    n_sequences = len(obs_sequences)

    for i in range(0, n_sequences):
        obs_sequences[i] = preprocessing.scale(obs_sequences[i])
        if (i == 0):
            tmp = obs_sequences[i]
            tmp_labels = labels_sequences[i]
        else:
            tmp = np.concatenate((tmp, obs_sequences[i]))
            tmp_labels = np.concatenate((tmp_labels, labels_sequences[i]))

    sequences_concatenated = tmp
    labels_concatenated = tmp_labels


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

    #We concatenate the sequences


    kmeans = KMeans(n_clusters=n, random_state=0).fit(sequences_concatenated)
    kmeans_clusters_centers_ = kmeans.cluster_centers_


    for i in range(0, np.shape(means)[0]):
        for j in xrange(m):
            # PENDIENTE DE CAMBIAR EN EL FUTURO: estoy inicializando TODOS LOS GMM con las mismas medias
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

    gmmhmm.train(obs_sequences, 100)
    print
    print "Pi", gmmhmm.pi
    print "A", gmmhmm.A
    print "weights", gmmhmm.w
    print "means", gmmhmm.means
    print "covars", gmmhmm.covars

    predictions_sequences = gmmhmm.decode(obs_sequences)
    #predictions is a list of predictions for each sequence

    plotResults(n_sequences, labels_sequences, predictions_sequences)


    #Measuring error
    error_percentage = obtainAccuracy(n_sequences, n, predictions, labels_concatenated)
    print ("Error: " + str(error_percentage) + " %")

    print ("hello")





test_main_Synthetic_Data()
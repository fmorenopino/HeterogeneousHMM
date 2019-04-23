# !/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Mar 20, 2018

@author: fmorenopino
'''
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import sys
from hmmlearn import hmm
import matplotlib.pyplot as plt

################################################################
start = datetime(2019, 1, 18, 10, 0)
end = datetime(2019, 1, 20, 10, 0)
################################################################

class Data(object):

    ###########################################################################################
    """
    Functions defined for importing HAR data from certains files of my personal project.
    """

    def getDatetime(self, x):


        return (datetime.strptime(x, '%Y-%m-%d %H:%M:%S UTC'))

    def reading_CSV(self, name):

        #Esqueleto para la función que leerá los csv con todos los datos disponibles de cada usuario
        data = pd.read_csv(name, encoding="ISO-8859-1")
        testing = np.asarray(data)


        #NaN en Actígrafo
        nan_index = pd.isnull(testing[:, 2])
        act_nan_index = np.where(nan_index)[0]
        testing[:, 2][act_nan_index] = float(1568)

        # NaN en Luz
        nan_index = pd.isnull(testing[:, 3])
        lig_nan_index = np.where(nan_index)[0]
        testing[:, 3][lig_nan_index] = float(0)


        tmp = map(self.getDatetime, testing[:, 1]) #Pasamos las fechas a formato datetime

        all_users = np.hstack((testing[:, 0].reshape(-1,1), np.asarray(tmp).reshape(-1,1), testing[:, 2].reshape(-1,1), testing[:, 3].reshape(-1,1)))


        return all_users

    def obtainUserBetweenDates(self, data, use_dates, start, end, user):



        # data is a numpy array of actigrapy, light, steep data

        # Tengo que empezar por el final
        index_of_user = np.where(data[:, 0] == user)
        users_list = data[:, 0]
        data = data[index_of_user]  # Aquí obtengo el usuario que me dicen

        if (use_dates == True):

            index = 0
            found_start = False
            found_end = False
            index_start = 0
            index_end = 0

            while (found_end == False and index <= np.shape(data)[0] - 1):
                if ((found_start == False) and (data[index, 1] > start and data[index, 1] < end)):
                    index_start = index
                    found_start = True
                if ((found_end == False) and (data[index, 1] > end and data[index, 1] > start)):
                    index_end = index
                    found_end = True
                index = index + 1

            # print (found_start)
            # print (found_end)
            slice_of_vector = data[index_start:index_end]
            return slice_of_vector

        else:
            return data


    def getData(self):
        all_users = self.reading_CSV("users_actigraphylight.csv")
        all_users = np.flip(all_users, axis=0)
        all_users = np.nan_to_num(all_users)

        #all_users[all_users[:, 2] > 1750, 2] = 1750
        #all_users[all_users[:, 3] > 2000, 2] = 2000

        # Ahora vamos a obtener las longitudes de las secuencias correspondientes a cada usuario
        selected_users = all_users[:, :]
        lengths = []
        keys = sorted(set(selected_users[:, 0]), key=list(selected_users[:, 0]).index)

        for key in keys:
            lengths.append(list(selected_users[:, 0]).count(key))

        scores = []
        n_components = np.array(range(2, 5))

        required_user = self.obtainUserBetweenDates(all_users, True, start, end, 'AAR') #test-asun, acobo-das-firebase

        required_user[required_user[:, 2] > 2000, 2] = 2000
        required_user[required_user[:, 3] > 2000, 2] = 2000

        return required_user

    ###########################################################################################





    ###########################################################################################
    """
    Here we sample from a model builded with hmmlearn whose parameter are fixed. This data
    will be used to test the performance of the model while building the HMM.
    """

    def generateSyntheticData(self, length=500):

        startprob = np.array([0.6, 0.3, 0.1, 0.0])
        # The transition matrix, note that there are no transitions possible
        # between component 1 and 3
        transmat = np.array([[0.7, 0.2, 0.0, 0.1],
                             [0.3, 0.5, 0.2, 0.0],
                             [0.0, 0.3, 0.5, 0.2],
                             [0.2, 0.0, 0.2, 0.6]])
        # The means of each component
        means = np.array([[0.0, 0.0],
                          [0.0, 11.0],
                          [9.0, 10.0],
                          [11.0, -1.0]])
        # The covariance of each component
        covars = .5 * np.tile(np.identity(2), (4, 1, 1))

        # Build an HMM instance and set parameters
        model = hmm.GaussianHMM(n_components=4, covariance_type="full")

        # Instead of fitting it from the data, we directly set the estimated
        # parameters, the means and covariance of the components
        model.startprob_ = startprob
        model.transmat_ = transmat
        model.means_ = means
        model.covars_ = covars

        X, Z = model.sample(length)

        # Plot the sampled data
        plt.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=6,
                 mfc="orange", alpha=0.7)

        # Indicate the component numbers
        for i, m in enumerate(means):
            plt.text(m[0], m[1], 'Component %i' % (i + 1),
                     size=17, horizontalalignment='center',
                     bbox=dict(alpha=.7, facecolor='w'))
        plt.legend(loc='best')
        #plt.show()

        return (X,Z,transmat)



    def generateSyntheticDataSeveralSequences(self, n_sequences = 5):

        list_of_lenghts = [600, 400]
        #list_of_lenghts = [1000, 1500, 2000, 1355, 4000]
        obs_sequences = []
        labels_sequences = []

        for i in range(0, n_sequences):

            (obs, labels, A_original) = self.generateSyntheticData(list_of_lenghts[i])
            obs_sequences.append(obs)
            labels_sequences.append(labels)

        return (obs_sequences, labels_sequences)

    ###########################################################################################





# !/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Mar 20, 2018

@author: fmorenopino

This code is based on:
 - HMM implementation by guyz- https://github.com/guyz/HMM
'''

from hmm._BaseHMM import _BaseHMM
import numpy as np

class _ContinuousHMM(_BaseHMM):
    '''
    A Continuous HMM - This is a base class implementation for HMMs with
    mixtures. A mixture is a weighted sum of several continuous distributions,
    which can therefore create a more flexible general PDF for each hidden state.
    
    This class can be derived, but should not be used directly. Deriving classes
    should generally only implement the PDF function of the mixtures.
    
    Model attributes:
    - n            number of hidden states
    - m            number of mixtures in each state (each 'symbol' like in the discrete case points to a mixture)
    - d            number of features (an observation can contain multiple features)
    - A            hidden states transition probability matrix ([NxN] numpy array)
    - means        means of the different mixtures ([NxMxD] numpy array)
    - covars       covars of the different mixtures ([NxM] array of [DxD] covar matrices)
    - w            weighing of each state's mixture components ([NxM] numpy array)
    - pi           initial state's PMF ([N] numpy array).
    
    Additional attributes:
    - min_std      used to create a covariance prior to prevent the covariances matrices from underflowing
    - precision    a numpy element size denoting the precision
    - verbose      a flag for printing progress information, mainly when learning
    '''

    def __init__(self,n,m,d=1,A=None,means=None,covars=None,w=None,pi=None,min_std=0.01,init_type='uniform',precision=np.double,verbose=False):
        '''
        Construct a new Continuous HMM.
        In order to initialize the model with custom parameters,
        pass values for (A,means,covars,w,pi), and set the init_type to 'user'.
        
        Normal initialization uses a uniform distribution for all probablities,
        and is not recommended.
        '''
        _BaseHMM.__init__(self,n,m,precision,verbose) #@UndefinedVariable
        
        self.d = d
        self.A = A
        self.pi = pi
        self.means = means
        self.covars = covars
        self.w = w
        self.min_std = min_std

        self.reset(init_type=init_type)

    def reset(self,init_type='uniform'):
        '''
        If required, initalize the model parameters according the selected policy
        '''        
        if init_type == 'uniform':
            self.pi = np.ones( (self.n), dtype=self.precision) *(1.0/self.n)
            self.A = np.ones( (self.n,self.n), dtype=self.precision)*(1.0/self.n)
            self.w = np.ones( (self.n,self.m), dtype=self.precision)*(1.0/self.m)
            self.means = np.zeros( (self.n,self.m,self.d), dtype=self.precision)
            self.covars = [[ np.matrix(np.ones((self.d,self.d), dtype=self.precision)) for j in xrange(self.m)] for i in xrange(self.n)]
        elif init_type == 'user':
            # if the user provided a 4-d array as the covars, replace it with a 2-d array of np matrices.
            covars_tmp = [[ np.matrix(np.ones((self.d,self.d), dtype=self.precision)) for j in xrange(self.m)] for i in xrange(self.n)]
            for i in xrange(self.n):
                for j in xrange(self.m):
                    if type(self.covars[i][j]) is np.ndarray:
                        covars_tmp[i][j] = np.matrix(self.covars[i][j])
                    else:
                        covars_tmp[i][j] = self.covars[i][j]
            self.covars = covars_tmp
              
    def _mapB(self,observations_sequences):
        '''
        Required implementation for _mapB. Refer to _BaseHMM for more details.
        This method highly optimizes the running time, since all PDF calculations
        are done here once in each training iteration.
        
        - self.Bmix_map - computesand maps Bjm(Ot) to Bjm(t).
        '''

        n_sequences = len(observations_sequences)
        self.B_map_list = []
        self.Bmix_map_list = []

        for sn in range(0, n_sequences):
            obs = observations_sequences[sn]
            B_map = np.zeros( (self.n,len(obs)), dtype=self.precision)
            Bmix_map = np.zeros( (self.n,self.m,len(obs)), dtype=self.precision)
            self.Bmix_map_list.append(Bmix_map)
            for j in xrange(self.n):
                for t in xrange(len(obs)):
                    #print (t)
                    if (np.any(np.isnan(obs[t]))):  # if any of the features at time 't' is nan, bjt = 1
                        """
                        When we have a missing observation, the value of p(y_t | s_t = i) is equal to 
                        p(means[j]), which is 1 (because we are integrating over p(y_t | s_t = i).
                        """
                        B_map[j][t] = self._calcbjt(sn, j, t, (self.means[j])[0].flat)

                    B_map[j][t] = self._calcbjt(sn, j, t, obs[t])

            self.B_map_list.append(B_map)

                
    """
    b[j][Ot] = sum(1...M)w[j][m]*b[j][m][Ot]
    Returns b[j][Ot] based on the current model parameters (means, covars, weights) for the mixtures.
    - j - state
    - Ot - the current observation
    Note: there's no need to get the observation itself as it has been used for calculation before.
    """    
    def _calcbjt(self,sn, j,t,Ot):
        '''
        Helper method to compute Bj(Ot) = sum(1...M){Wjm*Bjm(Ot)}
        '''

        #IMPORTANTE: añadido este if para el caso en que la observación es nula poner el bjt = 1
        if  (np.any(np.isnan(Ot)) ): #if any of the features at time 't' is nan, bjt = 1
            bjt = 1
            for m in xrange(self.m):
                self.Bmix_map_list[sn][j][m][t] = 1
        else:
            bjt = 0
            for m in xrange(self.m):
                self.Bmix_map_list[sn][j][m][t] = self._pdf(Ot, self.means[j][m], self.covars[j][m])
                bjt += (self.w[j][m]*self.Bmix_map_list[sn][j][m][t])
        return bjt
        
    def _calcgammamix(self,alpha,beta,observations_sequences):
        '''
        Calculates 'gamma_mix'.
        
        Gamma_mix is a (TxNxK) numpy array, where gamma_mix[t][i][m] = the probability of being
        in state 'i' at time 't' with mixture 'm' given the full observation sequence.
        '''



        n_sequences = len(observations_sequences)
        gamma_mix_list = []

        for sn in range(0, n_sequences):

            observations = observations_sequences[sn]

            gamma_mix = np.zeros((len(observations),self.n,self.m),dtype=self.precision)

            for t in xrange(len(observations)):
                for j in xrange(self.n):
                    for m in xrange(self.m):
                        alphabeta = 0.0
                        for jj in xrange(self.n):
                            alphabeta += alpha[sn][t][jj]*beta[sn][t][jj]
                        comp1 = (alpha[sn][t][j]*beta[sn][t][j]) / alphabeta

                        bjk_sum = 0.0
                        for k in xrange(self.m):
                            bjk_sum += (self.w[j][k]*self.Bmix_map_list[sn][j][k][t]) #PENDIENTE: Bmix_map_list está siendo 0 por los NaN y hace que divida entre 0
                        comp2 = (self.w[j][m]*self.Bmix_map_list[sn][j][m][t])/bjk_sum

                        gamma_mix[t][j][m] = comp1*comp2

            gamma_mix_list.append(gamma_mix)

        return gamma_mix_list
    
    def _updatemodel(self,new_model):
        '''
        Required extension of _updatemodel. Adds 'w', 'means', 'covars',
        which holds the in-state information. Specfically, the parameters
        of the different mixtures.
        '''        
        _BaseHMM._updatemodel(self,new_model) #@UndefinedVariable
        
        self.w = new_model['w']
        self.means = new_model['means']
        self.covars = new_model['covars']
        
    def E_step(self,observations):
        '''
        Extension of the original method so that it includes the computation
        of 'gamma_mix' stat.
        '''
        stats = _BaseHMM.E_step(self,observations) #@UndefinedVariable
        stats['gamma_mix'] = self._calcgammamix(stats['alpha'],stats['beta'],observations)

        return stats
    
    def M_step(self,stats,observations):
        '''
        Required extension of M_step. 
        Adds a re-estimation of the mixture parameters 'w', 'means', 'covars'.
        '''        
        # re-estimate A, pi
        new_model = _BaseHMM.M_step(self,stats,observations) #@UndefinedVariable
        """
        At this point, in "new_model" we store the value of "A" and "pi", we proceed now to compute 
        the weights, means and covars
        """
        
        # re-estimate the continuous probability parameters of the mixtures
        w_new, means_new, covars_new = self.M_stepMixtures(observations,stats['gamma_mix'])


        new_model['w'] = w_new
        new_model['means'] = means_new
        new_model['covars'] = covars_new
        
        return new_model
    
    def M_stepMixtures(self,observations_sequences,gamma_mix):
        '''
        Helper method that performs the Baum-Welch 'M' step
        for the mixture parameters - 'w', 'means', 'covars'.
        '''        
        w_new = np.zeros( (self.n,self.m), dtype=self.precision)
        means_new = np.zeros( (self.n,self.m,self.d), dtype=self.precision)
        covars_new = [[ np.matrix(np.zeros((self.d,self.d), dtype=self.precision)) for j in xrange(self.m)] for i in xrange(self.n)]

        n_sequences = len(observations_sequences)

        for sn in range(0, n_sequences):

            observations = observations_sequences[sn]

            for j in xrange(self.n):
                for m in xrange(self.m):
                    numer = 0.0
                    denom = 0.0
                    for t in xrange(len(observations)):
                        for k in xrange(self.m):
                            denom += (self._eta(t,len(observations)-1)*gamma_mix[sn][t][j][k])
                        numer += (self._eta(t,len(observations)-1)*gamma_mix[sn][t][j][m])
                    w_new[j][m] = numer/denom
                w_new[j] = self._normalize(w_new[j])
            """
            El siguiente bloque está devolviendo NaN en means (en el anterior está todo bien aparentemente)
            """
            for j in xrange(self.n):
                for m in xrange(self.m):
                    numer = np.zeros( (self.d), dtype=self.precision)
                    denom = np.zeros( (self.d), dtype=self.precision)
                    for t in xrange(len(observations)):

                        #np.any(np.isnan(observations[t]))

                        numer += (self._eta(t,len(observations)-1)*gamma_mix[sn][t][j][m]*observations[t])
                        denom += (self._eta(t,len(observations)-1)*gamma_mix[sn][t][j][m])
                    means_new[j][m] = numer/denom

            cov_prior = [[ np.matrix(self.min_std*np.eye((self.d), dtype=self.precision)) for j in xrange(self.m)] for i in xrange(self.n)]
            for j in xrange(self.n):
                for m in xrange(self.m):
                    numer = np.matrix(np.zeros( (self.d,self.d), dtype=self.precision))
                    denom = np.matrix(np.zeros( (self.d,self.d), dtype=self.precision))
                    for t in xrange(len(observations)):
                        vector_as_mat = np.matrix( (observations[t]-self.means[j][m]), dtype=self.precision )
                        numer += (self._eta(t,len(observations)-1)*gamma_mix[sn][t][j][m]*np.dot( vector_as_mat.T, vector_as_mat))
                        denom += (self._eta(t,len(observations)-1)*gamma_mix[sn][t][j][m])
                    covars_new[j][m] = numer/denom
                    covars_new[j][m] = covars_new[j][m] + cov_prior[j][m]

            return w_new, means_new, covars_new
    
    def _normalize(self, arr):
        '''
        Helper method to normalize probabilities, so that
        they all sum to '1'
        '''
        summ = np.sum(arr)
        for i in xrange(len(arr)):
            arr[i] = (arr[i]/summ)
        return arr
    
    def _pdf(self,x,mean,covar):
        '''
        Deriving classes should implement this method. This is the specific
        Probability Distribution Function that will be used in each
        mixture component.
        '''        
        raise NotImplementedError("PDF function must be implemented")
    
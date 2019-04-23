# !/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on Mar 20, 2018

@author: fmorenopino

This code is based on:
 - HMM implementation by guyz- https://github.com/guyz/HMM
'''

import numpy
import math
from sklearn import preprocessing


class _BaseHMM(object):

    '''
    Basics of the HMM.
    '''

    def __init__(self,n,m,precision=numpy.double,verbose=False):
        self.n = n
        self.m = m

        self.precision = precision
        self.verbose = verbose
        self._eta = self._eta1

        self.old_stats = None

    def train(self, observations, iterations=1, epsilon=0.0001,
              thres=(0.0015 / 100)):  # poner el número en tanto por ciento
        '''
        Updates the HMMs parameters given a new set of observed sequences.

        observations can either be a single (1D) array of observed symbols, or when using
        a continuous HMM, a 2D array (matrix), where each row denotes a multivariate
        time sample (multiple features).

        Training is repeated 'iterations' times, or until log likelihood of the model
        increases by less than 'epsilon'.

        'thres' denotes the algorithms sensitivity to the log likelihood decreasing
        from one iteration to the other.
        '''

        self._mapB(observations)

        for i in xrange(iterations):
            (prob_old, prob_new, Q_value_tmp) = self.trainiter(observations)

            if (self.verbose):
                # print "iter: ", i, ", L(model|O) =", prob_old, ", L(model_new|O) =", prob_new, ", converging =", ( abs( (prob_new-prob_old) / prob_old) < thres ) #converging =", ( prob_new-prob_old > thres )
                print "iter: ", i, ", L(model|O) =", prob_old, ", L(model_new|O) =", prob_new  # converging =", ( prob_new-prob_old > thres )

                """
                if (i == 0):
                    Q_value_new = Q_value_tmp
                    Q_value_old = None
                else:
                    Q_value_old = Q_value_new
                    Q_value_new = Q_value_tmp

                    print "iter: ", i, ", L(model|O) =", Q_value_old, ", L(model_new|O) =", Q_value_new  # converging =", ( prob_new-prob_old > thres )
                """

            # if ( abs( (prob_new-prob_old) / prob_old) < thres ): #if ( abs(prob_new-prob_old) < epsilon ):
            # converged
            # break
            if (prob_new == prob_old):
                break



    def trainiter(self, observations):
        '''
        A single iteration of an EM algorithm, which given the current HMM,
        computes new model parameters and internally replaces the old model
        with the new one.

        Returns the log likelihood of the old model (before the update),
        and the one for the new model.
        '''
        # call the EM algorithm
        (new_model, stats) = self._baumwelch(observations)

        # calculate the log likelihood of the previous model
        prob_old = self.forwardbackward(observations,
                                        cache=True)  # we compute the P(O|model) for the set of old parameters

        # update the model with the new estimation
        self._updatemodel(new_model)

        # calculate the log likelihood of the new model. Cache set to false in order to recompute probabilities of the observations give the model.
        prob_new = self.forwardbackward(observations, cache=False)

        # Q_value = self.Q_computation(stats, new_model, observations)
        Q_value = None

        return (prob_old, prob_new, Q_value)

    def _updatemodel(self, new_model):
        '''
        Replaces the current model parameters with the new ones.
        '''
        self.pi = new_model['pi']
        self.A = new_model['A']

    def _eta1(self,t,T):
        '''
        Governs how each sample in the time series should be weighed.
        This is the default case where each sample has the same weigh, 
        i.e: this is a 'normal' HMM.
        '''
        return 1.

    def forwardbackward(self,observations_sequences, cache=False):
        '''
        Forward-Backward procedure is used to efficiently calculate the probability of the observation, given the model - P(O|model)
        alpha_t(x) = P(O1...Ot,qt=Sx|model) - The probability of state x and the observation up to time t, given the model.

        We do it for each of the observations sequences provided.

        '''

        """
        PENDIENTE: las dos líneas siguientes las he quitado: hay que checkear que no está afectando al algoritmo
        """
        #if (cache==False):
        #    self._mapB(observations_sequences)

        (alpha, normalization_coefficients) = self._calcalpha(observations_sequences)


        #We have to compute the log-likelihood of all the provided sequences
        n_sequences = len(observations_sequences)
        log_prob = 0
        for sn in range(0, n_sequences):
            log_prob += -numpy.sum(numpy.log(normalization_coefficients[sn]))

        return log_prob

    def _calcalpha(self,observations_sequences):
        '''
        Calculates 'alpha' the forward variable.
    
        The alpha variable list (# sequences) where each element is a numpy array indexed by time, then state (TxN).
        alpha[sn][t][i] = the probability of being in state 'i' after observing the first t symbols of the 'sn' sequence.
        '''

        n_sequences = len(observations_sequences)
        alpha_list = []
        normalization_coefficients_list = []

        for sn in range(0, n_sequences):

            observations = observations_sequences[sn]
            alpha = numpy.zeros((len(observations),self.n),dtype=self.precision)

            normalization_coefficients = numpy.zeros((len(observations)),dtype=self.precision)

            # init stage - alpha_1(x) = pi(x)b_x(O1)
            for x in xrange(self.n):
                alpha[0][x] = self.pi[x]*self.B_map_list[sn][x][0]

            sum_tmp = numpy.sum(alpha[0])
            alpha[0] = alpha[0] / sum_tmp #We NORMALIZE alpha to avoid overflow problems
            normalization_coefficients[0] = sum_tmp


            # induction
            for t in xrange(1,len(observations)):
                for j in xrange(self.n):
                    for i in xrange(self.n):
                        alpha[t][j] += alpha[t-1][i]*self.A[i][j]
                    alpha[t][j] *= self.B_map_list[sn][j][t]
                sum_tmp = numpy.sum(alpha[t])
                alpha[t] = alpha[t] / sum_tmp #We NORMALIZE alpha to avoid overflow problems
                normalization_coefficients[t] = sum_tmp

            alpha_list.append(alpha)
            normalization_coefficients_list.append(normalization_coefficients)

        return (alpha_list, normalization_coefficients_list)

    def _calcbeta(self,observations_sequences, normalization_coefficients):
        '''
        Calculates 'beta' the backward variable.
        
        The beta variable list (# sequences) where each element is a numpy array indexed by time, then state (TxN).
        beta[t][i] = the probability of being in state 'i' and then observing the
        symbols from t+1 to the end (T).
        '''

        n_sequences = len(observations_sequences)
        beta_list = []

        for sn in range(0, n_sequences):

            observations = observations_sequences[sn]

            beta = numpy.zeros((len(observations),self.n),dtype=self.precision)

            # init stage
            for s in xrange(self.n):
                beta[len(observations)-1][s] = 1. #We do not apply the normalization to the first time instant

            # induction
            for t in xrange(len(observations)-2,-1,-1):
                for i in xrange(self.n):
                    for j in xrange(self.n):
                        beta[t][i] += self.A[i][j]*self.B_map_list[sn][j][t+1]*beta[t+1][j]
                beta[t] = beta[t] / normalization_coefficients[sn][t]

            beta_list.append(beta)

        return (beta_list)

    def _calcxi(self, observations_sequences, normalization_coefficients, alpha=None, beta=None):

        '''
        Calculates 'xi', a joint probability from the 'alpha' and 'beta' variables.

        The xi variable list (# sequences) where each element is a numpy array indexed by time, state, and state (TxNxN).
        xi[t][i][j] = the probability of being in state 'i' at time 't', and 'j' at
        time 't+1' given the entire observation sequence.
        '''

        if alpha is None:
            (alpha, tmp) = self._calcalpha(observations_sequences)
        if beta is None:
            beta = self._calcbeta(observations_sequences)

        n_sequences = len(observations_sequences)
        xi_list = []

        for sn in range(0, n_sequences):
            observations = observations_sequences[sn]

            xi = numpy.zeros((len(observations), self.n, self.n), dtype=self.precision)

            for t in xrange(len(observations) - 1):
                denom = 0.0
                for i in xrange(self.n):
                    for j in xrange(self.n):
                        tmp = 1.0
                        tmp *= alpha[sn][t][i]
                        tmp *= self.A[i][j]
                        tmp *= self.B_map_list[sn][j][
                            t + 1]  # B_map es la variable, mientras que _mapB es la función (implementada en _ContinuousHMM
                        tmp *= beta[sn][t + 1][j]
                        tmp *= normalization_coefficients[sn][t+1]
                        denom += tmp
                for i in xrange(self.n):
                    for j in xrange(self.n):
                        numer = 1.0
                        numer *= alpha[sn][t][i]
                        numer *= self.A[i][j]
                        numer *= self.B_map_list[sn][j][t + 1]
                        numer *= beta[sn][t + 1][j]
                        xi[t][i][j] = numer / denom

            xi_list.append(xi)
        return (xi_list)

    def _calcgamma(self, xi, observations_sequences): #En seqlen antes me llegaba "len(observations)", lo he cambiado para que ahora llegue observations directamente
        '''
        Calculates 'gamma' from xi.

        The Gamma variable list (# sequences) where each element is a (TxN) numpy array, where gamma[t][i] = the probability of being
        in state 'i' at time 't' given the full observation sequence.
        '''


        gamma_list = []
        n_sequences = len(observations_sequences)

        for sn in range(0, n_sequences):

            observations = observations_sequences[sn]
            seqlen = len(observations)

            gamma = numpy.zeros((seqlen, self.n), dtype=self.precision)

            for t in xrange(seqlen):
                for i in xrange(self.n):
                    gamma[t][i] = sum(xi[sn][t][i])

            gamma_list.append(gamma)

        return (gamma_list)


    def decode(self, observations):
        '''
        Find the best state sequence (path), given the model and an observation. i.e: max(P(Q|O,model)).
        
        This method is usually used to predict the next state after training. 
        '''
        # use Viterbi's algorithm. It is possible to add additional algorithms in the future.
        return self._viterbi(observations)

    def _viterbi(self, observations_sequences):
        '''
        Find the best state sequence (path) using viterbi algorithm - a method of dynamic programming,
        very similar to the forward-backward algorithm, with the added step of maximization and eventual
        backtracing.
        
        delta[t][i] = max(P[q1..qt=i,O1...Ot|model] - the path ending in Si and until time t,
        that generates the highest probability.
        
        psi[t][i] = argmax(delta[t-1][i]*aij) - the index of the maximizing state in time (t-1), 
        i.e: the previous state.
        '''
        # similar to the forward-backward algorithm, we need to make sure that we're using fresh data for the given observations.

        n_sequences = len(observations_sequences)
        path_list = []

        self._mapB(observations_sequences)

        for sn in range(0, n_sequences):

            observations = observations_sequences[sn]

            delta = numpy.zeros((len(observations),self.n),dtype=self.precision)
            psi = numpy.zeros((len(observations),self.n),dtype=self.precision)

            # init
            for x in xrange(self.n):
                delta[0][x] = self.pi[x]*self.B_map_list[sn][x][0]
                psi[0][x] = 0

            # induction
            for t in xrange(1,len(observations)):
                for j in xrange(self.n):
                    for i in xrange(self.n):
                        if (delta[t][j] < delta[t-1][i]*self.A[i][j]):
                            delta[t][j] = delta[t-1][i]*self.A[i][j]
                            psi[t][j] = i
                    delta[t][j] *= self.B_map_list[sn][j][t]

                """
                We perform the next normalization to avoid underflow problems:
                """
                delta[t] = preprocessing.normalize(delta[t].reshape(1, -1))



            # termination: find the maximum probability for the entire sequence (=highest prob path)
            p_max = 0 # max value in time T (max)
            path = numpy.zeros((len(observations)),dtype=self.precision)
            for i in xrange(self.n):
                if (p_max < delta[len(observations)-1][i]):
                    p_max = delta[len(observations)-1][i]
                    path[len(observations)-1] = i

            # path backtracing
            path = numpy.zeros((len(observations)),dtype=self.precision) ### 2012-11-17 - BUG FIX: wrong reinitialization destroyed the last state in the path
            for i in xrange(1, len(observations)):
                path[len(observations)-i-1] = psi[len(observations)-i][ int(path[len(observations)-i] )]
            path_list.append(path)
        return path_list



    def M_stepA(self,observations_sequences,xi,gamma):
        '''
        Reestimation of the transition matrix (part of the 'M' step of Baum-Welch).
        Computes A_new = expected_transitions(i->j)/expected_transitions(i)
        
        Returns A_new, the modified transition matrix. 
        '''

        n_sequences = len(observations_sequences)

        for sn in range(0, n_sequences):

            observations = observations_sequences[sn]

            A_new = numpy.zeros((self.n,self.n),dtype=self.precision)
            for i in xrange(self.n):
                for j in xrange(self.n):
                    numer = 0.0
                    denom = 0.0
                    for t in xrange(len(observations)-1):
                        numer += (self._eta(t,len(observations)-1)*xi[sn][t][i][j])
                        denom += (self._eta(t,len(observations)-1)*gamma[sn][t][i])
                    A_new[i][j] = numer/denom
            return A_new

    def E_step(self,observations):
        '''
        Calculates required statistics of the current model, as part
        of the Baum-Welch 'E' step.
        
        Deriving classes should override (extend) this method to include
        any additional computations their model requires.
        
        Returns 'stat's, a dictionary containing required statistics.
        '''
        stats = {}

        (stats['alpha'], normalization_coefficients) = self._calcalpha(observations)
        stats['beta'] = self._calcbeta(observations, normalization_coefficients)
        stats['xi'] = self._calcxi(observations, normalization_coefficients, stats['alpha'],stats['beta'])
        stats['gamma'] = self._calcgamma(stats['xi'], observations)

        return stats

    def M_step(self,stats,observations_sequences):
        '''
        Performs the 'M' step of the Baum-Welch algorithm.
        
        Deriving classes should override (extend) this method to include
        any additional computations their model requires.
        
        Returns 'new_model', a dictionary containing the new maximized
        model's parameters.
        '''
        new_model = {}

        # new init vector is set to the frequency of being in each step at t=0 
        #new_model['pi'] = stats['gamma'][0] #PENDIENTE: ÉSTO HAY QUE CAMBIARLO, AHORA PI SE CALCULA CON TODAS LAS SECUENCIAS


        #ahora tengo que hacer que la inicialización de pi sea la media de las primeras filas (t==0) de todas las secuencias

        gamma_0_each_sequence = [item[0] for item in stats['gamma']]
        new_model['pi'] = numpy.mean(gamma_0_each_sequence, axis=0)

        new_model['A'] = self.M_stepA(observations_sequences,stats['xi'],stats['gamma'])

        return new_model

    def _baumwelch(self,observations):
        '''
        An EM(expectation-modification) algorithm devised by Baum-Welch. Finds a local maximum
        that outputs the model that produces the highest probability, given a set of observations.
        
        Returns the new maximized model parameters
        '''
        # E step - calculate statistics
        stats = self.E_step(observations)

        # M step
        return (self.M_step(stats,observations), stats)





    def _mapB(self,observations):
        '''
        Deriving classes should implement this method, so that it maps the observations'
        mass/density Bj(Ot) to Bj(t).
        
        This method has no explicit return value, but it expects that 'self.B_map' is internally computed
        as mentioned above. 'self.B_map' is an (TxN) numpy array.
        
        The purpose of this method is to create a common parameter that will conform both to the discrete
        case where PMFs are used, and the continuous case where PDFs are used.
        
        For the continuous case, since PDFs of vectors could be computationally 
        expensive (Matrix multiplications), this method also serves as a caching mechanism to significantly
        increase performance.
        '''
        raise NotImplementedError("a mapping function for B(observable probabilities) must be implemented")


    def Q_computation(self, stats, new_model, observations):

        A = 0
        for i in xrange(self.n):
            if(new_model['pi'][i] != 0):
                A += stats['gamma'][0][i] * math.log(new_model['pi'][i])#En un momento new_model['pi'][i] es = 0, tendré que tenerlo en cuenta para no calcular dicho logaritmo



        B = 0
        tmp = 0
        for i in xrange(self.n):
            for j in xrange(self.n):
                for t in xrange(len(observations)):
                    tmp += stats['xi'][t][i][j]
                B += tmp * math.log(new_model['A'][i][j])

        C = 0
        for i in xrange(self.n):
            for t in xrange(len(observations)):
                C += stats['gamma'][t][i] * math.log(self._pdf(0, new_model['means'][i][0], new_model['covars'][i][0]))



        return (A + B + C)
        
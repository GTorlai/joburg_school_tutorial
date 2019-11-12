import numpy as np
import argparse
import math as m
import matplotlib.pyplot as plt
from itertools import product

class RBM(object):

    def __init__(self,pars):
        self.N             = pars.N   # Number of visible units  
        self.nh            = pars.nh  # Number of hidden units
        self.learning_rate = pars.lr  # Learning rate
        self.num_iter      = pars.ep  # Number of training iterations
        self.cd_steps      = 1        # Number of sampling steps in CD
        self.batch_size    = pars.bs  # Batch size
        
        # Initialize the RBM parameters
        sigma = pars.sigma             # Width of the uniform random distribution
        np.random.seed(1234)    # Random number generator
        
        # Weigths connecting visible and hidden units
        self.W = np.random.uniform(low=-sigma,high=sigma, size=[self.nh,self.N])
        # Visible biases
        self.b = np.random.uniform(low=-sigma,high=sigma, size=[self.N])
        # Hidden biases
        self.c = np.random.uniform(low=-sigma,high=sigma, size=[self.nh])
        
        # Basis of the full Hilbert space
        self.basis = np.asarray(list(product([0,1], repeat=self.N)))
        
    # Derivatives of the log-probability
    def DerLog(self,visible):
        ''' Compute the average derivatives of the log-probability across a batch of 
            configurations of the visible units.

            Input:  visible -> [M,N] binary array containing M configurations
            Output: der_W,der_b,der_c -> average derivatives over the M configurations '''

        ''' FILL HERE '''

        return der_W,der_b,der_c 

    def GradientUpdate(self,batch):
        ''' Compute the average derivatives of the negative log-likelihood on a batch of data
            
            Input:  batch -> [M,num_vis] binary array containing M configurations'''
        
        # Compute the positive phase of the gradients:
        # derivatives on the batch
        ''' FILL HERE '''

        # Compute the negative phase of the gradients:
        # derivatives computes on the samples
        samples = self.BlockGibbsSampling(self.cd_steps,batch)
        ''' FILL HERE '''

        # Update the parameters
        ''' FILL HERE '''

    def Probability(self,batch):
        ''' Compute the (unnormalized) probability of a batch of visible data 
            
            Input:  batch -> [M,num_vis] binary array containing M configurations
            Output: output probability -> 1-dimensional array with M entries'''

        ''' FILL HERE ''' 
    
    
    ### BUILT-IN FUNCTION ###
    
    def Train(self,training_data):
        frequency = 10
        for k in range(1,self.num_iter):
            self.GradientUpdate(training_data)
            if (k%frequency==0):
                Z = self.PartitionFunction()
                # Compute Negative Log Likelihood
                rbm_prob = self.Probability(training_data)/Z
                NLL= -np.mean(np.log(rbm_prob))
                print('Epoch: ',k,end='\t')
                print('NLL = %.5f' % NLL,end='\t')
                print()
    
    def PartitionFunction(self):
        prob = self.Probability(self.basis)
        return np.sum(prob)
    
    def SampleHiddenGiven(self,visible_batch):
        ''' Sample hidden units given a batch of visible units
            
            Input : visible_batch -> [M,N] binary array
            Output: hidden_batch  -> [M,nh] binary array '''
        argument = np.dot(visible_batch,np.transpose(self.W)) + np.tile(self.c,[visible_batch.shape[0],1])
        prob = 1.0 / (1.0 + np.exp(-argument)) 
        r_num = np.random.rand(visible_batch.shape[0],self.nh)
        hidden_batch = 1*np.less(r_num,prob)
        return hidden_batch
    
    def SampleVisibleGiven(self,hidden_batch):
        ''' Sample visible units given a batch of hidden units
           
            Input : hidden_batch   -> [M,nh] binary array
            Output: visible_batch  -> [M,N] binary array  '''
        argument = np.dot(hidden_batch,self.W) + np.tile(self.b,[hidden_batch.shape[0],1])
        prob = 1.0 / (1.0 + np.exp(-argument)) 
        r_num = np.random.rand(hidden_batch.shape[0],self.N)
        visible_batch = 1*np.less(r_num,prob)
        return visible_batch

    def BlockGibbsSampling(self,steps,visible_0):
        ''' Perform alternate block Gibbs sampling

            Input: steps -> int, number of steps
                   visible_0 -> [M,N] binary array (initial state of the chain) '''
        visible_batch = visible_0
        for k in range(steps):
            hidden_batch  = self.SampleHiddenGiven(visible_batch)
            visible_batch = self.SampleVisibleGiven(hidden_batch)
        return visible_batch
                
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-N',type=int,default=4)
    parser.add_argument('-nh',type=int,default=4)
    parser.add_argument('-B',type=float,default=1.0)
    parser.add_argument('-lr',type=float,default=0.01)
    parser.add_argument('-bs',type=int,default=100)
    parser.add_argument('-sigma',type=float,default=0.1)
    parser.add_argument('-ep',type=int,default=10000)
     
    pars = parser.parse_args()
    rbm = RBM(pars)
    
    training_data = np.loadtxt('data/debug_data.txt') 
    
    rbm.Train(training_data)
    
    

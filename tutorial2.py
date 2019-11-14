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
        
        data = np.loadtxt('data/ising_observables.txt')
        for i in range(data.shape[0]):
            if (pars.B == data[i][0]):
                self.true_sz = data[i][2]
                self.true_sx = data[i][3]
        
    # Derivatives of the log-probability
    def DerLog(self,visible):
        ''' Compute the average derivatives of the log-probability across a batch of 
            configurations of the visible units.

            Input:  visible -> [M,N] binary array containing M configurations
            Output: der_W,der_b,der_c -> average derivatives over the M configurations '''

        argument = np.dot(visible,np.transpose(self.W)) + np.tile(self.c,[visible.shape[0],1])
        sigmoid = 1.0 / (1.0 + np.exp(-argument))
        der_W = np.dot(np.transpose(sigmoid),visible)/float(visible.shape[0])
        der_b = np.mean(visible,axis=0)  
        der_c = np.mean(sigmoid,axis=0) 
        return der_W,der_b,der_c 

    def GradientUpdate(self,batch):
        ''' Compute the average derivatives of the negative log-likelihood on a batch of data
            
            Input:  batch -> [M,num_vis] binary array containing M configurations'''
        
        # Compute the positive phase of the gradients:
        # derivatives on the batch
        derivatives = self.DerLog(batch)
        der_W = - derivatives[0] 
        der_b = - derivatives[1]
        der_c = - derivatives[2]
        
        # Compute the negative phase of the gradients:
        # derivatives computes on the samples
        samples = self.BlockGibbsSampling(self.cd_steps,batch)
        derivatives = self.DerLog(samples)
        der_W += derivatives[0] 
        der_b += derivatives[1]
        der_c += derivatives[2]
       
        # Update the parameters
        self.W = self.W - self.learning_rate * der_W
        self.b = self.b - self.learning_rate * der_b
        self.c = self.c - self.learning_rate * der_c
    
    def Probability(self,batch):
        ''' Compute the (unnormalized) probability of a batch of visible data 
            
            Input:  batch -> [M,num_vis] binary array containing M configurations
            Output: output probability -> 1-dimensional array with M entries'''

        log_prob = np.dot(batch,self.b)
        log_prob += np.sum(np.log(1.0 + np.exp(np.dot(batch,np.transpose(self.W))+np.tile(self.c,[batch.shape[0],1]))),axis=1)
        return np.exp(log_prob)
    
    def Psi(self,state):
        log_prob = np.dot(state,self.b)
        log_prob += np.sum(np.log(1.0 + np.exp(np.dot(self.W,state)+self.c))) 
        return m.exp(0.5*log_prob)
    
    def Measurement(self,steps,nchains):
        initial_state = np.random.randint(2,size=[nchains,self.N])
        samples = self.BlockGibbsSampling(steps,initial_state)
        
        #Measure magnetization along Z direction
        sigma_z = np.absolute(np.mean(1.0 - 2.0*samples,axis=1))
        sigma_z_err = np.sqrt(np.var(sigma_z)/float(nchains))
        sigma_z = np.mean(sigma_z)
        #sigma_z_err = np.std(sigma_z_raw,ddof=1)
        
        sigma_x = np.zeros((samples.shape[0]))
        for n in range(samples.shape[0]):
            tmp = np.zeros((self.N))
            for j in range(self.N):
                psi = self.Psi(samples[n])
                samples[n,j] = 1 - samples[n,j]
                psi_flip = self.Psi(samples[n])
                samples[n,j] = 1 - samples[n,j]
                tmp[j] = psi_flip / psi
            sigma_x[n] = np.mean(tmp)
        sigma_x_err = np.sqrt(np.var(sigma_x)/float(nchains))
        sigma_x = np.mean(sigma_x)

        return sigma_z,sigma_z_err,sigma_x,sigma_x_err

    def Train(self,training_data,target_prob = None,plot=None):
        frequency = 10
        sz = []
        sx = []
        sz_err = []
        sx_err = []
        kl = []
        ov = []

        num_batches = int(training_data.shape[0]/self.batch_size)
        plt.ion() 
        if (plot is not None):
            plt.figure(figsize=(9,6), facecolor='w', edgecolor='k')
        
        for k in range(1,self.num_iter):
            for b in range(num_batches):
                batch = training_data[b*self.batch_size:(b+1)*self.batch_size]
                self.GradientUpdate(batch)
            
            if (k%frequency==0):
                Z = self.PartitionFunction()
                rbm_prob = self.Probability(self.basis)/Z
                # Compute the KL divergence
                KL = np.sum(np.multiply(target_prob,np.log(np.divide(target_prob,rbm_prob))))
                overlap = np.sum(np.multiply(np.sqrt(target_prob),np.sqrt(rbm_prob)))
                kl.append(KL)
                ov.append(overlap)
                # Compute Negative Log Likelihood
                #rbm_prob = self.Probability(batch)/Z
                #NLL= -np.mean(np.log(rbm_prob))
                sigma_z,err_z,sigma_x,err_x = self.Measurement(10,1000)
                sz.append(sigma_z)
                sx.append(sigma_x)
                sz_err.append(err_z)
                sx_err.append(err_x)
                
                print('Epoch: ',k,end='\t')
                print('KL = %.5f' % KL,end='\t')
                print('Overlap = %.5f' % overlap,end='\t')
                print('<sz> = %.4f+-%.4f' % (sigma_z,err_z),end='\t')
                print('<sx> = %.4f+-%.4f' % (sigma_x,err_x),end='\t')
                print()
                
                if (plot is not None):
                    plt.clf()
                    x = [i for i in range(0,k,frequency)]
                    plt.subplot(2,2,1)
                    plt.plot(x,kl,marker='o',markersize=3)
                    plt.ylim([0,kl[0]+0.1])
                    plt.subplot(2,2,2)
                    plt.axhline(y=1.0, linewidth=2,color='k', linestyle='-')
                    plt.plot(x,ov,marker='o',markersize=3)
                    plt.ylim([0,1.05])
                    plt.subplot(2,2,3)
                    plt.errorbar(x,sz,yerr=sz_err,marker='o',markersize=3)
                    plt.axhline(y=self.true_sz, linewidth=2,color='k', linestyle='-')
                    plt.ylim([0,1.05])
                    plt.subplot(2,2,4)
                    plt.errorbar(x,sx,yerr=sx_err,marker='o',markersize=3)
                    plt.axhline(y=self.true_sx, linewidth=2,color='k', linestyle='-')
                    plt.ylim([0,1.05])
                    plt.pause(0.001)
 

    ### BUILT-IN FUNCTION ###
    
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
    parser.add_argument('-N',type=int,default=10)
    parser.add_argument('-nh',type=int,default=1)
    parser.add_argument('-B',type=float,default=1.0)
    parser.add_argument('-lr',type=float,default=0.01)
    parser.add_argument('-bs',type=int,default=100)
    parser.add_argument('-sigma',type=float,default=0.1)
    parser.add_argument('-ep',type=int,default=10000)
     
    pars = parser.parse_args()
    rbm = RBM(pars)
    
    training_data = np.loadtxt('data/datasets/ising_N'+str(pars.N)+'_B'+"{:.1f}".format(pars.B)+'_dataset.txt')
    target_psi    = np.loadtxt('data/wavefunctions/ising_N'+str(pars.N)+'_B'+"{:.1f}".format(pars.B)+'_psi.txt')
    target_prob = target_psi*target_psi
    
    #rbm.TestDerivativesKL(target_prob)
    rbm.Train(training_data,target_prob,plot=True)
    
   

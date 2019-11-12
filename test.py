import numpy as np
import math as m
from tutorial1 import RBM
import sys
import argparse

class bcolors:
    FAIL = '\033[91m'
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'
    ORANGE = '\033[33m'

def TestDerivativesLogProb(rbm,epsilon=1e-6):
    print(bcolors.ORANGE+'\nTesting gradients of log-probabilities:\n\n'+bcolors.ENDC,end='')
    threshold=1e-9
    
    data=np.loadtxt('data/debug_data.txt')
    
    der_W_alg,der_b_alg,der_c_alg = rbm.DerLog(data)
    for i in range(rbm.nh):
        for j in range(rbm.N):
            rbm.W[i,j] += epsilon
            log_prob_plus = np.dot(data,rbm.b)
            log_prob_plus += np.sum(np.log(1.0 + np.exp(np.dot(data,np.transpose(rbm.W))+np.tile(rbm.c,[data.shape[0],1]))),axis=1)
            log_prob_plus = np.mean(log_prob_plus)
            rbm.W[i,j] -= 2*epsilon
            log_prob_minus = np.dot(data,rbm.b)
            log_prob_minus += np.sum(np.log(1.0 + np.exp(np.dot(data,np.transpose(rbm.W))+np.tile(rbm.c,[data.shape[0],1]))),axis=1)
            log_prob_minus = np.mean(log_prob_minus)
            rbm.W[i,j] += epsilon 
            der_num = (log_prob_plus - log_prob_minus) / (2*epsilon)
            print('Der W[',i,',',j,']',end=' ')
            print('Algorithm: %.8f\t' % der_W_alg[i,j],end='')
            print('Numeric: %.8f' % der_num,end='\t')
            if(abs(der_W_alg[i,j]-der_num)>threshold):
                print (bcolors.FAIL + "FAILED" + bcolors.ENDC)
            else:
                print(bcolors.OKGREEN + 'PASSED' + bcolors.ENDC)
    print()
    for j in range(rbm.N):
        rbm.b[j] += epsilon
        log_prob_plus = np.dot(data,rbm.b)
        log_prob_plus += np.sum(np.log(1.0 + np.exp(np.dot(data,np.transpose(rbm.W))+np.tile(rbm.c,[data.shape[0],1]))),axis=1)
        log_prob_plus = np.mean(log_prob_plus)
        rbm.b[j] -= 2*epsilon
        log_prob_minus = np.dot(data,rbm.b)
        log_prob_minus += np.sum(np.log(1.0 + np.exp(np.dot(data,np.transpose(rbm.W))+np.tile(rbm.c,[data.shape[0],1]))),axis=1)
        log_prob_minus = np.mean(log_prob_minus)
        rbm.b[j] += epsilon 
        der_num = (log_prob_plus - log_prob_minus) / (2*epsilon)
        print('Der b[',j,']',end=' ')
        print('Algorithm: %.8f\t' % der_b_alg[j],end='')
        print('Numeric: %.8f' % der_num,end='\t') 
        if(abs(der_b_alg[j]-der_num)>threshold):
            print (bcolors.FAIL + "FAILED" + bcolors.ENDC)
        else:
            print(bcolors.OKGREEN + 'PASSED' + bcolors.ENDC)
    print()
    for i in range(rbm.nh):
        rbm.c[i] += epsilon
        log_prob_plus = np.dot(data,rbm.b)
        log_prob_plus += np.sum(np.log(1.0 + np.exp(np.dot(data,np.transpose(rbm.W))+np.tile(rbm.c,[data.shape[0],1]))),axis=1)
        log_prob_plus = np.mean(log_prob_plus)
        rbm.c[i] -= 2*epsilon
        log_prob_minus = np.dot(data,rbm.b)
        log_prob_minus += np.sum(np.log(1.0 + np.exp(np.dot(data,np.transpose(rbm.W))+np.tile(rbm.c,[data.shape[0],1]))),axis=1)
        log_prob_minus = np.mean(log_prob_minus)
        der_num = (log_prob_plus - log_prob_minus) / (2*epsilon)
        print('Der c[',i,']',end=' ')
        print('Algorithm: %.8f\t' % der_c_alg[i],end='')
        print('Numeric: %.8f' % der_num,end='\t') 
        if(abs(der_c_alg[i]-der_num)>threshold):
            print (bcolors.FAIL + "FAILED" + bcolors.ENDC)
        else:
            print(bcolors.OKGREEN + 'PASSED' + bcolors.ENDC)

parser = argparse.ArgumentParser()
parser.add_argument('-N',type=int,default=4)
parser.add_argument('-nh',type=int,default=4)
parser.add_argument('-B',type=float,default=1.0)
parser.add_argument('-lr',type=float,default=0.5)
parser.add_argument('-sigma',type=float,default=1.0)
parser.add_argument('-bs',type=int,default=100)
parser.add_argument('-ep',type=int,default=10000)
pars = parser.parse_args()
rbm = RBM(pars)

TestDerivativesLogProb(rbm)


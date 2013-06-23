'''
Created on Apr 20, 2013

@author: dimitris
'''
################################################################################

import numpy as np
import stats as st

################################################################################
        
def rParams(x0):
    return 0., 

################################################################################
    
def rPrior(N, *pars):    
    var  = st.invgamma(N = (N, 1))
    mean = st.normal(0., 10., N = (N, 1)) 
    
    return np.hstack((mean, var))
    
################################################################################
   
def rPost(x0, idx, C, Z, data, *pars):     
    ## collect data
    data = np.asarray([ counts[c[z] == idx] for c, z, counts in zip(C, Z, data.counts) ])
    
    ## sample new mean and var
    mean, var = st.normal_mean_var(data.ravel())
        
    # return        
    return mean, var

################################################################################
    
def dLogLik(X0, counts, exposure):
    X0 = np.atleast_2d(X0)
    
    mean = X0[:,0] 
    var  = X0[:,1]
        
    return st.dLogNormal(counts.reshape(-1,1), mean, var)
    
################################################################################
                
def generateData(N = 1000, M = 2, L = 4, ):
    pass

################################################################################
        

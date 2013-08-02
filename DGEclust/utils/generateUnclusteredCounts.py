## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import numpy as np

################################################################################

def generateUnclusteredCounts(x, ngenes = 1e3, nreplicas = 10, nDE = 100, c0 = [7e6, 14e6], w = 1.5):    
    ## fetch dispersions and means for the control condition
    idxs  = np.random.randint(0, len(x), size = ngenes)
    phi1  = x[idxs,0]
    mu1   = x[idxs,1]
    
    ## indicators of differentially expressed genes
    z = np.zeros(ngenes, dtype='int')
    z[:nDE] = np.random.choice([-1,1], size = nDE)    ## -1/1 stand for down-/up-regulation   

    ## size of regulation
    s = w + np.random.exponential(size = ngenes)

    ## library sizes
    c = c0[0] + (c0[1] - c0[0]) * np.random.rand(nreplicas * 2)
    
    ## compute gene- and sample-specific mean and dispersion parameters for conditon 2
    mu2   = s**z * mu1
    phi2  = phi1
    
    ## draw from the negative binomial
    y1 = [ np.random.poisson(np.random.gamma(1 / phi1, ci * mu1 / mu1.sum() * phi1)) for i, ci in zip(range(nreplicas), c[:nreplicas]) ]          
    y2 = [ np.random.poisson(np.random.gamma(1 / phi2, ci * mu2 / mu2.sum() * phi2)) for i, ci in zip(range(nreplicas), c[nreplicas:]) ]          
    
    ## return
    return np.vstack((y1,y2)).T, z



# def generateUnclusteredCounts(ngenes = 1e3, nreplicas = 10, nDE = 100, c0 = [7e6, 14e6], w = 1.5, mu = 5.7, var = 5., shape = 1., scale = 0.45):
#     ## indicators of differentially expressed genes
#     z = np.zeros(ngenes, dtype='int')
#     z[:nDE] = np.random.choice([-1,1], size = nDE)    ## -1/1 stand for down-/up-regulation   
# 
#     ## size of regulation
#     s = w + np.random.exponential(size = ngenes)
# 
#     ## library sizes
#     c = c0[0] + (c0[1] - c0[0]) * np.random.rand(nreplicas * 2)
#     
#     ## compute gene- and sample-specific mean and dispersion parameters for conditon 1
#     mu1  = np.exp(np.random.randn(ngenes) * np.sqrt(var) + mu)
#     mu1  = mu1 / mu1.sum()
#     phi1 = np.random.gamma(shape, scale, ngenes)        
#     
#     ## compute gene- and sample-specific mean and dispersion parameters for conditon 2
#     mu2  = s**z * mu1
#     mu2  = mu2 / mu2.sum()
#     phi2 = phi1
#     
#     ## draw from the negative binomial
#     y1 = [ np.random.poisson(np.random.gamma(1 / phi1, ci * mu1 * phi1)) for i, ci in zip(range(nreplicas), c[:nreplicas]) ]          
#     y2 = [ np.random.poisson(np.random.gamma(1 / phi2, ci * mu2 * phi2)) for i, ci in zip(range(nreplicas), c[nreplicas:]) ]          
#     
#     ## return
#     return np.vstack((y1,y2)).T, z

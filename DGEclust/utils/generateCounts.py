## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import numpy as np

################################################################################

def generateUnclusteredCounts(x, ngenes = 1000, nclasses = 2, nreplicas = 2, nDE = 100, c0 = [7e6, 14e6], w = 1.5):    
    nclasses -= 1
    
    ## parameters for the control condition
    idxs = np.random.choice(range(len(x)), size = ngenes, replace = False) 
    phi1 = x[idxs,0].reshape(-1,1)   
    mu1  = x[idxs,1].reshape(-1,1)
    z1   = np.zeros((ngenes,1))
    c1   = c0[0] + (c0[1] - c0[0]) * np.random.rand(nreplicas)

    ## counts for the control condition
    y1 = np.random.poisson(np.random.gamma(1. / phi1, c1 * mu1 / mu1.sum() * phi1))
    
    ## indicators for DE genes
    Z   = [ np.zeros((ngenes,1))                                       for i in range(nclasses) ]
    ZDE = [ np.random.choice(range(ngenes), size=nDE, replace = False) for i in range(nclasses) ]
    DE  = [ np.random.choice([-1,1], size=(nDE,1))                     for i in range(nclasses) ]

    for z, zde, de in zip(Z,ZDE,DE):        
        z[zde] = de
        
    C   = [ c0[0] + (c0[1] - c0[0]) * np.random.rand(1,nreplicas) for i in range(nclasses) ]        ## library sizes
    S   = [ w + np.random.exponential(size = (ngenes,1))          for i in range(nclasses) ]        ## regulation sizes
    PHI = [ phi1                                                  for i in range(nclasses) ]        ## dispersions
    MU  = [ s**z * mu1                                            for s, z in zip(S,Z) ]            ## means
    
    ## draw from the negative binomial
    Y  = [ np.random.poisson(np.random.gamma(1. / phi, c * mu / mu.sum() * phi)) for c, phi, mu in zip(C, PHI, MU) ]          
    
    ## return
    return np.hstack([y1]+Y), np.hstack([z1]+Z)


################################################################################
################################################################################

def generateUnclusteredCounts2(ngenes = 1000, nclasses = 2, nreplicas = 2, nDE = 100, c0 = [7e6, 14e6], w = 1.5, sh1 = 0.28, sc1 = 666., sh2 = 0.85, sc2 = 0.5):      
    ## sh and sc parameters from Kvam et al. 2012    
    
    nclasses -= 1
    
    ## parameters for the control condition
    mu1  = np.random.gamma(shape = sh1, scale = sc1, size = (ngenes,1))
    phi1 = np.random.gamma(shape = sh2, scale = sc2, size = (ngenes,1))
    z1   = np.zeros((ngenes,1))
    c1   = c0[0] + (c0[1] - c0[0]) * np.random.rand(nreplicas)

    ## counts for the control condition
    y1 = np.random.poisson(np.random.gamma(1. / phi1, c1 * mu1 / mu1.sum() * phi1))
    
    ## indicators for DE genes
    Z   = [ np.zeros((ngenes,1))                                       for i in range(nclasses) ]
    ZDE = [ np.random.choice(range(ngenes), size=nDE, replace = False) for i in range(nclasses) ]
    DE  = [ np.random.choice([-1,1], size=(nDE,1))                     for i in range(nclasses) ]

    for z, zde, de in zip(Z,ZDE,DE):        
        z[zde] = de
        
    C   = [ c0[0] + (c0[1] - c0[0]) * np.random.rand(1,nreplicas) for i in range(nclasses) ]        ## library sizes
    S   = [ w + np.random.exponential(size = (ngenes,1))          for i in range(nclasses) ]        ## regulation sizes
    PHI = [ phi1                                                  for i in range(nclasses) ]        ## dispersions
    MU  = [ s**z * mu1                                            for s, z in zip(S,Z) ]            ## means
    
    ## draw from the negative binomial
    Y  = [ np.random.poisson(np.random.gamma(1. / phi, c * mu / mu.sum() * phi)) for c, phi, mu in zip(C, PHI, MU) ]          
    
    ## return
    return np.hstack([y1]+Y), np.hstack([z1]+Z)


################################################################################
################################################################################

def generateClusteredCounts(x0, w0, ngenes = 1000, nclasses = 2, nreplicas = 2, nDE = 100, c0 = [7e6, 14e6]):        
    nclasses -= 1    
    nclusters = len(x0)

    ## parameters for the control condition
    z1    = np.random.choice(range(nclusters), size = (ngenes,1), p = w0)
    c1    = c0[0] + (c0[1] - c0[0]) * np.random.rand(nreplicas) 
    phi1  = x0[z1,0]
    beta1 = x0[z1,1]
    mu1   = np.exp(beta1)

    ## counts for the control condition
    y1 = np.random.poisson(np.random.gamma(1. / phi1, c1 * mu1 / mu1.sum() * phi1))

    ## indicators for DE genes
    Z   = [ z1.copy()                                                  for i in range(nclasses) ]
    ZDE = [ np.random.choice(range(ngenes), size=nDE, replace = False) for i in range(nclasses) ]
    DE  = [ np.random.choice(range(nclusters), size = (nDE,1), p = w0) for i in range(nclasses) ]

    for z, zde, de in zip(Z,ZDE,DE):        
        z[zde] = de
    
    C    = [ c0[0] + (c0[1] - c0[0]) * np.random.rand(1,nreplicas) for i in range(nclasses) ]   ## library sizes
    PHI  = [ x0[z,0] for z in Z ]                                                               ## dispersions
    BETA = [ x0[z,1] for z in Z ]                                                               ## means    
    MU   = np.exp(BETA)
    
    ## draw from the negative binomial
    Y = [ np.random.poisson( np.random.gamma(1. / phi, c * mu / mu.sum() * phi) ) for c, mu, phi in zip(C, MU, PHI) ]          
    
    ## return
    return np.hstack([y1]+Y), np.hstack([z1]+Z)


################################################################################
################################################################################

def generateClusteredCounts2(nclusters = 10, ngenes = 1000, nclasses = 2, nreplicas = 2, c0 = [7e6, 14e6], sh1 = 0.28, sc1 = 666., sh2 = 3., sc2 = 1.):        
    mu0  = np.random.gamma(shape = sh1, scale = sc1, size = nclusters)     
    phi0 = np.random.gamma(shape = sh2, scale = sc2, size = nclusters)
    w0   = np.random.rand(nclusters) 
    w0   = w0 / w0.sum()

    ## class- and gene-specific quantities
    Z    = [ np.random.choice(range(nclusters), size = (ngenes,1), p = w0) for i in range(nclasses)]   ## cluster indicators
    C    = [ c0[0] + (c0[1] - c0[0]) * np.random.rand(1,nreplicas)         for i in range(nclasses)]   ## library sizes
    PHI  = [ phi0[z]                                                       for z    in Z           ]   
    MU   = [ mu0[z]                                                        for z    in Z           ]
    
    ## counts 
    Y = [ np.random.poisson(np.random.gamma(1. / phi, c * mu / mu.sum() * phi)) for phi, mu, c in zip(PHI,MU,C) ]

    ## return
    return np.hstack(Y), np.hstack(Z)





## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import numpy as np

################################################################################

def generateClusteredCounts(nclusters = 10, ngenes = 1e3, nreplicas = 10, nDE = 100, c0 = [7e6, 14e6], sh = 1., sc = 0.45, mu = 5.7, var = 10., scale = 1):
    phi0 = np.random.gamma(sh, sc, size = nclusters) * scale
    mu0  = np.exp(np.random.randn(nclusters) * np.sqrt(var) / scale + mu)
        
    ## library sizes
    c = c0[0] + (c0[1] - c0[0]) * np.random.rand(nreplicas * 2)
        
    ## generate cluster occupany indicators
    z1 = np.random.randint(0, nclusters, size = ngenes)
    z2 = np.random.randint(0, nclusters, size = ngenes)
    
    z2[nDE:] = z1[nDE:]  ## ensure there are only nDE differentially expressed genes 
    
    ## draw from the negative binomial
    y1 = [ np.random.poisson(np.random.gamma(1 / phi0[z1], ci * mu0[z1] / mu0[z1].sum() * phi0[z1])) for i, ci in zip(range(nreplicas), c[:nreplicas]) ]          
    y2 = [ np.random.poisson(np.random.gamma(1 / phi0[z2], ci * mu0[z2] / mu0[z2].sum() * phi0[z2])) for i, ci in zip(range(nreplicas), c[nreplicas:]) ]          
    
    ## return
    return np.vstack((y1,y2)).T, z1 != z2, np.vstack((z1,z2)).T

## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import numpy as np

################################################################################

def generateClusteredCounts(x0, z, ngenes = 1e3, nreplicas = 10, nDE = 100, c0 = [7e6, 14e6]):        
    ## condition 1
    z1   = np.random.choice(z, size = ngenes)
    phi1 = x0[z1,0]
    mu1  = np.exp(x0[z1,1])
    
    ## condition 2
    z2       = z1.copy()
    z2[:nDE] = np.random.choice(z, size = nDE)
    phi2     = x0[z2,0]
    mu2      = np.exp(x0[z2,1])
    
    ## library sizes
    c = c0[0] + (c0[1] - c0[0]) * np.random.rand(nreplicas * 2)
        
    ## draw from the negative binomial
    y1 = [ np.random.poisson(np.random.gamma(1. / phi1, ci * mu1 / mu1.sum() * phi1)) for ci in c[nreplicas:] ]          
    y2 = [ np.random.poisson(np.random.gamma(1. / phi2, ci * mu2 / mu2.sum() * phi2)) for ci in c[:nreplicas] ]          
    
    ## return
    return np.vstack((y1,y2)).T, z1 != z2, np.vstack((z1,z2)).T



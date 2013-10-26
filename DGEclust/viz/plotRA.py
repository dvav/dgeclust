## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import pylab as pl
import numpy as np

################################################################################

def plotRA(samples1, samples2, ids = None, epsilon = 1., *args, **kargs):        
    ## set zero elements to epsilon
    samples1[samples1 < 1.] = epsilon     
    samples2[samples2 < 1.] = epsilon     
    
    ## compute means
    lmeans1 = np.log2(samples1).mean(0)
    lmeans2 = np.log2(samples2).mean(0)
     
    ## compute A and R
    A = ( lmeans1 + lmeans2 ) * 0.5
    R =   lmeans1 - lmeans2
        
    ## generate RA plot
    if ids is not None:
        pl.plot(A[~ids], R[~ids], 'k.', A[ids], R[ids], 'r.')
    else:
        pl.plot(A, R, 'k.')
        
    pl.plot(pl.gca().get_xlim(),(0.,0.),'k--')
    pl.xlabel('mean')
    pl.ylabel('log2 fold change')
    
    return A, R
        
################################################################################

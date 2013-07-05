## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import sys, os
import numpy as np
 
################################################################################

def computeAverageSimilarityMatrix(res, T0, T, dt = 1, compare_samples = True):    
    ## loop
    nsamples, sim = 0, 0.    
    for _, Z in res.clusts(T0, T, dt): 
        ## transpose, in case genes are compared, instead of samples
        Z = Z.T if compare_samples is False else Z
                
        ## update average similarity matrix
        sim += _computeSimilarityMatrix(Z)
        nsamples += 1

    ## log
    print >> sys.stderr, '{0} samples processed'.format(nsamples)        
    
    ## return
    return sim / nsamples
    
################################################################################

def _computeSimilarityMatrix(Z):
    nrows, ncols = Z.shape
    sim = np.zeros((nrows,nrows))
    for i in range(nrows):
        sim[i,i:] = ( Z[i] == Z[i:] ).sum(1)
        sim[i:,i] = sim[i,i:]

    ## return
    return sim / ncols

################################################################################

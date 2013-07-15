## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import sys, os
import numpy as np
import scipy.cluster.hierarchy as hr

################################################################################

def computeAverageDistanceMatrix(res, T0, T, dt = 1, compare_samples = True, idxs = None):    
    ## loop
    nsamples, dist = 0, 0.    
    for t, Z in res.clusts(T0, T, dt): 
        ## transpose, in case genes are compared, instead of samples
        Z = Z.T if compare_samples is False else Z
        Z = Z   if idxs            is None  else Z[idxs]
                    
        ## update average similarity matrix
        dist += _computeDistanceMatrix(Z)
        # dist += hr.distance.pdist(Z, 'hamming')
        nsamples += 1

        ## log
        print >> sys.stderr, t, 
         
    ## log
    print >> sys.stderr, '\n{0} samples processed'.format(nsamples)        
    
    ## return
    return dist / nsamples
    
################################################################################

def _computeDistanceMatrix(Z):
    nrows, ncols = Z.shape
    dist = np.zeros((nrows,nrows))
    for i in range(nrows):
        dist[i,i:] = ( Z[i] != Z[i:] ).sum(1)
        dist[i:,i] = dist[i,i:]

    ## return
    return dist / ncols

################################################################################

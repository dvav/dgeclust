## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import sys, os
import numpy as np
 
################################################################################

def computeAverageSimilarityMatrix(path, T0, T, dt = 1, compare_samples = True):
    ## read directory
    fnames = os.listdir(path)
    fnames = np.sort(np.asarray(fnames, dtype='int'))
    fnames = fnames[(fnames >= T0) & (fnames <= T)]    ## T0 <= fnames <= T 
    fnames = fnames[range(0, fnames.size, dt)]         ## keep every dt-th sample
        
    ## sample size
    nsamples = fnames.size
    
    ## loop
    sim = 0.
    print >> sys.stderr, 'Start sampling at iteration {0} ...'.format(fnames[0])        
    for fname in fnames.astype('str'): 
        ## read indicators
        Z = np.loadtxt(os.path.join(path,fname), dtype = 'int')
        Z = Z.T if compare_samples is False else Z
                
        ## update average similarity matrix
        sim += _computeSimilarityMatrix(Z)

    ## log
    print >> sys.stderr, 'Finish sampling at iteration {0} ...'.format(fnames[-1])                    
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

## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import sys, os
import numpy as np

################################################################################

def computeGenePosteriorProbabilities(path, T0, T, dt, idxs1, idxs2):
    ## read directory
    fnames = os.listdir(path)
    fnames = np.asarray(fnames, dtype='int')
    fnames = fnames[(fnames >= T0) & (fnames <= T)]    ## T0 <= fnames <= T 
    fnames = fnames[range(0, len(fnames), dt)]         ## keep every dt-th sample
    
    ## sample size
    nsamples = fnames.size

    ## loop
    N1 = len(idxs1)
    N2 = len(idxs2)  
    p  = 0.
    print >> sys.stderr, 'Start sampling at iteration {0} ...'.format(fnames[0])        
    for fname in fnames.astype('str'): 
        ## read indicators
        Z = np.loadtxt(os.path.join(path,fname), dtype = 'int')        
        
        ## update number of co-occurances
        s  = [ ( Z[idx] == Z[idxs2] ).sum(0) for idx in idxs1 ]
        p += np.sum(s, 0)

    ## log
    print >> sys.stderr, 'Finish sampling at iteration {0} ...'.format(fnames[-1])                    
    print >> sys.stderr, '{0} samples processed'.format(nsamples)        
        
    ## return
    return p / (N1 * N2 * nsamples)

################################################################################

# 
# def computeGenePosteriorProbabilities2(path, T0, T, step, idxs1, idxs2):
#     ## read directory
#     fnames = os.listdir(path)
#     fnames = np.asarray(fnames, dtype='int')
#     fnames = fnames[(fnames >= T0) & (fnames <= T)]    ## T0 <= fnames <= T 
#     fnames = fnames[range(0, len(fnames), step)]       ## keep every step-th sample
#     
#     ## loop
#     nsamples = 0
#     idxs = np.r_[idxs1,idxs2]
#     p  = 0.
#     print >> sys.stderr, 'Start sampling at iteration {0} ...'.format(fnames[0])        
#     for fname in fnames.astype('str'): 
#         ## read indicators
#         Z = np.loadtxt(path + fname, dtype = 'int')        
#         
#         ## check groups of replicas
#         i1 = ( Z[idxs1[0]] == Z[idxs1[1:]] ).all(0)
#         i2 = ( Z[idxs2[0]] == Z[idxs2[1:]] ).all(0)
#         
#         nsamples += (i1 & i2)
#         
#         ## update number of co-occurances
#         p += ( Z[idxs[0]] == Z[idxs[1:]] ).all(0)
# 
#     ## log
#     print >> sys.stderr, 'Finish sampling at iteration {0} ...'.format(fnames[-1])                    
#     print >> sys.stderr, '{0}/{1} samples processed'.format(min(nsamples),max(nsamples))        
#         
#     ## return
#     return p / nsamples
#     
# ################################################################################
# 
# def computeGenePosteriorProbabilities3(path, T0, T, step, idxs1, idxs2):
#     ## read directory
#     fnames = os.listdir(path)
#     fnames = np.asarray(fnames, dtype='int')
#     fnames = fnames[(fnames >= T0) & (fnames <= T)]    ## T0 <= fnames <= T 
#     fnames = fnames[range(0, len(fnames), step)]       ## keep every step-th sample
#     
#     ## sample size
#     nsamples = fnames.size
# 
#     ## loop
#     idxs = np.r_[idxs1,idxs2]
#     p = p1 = p2 = 0.
#     print >> sys.stderr, 'Start sampling at iteration {0} ...'.format(fnames[0])        
#     for fname in fnames.astype('str'): 
#         ## read indicators
#         Z = np.loadtxt(path + fname, dtype = 'int')        
# 
#         ## update number of co-occurances        
#         p1 += ( Z[idxs1[0]] == Z[idxs1[1:]] ).all(0)
#         p2 += ( Z[idxs2[0]] == Z[idxs2[1:]] ).all(0)
#         p  += ( Z[idxs[0]]  == Z[idxs[1:]]  ).all(0)
# 
#     p  /= nsamples
#     p1 /= nsamples
#     p2 /= nsamples 
#     
#     ## log
#     print >> sys.stderr, 'Finish sampling at iteration {0} ...'.format(fnames[-1])                    
#     print >> sys.stderr, '{0} samples processed'.format(nsamples)        
#         
#     ## return
#     return p / (p1 * p2)
#     
# ################################################################################
# 
# def findOptimalClustering(ref, fname, fmap, compare_samples = True):
#     ## process up to T (starting at T0, if possible)
#     print >> sys.stderr, 'Start searching at iteration {0} ...'.format(min(fmap2.keys()))        
#     p = 1. 
#     for key in fmap2.keys(): 
#         ## process candidate clustering
#         Z_ = _readArray(fhandle, fmap2[key])            
#         if compare_samples is False:    ## ref is a gene similarity matrix
#             Z_ = Z_.T
#         val = _computeSimilarityMatrix(Z_)
#     
#         ## make a decision
#         p_ = val ref
#         if p_ <= p:
#             Z = Z_
#             p = p_
#     print >> sys.stderr, '{0} samples processed'.format(nsamples)        
# 
#     ## return
#     return Z, p
# 
# ################################################################################

             
        

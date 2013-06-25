'''
Created on Apr 19, 2013

@author: dimitris
'''

################################################################################

import sys, os
import numpy  as np
import pandas as pd

################################################################################

class Data(object):
    def __init__(self, fname, *args, **kargs):        
        data           = pd.read_table(fname, *args, **kargs)
        
        # self.counts    = data.values.astype('double').T
        self.counts    = data.values.astype('double').T
        self.exposures = self.counts.sum(1)
        self.samples   = data.columns.values.tolist()
        self.genes     = data.index.values.tolist()
        
################################################################################

class Recorder(object):    
    def __init__(self, outDir, dt, extend):                
        ## member data
        self.fX0   = os.path.join(outDir,'X0.txt')
        self.flw0  = os.path.join(outDir,'lw0.txt')
        self.fLW   = os.path.join(outDir,'LW.txt')
        self.fC    = os.path.join(outDir,'C.txt')
        self.fZ    = os.path.join(outDir,'Z.txt')
        self.feta  = os.path.join(outDir,'eta.txt')
        self.fpars = os.path.join(outDir,'pars.txt') 
        self.dZ    = os.path.join(outDir,'clust')

        self.t0    = 0
        self.dt    = dt

        ## check for previous data on disk
        if os.path.exists(outDir): 
            if extend is not True:
                raise Exception("Directory '{0}' already exists!".format(outDir))
            else:
                print >> sys.stderr, "Directory '{0}' already exists! Extending previous simulation...".format(outDir)
                t0      =  np.loadtxt(self.fpars)[-1,0]   ## the last iteration of the previous simulation
                self.t0 = int(t0)  
        else:
            os.makedirs(self.dZ)
        
    ############################################################################    
        
    def update(self, t, hdp, Z, Ka):    
        t += self.t0
        
        ## write X0
        with open(self.fX0, 'w') as f: 
            np.savetxt(f, hdp.X0, fmt='%f', delimiter = '\t')

        ## write lw0
        with open (self.flw0, 'w') as f: 
            np.savetxt(f, hdp.lw0, fmt='%f', delimiter = '\t')

        ## write C
        with open (self.fC, 'w') as f: 
            np.savetxt(f, hdp.C, fmt='%d', delimiter = '\t')

        ## write Z
        with open (self.fZ, 'w') as f: 
            np.savetxt(f, hdp.Z, fmt='%d', delimiter = '\t')

        ## write LW
        with open (self.fLW, 'w') as f: 
            np.savetxt(f, hdp.LW, fmt='%f', delimiter = '\t')

        ## write eta
        with open (self.feta, 'a') as f: 
            np.savetxt(f, np.atleast_2d(np.r_[t, hdp.eta0, hdp.eta]), fmt='\t%d\t%f' + '\t%f ' * np.size(hdp.eta))

        ## write pars
        with open (self.fpars, 'a') as f: 
            np.savetxt(f, np.atleast_2d(np.r_[t, Ka, hdp.pars]), fmt='\t%d\t%d' + '\t%f' * np.size(hdp.pars))

        ## write Z
        if (self.dt > 0) and not (t % self.dt):
            with open (os.path.join(self.dZ, str(t)), 'w') as f: 
                np.savetxt(f, Z, fmt='%d', delimiter = '\t')

        
################################################################################

def normalizeLogWeights(lw):
    ref  = lw.max(0)
    lsum = np.log( np.exp(lw - ref).sum(0) ) + ref

    return lw - lsum
 
################################################################################

def computeClusterOccupancies(K, indicators):
    labs = np.arange(K).reshape(-1,1);                      ## cluster labels
    Z    = (indicators == labs)                             ## indicators
    
    return Z.sum(1)

################################################################################

def getClusterInfo(K, indicators):
    Ko = computeClusterOccupancies(K, indicators)
    Ki = Ko > 0               ## active clusters
    Ka = np.count_nonzero(Ki) ## number of active clusters

    return Ko, Ki, Ka

################################################################################

def computeAverageSimilarityMatrix(path, T0, T, step, compare_samples = True):
    ## read directory
    fnames = os.listdir(path)
    fnames = np.sort(np.asarray(fnames, dtype='int'))
    fnames = fnames[(fnames >= T0) & (fnames <= T)]    ## T0 <= fnames <= T 
    fnames = fnames[range(0, fnames.size, step)]       ## keep every step-th sample
        
    ## sample size
    nsamples = fnames.size
    
    ## loop
    sim = 0.
    print >> sys.stderr, 'Start sampling at iteration {0} ...'.format(fnames[0])        
    for fname in fnames.astype('str'): 
        ## read indicators
        Z = np.loadtxt(path + fname, dtype = 'int')
        Z = Z.T if compare_samples is False else Z
        
        ## update average similarity matrix
        sim += _computeSimilarityMatrix(Z)

    ## log
    print >> sys.stderr, 'Finish sampling at iteration {0} ...'.format(fnames[-1])                    
    print >> sys.stderr, '{0} samples processed'.format(nsamples)        
    
    ## return
    return sim / nsamples
    
################################################################################

def computeGenePosteriorProbabilities(path, T0, T, step, idxs1, idxs2):
    ## read directory
    fnames = os.listdir(path)
    fnames = np.asarray(fnames, dtype='int')
    fnames = fnames[(fnames >= T0) & (fnames <= T)]    ## T0 <= fnames <= T 
    fnames = fnames[range(0, len(fnames), step)]       ## keep every step-th sample
    
    ## sample size
    nsamples = fnames.size

    ## loop
    N1 = len(idxs1)
    N2 = len(idxs2)  
    p  = 0.
    print >> sys.stderr, 'Start sampling at iteration {0} ...'.format(fnames[0])        
    for fname in fnames.astype('str'): 
        ## read indicators
        Z = np.loadtxt(path + fname, dtype = 'int')        
        
        ## update number of co-occurances
        s  = [ ( Z[idx] == Z[idxs2] ).sum(0) for idx in idxs1 ]
        p += np.sum(s, 0)

    ## log
    print >> sys.stderr, 'Finish sampling at iteration {0} ...'.format(fnames[-1])                    
    print >> sys.stderr, '{0} samples processed'.format(nsamples)        
        
    ## return
    return p / (N1 * N2 * nsamples)

################################################################################

def computeGenePosteriorProbabilities2(path, T0, T, step, idxs1, idxs2):
    ## read directory
    fnames = os.listdir(path)
    fnames = np.asarray(fnames, dtype='int')
    fnames = fnames[(fnames >= T0) & (fnames <= T)]    ## T0 <= fnames <= T 
    fnames = fnames[range(0, len(fnames), step)]       ## keep every step-th sample
    
    ## loop
    nsamples = 0
    idxs = np.r_[idxs1,idxs2]
    p  = 0.
    print >> sys.stderr, 'Start sampling at iteration {0} ...'.format(fnames[0])        
    for fname in fnames.astype('str'): 
        ## read indicators
        Z = np.loadtxt(path + fname, dtype = 'int')        
        
        ## check groups of replicas
        i1 = ( Z[idxs1[0]] == Z[idxs1[1:]] ).all(0)
        i2 = ( Z[idxs2[0]] == Z[idxs2[1:]] ).all(0)
        
        nsamples += (i1 & i2)
        
        ## update number of co-occurances
        p += ( Z[idxs[0]] == Z[idxs[1:]] ).all(0)

    ## log
    print >> sys.stderr, 'Finish sampling at iteration {0} ...'.format(fnames[-1])                    
    print >> sys.stderr, '{0}/{1} samples processed'.format(min(nsamples),max(nsamples))        
        
    ## return
    return p / nsamples
    
################################################################################

def computeGenePosteriorProbabilities3(path, T0, T, step, idxs1, idxs2):
    ## read directory
    fnames = os.listdir(path)
    fnames = np.asarray(fnames, dtype='int')
    fnames = fnames[(fnames >= T0) & (fnames <= T)]    ## T0 <= fnames <= T 
    fnames = fnames[range(0, len(fnames), step)]       ## keep every step-th sample
    
    ## sample size
    nsamples = fnames.size

    ## loop
    idxs = np.r_[idxs1,idxs2]
    p = p1 = p2 = 0.
    print >> sys.stderr, 'Start sampling at iteration {0} ...'.format(fnames[0])        
    for fname in fnames.astype('str'): 
        ## read indicators
        Z = np.loadtxt(path + fname, dtype = 'int')        

        ## update number of co-occurances        
        p1 += ( Z[idxs1[0]] == Z[idxs1[1:]] ).all(0)
        p2 += ( Z[idxs2[0]] == Z[idxs2[1:]] ).all(0)
        p  += ( Z[idxs[0]]  == Z[idxs[1:]]  ).all(0)

    p  /= nsamples
    p1 /= nsamples
    p2 /= nsamples 
    
    ## log
    print >> sys.stderr, 'Finish sampling at iteration {0} ...'.format(fnames[-1])                    
    print >> sys.stderr, '{0} samples processed'.format(nsamples)        
        
    ## return
    return p / (p1 * p2)
    
################################################################################

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

################################################################################

def _computeSimilarityMatrix(Z):
    nrows, ncols = Z.shape
    sim = np.zeros((nrows,nrows))
    for i in range(nrows):
        sim[i,i:] = ( Z[i] == Z[i:] ).sum(1)
        sim[i:,i] = sim[i,i:]

    ## return
    return sim / ncols

             
        

## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import os
import numpy  as np
import pandas as pd

################################################################################

class Results(object):
    def __init__(self, fld, samples = None, genes = None, *args, **kargs):    
        ## load results from disk
        pars = np.loadtxt(os.path.join(fld,'pars.txt'))
        eta  = np.loadtxt(os.path.join(fld,'eta.txt'))
        lw0  = np.loadtxt(os.path.join(fld,'lw0.txt'))
        LW   = np.loadtxt(os.path.join(fld,'LW.txt'))
        X0   = np.loadtxt(os.path.join(fld,'X0.txt'))
        C    = np.loadtxt(os.path.join(fld,'C.txt'), dtype='int')
        Z    = np.loadtxt(os.path.join(fld,'Z.txt'), dtype='int')
        path = os.path.join(fld,'clust')
        Zf   = os.listdir(path)

        ## read everything into object attributes        
        t = pars[:,0]
        
        self.Ka = pd.Series(pars[:,1], index = t)
        self.sh = pd.Series(pars[:,2], index = t)
        self.sc = pd.Series(pars[:,3], index = t)
        
        self.eta = eta
        self.C   = C
        self.Z   = Z
        self.lw0 = lw0
        self.LW  = LW
        self.X0  = X0
        
        self.Zd  = np.asarray([ c[z] for c, z in zip(C,Z) ])     ## current direct clustering

        self._Zf   = np.sort(np.asarray(Zf, dtype='int'))        ## ordered list of clustering file names                       
        self._path = path                                        ## directory where these filenames are kept
        
        self.samples = np.arange(self.Zd.shape[0]) if samples is None else samples
        self.genes   = np.arange(self.Zd.shape[1]) if genes   is None else genes

    ##############################################################################

    def clusts(self, T0, T, dt):
        ids = (self._Zf >= T0) & (self._Zf <= T) & (np.arange(self._Zf.size) % dt == 0)   ## keep every dt-th sample between T0 and T
        Zf  = self._Zf[ids] 
        
        for f in Zf: yield f, np.loadtxt(os.path.join(self._path, str(f)), dtype = 'int')

################################################################################


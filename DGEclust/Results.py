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
    def __init__(self, fld, *args, **kargs):    
        ## load results from disk
        pars = np.loadtxt(os.path.join(fld,'pars.txt'))
        eta  = np.loadtxt(os.path.join(fld,'eta.txt'))
        lw0  = np.loadtxt(os.path.join(fld,'lw0.txt'))
        LW   = np.loadtxt(os.path.join(fld,'LW.txt'))
        X0   = np.loadtxt(os.path.join(fld,'X0.txt'))
        C    = np.loadtxt(os.path.join(fld,'C.txt'), dtype='int')
        Z    = np.loadtxt(os.path.join(fld,'Z.txt'), dtype='int')
                
        ## read everything into object attributes        
        t = pars[:,0]
        
        self.Ka = pd.Series(pars[:,1], index = t)
        self.mu = pd.Series(pars[:,2], index = t)
        self.s2 = pd.Series(pars[:,3], index = t)
        self.sh = pd.Series(pars[:,4], index = t)
        self.sc = pd.Series(pars[:,5], index = t)
        
        self.eta = eta
        self.C   = C
        self.Z   = Z
        self.lw0 = lw0
        self.LW  = LW
        self.X0  = X0
        
        self.Zd  = np.asarray([ c[z] for c, z in zip(C,Z) ])
        
################################################################################

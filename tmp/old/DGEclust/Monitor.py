## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import sys, os
import numpy as np

################################################################################

class Monitor(object):    
    def __init__(self, outDir, T0, dt, extend):                
        ## attributes
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
        self.T0    = T0

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
            np.savetxt(f, np.atleast_2d(np.r_[t, hdp.eta0, hdp.eta]), fmt='%d\t%f' + '\t%f ' * np.size(hdp.eta))

        ## write pars
        with open (self.fpars, 'a') as f: 
            np.savetxt(f, np.atleast_2d(np.r_[t, Ka, hdp.pars]), fmt='%d\t%d' + '\t%f' * np.size(hdp.pars))

        ## write Z
        if (t > self.T0) and (self.dt > 0) and not (t % self.dt):
            with open (os.path.join(self.dZ, str(t)), 'w') as f: 
                np.savetxt(f, Z, fmt='%d', delimiter = '\t')

################################################################################

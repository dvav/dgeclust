'''
Created on Apr 19, 2013

@author: dimitris
'''

################################################################################

import numpy as np
import utils as ut

################################################################################

class HDP(object):
    '''
    classdocs
    '''

    ############################################################################

    def __init__(self, X0, lw0, LW, C, Z, eta0, eta, pars):
        '''
        Constructor
        '''        
        self.X0   = X0
        self.lw0  = lw0
        self.LW   = LW
        self.C    = C
        self.Z    = Z
        self.eta0 = eta0
        self.eta  = eta
        self.pars = pars
                        
    ############################################################################

    def getClusterInfo(self):
        Z  = [ c[z] for c, z in zip(self.C, self.Z) ]
        Ko = ut.computeClusterOccupancies(self.lw0.size, np.asarray(Z).ravel())
        Ka = np.count_nonzero(Ko > 0)  
        
        return Z, Ko, Ka
    
    
    
    ############################################################################


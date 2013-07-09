## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import numpy as np

from . import getClusterInfo

################################################################################

def computeFittedModel(X0, zd, exposure, model, xlimits = [-1., 12.], npoints = 1000):        
    ## compute cluster occupancies
    Ko, Ki, _ = getClusterInfo(X0.shape[0], zd)
    Ko = Ko[Ki]    
    
    ## read active alpha and beta, compute mu and p
    x = np.linspace(xlimits[0], xlimits[1], npoints).reshape(-1,1)   
    y = np.exp(x) * np.exp(model.dLogLik(X0[Ki], np.exp(x), exposure))        
    y = y * Ko / zd.size      ## notice the normalisation of y
    
    ## return
    return x, y 
    
################################################################################

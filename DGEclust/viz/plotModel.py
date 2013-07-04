## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import pylab            as pl
import numpy            as np
import DGEclust.utils   as ut
import DGEclust.stats   as st

################################################################################

def plotModel(X0, c, z, exposure, model = 'NegBinomial', xmin = -1., xmax = 12., npoints = 100, color = 'red', draw_clusters = False, *args, **kargs):        
    ## compute cluster occupancies
    Ko = ut.computeClusterOccupancies(X0.shape[0], c[z])
    Ki = Ko > 0
    
    ## read active alpha and beta, compute mu and p
    x = np.linspace(xmin, xmax, npoints).reshape(-1,1)   
    if model == 'Poisson':
        beta = X0[Ki]
        mu   = np.exp(beta) * exposure
        Y    = np.exp(x) * np.exp(st.dLogPoisson(np.exp(x), mu))
    else:
        phi   = X0[Ki, 0]
        beta  = X0[Ki, 1]       
        alpha = 1. / phi
        mu    = np.exp(beta) * exposure
        p     = alpha /  (alpha + mu)
        Y     = np.exp(x) * np.exp(st.dLogNegBinomial(np.exp(x), alpha, p))
        
    Ko = Ko[Ki]    
    y  = (Y * Ko).sum(1) / z.size   ## notice the normalisation of y

    ## plot    
    if draw_clusters:
        pl.plot(x, Y, color='0.75', linewidth=0.5)        
            
    pl.plot(x, y, color = color, *args, **kargs)    
    
    return x, y, Y      
    
################################################################################

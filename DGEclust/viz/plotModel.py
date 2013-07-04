## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import pylab            as pl
import numpy            as np
import DGEclust.utils   as ut

from . import plotSample

################################################################################

def plotModel(isample, res, data, model, xmin = -1., xmax = 12., npoints = 1000, color = 'red', draw_clusters = False, *args, **kargs):        
    ## compute cluster occupancies
    Ko, Ki, _ = ut.getClusterInfo(res.X0.shape[0], res.Zd[isample])
    Ko = Ko[Ki]    
    
    ## read active alpha and beta, compute mu and p
    x = np.linspace(xmin, xmax, npoints).reshape(-1,1)   
    Y = np.exp(x) * np.exp(model.dLogLik(res.X0[Ki], np.exp(x), data.exposures[isample]))        
    y = (Y * Ko).sum(1) / data.genes.size   ## notice the normalisation of y

    ## plot
    plotSample(data.icol(isample))    
    if draw_clusters:
        pl.plot(x, Y, color='0.75', linewidth=0.5)        
    pl.plot(x, y, color = color, *args, **kargs)    
    
    ## return
    return x, y, Y      
    
################################################################################

## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import pylab               as pl
import numpy               as np
import DGEclust.utils      as ut
import DGEclust.stats.dist as ds

################################################################################

def plotBaseDists(res, xlimits_gamma = [1e-6, 5], npoints = 1000):
    _, Ki, _ = ut.getClusterInfo(len(res.X0), res.Zd.ravel())

    ## plot base dist for phi
    x  = np.linspace(xlimits_gamma[0], xlimits_gamma[1], npoints);
    y  = np.exp(ds.dLogGamma(x,  res.shape[-1], res.scale[-1])); 
    y0 = np.exp(ds.dLogGamma(x,  res.shape[0], res.scale[0])); 
    pl.plot(x, y, 'k-', x, y0, 'k--', res.X0[Ki,0], [-0.011]*sum(Ki), 'ro'); 
    pl.xlabel(r'$\phi^*$'); pl.ylabel('density');  
        
################################################################################
    
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

def plotBaseDists(res, xlimits_normal = [-20, 20], xlimits_gamma = [0, 5], npoints = 1000):
    _, Ki, _ = ut.getClusterInfo(len(res.X0), res.Zd.ravel())

    ## plot base dist for beta
    x  = np.linspace(xlimits_normal[0], xlimits_normal[1], npoints); 
    y  = np.exp(ds.dLogNormal(x, res.mu[-1], res.s2[-1])); 
    y0 = np.exp(ds.dLogNormal(x, res.mu[0],  res.s2[0])); 
    pl.subplot(1,2,1); 
    pl.plot(x, y, 'k-', x, y0, 'k--', res.X0[Ki,1], [-0.011]*sum(Ki), 'ro'); 
    pl.xlabel(r'$\beta^*$'); pl.ylabel('density');  

    ## plot base dist for phi
    x  = np.linspace(xlimits_gamma[0], xlimits_gamma[1], npoints);
    y  = np.exp(ds.dLogGamma(x,  res.sh[-1], res.sc[-1])); 
    y0 = np.exp(ds.dLogGamma(x,  res.sh[0], res.sc[0])); 
    pl.subplot(1,2,2); 
    pl.plot(x, y, 'k-', x, y0, 'k--', res.X0[Ki,0], [-0.011]*sum(Ki), 'ro'); 
    pl.xlabel(r'$\phi^*$'); pl.ylabel('density');  
        
################################################################################
    
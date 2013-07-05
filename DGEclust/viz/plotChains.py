## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import pylab          as pl
import DGEclust.utils as ut 

################################################################################

def plotChains(res, T0, T, dt = 1):
    ii = (res.Ka.index >= T0) & (res.Ka.index <= T) & (res.Ka.index % dt == 0) 

    ## plot histogram of clusters
    pl.subplot(3,2,1); 
    res.Ka[ii].hist(bins=100, normed=True, color='grey'); 
    pl.plot([res.Ka[ii].mean()]*2, pl.gca().get_ylim(), 'k--'); 
    pl.xlabel('cluster #'); pl.ylabel('density');  
    
    ## plot clusters
    pl.subplot(3,2,2); 
    res.Ka[ii].plot(color='grey');
    pl.plot(pl.gca().get_xlim(), [res.Ka[ii].mean()]*2, 'k--'); 
    pl.xlabel('iteration #'); pl.ylabel('clusters #');  
    
    ## plot mu
    pl.subplot(3,2,3); 
    res.mu[ii].plot(color='grey');  
    pl.plot(pl.gca().get_xlim(), [res.mu[ii].mean()]*2, 'k--'); 
    pl.xlabel('iteration #'); pl.ylabel('mean');
    
    ## plot s2
    pl.subplot(3,2,4); 
    res.s2[ii].plot(color='grey');  
    pl.plot(pl.gca().get_xlim(), [res.s2[ii].mean()]*2, 'k--'); 
    pl.xlabel('iteration #'); pl.ylabel('variance');  
    
    ## plot sh
    pl.subplot(3,2,5); 
    res.sh[ii].plot(color='grey');  
    pl.plot(pl.gca().get_xlim(), [res.sh[ii].mean()]*2, 'k--'); 
    pl.xlabel('iteration #'); pl.ylabel('shape');
    
    #3 plot sc
    pl.subplot(3,2,6); 
    res.sc[ii].plot(color='grey');  
    pl.plot(pl.gca().get_xlim(), [res.sc[ii].mean()]*2, 'k--'); 
    pl.xlabel('iteration #'); pl.ylabel('scale');

    ## plot concentration parameters
    # figure(figsize=(20,6)); 
    # subplot(1,2,1); plot(res.eta[::10,0], res.eta[::10,1],  color='grey'); xlabel('iteration #'); ylabel(r'$\eta_0$');  
    # subplot(1,2,2); plot(res.eta[::10,0], res.eta[::10,2:], color='grey'); xlabel('iteration #'); ylabel(r'$\eta_j$');  
    # 

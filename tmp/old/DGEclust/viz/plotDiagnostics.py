## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import numpy as np
import pylab as pl
import pymc  as mc

################################################################################

def _plotGeweke(x, xlabel='First iteration', ylabel = ''):
    scores = np.asarray(mc.geweke(x));  
    pl.plot(scores[:,0], scores[:,1], 'o'); 
    pl.hlines([-2,0,2], scores[0,0],scores[-1,0], color='k', linestyle='--'); 
    pl.ylim([-2.5, 2.5]); 
    pl.xlim([scores[0,0], scores[-1,0]]); 
    pl.xlabel(xlabel); 
    pl.ylabel(ylabel); 
    
################################################################################

def _plotACorr(x, xlabel ='lag', ylabel='', maxlags = 100):
    pl.acorr(x, normed=True, detrend=pl.mlab.detrend_mean, usevlines=True, maxlags=maxlags);
    pl.xlabel(xlabel)
    pl.ylabel(ylabel)    

################################################################################

def plotDiagnostics(res, T0, T, dt = 1, maxlags = 100):
    ii = (res.Ka.index >= T0) & (res.Ka.index <= T) & (res.Ka.index % dt == 0) 

    ## Geweke's test
    pl.subplot(5,2,1); _plotGeweke(res.Ka[ii], ylabel='Z-score for clusters #')
    pl.subplot(5,2,3); _plotGeweke(res.mu[ii], ylabel='Z-score for mean')    
    pl.subplot(5,2,5); _plotGeweke(res.s2[ii], ylabel='Z-score for variance')    
    pl.subplot(5,2,7); _plotGeweke(res.sh[ii], ylabel='Z-score for shape')
    pl.subplot(5,2,9); _plotGeweke(res.sc[ii], ylabel='Z-score for scale')
    
    ## auto-correlation
    pl.subplot(5,2,2);  _plotACorr(res.Ka[ii], ylabel='autocorrelation for clusters #', maxlags = maxlags);
    pl.subplot(5,2,4);  _plotACorr(res.mu[ii], ylabel='autocorrelation for mean', maxlags = maxlags);
    pl.subplot(5,2,6);  _plotACorr(res.s2[ii], ylabel='autocorrelation for variance', maxlags = maxlags);
    pl.subplot(5,2,8);  _plotACorr(res.sh[ii], ylabel='autocorrelation for shape', maxlags = maxlags);
    pl.subplot(5,2,10); _plotACorr(res.sc[ii], ylabel='autocorrelation for scale', maxlags = maxlags);



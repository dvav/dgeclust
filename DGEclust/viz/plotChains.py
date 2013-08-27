## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import pylab          as pl

################################################################################

def _plotChain(x, color = 'grey', xlabel='# of iterations', ylabel=''):
    x.plot(color=color);
    pl.plot(pl.gca().get_xlim(), [x.mean()]*2, 'k--'); 
    pl.xlabel(xlabel) 
    pl.ylabel(ylabel)  

def _plotHistogram(x, bins = 100, color = 'grey', ylabel = 'density', xlabel = ''):
    pl.hist(x, bins=bins, normed=True, color=color); 
    pl.plot([x.mean()]*2, pl.gca().get_ylim(), 'k--'); 
    pl.xlabel(xlabel); pl.ylabel(ylabel);  
    pl.grid()
        
################################################################################

def plotChains(res, T0, T, dt = 1, bins = 100, color = 'grey'):
    ii = (res.Ka.index >= T0) & (res.Ka.index <= T) & (res.Ka.index % dt == 0) 

    ## plot histogram of clusters
    pl.subplot(3,2,1); _plotChain(res.Ka[ii], color=color, ylabel='# of clusters')    
    pl.subplot(3,2,2); _plotHistogram(res.Ka[ii], bins=bins, color=color, xlabel='# of clusters')

    pl.subplot(3,2,3); _plotChain(res.shape[ii], color=color, ylabel='shape')                    
    pl.subplot(3,2,4); _plotHistogram(res.shape[ii], bins=bins, color=color, xlabel='shape')    

    pl.subplot(3,2,5); _plotChain(res.scale[ii], color=color, ylabel='scale')                    
    pl.subplot(3,2,6); _plotHistogram(res.scale[ii], bins=bins, color=color, xlabel='scale')    

    ## plot concentration parameters
    # figure(figsize=(20,6)); 
    # subplot(1,2,1); plot(res.eta[::10,0], res.eta[::10,1],  color='grey'); xlabel('iteration #'); ylabel(r'$\eta_0$');  
    # subplot(1,2,2); plot(res.eta[::10,0], res.eta[::10,2:], color='grey'); xlabel('iteration #'); ylabel(r'$\eta_j$');  
    # 

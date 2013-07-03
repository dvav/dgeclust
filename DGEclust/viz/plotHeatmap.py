## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

## NOTE: Based on code found in the following link http://stackoverflow.com/questions/2455761/reordering-matrix-elements-to-reflect-column-and-row-clustering-in-naiive-python/3017704#3017704

################################################################################

import numpy                    as np
import pylab                    as pl
import scipy.cluster.hierarchy  as hr

################################################################################

def plotHeatmap(D, fig = None, method = 'average', metric = 'euclidean', xticklabels = None, yticklabels = None, cmap = pl.cm.warm): 
    ## create figure
    if fig is None:
        fig = pl.figure(figsize = (10,10))

    ## normalize D
    D = ( D - D.min() ) / ( D.max() - D.min() )
    
    # Compute and plot first dendrogram
    ax1 = fig.add_axes([0,0,0.2,0.6], frame_on = False)
    Y1  = hr.linkage(D, method = method, metric = metric);
    Z1  = hr.dendrogram(Y1, orientation = 'right')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Compute and plot second dendrogram
    ax2 = fig.add_axes([0.21,0.61,0.6,0.2], frame_on = False)
    Y2  = hr.linkage(D, method = method, metric = metric);
    Z2  = hr.dendrogram(Y2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Plot distance matrix
    axmatrix = fig.add_axes([0.21,0,0.6,0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    yticklabels = np.asarray(yticklabels)[idx2].tolist()
    xticklabels = np.asarray(xticklabels)[idx1].tolist()
    D    = D[idx1,:]
    D    = D[:,idx2]
    
    im = axmatrix.matshow(D, aspect = 'auto', origin = 'lower', cmap = cmap)   
    axmatrix.xaxis.set_ticks_position('bottom')
    axmatrix.yaxis.set_ticks_position('right')    
    axmatrix.set_xticks(range(len(xticklabels)))
    axmatrix.set_yticks(range(len(yticklabels)))
    axmatrix.set_xticklabels(xticklabels, rotation = 'vertical') 
    axmatrix.set_yticklabels(yticklabels) 
    
    # Plot colorbar
    axcolor = fig.add_axes([0.05,0.7,0.15,0.05], frame_on = False)
    pl.colorbar(im, cax = axcolor, orientation = 'horizontal', ticks = (0,0.5,1))
    
    

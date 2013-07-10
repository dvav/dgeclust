## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

## NOTE: Based on code found in the following link: http://stackoverflow.com/questions/2455761/reordering-matrix-elements-to-reflect-column-and-row-clustering-in-naiive-python/3017704#3017704

################################################################################

import numpy                    as np
import pylab                    as pl
import matplotlib.gridspec      as gr
import scipy.cluster.hierarchy  as hr

################################################################################

def plotHeatmap(D, method = 'average', metric = 'euclidean', xticklabels = None, yticklabels = None, cmap = pl.cm.coolwarm): 
    grid = gr.GridSpec(2,2, width_ratios=[1,3.5], height_ratios=[1,3.5], wspace=0, hspace=0)
    
    ## normalize D
    D = ( D - D.min() ) / ( D.max() - D.min() )
        
    # Compute and plot first dendrogram
    ax2 = pl.subplot(grid[1])
    Y2  = hr.linkage(D, method = method, metric = metric);
    Z2  = hr.dendrogram(Y2)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_frame_on(False)

    # Compute and plot second dendrogram
    ax1 = pl.subplot(grid[2]); 
    Y1  = hr.linkage(D, method = method, metric = metric);
    Z1  = hr.dendrogram(Y1, orientation = 'right')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_frame_on(False)
    
    # Plot distance matrix
    axmatrix = pl.subplot(grid[3])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    yticklabels = np.asarray(yticklabels)[idx2].tolist()
    xticklabels = np.asarray(xticklabels)[idx1].tolist()
    D    = D[idx1,:]
    D    = D[:,idx2]
    
    im = axmatrix.matshow(D, aspect = 'auto', origin = 'lower', cmap = cmap)   
    axmatrix.set_xticks(range(len(xticklabels)))
    axmatrix.set_yticks(range(len(yticklabels)))
    axmatrix.xaxis.set_tick_params(which = 'major', size=0)
    axmatrix.yaxis.set_tick_params(which = 'major', size=0)
    axmatrix.xaxis.set_ticks_position('bottom')
    axmatrix.yaxis.set_ticks_position('right')        
    axmatrix.set_xticklabels(xticklabels, rotation = 'vertical') 
    axmatrix.set_yticklabels(yticklabels)
    
    # Plot colorbar
    axcolor = pl.gcf().add_axes([0.02,0.9,0.12,0.05])
    pl.colorbar(im, cax = axcolor, orientation = 'horizontal', ticks = (0,0.5,1))
    
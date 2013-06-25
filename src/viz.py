'''
Created on Apr 20, 2013

@author: dimitris
'''
################################################################################

import numpy                    as np
import pylab                    as pl
import scipy.cluster.hierarchy  as hr
import utils                    as ut
import stats                    as st

################################################################################

def plotRA(samples1, samples2, ids = None, epsilon = 0.5, *args, **kargs):        
    samples1 = samples1.astype('double')
    samples2 = samples2.astype('double')
    
    ## set zero elements to epsilon
    samples1[samples1 < 1.] = epsilon     
    samples2[samples2 < 1.] = epsilon     
    
    ## compute means
    lmeans1 = np.log2(samples1).mean(0)
    lmeans2 = np.log2(samples2).mean(0)
     
    ## compute A and R
    A = ( lmeans1 + lmeans2 ) * 0.5
    R =   lmeans1 - lmeans2
        
    ## generate RA plot
    if ids is not None:
        pl.plot(A[~ids], R[~ids], 'k.', A[ids], R[ids], 'r.')
    else:
        pl.plot(A, R, 'k.')
        
    pl.plot(pl.gca().get_xlim(),(0.,0.),'k--')
    pl.xlabel('mean')
    pl.ylabel('log2 fold change')
    
    return A, R
        
################################################################################

def plotSample(sample, epsilon = 0.5, bins = 100, normed = True, *args, **kargs):        
    sample = sample.astype('double')
    
    ## set zero elements to epsilon
    sample[sample < 1.] = epsilon     
    
    ## generate plot
    pl.hist(np.log(sample), bins = bins, normed = normed, *args, **kargs)
    pl.xlabel('log(S)')
    pl.ylabel('frequency')
            
################################################################################

def plotModel(X0, c, z, exposure, model = 'NegBinomial', xmin = -1., xmax = 12., npoints = 100, color = 'red', draw_clusters = False, *args, **kargs):        
    x = np.linspace(xmin, xmax, npoints).reshape(-1,1)   

    ## compute cluster occupancies
    Ko = ut.computeClusterOccupancies(X0.shape[0], c[z])
    Ki = Ko > 0
    
    ## read active alpha and beta, compute mu and p
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
    
################################################################################

def plotHeatmap(D, fig = None, xmethod = 'average', xmetric = 'euclidean', ymethod = 'average', ymetric = 'euclidean', xticklabels = None, yticklabels = None, cmap = None):    ## pl.cm.hot
    ## create figure
    if fig is None:
        fig = pl.figure(figsize = (10,10))

    ## normalize D
    D = ( D - D.min() ) / ( D.max() - D.min() )
    
    # Compute and plot first dendrogram
    ax1 = fig.add_axes([0,0,0.2,0.6], frame_on = False)
    Y   = hr.linkage(D, method = ymethod, metric = ymetric);
    Z1  = hr.dendrogram(Y, orientation = 'right')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Compute and plot second dendrogram
    ax2 = fig.add_axes([0.21,0.61,0.6,0.2], frame_on = False)
    Y = hr.linkage(D, method = xmethod, metric = xmetric);
    Z2 = hr.dendrogram(Y)
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
    
    

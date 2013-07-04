## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import pylab as pl
import numpy as np

################################################################################

def plotSample(sample, epsilon = 0.5, bins = 100, normed = True, *args, **kargs):        
    sample = sample.astype('double')
    
    ## set zero elements to epsilon
    sample[sample < 1.] = epsilon     
    
    ## generate plot
    pl.hist(np.log(sample), bins = bins, normed = normed, *args, **kargs)
            
################################################################################

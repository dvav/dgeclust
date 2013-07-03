## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################
 
import numpy as np

################################################################################

def normalizeLogWeights(lw):
    ref  = lw.max(0)
    lsum = np.log( np.exp(lw - ref).sum(0) ) + ref

    return lw - lsum
 
################################################################################

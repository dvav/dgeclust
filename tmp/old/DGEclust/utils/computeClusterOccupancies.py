## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################
 
import numpy as np

################################################################################

def computeClusterOccupancies(K, indicators):
    labs = np.arange(K).reshape(-1,1);                      ## cluster labels
    Z    = (indicators == labs)                             ## indicators
    
    return Z.sum(1)

################################################################################

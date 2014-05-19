## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################
 
import numpy as np
from . import computeClusterOccupancies

################################################################################

def getClusterInfo(K, indicators):
    Ko = computeClusterOccupancies(K, indicators)
    Ki = Ko > 0               ## active clusters
    Ka = np.count_nonzero(Ki) ## number of active clusters

    return Ko, Ki, Ka

################################################################################

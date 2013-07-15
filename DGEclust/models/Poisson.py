# # Copyright (C) 2012-2013 Dimitrios V. Vavoulis
# # Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
# # Department of Computer Science
# # University of Bristol
# 
# ################################################################################
# 
import numpy               as np
import DGEclust.stats.rand as rn
import DGEclust.stats.conj as cj
import DGEclust.stats.dist as ds

################################################################################
        
def rParams(beta, shape, scale):
    shape, scale, _,  _, _  = cj.gamma_shape_scale(beta, shape, scale)           

    return shape, scale
     
################################################################################
    
def rPrior(N, shape, scale):    
    return rn.gamma(shape, scale, N)
        
################################################################################
   
def rPost(beta, idx, C, Z, counts, exposures, shape, scale):     
    Z     = [ c[z] == idx for c, z in zip(C, Z) ]
    ndata = np.sum([ z.sum()       for z in Z ]) 
    dsum  = np.sum([ cnts[z].sum() for cnts, z in zip(counts,Z) ]) 

    # return        
    return rn.gamma(shape + dsum, scale / (scale * ndata + 1.))

################################################################################
    
def dLogLik(beta, counts, exposure):        
    return ds.dLogPoisson(counts.reshape(-1,1), beta)
    
################################################################################
    
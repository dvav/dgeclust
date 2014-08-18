## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import numpy  as np

################################################################################

## Estimate normalization factors as in DESeq
def estimateSizeFactors(counts, locfcn = np.median):
    ## compute geometric mean over genes
    lsum   = np.log(counts).sum(0)
    gmeans = np.exp(lsum / counts.shape[0])          
    
    ## divide samples by geometric means
    counts = counts / gmeans     ## counts /= gmeans gives wrong results!!!????        

    ## get median (or other central tendency metric) of samples excluding genes with 0 gmean 
    sizes = locfcn(counts[:,gmeans > 0.], 1)      
    
    ## return
    return sizes
            
################################################################################

class CountData(object):
    def __init__(self, counts, exposures = None, groups = None):
        self.counts     = counts.values.T
        self.exposures  = estimateSizeFactors(self.counts) if exposures is None else exposures
        self.countsNorm = self.counts / self.exposures.reshape(-1,1)
        self.libSizes   = self.counts.sum(1)
        self.groups     = np.arange(self.counts.shape[0])  if groups    is None else groups  
    
        self.ngenes     = self.counts.shape[1]
        self.ngroups    = len(self.groups)
        self.nreplicas  = [ np.size(group) for group in self.groups ] 
        
        self.genes      = counts.index.values
        self.samples    = counts.columns.values
        
################################################################################
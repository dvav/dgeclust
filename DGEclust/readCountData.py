## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import pandas as pd

################################################################################

def readCountData(fname, classes = None, *args, **kargs):    
    df = pd.read_table(fname, *args, **kargs))

    ## add attributes      
    df.counts    = df.values  
    df.exposures = df.sum() 
    df.samples   = df.columns
    df.genes     = df.index
    
    ## classes ??
                    
    return df
    
################################################################################

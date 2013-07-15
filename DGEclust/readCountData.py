## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import numpy  as np
import pandas as pd

################################################################################

def readCountData(fname, classes = None, *args, **kargs):    
    df = pd.read_table(fname, *args, **kargs)

    ## add attributes      
    df.counts    = df.values  
    df.exposures = df.sum() / df.sum().astype('double') #df.sum() / df.sum().max().astype('double') 
    df.samples   = df.columns
    df.genes     = df.index
    
    ## classes
    if classes is None:
        df.classes = np.arange(df.samples.size).astype('str')
    else:
        df.classes = classes
                    
    return df
    
################################################################################

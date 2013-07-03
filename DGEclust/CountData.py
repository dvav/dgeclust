## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import pandas as pd

################################################################################

class CountData(pd.DataFrame):
    def __init__(self, fname, *args, **kargs):    
        super(CountData, self).__init__(pd.read_table(fname, *args, **kargs))
    
        ## add attributes        
        self.exposures = self.sum() 
                    
################################################################################

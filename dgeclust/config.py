from __future__ import division

import os
import json

########################################################################################################################

## read configuration file
config_file_name = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_file_name) as f:
    config = json.load(f)

models = config['models']
fnames = config['fnames']
nthreads = config['nthreads']
clust = config['clust']
post = config['post']

########################################################################################################################
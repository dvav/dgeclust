from __future__ import division

import os
import json
import collections as cl

########################################################################################################################

## read configuration file
config_file_name = os.path.join(os.path.dirname(__file__), '..', 'config.json')
with open(config_file_name) as f:
    config = json.load(f, object_pairs_hook=cl.OrderedDict)

models = config['models']
norm = config['norm']
fnames = config['fnames']
nthreads = config['nthreads']
clust = config['clust']
post = config['post']

########################################################################################################################

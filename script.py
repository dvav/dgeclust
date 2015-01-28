## call like: python script.py base max_nreps_per_group nreps ngroups sample &

import sys

import numpy as np
import pandas as pd

from dgeclust import CountData, SimulationManager
from dgeclust.models.tmp.nbinom_noclust_2 import NBinomModel


_, base, max_nsamples_per_group, r, g, s = sys.argv
max_nsamples_per_group = int(max_nsamples_per_group)
r = int(r)
g = int(g)
s = int(s)

## run simulation
infile = '{}/data/simdata{}.txt'.format(base, s)
outdir = '{}/{}/{}/_clust{}'.format(base, r, g, s)
samples = ['sample' + str(item) for item in range(1, r+1)*g + np.repeat(range(0, g), r)*max_nsamples_per_group]
groups = np.repeat(range(1, g+1), r)

counts = pd.read_table(infile, index_col=0)
data = CountData(counts[samples], groups=groups)
model = NBinomModel(data)
SimulationManager().new(data, model, outdir=outdir, bg=False, niters=10000)

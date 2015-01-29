import pandas as pd

from dgeclust import CountData, SimulationManager
from dgeclust.models.tmp.nbinom_HDP5 import NBinomModel


counts = pd.read_table('/Users/dimitris/Repositories/benchmarks/fly/modencodefly_pooledreps_count_table_filt.txt', index_col=0)
# data = CountData(counts[['C1', 'C2', 'F1', 'F2']], groups=['C', 'C', 'F', 'F'])
# data = CountData(counts[['C1', 'F1', 'H1', 'P1', 'T1']], groups=['C', 'F', 'H', 'P', 'T'])
# data = CountData(counts[['C1', 'C2', 'F1', 'F2', 'H1', 'H2', 'P1', 'P2', 'T1', 'T2']], groups=list(np.repeat(['C', 'F', 'H', 'P', 'T'], 2)))
# data = CountData(counts, groups=list(np.repeat(['C', 'F', 'H', 'P', 'T'], 5)))
data = CountData(counts[['SRX008013','SRX008022','SRX008012','SRX008028']], groups=['A', 'A', 'B', 'B'])
model = NBinomModel(data)
# model = NBinomModel.load('_cage/')
SimulationManager().new(data, model, outdir='_clust', niters=10000)

# from dgeclust import compare_groups, compute_similarity_vector
# pvals, nsamples = compare_groups('_cage2/', 'C', 'F', t0=10000, tend=20000, dt=10, nthreads=1)

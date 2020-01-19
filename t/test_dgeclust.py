import pytest
import sys
from os.path import isfile
import numpy as np
import pandas as pd
import pickle as pkl
from dgeclust import SimulationManager
from dgeclust.models import NBinomModel
from dgeclust import compare_groups, compute_similarity_vector
import matplotlib.pylab as pl
import scipy.cluster.hierarchy as hr
from dgeclust.utils import plot_ra


def test_tutorial_passthrough(counts_filt, data, tmp_file_map):

    assert (8647, 7) == counts_filt.shape, 'incorrect number of filtered features'
    mgr = SimulationManager()

    mdl = NBinomModel(data, outdir=tmp_file_map['tmp_dir'])

    assert counts_filt.shape == (mdl.nfeatures, mdl.nsamples), 'incorrect number of samples and features'
    mgr.new(data, mdl, bg=False, niters=10)

    assert isfile(tmp_file_map['pars.txt'])
    assert isfile(tmp_file_map['state.pkl'])
    assert isfile(tmp_file_map['z/0'])

    pars = pd.read_csv(tmp_file_map['pars.txt'], sep='\t', header=None)
    assert (11, 7) == pars.shape

    model_state = pkl.load(open(tmp_file_map['state.pkl'], 'rb'))

    assert counts_filt.shape[0] == model_state.nfeatures
    assert counts_filt.shape[1] == model_state.nsamples

    mdl = NBinomModel.load(tmp_file_map['tmp_dir'])
    mgr.new(data, mdl, niters=5, bg=False)

    pars_ext = pd.read_csv(tmp_file_map['pars.txt'], sep='\t', header=None)
    assert (16, 7) == pars_ext.shape

    if sys.version_info[0] > 2:
        assert pars[1][0:11].to_list() == pars_ext[1][0:11].to_list()
    else:
        assert pars[1][0:11].tolist() == pars_ext[1][0:11].tolist()

    res, nsamples = compare_groups(data, mdl, group1='treated', group2='untreated', nthreads=1, t0=11, tend=16)

    idxs = np.argsort(res.FDR)[:50]
    simvec, nsamples = compute_similarity_vector(mdl, inc=idxs, compare_genes=True, t0=11, tend=16)
    assert simvec is not None, 'failed to calculate similarity vector'
    assert nsamples is not None, 'failed to return number os samples'
    assert 1225 == simvec.shape[0], 'incorrect number of similarity values'
    assert 1 == simvec.max(), 'incorrect max similarity value'
    assert 0.5 == simvec.min(), 'incorrect minimum similarity value'


def test_tutorial_few_iterations(counts_filt, data, tmp_file_map):

    assert (8647, 7) == counts_filt.shape, 'incorrect number of filtered features'
    mgr = SimulationManager()

    mdl = NBinomModel(data, outdir=tmp_file_map['tmp_dir'])

    assert counts_filt.shape == (mdl.nfeatures, mdl.nsamples), 'incorrect number of samples and features'
    mgr.new(data, mdl, bg=False, niters=50)

    assert isfile(tmp_file_map['pars.txt']), 'failed to write pars.txt file'
    assert isfile(tmp_file_map['state.pkl']), 'failed to write state.pkl file'
    assert isfile(tmp_file_map['z/0']), 'failed to write z/0 file'

    pars = pd.read_csv(tmp_file_map['pars.txt'], sep='\t', header=None)
    assert (51, 7) == pars.shape, 'incorrect number of iterations'

    model_state = pkl.load(open(tmp_file_map['state.pkl'], 'rb'))

    assert counts_filt.shape[0] == model_state.nfeatures, 'incorrect number of features from loaded state object'
    assert counts_filt.shape[1] == model_state.nsamples, 'incorrect number of samples from loaded state object'

    mdl = NBinomModel.load(tmp_file_map['tmp_dir'])
    mgr.new(data, mdl, niters=130, bg=False)

    pars_ext = pd.read_csv(tmp_file_map['pars.txt'], sep='\t', header=None)
    assert (181, 7) == pars_ext.shape, 'failed to update pars.txt with new iterations'

    if sys.version_info[0] > 2:
        assert pars[1][0:51].to_list() == pars_ext[1][0:51].to_list(), 'change to values from initial simulation'
    else:
        assert pars[1][0:51].tolist() == pars_ext[1][0:51].tolist(), 'change to values from initial simulation'

    res, nsamples = compare_groups(data, mdl, group1='treated', group2='untreated', nthreads=1, t0=150, tend=181)
    idxs = res.FDR < 0.01  # identify DE genes at 1% FDR
    assert np.abs(sum(idxs) - 400) < 200, 'incorrect number of DE genes at 1% FDR'
    assert np.abs(sum(res.Posteriors == 0) - 400) < 100, 'incorrect number of genes with posterior probability of zero'

    idxs = np.argsort(res.FDR)[:50]  # identifies the top 20 DE genes
    simvec, nsamples = compute_similarity_vector(mdl, inc=idxs, compare_genes=True, t0=150, tend=181)
    assert simvec is not None, 'failed to calculate similarity vector'
    assert nsamples is not None, 'failed to return number os samples'
    assert 1225 == simvec.shape[0], 'incorrect number of similarity values'
    assert 0.02 > np.abs(simvec.max() - 1), 'max similarity outside tolerance'
    assert 0.5 == simvec.min(), 'incorrect minimum similarity value'
    assert 0.15 > np.abs(simvec.mean() - 0.5), 'mean similarity outside tolerance'
    assert 0.12 > np.abs(simvec.std() - 0.21), 'standard deviation from similarities outside tolerance'


@pytest.mark.skipif(True, reason='slow test with default number of iterations')
def test_tutorial_complete(counts_filt, data, tmp_file_map):

    mgr = SimulationManager()

    model = NBinomModel(data, outdir=tmp_file_map['tmp_dir'])

    assert counts_filt.shape == (model.nfeatures, model.nsamples), 'incorrect number of samples and features'
    mgr.new(data, model, bg=False)

    # model.plot_progress(fig=pl.figure(figsize=(10, 7)))
    # pl.show()
    #
    # model.plot_fitted_model('treated1fb', data)
    # pl.show()

    assert isfile(tmp_file_map['pars.txt']), 'failed to write pars.txt file'
    assert isfile(tmp_file_map['state.pkl']), 'failed to write state.pkl file'
    assert isfile(tmp_file_map['z/0']), 'failed to write z/0 file'

    pars = pd.read_csv(tmp_file_map['pars.txt'], sep='\t', header=None)
    assert (10001, 7) == pars.shape, 'incorrect number of iterations'

    res, nsamples = compare_groups(data, model, group1='treated', group2='untreated', nthreads=1)
    idxs = res.FDR < 0.01  # identify DE genes at 1% FDR
    assert 5 > np.abs(sum(idxs) - 157), 'incorrect number of DE genes at 1% FDR'
    assert 10 > np.abs(sum(res.Posteriors == 0) - 30), 'incorrect number of genes with posterior probability of zero'

    # plot_ra(data.counts_norm['treated1fb'], data.counts_norm['untreated1fb'], idxs=idxs)
    # pl.show()

    idxs = np.argsort(res.FDR)[:20]  # identifies the top 20 DE genes
    simvec, nsamples = compute_similarity_vector(model, inc=idxs, compare_genes=True)

    assert simvec is not None, 'failed to calculate similarity vector'
    assert nsamples is not None, 'failed to return number os samples'
    assert 190 == simvec.shape[0], 'incorrect number of similarity values'
    assert 0.01 > np.abs(simvec.max() - 1), 'max similarity outside tolerance'
    assert 0.5 == simvec.min(), 'incorrect minimum similarity value'
    assert 0.2 > np.abs(simvec.mean() - 0.5), 'mean similarity outside tolerance'
    assert 0.1 > np.abs(simvec.std() - 0.21), 'standard deviation from similarities outside tolerance'

    # pl.figure(figsize=(4, 6))
    # hr.dendrogram(hr.linkage(1 - simvec), labels=res.FDR[idxs].index, orientation='right')
    # pl.show()


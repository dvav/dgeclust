import pytest
import sys
import pandas as pd
import numpy as np
import shutil
from os.path import dirname, join


@pytest.fixture
def counts_filt():

    resources = 'resources/data.csv'

    filename = join(dirname(__file__), resources)

    counts = pd.read_csv(filename)
    counts = counts.set_index('gene_id')
    row_sums = counts.sum(1)
    idxs = row_sums > np.percentile(row_sums, 40)  # identifies the upper 60% of the data
    counts_filt = counts[idxs]

    return counts_filt


@pytest.fixture
def data(counts_filt):
    from dgeclust import CountData
    return CountData(counts_filt,
                     groups=['treated', 'treated', 'treated', 'untreated', 'untreated', 'untreated', 'untreated'])


@pytest.fixture
def model():
    from dgeclust.models import NBinomModel
    resources = 'resources/run002'
    run_dir = join(dirname(__file__), resources)
    return NBinomModel.load(run_dir)


@pytest.fixture
def tmp_file_map():

    if sys.version_info[0] > 2:
        import tempfile
    else:
        from backports import tempfile

    tmp_dir = str(tempfile.TemporaryDirectory().name)

    yield {'tmp_dir': tmp_dir,
           'pars.txt': join(tmp_dir, 'pars.txt'),
           'state.pkl': join(tmp_dir, 'state.pkl'),
           'z/0': join(tmp_dir, 'z/0'),
           }
    print('teardown tmp directory: {}'.format(tmp_dir))
    shutil.rmtree(tmp_dir)

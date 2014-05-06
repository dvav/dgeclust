from __future__ import division

import os
import sys
import multiprocessing as mp
import argparse as ap

import numpy as np

import utils as ut
import gibbs.state as gs
import gibbs.alg as alg
import models.nbinom as nbinom
import models.poisson as poisson
import config as cfg

########################################################################################################################

## parse command-line arguments (default values are from the config file)
parser = ap.ArgumentParser(prog='clust',
                           description='Hierarchical Non-Parametric Bayesian Clustering of Digital Expression Data')
parser.add_argument('data', type=str, help='data file to process')
parser.add_argument('-n', type=str, dest='norm', help='normalisation factors', default=None)
parser.add_argument('-g', type=str, dest='groups', help='grouping of samples', default=None)
parser.add_argument('-o', type=str, dest='outdir', help='output directory', default=cfg.fnames['clust'])
parser.add_argument('-t', type=int, dest='niters', help='number of iterations', default=cfg.clust['niters'])
parser.add_argument('-t0', type=int, dest='burnin',  help='burn-in period', default=cfg.clust['burnin'])
parser.add_argument('-dt', type=int, dest='nlog', help='save-state interval', default=cfg.clust['nlog'])
parser.add_argument('-k', type=int, dest='nglobal', help='truncation at level 0', default=cfg.clust['nglobal'])
parser.add_argument('-l', type=int, dest='nlocal', help='truncation at level 1', default=cfg.clust['nlocal'])
parser.add_argument('-r', type=int, dest='nthreads', help='number of threads', default=cfg.nthreads)
parser.add_argument('-e', dest='extend', help='extend simulation', action='store_true', default=cfg.clust['extend'])
parser.add_argument('-m', type=str, dest='model', help='model to use', default=cfg.model['distribution'],
                    choices=cfg.model['choices'])
parser.add_argument('-p', type=str, dest='pars', help='initial model parameters', default=str(cfg.model['pars']))

args = parser.parse_args()

model = {'NegBinom': nbinom, 'Poisson': poisson, 'Normal': None}[args.model]  # Normal and Poisson not implemented yet
norm = None if args.norm is None else eval(args.norm)
groups = None if args.groups is None else eval(args.groups)
pars = eval(args.pars)
nthreads = args.nthreads if args.nthreads > 0 else mp.cpu_count()

########################################################################################################################

## prepare output file names
fnames = {
    'theta': os.path.join(args.outdir, cfg.fnames['theta']),
    'lw': os.path.join(args.outdir, cfg.fnames['lw']),
    'lu': os.path.join(args.outdir, cfg.fnames['lu']),
    'c': os.path.join(args.outdir, cfg.fnames['c']),
    'z': os.path.join(args.outdir, cfg.fnames['z']),
    'pars': os.path.join(args.outdir, cfg.fnames['pars']),
    'eta': os.path.join(args.outdir, cfg.fnames['eta']),
    'zz': os.path.join(args.outdir, cfg.fnames['zz'])
}

########################################################################################################################

## load data
data = ut.read_count_data(args.data, norm, groups)


## generate initial state
if os.path.exists(args.outdir):
    if args.extend is False:
        raise Exception("Directory '{0}' already exists!".format(args.outdir))
    else:
        print >> sys.stderr, "Extending previous simulation...".format(args.outdir)
        state = gs.GibbsState.from_disk(fnames)
else:
    os.makedirs(fnames['zz'])
    state = gs.GibbsState.random(data.ngroups, data.nfeatures, model.sample_prior, pars, args.nglobal, args.nlocal)

    ## write feature and sample names on disk
    np.savetxt(os.path.join(args.outdir, cfg.fnames['featureNames']), data.feature_names, fmt='%s')
    np.savetxt(os.path.join(args.outdir, cfg.fnames['sampleNames']), data.sample_names, fmt='%s')

## use multiple cores
pool = mp.Pool(processes=nthreads)

## execute
alg.do_gibbs_sampling(data, model, state, pool, fnames, args.niters, args.burnin, args.nlog)

########################################################################################################################

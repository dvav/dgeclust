from __future__ import division

import os
import sys
import json
import multiprocessing as mp
import collections as cl
import argparse as ap
import numpy as np

import dgeclust.config as cfg

from dgeclust.data import CountData
from dgeclust.gibbs.state import GibbsState
from dgeclust.gibbs.alg import GibbsSampler
from dgeclust.models import nbinom

########################################################################################################################

## parse command-line arguments (default values are from the config file)
parser = ap.ArgumentParser(prog='clust',
                           description='Hierarchical Non-Parametric Bayesian Clustering of Digital Expression Data')
parser.add_argument('data', type=str, help='data file to process')
parser.add_argument('-n', type=str, dest='norm', help='normalisation method', default=cfg.norm['default'],
                    choices=cfg.norm['options'].keys())
parser.add_argument('-g', type=str, nargs='+', dest='groups', help='grouping of samples', default=None)
parser.add_argument('-s', type=str, nargs='+', dest='samples', help='samples to load', default=None)
parser.add_argument('-o', type=str, dest='outdir', help='output directory', default=cfg.fnames['clust'])
parser.add_argument('-t', type=int, dest='niters', help='number of iterations', default=cfg.clust['niters'])
parser.add_argument('-t0', type=int, dest='burnin',  help='burn-in period', default=cfg.clust['burnin'])
parser.add_argument('-dt', type=int, dest='nlog', help='save-state interval', default=cfg.clust['nlog'])
parser.add_argument('-k', type=int, dest='nclusters_max', help='maximum number of gene-wise clusters',
                    default=cfg.clust['nclusters_max'])
parser.add_argument('-r', type=int, dest='nthreads', help='number of threads', default=cfg.nthreads)
parser.add_argument('-e', dest='extend', help='extend simulation', action='store_true', default=cfg.clust['extend'])
parser.add_argument('-m', type=str, dest='model', help='model to use', default=cfg.models['default'],
                    choices=cfg.models['options'].keys())
parser.add_argument('-p', type=float, nargs='+', dest='hpars', help='initial hyper-parameter values', default=None)

args = parser.parse_args()

args.hpars = np.asarray(cfg.models['options'][args.model]['hpars'].values() if args.hpars is None else args.hpars)
args.nthreads = args.nthreads if args.nthreads > 0 else mp.cpu_count()

########################################################################################################################

## prepare output file names
args.fnames = {
    'state': os.path.join(args.outdir, cfg.fnames['state']),
    'pars': os.path.join(args.outdir, cfg.fnames['pars']),
    'cc': os.path.join(args.outdir, cfg.fnames['cc']),
}

########################################################################################################################

## load data
data = CountData.load(args.data, args.norm, args.groups, args.samples)

counts = [data.counts[samples].values for samples in data.groups.values()]
lib_sizes = [data.lib_sizes[samples].values.ravel() for samples in data.groups.values()]
nreplicas = data.nreplicas.values()

## prepare model
model = {
    'NegativeBinomial': nbinom
}[args.model]

## generate initial sampler state
if os.path.exists(args.outdir):
    if args.extend is False:
        raise Exception("Directory '{0}' already exists!".format(args.outdir))
    else:
        print >> sys.stderr, "Extending previous simulation...".format(args.outdir)
        state = GibbsState.load(args.fnames['state'])
else:
    os.makedirs(args.fnames['cc'])
    state = GibbsState.random(len(data.counts), len(data.groups), model.sample_pars_prior,
                              args.hpars, args.nclusters_max)

    ## save groups, feature and sample names
    with open(os.path.join(args.outdir, cfg.fnames['config']), 'w') as f:
        json.dump(cl.OrderedDict([
            ("data", args.data),
            ("norm", args.norm),
            ("groups", data.groups),
            ("nclusters_max", args.nclusters_max),
            ("model", args.model),
            ("pars", cfg.models['options'][args.model]['pars']),
            ("hpars", cl.OrderedDict(zip(cfg.models['options'][args.model]['hpars'].keys(), args.hpars))),
            ("featureNames", data.counts.index.tolist())
        ]), f, indent=4, separators=(',', ':'))

## use multiple cores
pool = mp.Pool(processes=args.nthreads)

## execute
GibbsSampler((counts, lib_sizes, nreplicas), model, state, args.niters, args.burnin, args.nlog, args.fnames, pool).run()

########################################################################################################################

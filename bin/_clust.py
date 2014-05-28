from __future__ import division

import os
import sys
import json
import multiprocessing as mp
import collections as cl
import argparse as ap

import dgeclust.config as cfg

from dgeclust.data import CountData
from dgeclust.gibbs.state import GibbsState
from dgeclust.gibbs.alg import GibbsSampler
from dgeclust.models import nbinom, poisson, bbinom, binom, normal

########################################################################################################################

## parse command-line arguments (default values are from the config file)
parser = ap.ArgumentParser(prog='clust',
                           description='Hierarchical Non-Parametric Bayesian Clustering of Digital Expression Data')
parser.add_argument('data', type=str, help='data file to process')
parser.add_argument('-n', type=str, dest='norm', help='normalisation method', default=cfg.norm['default'],
                    choices=cfg.norm['options'].keys())
parser.add_argument('-g', type=str, nargs='+', dest='groups', help='grouping of samples', default=None)
parser.add_argument('-o', type=str, dest='outdir', help='output directory', default=cfg.fnames['clust'])
parser.add_argument('-t', type=int, dest='niters', help='number of iterations', default=cfg.clust['niters'])
parser.add_argument('-t0', type=int, dest='burnin',  help='burn-in period', default=cfg.clust['burnin'])
parser.add_argument('-dt', type=int, dest='nlog', help='save-state interval', default=cfg.clust['nlog'])
parser.add_argument('-k', type=int, dest='nglobal', help='truncation at level 0', default=cfg.clust['nglobal'])
parser.add_argument('-l', type=int, dest='nlocal', help='truncation at level 1', default=cfg.clust['nlocal'])
parser.add_argument('-r', type=int, dest='nthreads', help='number of threads', default=cfg.nthreads)
parser.add_argument('-e', dest='extend', help='extend simulation', action='store_true', default=cfg.clust['extend'])
parser.add_argument('-m', type=str, dest='model', help='model to use', default=cfg.models['default'],
                    choices=cfg.models['options'].keys())
parser.add_argument('-p', type=float, nargs='+', dest='hpars', help='initial hyper-parameter values', default=None)

args = parser.parse_args()

args.hpars = cfg.models['options'][args.model]['hpars'].values() if args.hpars is None else args.hpars
args.nthreads = args.nthreads if args.nthreads > 0 else mp.cpu_count()

########################################################################################################################

## prepare output file names
args.fnames = {
    'pars': os.path.join(args.outdir, cfg.fnames['pars']),
    'lw': os.path.join(args.outdir, cfg.fnames['lw']),
    'lu': os.path.join(args.outdir, cfg.fnames['lu']),
    'c': os.path.join(args.outdir, cfg.fnames['c']),
    'z': os.path.join(args.outdir, cfg.fnames['z']),
    'hpars': os.path.join(args.outdir, cfg.fnames['hpars']),
    'eta': os.path.join(args.outdir, cfg.fnames['eta']),
    'nact': os.path.join(args.outdir, cfg.fnames['nact']),
    'zz': os.path.join(args.outdir, cfg.fnames['zz'])
}

########################################################################################################################

## load data
data = CountData.load(args.data, args.norm, args.groups)

## prepare model
model = {
    'NegativeBinomial': nbinom,
    'Poisson': poisson,
    'BetaBinomial': bbinom,
    'Binomial': binom,
    'Normal': normal
}[args.model]

## generate initial sampler state
if os.path.exists(args.outdir):
    if args.extend is False:
        raise Exception("Directory '{0}' already exists!".format(args.outdir))
    else:
        print >> sys.stderr, "Extending previous simulation...".format(args.outdir)
        state = GibbsState.load(args.fnames)
else:
    os.makedirs(args.fnames['zz'])
    state = GibbsState.random(len(data.groups), len(data.counts), model.sample_prior, args.hpars,
                              args.nglobal, args.nlocal)

    ## write groups, feature and sample names on disk
    with open(os.path.join(args.outdir, cfg.fnames['config']), 'w') as f:
        json.dump(cl.OrderedDict([
            ("data", args.data),
            ("norm", args.norm),
            ("groups", data.groups),
            ("nglobal", args.nglobal),
            ("nlocal", args.nlocal),
            ("model", args.model),
            ("pars", cfg.models['options'][args.model]['pars']),
            ("hpars", cl.OrderedDict(zip(cfg.models['options'][args.model]['hpars'].keys(), args.hpars))),
            ("sampleNames", data.counts.columns.tolist()),
            ("featureNames", data.counts.index.tolist())
        ]), f, indent=4, separators=(',', ':'))

## use multiple cores
pool = mp.Pool(processes=args.nthreads)

## execute
GibbsSampler(data, model, state, args.niters, args.burnin, args.nlog, args.fnames, pool).run()

########################################################################################################################

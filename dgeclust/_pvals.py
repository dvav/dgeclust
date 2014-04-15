from __future__ import division

import os
import sys
import multiprocessing as mp
import argparse as ap

import gibbs.post as post
import config as cfg

################################################################################

## parse command-line arguments (default values are from the config file)
parser = ap.ArgumentParser(prog='pvals', description='Compute posterior probabilities')
parser.add_argument('-i', type=str, dest='indir', help='input directory', default=cfg.fnames['clust'])
parser.add_argument('-t0', type=int, dest='t0', help='first sample to process', default=cfg.post['t0'])
parser.add_argument('-tend', type=int, dest='tend', help='last sample to process', default=cfg.post['tend'])
parser.add_argument('-dt', type=int, dest='dt', help='process every dt samples', default=cfg.post['dt'])
parser.add_argument('-g1', type=int, dest='group1', help='first group of samples', default=cfg.post['group1'])
parser.add_argument('-g2', type=int, dest='group2', help='second group of samples', default=cfg.post['group2'])
parser.add_argument('-o', type=str, dest='outfile', help='output file', default=cfg.fnames['pvals'])
parser.add_argument('-r', type=int, dest='nthreads', help='number of threads', default=cfg.nthreads)

args = parser.parse_args()

################################################################################

## use multiple cores
nthreads = args.nthreads if args.nthreads > 0 else mp.cpu_count()
pool = mp.Pool(processes=nthreads)

## compute p values
indir = os.path.join(args.indir, cfg.fnames['zz'])   # input directory
fname = os.path.join(args.indir, cfg.fnames['featureNames'])   # feature names
pvals, nsamples = post.compute_pvals(indir, fname, args.t0, args.tend, args.dt, args.group1, args.group2, pool)
print >> sys.stderr, '{0} samples processed from directory "{1}"'.format(nsamples, indir)

## save pvals to output file
pvals.to_csv(args.outfile, sep='\t')

################################################################################

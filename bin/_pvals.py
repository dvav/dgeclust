from __future__ import division

import os
import sys
import multiprocessing as mp
import argparse as ap

import dgeclust.gibbs.post as post
import dgeclust.config as cfg

########################################################################################################################

## parse command-line arguments (default values are from the config file)
parser = ap.ArgumentParser(prog='pvals', description='Compute posterior probabilities')
parser.add_argument('group1', type=str, help='first group of samples')
parser.add_argument('group2', type=str, help='second group of samples')
parser.add_argument('-i', type=str, dest='indir', help='input directory', default=cfg.fnames['clust'])
parser.add_argument('-t0', type=int, dest='t0', help='first sample to process', default=cfg.post['t0'])
parser.add_argument('-tend', type=int, dest='tend', help='last sample to process', default=cfg.post['tend'])
parser.add_argument('-dt', type=int, dest='dt', help='process every dt samples', default=cfg.post['dt'])
parser.add_argument('-o', type=str, dest='outfile', help='output file', default=cfg.fnames['pvals'])
parser.add_argument('-r', type=int, dest='nthreads', help='number of threads', default=cfg.nthreads)

args = parser.parse_args()

args.nthreads = args.nthreads if args.nthreads > 0 else mp.cpu_count()
args.cc = os.path.join(args.indir, cfg.fnames['cc'])       # input directory
args.config = os.path.join(args.indir, cfg.fnames['config'])   # includes config info

########################################################################################################################

## use multiple cores
pool = mp.Pool(processes=args.nthreads)

## compute p values
pvals, nsamples = post.compute_pvals(args.cc, args.config, args.t0, args.tend, args.dt,
                                     args.group1, args.group2, pool)
print >> sys.stderr, '{0} samples processed from directory "{1}"'.format(nsamples, args.cc)


## save pvals to output file
pvals.to_csv(args.outfile, sep='\t')

########################################################################################################################

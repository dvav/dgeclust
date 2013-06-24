'''
Created on Apr 19, 2013

@author: dimitris
'''

################################################################################

#!/usr/bin/env python

################################################################################

import negbinom
import poisson
import gaussian
import gibbs
import HDP
import utils as ut
import numpy as np

from argparse import ArgumentParser

################################################################################

## parse command-line arguments
parser = ArgumentParser(description='Cluster digital gene expression data')    
parser.add_argument('data', type=str, help='data file to process')

parser.add_argument('-O',  metavar='OUTDIR',     type=str,   help='output directory',                       default='./DGEclust_output/')    
parser.add_argument('-T',  metavar='NITERS',     type=int,   help='number of iterations',                   default=10000)    
parser.add_argument('-DT', metavar='NLOG',       type=int,   help='save-to-disk interval',                  default=100)    
parser.add_argument('-P',  metavar='PARS',       type=float, help='initial model parameters',               default=[-10., 100., 1., 0.1], nargs='+')
parser.add_argument('-TP', metavar='TPARS',      type=int,   help='sample interval for model parameters',   default=1)
parser.add_argument('-K0', metavar='TRUNC0',     type=int,   help='truncation at level 0',                  default=100)
parser.add_argument('-K',  metavar='TRUNC1',     type=int,   help='truncation at level 1',                  default=100)
parser.add_argument('-M',  metavar='MODEL',      type=str,   help='model to use',                           default='NegBinomial', choices=['NegBinomial','Poisson','Gaussian'])
parser.add_argument('-R',  metavar='NTHREADS',   type=int,   help='number of threads',                      default=1)

parser.add_argument('-E',  dest='E', help='extend previous simulation', action='store_true', default=False)
            
args = parser.parse_args()

dataFile    = args.data    
outDir      = args.O
niters      = args.T
dt          = args.DT
pars        = args.P
tpars       = args.TP
K0          = args.K0
K           = args.K
model       = {'NegBinomial':negbinom, 'Poisson':poisson, 'Gaussian':gaussian}[args.M]
nthreads    = args.R
extend      = args.E 

################################################################################

## load data and prepare output
data = ut.Data(dataFile)
rec  = ut.Recorder(outDir, dt, extend)   

################################################################################

## prepare HDP object
if extend is True:        
    X0   = np.loadtxt(rec.fX0)
    lw0  = np.loadtxt(rec.flw0)                        
    LW   = np.loadtxt(rec.fLW)
    C    = np.loadtxt(rec.fC, dtype='int')
    Z    = np.loadtxt(rec.fZ, dtype='int')            
    eta0 = np.loadtxt(rec.feta)[-1,1]
    eta  = np.loadtxt(rec.feta)[-1,2:]
    pars = np.loadtxt(rec.fpars)[-1,2:]        
else:
    M, N = data.counts.shape
    
    X0   = model.rPrior(K0, *pars)
    lw0  = np.tile(-np.log(K0), K0)
    LW   = np.tile(-np.log(K), (M,K))    
    C    = np.random.randint(0, K0, (M,K))   #[ np.zeros(K, dtype = 'int') for i in range(M) ]
    Z    = np.random.randint(0, K,  (M,N))   #[ np.zeros(N, dtype = 'int') for i in range(M) ]
    eta0 = 1.
    eta  = np.ones(M)
    pars = pars

hdp  = HDP.HDP(X0, lw0, LW, C, Z, eta0, eta, pars)

## execute
gibbs.loop(niters, tpars, data, model, hdp, rec, nthreads)
    

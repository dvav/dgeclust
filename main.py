## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import DGEclust         as cl
import DGEclust.models  as mdl
import numpy            as np   
import argparse         as ap

################################################################################

## parse command-line arguments
def parseArgs():
    parser = ap.ArgumentParser(prog='dgeclust', description='Non-parametric Bayesian clustering of digital gene expression data')    
    parser.add_argument('data', type=str, help='data file to process')

    parser.add_argument('-O',  metavar='OUTDIR',     type=str,   help='output directory',                       default='./DGEclust_output/')    
    parser.add_argument('-T',  metavar='NITERS',     type=int,   help='number of iterations',                   default=100000)    
    parser.add_argument('-DT', metavar='NLOG',       type=int,   help='save-to-disk interval',                  default=10)    
    parser.add_argument('-P',  metavar='PARS',       type=float, help='initial model parameters',               default=[-7., 10., 10., 0.1], nargs='+')
    parser.add_argument('-K0', metavar='TRUNC0',     type=int,   help='truncation at level 0',                  default=100)
    parser.add_argument('-K',  metavar='TRUNC1',     type=int,   help='truncation at level 1',                  default=100)
    parser.add_argument('-M',  metavar='MODEL',      type=str,   help='model to use',                           default='NegBinom', choices=['NegBinom','Poisson','Gaussian'])
    parser.add_argument('-R',  metavar='NTHREADS',   type=int,   help='number of threads',                      default=1)
    
    parser.add_argument('-E', dest='E', help='extend previous simulation', action='store_true', default=False)
    parser.add_argument('-U', dest='U', help='update parameters', action='store_true',default=True)
            
    return parser.parse_args()
    

################################################################################

if __name__ == '__main__':
    ## parse command-line arguments
    args = parseArgs()        
    
    dataFile    = args.data    
    outDir      = args.O
    niters      = args.T
    dt          = args.DT
    pars        = args.P
    K0          = args.K0
    K           = args.K
    model       = {'NegBinom':mdl.NegBinom, 'Poisson':mdl.Poisson, 'Gaussian':mdl.Gaussian}[args.M]
    nthreads    = args.R
    extend      = args.E 
    update      = args.U

    ################################################################################

    ## load data and prepare output
    data = cl.readCountData(dataFile)
    mtr  = cl.Monitor(outDir, dt, extend)   

    ################################################################################

    ## prepare HDP object
    if extend is True:        
        X0   = np.loadtxt(mtr.fX0)
        lw0  = np.loadtxt(mtr.flw0)                        
        LW   = np.loadtxt(mtr.fLW)
        C    = np.loadtxt(mtr.fC, dtype='int')
        Z    = np.loadtxt(mtr.fZ, dtype='int')            
        eta0 = np.loadtxt(mtr.feta)[-1,1]
        eta  = np.loadtxt(mtr.feta)[-1,2:]
        pars = np.loadtxt(mtr.fpars)[-1,2:]        
    else:
        N, M = data.values.shape
    
        X0   = model.rPrior(K0, *pars)
        lw0  = np.tile(-np.log(K0), K0)
        LW   = np.tile(-np.log(K), (M,K))    
        C    = np.random.randint(0, K0, (M,K))   #[ np.zeros(K, dtype = 'int') for i in range(M) ]
        Z    = np.random.randint(0, K,  (M,N))   #[ np.zeros(N, dtype = 'int') for i in range(M) ]
        eta0 = 1.
        eta  = np.ones(M)

    hdp  = cl.HDP(X0, lw0, LW, C, Z, eta0, eta, pars)

    ## execute
    sampler = cl.GibbsSampler(nthreads)
    sampler.loop(niters, update, data, model, hdp, mtr)
    

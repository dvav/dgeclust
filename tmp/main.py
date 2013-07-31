## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

import DGEclust         as cl
import DGEclust.models  as mdl
import argparse         as ap

################################################################################

## parse command-line arguments
def parseArgs():
    parser = ap.ArgumentParser(prog='dgeclust', description='Non-parametric Bayesian clustering of digital gene expression data')    
    parser.add_argument('data', type=str, help='data file to process')

    parser.add_argument('-O',  metavar='OUTDIR',     type=str,   help='output directory',                       default='DGEclust_output/')    
    parser.add_argument('-T',  metavar='NITERS',     type=int,   help='number of iterations',                   default=100000)    
    parser.add_argument('-DT', metavar='NLOG',       type=int,   help='save-to-disk interval',                  default=10)    
    parser.add_argument('-P',  metavar='PARS',       type=float, help='initial model parameters',               default=[100., 1., 10.], nargs='+')
    parser.add_argument('-K0', metavar='TRUNC0',     type=int,   help='truncation at level 0',                  default=100)
    parser.add_argument('-K',  metavar='TRUNC1',     type=int,   help='truncation at level 1',                  default=100)
    parser.add_argument('-M',  metavar='MODEL',      type=str,   help='model to use',                           default='NegBinom', choices=['NegBinom','Poisson','Gaussian'])
    parser.add_argument('-R',  metavar='NTHREADS',   type=int,   help='number of threads',                      default=1)
    
    parser.add_argument('-E', dest='E', help='extend previous simulation', action='store_true', default=False)
    parser.add_argument('-U', dest='U', help='update parameters', action='store_true',default=False)
            
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
    
    countData = cl.CountData(dataFile)

    ################################################################################

    

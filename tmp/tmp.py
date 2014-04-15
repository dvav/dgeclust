import numpy           as np
import matplotlib.pyplot        as pl
import dgeclust        as cl
import dgeclust.nbinom as nbinom
import dgeclust.utils  as utils 
import dgeclust.gibbs  as gibbs


data = utils.read_count_data('/Users/dimitris/Desktop/benchmarks/data/macaque/macaque_counts_table.txt',groups=[range(5),range(5,10),range(10,15),range(15,20),range(20,25)])
res  = gibbs.read_results('results/macaque/'); res.theta[:,1] = 1/res.theta[:,0] / (1/res.theta[:,0]+res.theta[:,1]); 

## plot parameters
pl.rc('font',  family='Arial', size=8)
pl.rc('lines', linewidth=0.5)
golden_ratio = 1.61803398875
NROWS, NCOLS = 2, 3
fig_width    = 3.34645669 * 2           # in inches
fig_height   = fig_width / golden_ratio # in inches

pl.figure(figsize=(fig_width, fig_height))

pl.subplot(NROWS,NCOLS,1);
isample = 0; igroup = 0; 
utils.plot_fitted_model(isample, igroup, res, data, nbinom);
pl.xlabel('log(# of counts)',size=10); pl.ylabel('density',size=10)
pl.title('Caudate Nucleus',size=12,weight='bold')

pl.subplot(NROWS,NCOLS,2);
isample = 5; igroup = 1; 
utils.plot_fitted_model(isample, igroup, res, data, nbinom);
pl.xlabel('log(# of counts)',size=10); pl.ylabel('density',size=10)
pl.title('Frontal Gyrus',size=12,weight='bold')

pl.subplot(NROWS,NCOLS,3);
isample = 10; igroup = 2; 
utils.plot_fitted_model(isample, igroup, res, data, nbinom);
pl.xlabel('log(# of counts)',size=10); pl.ylabel('density',size=10)
pl.title('Hippocampus',size=12,weight='bold')

pl.subplot(NROWS,NCOLS,4);
isample = 15; igroup = 3; 
utils.plot_fitted_model(isample, igroup, res, data, nbinom);
pl.xlabel('log(# of counts)',size=10); pl.ylabel('density',size=10)
pl.title('Putamen',size=12,weight='bold')

pl.subplot(NROWS,NCOLS,5);
isample = 20; igroup = 4; 
utils.plot_fitted_model(isample, igroup, res, data, nbinom);
pl.xlabel('log(# of counts)',size=10); pl.ylabel('density',size=10)
pl.title('Temporal Gyrus',size=12,weight='bold')

pl.tight_layout()
pl.savefig('tmp.pdf')


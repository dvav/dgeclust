## Copyright (C) 2012-2013 Dimitrios V. Vavoulis
## Computational Genomics Group (http://bioinformatics.bris.ac.uk/)
## Department of Computer Science
## University of Bristol

################################################################################

from plotRA         import plotRA
from plotSample     import plotSample
from plotModel      import plotModel
from plotHeatmap    import plotHeatmap
from plotChains     import plotChains
from plotBaseDists  import plotBaseDists



# ## sampling interval
# tini = 0
# tend = 100000
# di   = 10
# 
# ii = (t >= tini) & (t <= tend) & (t % di == 0) 
# 
# ## plot diagnostics
# figure(figsize=(20,20)); 
# 
# ## Geweke's test
# subplot(5,2,1); scores = asarray(mc.geweke(N[ii]));  plot(scores[:,0],scores[:,1],'o'); hlines([-2,0,2], scores[0,0],scores[-1,0], color='k', linestyle='--'); ylim([-2.5, 2.5]); xlim([scores[0,0],scores[-1,0]]); xlabel('First iteration'); ylabel('Z-score for clusters #'); 
# subplot(5,2,3); scores = asarray(mc.geweke(mu[ii])); plot(scores[:,0],scores[:,1],'o'); hlines([-2,0,2], scores[0,0],scores[-1,0], color='k', linestyle='--'); ylim([-2.5, 2.5]); xlim([scores[0,0],scores[-1,0]]); xlabel('First iteration'); ylabel('Z-score for mean'); 
# subplot(5,2,5); scores = asarray(mc.geweke(s2[ii])); plot(scores[:,0],scores[:,1],'o'); hlines([-2,0,2], scores[0,0],scores[-1,0], color='k', linestyle='--'); ylim([-2.5, 2.5]); xlim([scores[0,0],scores[-1,0]]); xlabel('First iteration'); ylabel('Z-score for variance'); 
# subplot(5,2,7); scores = asarray(mc.geweke(sh[ii])); plot(scores[:,0],scores[:,1],'o'); hlines([-2,0,2], scores[0,0],scores[-1,0], color='k', linestyle='--'); ylim([-2.5, 2.5]); xlim([scores[0,0],scores[-1,0]]); xlabel('First iteration'); ylabel('Z-score for shape'); 
# subplot(5,2,9); scores = asarray(mc.geweke(sc[ii])); plot(scores[:,0],scores[:,1],'o'); hlines([-2,0,2], scores[0,0],scores[-1,0], color='k', linestyle='--'); ylim([-2.5, 2.5]); xlim([scores[0,0],scores[-1,0]]); xlabel('First iteration'); ylabel('Z-score for scale'); 
# 
# ## auto-correlation
# subplot(5,2,2);  acorr(N[ii],   normed=True, detrend=mlab.detrend_mean, usevlines=True, maxlags=100); xlabel('lag'); ylabel('autocorrelation for clusters #'); 
# subplot(5,2,4);  acorr(mu[ii],  normed=True, detrend=mlab.detrend_mean, usevlines=True, maxlags=100); xlabel('lag'); ylabel('autocorrelation for mean');
# subplot(5,2,6);  acorr(s2[ii],  normed=True, detrend=mlab.detrend_mean, usevlines=True, maxlags=100); xlabel('lag'); ylabel('autocorrelation for variance');
# subplot(5,2,8);  acorr(sh[ii],  normed=True, detrend=mlab.detrend_mean, usevlines=True, maxlags=100); xlabel('lag'); ylabel('autocorrelation for shape');
# subplot(5,2,10); acorr(sc[ii],  normed=True, detrend=mlab.detrend_mean, usevlines=True, maxlags=100); xlabel('lag'); ylabel('autocorrelation for scale');
# 
# 

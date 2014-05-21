---
title: Tutorial
layout: docs
---

Clustering the data
===================

Next, we need to cluster the data. For this purpose, we shall use the program `clust`, which comes with **DGEclust**. 
From your terminal, do the following:

{% highlight bash %}
$ cd path/to/dgeclust
$ bin/clust path/to/data_filt.txt -g [[0,1,2,3],[4,5]] &
{% endhighlight %}

The above command runs a Gibbs sampler for a default of 10K iterations. The argument `-g [[0,1,2,3],[4,5]]`
informs the simulator that the first four samples and the last two (remember that indexing starts at zero in Python)
form two different groups (i.e. cancerous and non-cancerous). Depending on the size of your data, this will take 
some time to finish. The output from the sampler is saved periodically in the directory `_clust`, which you can 
inspect to check the progress of your simulation, e.g. using `tail`:

{% highlight bash %}
$ tail -f _clust/pars.txt
0    -1.234089       0.400631        4.798040        6.434471
1    -1.063557       0.719506        6.253583        9.187044
2    -1.134126       0.372169        4.724750        3.580500
3    -1.241222       0.475262        5.225677        3.228103
...
{% endhighlight %}

The first column in `pars.txt` gives you the number of iterations, while the remaining 
columns correspond to hyper-parameters, which depend on the specific distribution being used
by the simulator to model count data (defaults to the **Negative Binomial** distribution).

There are more arguments that you can pass to `clust`. Type `bin/clust -h` for more details.

After the end of the simulation, you can visualize your results using **IPython**:

{% highlight python %}
cd path/to/dgeclust
from dgeclust.gibbs.results import GibbsOutput
res = GibbsOutput.read('_clust')
figure()
subplot(2,2,1); plot(res.t, res.nactive0); xlabel('# of iterations'); ylabel('# of clusters')
subplot(2,2,2); hist(res.nactive0, 100, range=(0,100), normed=True); xlabel('# of clusters'); ylabel('frequency')
subplot(2,2,3); plot(res.t, res.pars[:,[0,1]]); xlabel('# iterations'); ylabel('p1, p2')
subplot(2,2,4); plot(res.t, res.pars[:,[3,2]]); xlabel('# iterations'); ylabel('p3, p4')
{% endhighlight %}

<img class="img-responsive" alt="Simulation progress" title="Simulation progress" src="{{ site.baseurl }}/img/progress.png"></img>

It seems that the algorithm converges rapidly after ~1000 iterations. From the histogram on the top right, we can see that the data
support between 20 and 23 clusters with a peak at 21. If you need to extend the simulation for another 10K iterations (i.e. a total
of 20K iterations), you
can type:

{% highlight bash %}
bin/clust /path/to/data_filt.txt -g [[0,1,2,3],[4,5]] -t 20000 -e & 
{% endhighlight %}
 
The argument `-e` indicates that a previously terminated simulation should
be extended and the argument `-t` indicates the total duration of the simulation. 

If you wish, you can see how the fitted model at the end of the simulation compares
to the actual data:

{% highlight python %}
from dgeclust import utils
from dgeclust.data import CountData
from dgeclust.models import nbinom
data = CountData.load('path/to/data_filt.txt', groups=[[0,1,2,3],[4,5]])    
counts_norm = data.counts / data.norm_factors
isample = 0  # the index of the sample you want to visualise (between 0 and 5)
igroup = 0   # the group the above sample belongs to (0: cancerous, 1: non-cancerous)
x, y = utils.compute_fitted_model(igroup, res, nbinom); 
figure()
hist(log(counts_norm[:,isample]), 100, histtype='stepfilled', linewidth=0, normed=True, color='gray')
plot(x, y, 'k', x, y.sum(1), 'r');
xlabel('log counts'); ylabel('frequency')
{% endhighlight %}

<img class="img-responsive" alt="Fitted model" title="Fitted model" src="{{ site.baseurl }}/img/fitted.png"></img>

Of course, you can repeat the above for all possible values of `isample` and corresponding values of `igroup`.
The next step is to post-process your data in order to test for differential expression. 
<a href="{{ site.baseurl }}{{ site.data.nav.docs.tut.detest.url }}">Learn how!</a>


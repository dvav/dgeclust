---
title: Tutorial
layout: default
---

Quick start
===========

The data
--------

For the sake of this tutorial, let's assume that the downloaded **DGEclust** files 
are located in a directory named `dgeclust`. Also, let's assume that you 
have a matrix of count data stored in the file `data.txt`. In this tutorial, we 
shall use a Tag-Seq dataset derived from tissue cultures of neural stem cells
obtained by <a href="http://link.springer.com/article/10.1186%2Fgm377" target="_blank">Engström et al.</a> 
The data can be downloaded from [here](http://genomebiology.com/content/supplementary/gb-2010-11-10-r106-s3.tgz).

You can inspect the data using the `head` command at the terminal:
{% highlight bash %}
$ head /path/to/data.txt
        GliNS1  G144    G166    G179    CB541   CB660
13CDNA73        4       0       6       1       0       5
15E1.2  75      74      222     458     215     167
182-FIP 118     127     555     231     334     114
2'-PDE  39      38      98      127     34      40
3'HEXO  18      20      76      111     121     112
3.8-1   0       0       1       0       0       0
384D8-2 3       3       4       4       3       3
76P     61      51      129     108     358     232
7h3     4       0       3       0       9       2
{% endhighlight %}

The first four columns correspond to cultures from glioblastoma-derived neural stem cells and the 
remaining two to cultures from non-cancerous neural stem cells.

Let's use **IPython** to filter the data. Execute `ipython --pylab` at the terminal and, at the subsequent
**IPython** command prompt, type the following:

{% highlight python %}
import pandas as pd
data = pd.read_table('/path/to/data.txt', index_col=0)
idxs = all(data, 1)    # identify the rows where all entries are non-zero 
data_filt = data[idxs]
data_filt.head()    # inspect the data
data_filt.to_csv('data_filt.txt', sep='\t')
{% endhighlight %}

The filtered dataset contains 12221 features.


Clustering the data
-------------------

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

![Simulation progress]({{ site.baseurl }}/img/progress.png "Simulation progress")

It seems that the algorithm converges rapidly after ~1000 iterations. From the histogram on the top right, we can see that the data
support between 20 and 23 clusters with a peak at 21. If you need the extend the simulation for another 10K iterations (i.e. a total
of 20K iterations), you
can type:

{% highlight bash %}
bin/clust /path/to/data_filt.txt -g [[0,1,2,3],[4,5]] -t 20000 -e & 
{% endhighlight %}
 
The argument `-e` indicates that a previously terminated simulation should
be extended and the argument `-t` indicates the total duration of the simulation. 

If you wish, we can see how the fitted model at the end of the simulation compares
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

![Fitted model]({{ site.baseurl }}/img/fitted.png "Fitted model")

Of course, you can repeat the above for all possible values of `isample` and corresponding values of `igroup`.


Testing for differential expression
-----------------------------------

The output of `clust` can be used to test for differential expression. For this purpose, 
we use another program that comes with **DGEclust**, `pvals`, which computes the 
posterior probability that any two features between any two groups of samples in 
our dataset are differentially expressed. In a terminal, we type: 

{% highlight bash %}
$ bin/pvals -t0 1000
9001 samples processed from directory "clust/zz"
{% endhighlight %}

The above command post-processes the contents of the directory `_clust` and returns 
a file, `_pvals.txt`, which includes a list of features with decreasing posterior 
probabilities of being differentially expressed between the cancerous and non-cancerous 
samples in the data:

{% highlight bash %}
$ head _pvals.txt
        	Posteriors      FDR
SLC4A4  	0.0     		0.0
MYL9    	0.0     		0.0
IGF2    	0.0     		0.0
CD74    	0.0     		0.0
MT2A    	0.0     		0.0
TYRO3   	0.0     		0.0
HMGA2   	0.0     		0.0
RGAG4   	0.0     		0.0
C22orf16	0.0     		0.0
{% endhighlight %}

Notice that for the top-most features in this particular example, the probability is so low, that it is reported 
as being 0. 

Back in **IPython**, we can visualise our results using an **RA diagram**:

{% highlight python %}
pvals = pd.read_table('_pvals.txt',index_col=0)
idxs = (pvals.FDR < 0.01).values    # Use a False Discovery Rate of 1%
in1 = mean(counts_norm[:,[0,1,2,3]],1)
in2 = mean(counts_norm[:,[4,5]],1)
R, A = utils.compute_ra_plot(in1, in2)
figure()
plot(A[~idxs], R[~idxs], 'k.', markersize=1.8)
plot(A[idxs], R[idxs], 'r.', markersize=1.8)
xlabel('( log2(in2) + log2(in1) ) * 0.5')
ylabel('log2(in2) - log2(in1)')
{% endhighlight %}

![RA plot]({{ site.baseurl }}/img/RA_plot.png "RA plot")

<!-- Hierarchical clustering of samples
----------------------------------
(*under construction*)



Hierarchical clustering of genes
--------------------------------
(*under construction*)
 -->



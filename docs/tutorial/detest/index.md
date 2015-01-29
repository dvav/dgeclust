---
title: Tutorial
layout: docs
---

Testing for differential expression
===================================


**DGEclust** treats differential expression as a particular clustering configuration of the data. 
In particular, if the LFCs between two different treatments for a particular gene belong to two 
different clusters, then this gene is differentially expressed between these two treatments. 
**DGEclust** provides a `compare_groups` function for calculating the posterior probability
that the LFCs between any two treatments for a particular gene belong to the same cluster, i.e.
the posterior probability that the gene *is not* differentially expressed. Having processed the 
data as explained in the previous sections, this can be achieved as follows: 

{% highlight python linenos %}
from dgeclust import compare_groups
res, nsamples = compare_groups(data, mdl, group1='treated', group2='untreated')
{% endhighlight %}

The above statement checks each gene for differential expression between the `treated` and `untreated` 
groups of samples by post-processing the output of the Gibbs sampler (encapsulated by the `mdl` object), 
using a default burn-in period of 5K iterations. The burn-in period can be modified using the `t0` argument,
while the last sample to be processed is identified by the `tend` argument (default value: 10000). Every 
n-th sample can be post-processed by modifying the `dt` argument (default value: 1), while the number of cores
used is controlled by the `nthreads` argument (default: all available cores are used). The method returns a 
list raw and adjusted (FDR) posterior probabilities of no differential expression for each gene (`res`) and 
the number of samples that were processed (`nsamples`):

{% highlight python linenos %}
res.head()

               Posteriors   FDR
FBgn0000008    0.997001     0.834967
FBgn0000017    0.934213     0.580633
FBgn0000018    0.996401     0.830695
FBgn0000024    0.936813     0.586494
FBgn0000032    1.000000     0.880004
FBgn0000037    0.840232     0.421552
FBgn0000042    0.967606     0.669613
FBgn0000043    0.130774     0.043212
FBgn0000045    0.870626     0.455952
FBgn0000046    0.885023     0.482976
...
{% endhighlight %}

We can visualise our results using an **RA diagram**:

{% highlight python linenos %}
from dgelust.utils import plot_ra

idxs = res.FDR < 0.01    # identify DE genes at 1% FDR
plot_ra(data.counts_norm['treated1fb'], data.counts_norm['untreated1fb'], idxs=idxs)
{% endhighlight %}

<img class="img-responsive" alt="RA plot" title="RA plot" src="{{ site.baseurl }}/img/RA_plot.png"></img>

Notice that the `data` object contains a copy of the normalised data as one of its attributes.

The output of the Gibbs sampler can be further used as input to hierarchical clustering algorithms.
<a href="{{ site.baseurl }}{{ site.data.nav.docs.tut.clust.url }}">Learn how!</a>  


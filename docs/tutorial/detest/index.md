---
title: Tutorial
layout: docs
---

Testing for differential expression
===================================


The output of `clust` can be used to test for differential expression. For this purpose, 
we use another program that comes with **DGEclust**, `pvals`, which computes the 
posterior probability that any two features between any two groups of samples in 
our dataset are differentially expressed. In a terminal, we type: 

{% highlight bash linenos %}
$ bin/pvals treated untreated -t0 1000
9001 samples processed from directory "clust/cc"
{% endhighlight %}

The above command post-processes the contents of the directory `_clust` and returns 
a file, `_pvals.txt`, which includes a list of features along with their posterior 
probabilities of being differentially expressed between the treated and untreated 
samples in the data:

{% highlight bash linenos %}
$ head _pvals.txt
                Posteriors              FDR                                           
FBgn0000008     0.6344850572158649      0.41553854746732966   
FBgn0000017     0.1879791134318409      0.06813443720197582   
FBgn0000018     0.6990334407288079      0.45908014854398776   
FBgn0000024     0.4831685368292412      0.26083987371505857   
FBgn0000032     0.7033662926341517      0.46239982277180447   
FBgn0000037     0.4733918453505166      0.24896779084981993   
FBgn0000042     0.40939895567159207     0.19157589997351965   
FBgn0000043     0.16709254527274747     0.059146915319410236  
FBgn0000045     0.5090545494945006      0.29378201476431676   
{% endhighlight %}

Back in **IPython**, we can visualise our results using an **RA diagram**:

{% highlight python linenos %}
pvals = pd.read_table('_pvals.txt', index_col=0)
norm_factors = data.lib_sizes.astype('float') / sum(data.lib_sizes.values)    # a quick way to generate normalisation factors for visualisation purposes only
figure()
utils.plot_ra(data.counts['treated1fb']/norm_factors['treated1fb'], data.counts['untreated1fb']/norm_factors['untreated1fb'], pvals.FDR < 0.01)     # use an FDR of 1%
xlabel('( log2(untreated1fb) + log2(treated1fb) ) * 0.5')
ylabel('log2(untreated1fb) - log2(treated1fb)')
{% endhighlight %}

<img class="img-responsive" alt="RA plot" title="RA plot" src="{{ site.baseurl }}/img/RA_plot.png"></img>

<!-- Hierarchical clustering of samples
----------------------------------
(*under construction*)



Hierarchical clustering of genes
--------------------------------
(*under construction*)
 -->



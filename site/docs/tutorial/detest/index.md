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

<img class="img-responsive" alt="RA plot" title="RA plot" src="{{ site.baseurl }}/img/RA_plot.png"></img>

<!-- Hierarchical clustering of samples
----------------------------------
(*under construction*)



Hierarchical clustering of genes
--------------------------------
(*under construction*)
 -->



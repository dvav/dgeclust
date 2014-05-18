---
title: Tutorial
layout: default
---

Installation
============

Just download **DGEclust** using the links at the top of the page and unzip it 
in a directory of your choice. You can rename the unzipped directory to something 
more convenient, if you wish. That's it! No further installation procedure is 
necessary. In order to be able to use the software you will need a fairly recent 
version of the <a href="http://www.scipy.org/" target=”_blank”>Scipy</a> stack. 
**DGEclust** was developed and tested on a Mac, but it should be usable without 
problems on Linux machines, too. 

Quick start
===========

The data
--------

For the sake of this tutorial, let's assume that the downloaded **DGEclust** files 
are located in a directory named `dgeclust`. Also, let's assume that you 
have a matrix of count data stored in the file `data.txt`. In this tutorial, we 
shall use an RNA-seq dataset from yeast (*Saccharomyces cerevisiae*) cultures by 
<a href="http://www.ncbi.nlm.nih.gov/pubmed/18451266?dopt=Abstract&holding=f1000,f1000m,isrctn" target="_blank">Nagalakshmi et al</a>. 
The dataset can be downloaded from [here](http://genomebiology.com/content/supplementary/gb-2010-11-10-r106-s3.tgz).  


Clustering the data
-------------------

From your terminal, do the following:

{% highlight bash %}
$ cd path-to-dgeclust
$ bin/clust path-to-data-txt -g [[0,3],[1,2,4,5]] &
{% endhighlight %}

Depending on the size of your data, this will take some time to finish. The above 
command runs a Gibbs sampler for a default of 10K iterations. The output of the sampler 
is saved periodically in the directory `_clust`, which you can inspect to check the progress 
of your simulation, e.g. using `tail`:

{% highlight bash %}
$ tail -f _clust/pars.txt
0	10 	l;j
0	10 	l;j
0	10 	l;j
0	10 	l;j	
{% endhighlight %}

The first column in `pars.txt` gives you the number of iterations, the second gives 
you the estimated number of clusters at each iteration, while the remaining columns 
correspond to hyper-parameters, which depend on the specific distribution being used
by the simulator (defaults to the **Negative Binomial** distribution).

There are more arguments that you can pass to `clust`. Type `bin/clust -h` for more details.

After the end of the simulation, you can visualize your results using **IPython**.
Do `ipython --pylab` and then type: 

{% highlight python %}
In [1]: cd path-to-dgeclust
In [2]: from dgeclust.gibbs.results import GibbsOutput
In [3]: res = GibbsOutput.read('_clust')
In [4]: figure()
In [5]; subplot(3,1,1); plot(res.pars[:,0], res.pars[:,1], 'k'); xlabel('# iterations'); ylabel('# clusters')
In [6]; subplot(3,1,2); plot(res.pars[:,0], res.pars[:,[2,3]], 'k'); xlabel('# iterations'); ylabel('p1, p2')
In [7]; subplot(3,1,3); plot(res.pars[:,0], res.pars[:,[4,5]], 'k'); xlabel('# iterations'); ylabel('p3, p4')</code></pre>
{% endhighlight %}

It seems that the algorithm converges nicely after ~1000 iterations.

Testing for differential expression
-----------------------------------

The output of `clust` can be used to test for differential expression. For this purpose, 
we use another program that comes with **DGEclust**, `pvals`, which computes the 
posterior probability that any two features between any two groups of samples in 
our dataset are differentially expressed. In a terminal, we type: 

{% highlight bash %}
$ bin/pvals -t0 1000
???????
{% endhighlight %}

The above command post-processes the contents of the directory `_clust and returns 
a file, `_pvals.txt`, which includes a list of features with decreasing posterior 
probabilities of being differentially expressed between the two groups of samples 
in the yeast dataset:

{% highlight bash %}
$ head _pvals.txt
???????
{% endhighlight %}

Back in **IPython**, we can visualise our dataset using an **RA diagram**:

{% highlight python %}
    ??????
{% endhighlight %}


Hierarchical clustering of samples
----------------------------------
(*under construction*)


		
Hierarchical clustering of genes
--------------------------------
(*under construction*)




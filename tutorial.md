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
In [1]: import pandas as pd
In [2]: data = pd.read_table('/path/to/data.txt', index_col=0)
In [3]: idxs = all(data, 1)    # identify the rows where all entries are non-zero 
In [4]: data_filt = data[idxs]
In [5]: data_filt.head()    # inspect the data
In [6]: data_filt.to_csv('data_filt.txt', sep='\t')
{% endhighlight %}

This dataset contains 12221 features.


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
In [7]: cd path/to/dgeclust
In [8]: from dgeclust.gibbs.results import GibbsOutput
In [9]: res = GibbsOutput.read('_clust')
In [10]: figure()
In [11]: subplot(3,1,1); plot(res.pars[:,0], res.nactive0, 'k'); xlabel('# iterations'); ylabel('# clusters')
In [12]: subplot(3,1,2); plot(res.pars[:,0], res.pars[:,[2,3]], 'k'); xlabel('# iterations'); ylabel('p1, p2')
In [7]: subplot(3,1,3); plot(res.pars[:,0], res.pars[:,[4,5]], 'k'); xlabel('# iterations'); ylabel('p3, p4')
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




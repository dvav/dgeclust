---
title: Tutorial
layout: docs
---

Data preparation
================

For the sake of this tutorial, let's assume that the downloaded **DGEclust** files 
are located in a directory named `dgeclust`. Also, let's assume that you 
have a matrix of count data stored in the file `data.txt`. In this tutorial, we 
shall use the *pasilla* RNA-seq dataset, which is available through 
<a href="http://www.bioconductor.org/packages/release/data/experiment/html/pasilla.html" target="_blank">Bioconductor</a>. 
You can inspect the data using the `head` command at the terminal:
{% highlight bash %}
$ head /path/to/data.txt
treated1fb      treated2fb      treated3fb      untreated1fb    untreated2fb    untreated3fb    untreated4fb
FBgn0000003     0       0       1       0       0       0       0
FBgn0000008     78      46      43      47      89      53      27
FBgn0000014     2       0       0       0       0       1       0
FBgn0000015     1       0       1       0       1       1       2
FBgn0000017     3187    1672    1859    2445    4615    2063    1711
FBgn0000018     369     150     176     288     383     135     174
FBgn0000022     0       0       0       0       1       0       0
FBgn0000024     4       5       3       4       7       1       0
FBgn0000028     0       1       1       0       1       0       0
...
{% endhighlight %}

The data set consists of 7 libraries/samples with 14115 features each. The libraries are grouped in two different 
classes, *treated* and *untreated*. 

Let's use **IPython** to filter the data. Execute `ipython --pylab` at the terminal and, at the subsequent
**IPython** command prompt, type the following:

{% highlight python %}
import pandas as pd
data = pd.read_table('/path/to/data.txt')
row_sums = sum(data, 1)
idxs = rows_sums > percentile(row_sums, 40)  # identifies the upper 60% of the row sums
data_filt = data[idxs]
data_filt.head()    # inspect the data
data_filt.to_csv('data_filt.txt', sep='\t')
{% endhighlight %}

The filtered dataset contains 8461 features. The next step is to cluster your data. 
<a href="{{ site.baseurl }}{{ site.data.nav.docs.tut.clust.url }}">Learn how!</a>

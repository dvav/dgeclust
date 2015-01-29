---
title: Tutorial
layout: docs
---

Data preparation
================

Let's assume that you have a matrix of count data stored in the file `data.txt`. In this tutorial, we 
shall use the *pasilla* RNA-seq dataset, which is available through 
<a href="http://www.bioconductor.org/packages/release/data/experiment/html/pasilla.html" target="_blank">Bioconductor</a>. 
You can inspect the data using the `head` command at the terminal:
{% highlight python linenos %}
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

The data consists of 7 libraries/samples with 14115 features each. The libraries are grouped in two different 
classes, *treated* and *untreated*. 

Let's use **IPython** to filter the data. Execute `ipython` at the terminal and, at the subsequent
**IPython** command prompt, type the following:

{% highlight python linenos %}
import numpy as np
import pandas as pd
import matplotlib.pylab as pl

counts = pd.read_table('/path/to/data.txt')
row_sums = counts.sum(1)
idxs = row_sums > np.percentile(row_sums, 40)   # identifies the upper 60% of the data
counts_filt = counts[idxs]  
counts_filt.head()    # inspect the data
{% endhighlight %}

The filtered dataset contains 8461 features. The next step is to process your data. 
<a href="{{ site.baseurl }}{{ site.data.nav.docs.tut.processing.url }}">Learn how!</a>

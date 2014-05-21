---
title: Tutorial
layout: docs
---

Data preparation
================

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

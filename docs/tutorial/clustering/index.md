---
title: Tutorial
layout: docs
---

Clustering the data
===================

Next, we need to cluster the data. For this purpose, we shall use the program `clust`, which comes with **DGEclust**. 
From your terminal, do the following:

{% highlight bash linenos %}
$ cd path/to/dgeclust
$ bin/clust path/to/data_filt.txt -g treated treated treated untreated untreated untreated untreated &
{% endhighlight %}

The above command runs a Gibbs sampler for a default of 10K iterations. The argument `-g treated treated treated untreated untreated untreated untreated`
instructs the simulator that the first three samples and the last four form two different groups (i.e. treated and untreated). 
Depending on the size of your data, this will take some time to finish. The output from the sampler is saved periodically in the directory 
`_clust`, which you can inspect to check the progress of your simulation, e.g. using `tail`:

{% highlight bash linenos %}
$ tail -f _clust/pars.txt
0       100     100     1.000000        0.000001        0.500000        0.500000        0.500000        0.500000        0.000000        1.000000        0.000000        1.000000        0.000000        1.000000
1       100     25      1.000000        0.013232        0.592728        0.407272        0.789091        0.210909        1.019620        0.856941        -0.624808       1.008012        -0.215131       0.877321
2       93      36      1.000000        0.010479        0.690192        0.309808        0.747548        0.252452        1.206372        0.689313        -0.995780       1.226596        -0.449124       0.781314
3       96      34      1.000000        0.010251        0.781343        0.218657        0.729465        0.270535        1.251615        0.800146        -1.527719       0.639356        -0.673721       0.717895
4       85      29      1.000000        0.009616        0.862315        0.137685        0.716464        0.283536        1.347671        0.825410        -1.863795       1.276711        -0.883036       0.677475
5       88      32      1.000000        0.010956        0.925837        0.074163        0.705472        0.294528        1.241751        0.567440        -2.235118       1.246565        -1.102693       0.568704
6       86      31      1.000000        0.009139        0.958201        0.041799        0.691289        0.308711        1.182294        1.339433        -2.605717       1.943154        -1.279828       0.540100
7       87      36      1.000000        0.011891        0.981414        0.018586        0.675038        0.324962        1.025973        1.544493        -3.123108       1.817990        -1.444316       0.509955
8       84      36      1.000000        0.009918        0.990247        0.009753        0.649273        0.350727        0.859629        1.532394        -3.601945       1.525379        -1.613195       0.462447
9       84      41      1.000000        0.011009        0.998402        0.001598        0.621558        0.378442        0.662717        1.565444        -3.860092       1.305714        -1.772060       0.423376
...
{% endhighlight %}

The first column in `pars.txt` gives you the number of iterations, while the remaining 
columns correspond to parameters, some of which depend on the specific distribution being used
by the simulator to model count data (defaults to the **Negative Binomial** distribution).

There are more arguments that you can pass to `clust`. Type `bin/clust -h` for more details.

After the end of the simulation, you can visualize your results using **IPython**:

{% highlight python linenos %}
cd path/to/dgeclust
from dgeclust.gibbs.results import GibbsOutput
res = GibbsOutput.load('_clust')
figure()
subplot(2,2,1); plot(res.nclust.active); xlim([0, 500]); xlabel('# of iterations'); ylabel('# of clusters')
subplot(2,2,2); hist(res.nclust.active, 100, range=(0,100), normed=True); xlabel('# of clusters'); ylabel('frequency')
subplot(2,2,3); plot(res.hpars.iloc[:,:4]); xlabel('# of terations'); ylabel('p1 to p4')
subplot(2,2,4); plot(res.hpars.iloc[:,4:]); xlabel('# of iterations'); ylabel('p5, p6')
{% endhighlight %}

<img class="img-responsive" alt="Simulation progress" title="Simulation progress" src="{{ site.baseurl }}/img/progress.png"></img>

The algorithm converges rapidly after ~500 iterations. From the histogram on the top right, we can see that the data
supports around 70 clusters. The bottom panels indicate the progress of additional model parameters. Their exact meaning
is not important at the moment. What is important, however, is that they also reach a stable equilibrium quite rapidly 
during the course of the simulation. If you need to extend the simulation for another 10K iterations (i.e. a total
of 20K iterations), you can type:

{% highlight bash linenos %}
bin/clust /path/to/data_filt.txt -g treated treated treated untreated untreated untreated untreated -t 20000 -e & 
{% endhighlight %}
 
The argument `-e` indicates that a previously terminated simulation should
be extended and the argument `-t` indicates the total duration of the simulation. 

If you wish, you can see how the fitted model at the end of the simulation compares
to the actual data:

{% highlight python linenos %}
from dgeclust import utils
from dgeclust.data import CountData
from dgeclust.models import nbinom
data = CountData.load('path/to/data_filt.txt', groups=['treated', 'treated', 'treated', 'untreated', 'untreated', 'untreated', 'untreated'])    
figure()
x, y = utils.compute_fitted_model('treated1fb', res.state, data, nbinom);    # you can change 'treated1fb' to any other sample 
xlabel('log counts'); ylabel('frequency')
{% endhighlight %}

<img class="img-responsive" alt="Fitted model" title="Fitted model" src="{{ site.baseurl }}/img/fitted.png"></img>

The next step is to post-process your data in order to test for differential expression. 
<a href="{{ site.baseurl }}{{ site.data.nav.docs.tut.detest.url }}">Learn how!</a>


---
title: Tutorial
layout: docs
---

Processing the data
===================

Next, we need to process the data using **DGEclust**. At the **IPython** terminal, type the following:

{% highlight python linenos %}
from dgeclust import CountData, SimulationManager
from dgeclust.models import NBinomModel

mgr = SimulationManager()

data = CountData(counts_filt, groups=['treated', 'treated', 'treated', 'untreated', 'untreated', 'untreated', 'untreated'])
mdl = NBinomModel(data, outdir='sim_output')

mgr.new(data, mdl)
{% endhighlight %}

**DGEclust** is distributed as the python package `dgeclust`, which you can import and use at the 
**IPython** command prompt. At `line 4` above, we create a new `SimulationManager` object, which we
will use for initiating simulations. 

At `line 6`, we create a `CountData` object, which takes as input the filtered count data and a list of strings 
representing the group assigment (i.e. *treated* or *untreated*) of each sample. If `group` is omitted, each 
sample is assumed to be a group on its own. This is a very simple way to indicate the presence or absence of 
**biological replicates**. The `CountData` constructor also accepts a `lib_sizes` argument, which is a list of
normalisation factors, one for each sample. If omitted, as above, these normalisation factors are computed 
automatically using the same method `DESeq` uses.  

At `line 7`, we create an `NBinomModel` object, which models the data using a Negative Binomial distribution.
Simulation results are saved in the directory specified by the argument `outdir` (`sim_output` in this case). 
If this argument is omitted, the default destination is the sub-directory `_clust` in the current directory.
       
Finally, at `line 9`, we fire up a simulation using the `new` method of the `mgr` object and the data and model 
objects we created earlier as arguments. Notice that the `new` method returns immediately at the command prompt. 
This permits running additional simulations (i.e. by creating new `CountData` and `NBinomModel` objects and 
calling `new` with these as arguments) and inspecting them while they are running, as we shall see below. 
If this behavior is undesirable, we can modify it using the `bg=False` argument. 
The above command runs a Gibbs sampler for a default of 10K iterations. A different simulation length can be 
specified using the argument `niters`. By default, the data are processed in parallel using all available 
processing cores in the system. Again, this behaviour can be modified using the `nthreads` argument.

Depending on the size of your data and system resources, the above simulation will take several minutes to finish. 
Its progress can be checked periodically by calling the `plot_progress` method of the `mdl` object:

{% highlight python linenos %}
mdl.plot_progress(fig=figure(figsize=(10,7)))
{% endhighlight %}

A sample output of the above statement after the end of the simulation, is given below: 

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


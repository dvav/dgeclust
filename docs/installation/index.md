---
title: Installation
layout: docs
---

Installation
========

The easiest way to install **DGEclust** is using `pip`, as follows:

{% highlight bash linenos %}
$ sudo pip install --install-option='--install-scripts=/usr/local/bin' dgeclust==14.08a
{% endhighlight %} 

Alternatively, download the source code using the links at the top-right of the page and from
the source folder do:

{% highlight bash linenos %}
$ sudo python setup.py --install-scripts=/usr/local/bin
{% endhighlight %} 
   
Depending on your system, the `--install-option` and `--install-scripts` arguments may be omitted.

That's it! No further installation procedure is necessary. In order to be able to use the software 
you will need a fairly recent version of the <a href="http://www.scipy.org/" target=”_blank”>Scipy</a> stack. 
This will be installed, if necessary, upon installation using `pip`.

**DGEclust** was developed and tested on a Mac, but it should be usable without problems on Linux machines, too. 



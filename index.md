---
title: Home
layout: default
---

Welcome!
========

**DGEclust** is a program for clustering digital expression data, generated from next-generation sequencing
assays, such as RNA-seq, CAGE and others. It takes as input a table of count data and it estimates the
number and parameters of the clusters supported by the data. Internally, **DGEclust** uses a *Hierarchical Dirichlet
Process Mixture Model* for modelling (over-dispersed) count data, combined with a *blocked Gibbs sampler* for
efficient Bayesian learning.
			
This program is part of the software collection of the <a href="http://bioinformatics.bris.ac.uk/" target=”_blank”>Computational Genomics Group</a>
at the University of Bristol and it is still under heavy development. You can find
more technical details on the statistical methodologies used in this software in the following
papers:

1. **Vavoulis DV**, Gough J (2013). Non-Parametric Bayesian Modelling of Digital Gene Expression Data. 
*J Comput Sci Syst Biol* 7:001-009. doi: 10.4172/jcsb.1000131 \[[PDF](http://arxiv.org/pdf/1301.4144v1.pdf)\]
2. **Vavoulis DV**, Francescatto M, Heutink P, Gough J (2014). DGEclust: differential
expression analysis of clustered count data. (*submitted*) \[[PDF](http://arxiv.org/pdf/1405.0723v1.pdf)\]

For more information, send an email to <Dimitris.Vavoulis@bristol.ac.uk> or <Julian.Gough@bristol.ac.uk>

Enjoy!

Installation
============

Just download **DGEclust** using the links at the top of the page and unzip it 
in a directory of your choice. You can rename the unzipped directory to something 
more convenient, if you wish. That's it! No further installation procedure is 
necessary. In order to be able to use the software you will need a fairly recent 
version of the <a href="http://www.scipy.org/" target=”_blank”>Scipy</a> stack. 
**DGEclust** was developed and tested on a Mac, but it should be usable without 
problems on Linux machines, too. 


License
=======

**DGEclust** is covered by the MIT License:

The MIT License (MIT)

Copyright (c) 2012-2014 Dimitrios V. Vavoulis and the Computational Genomics Group at Bristol University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 


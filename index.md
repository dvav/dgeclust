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

If you find this software useful, please cite the second paper, above. 
For more information, send an email to <Dimitris.Vavoulis@bristol.ac.uk> or <Julian.Gough@bristol.ac.uk>

Enjoy!


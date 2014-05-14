`DGEclust` is a program for clustering digital expression data, generated from next-generation sequencing
assays, such as RNA-seq, CAGE and others. It takes as input a table of count data and it estimates the
number and parameters of the clusters supported by the data. Internally, `DGEclust` uses a Hierarchical Dirichlet
Process Mixture Model for modelling (over-dispersed) count data, combined with a blocked Gibbs sampler for
efficient Bayesian learning.

This program is part of the software collection of the Computational Genomics Group at the University
of Bristol (http://bioinformatics.bris.ac.uk/) and it is still under heavy development. You can find 
more technical details on the statistical methodologies used in this software in the following
papers:

1. http://arxiv.org/abs/1301.4144 (Vavoulis & Gough, -J Comput Sci Syst Biol- 7:001-009, 2013)
2. http://arxiv.org/abs/1405.0723 (Vavoulis et al., submitted, 2014)

For more information, send an email to [Dimitris.Vavoulis@bristol.ac.uk] or [Julian.Gough@bristol.ac.uk]

Enjoy!

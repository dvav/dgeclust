rm(list=ls())
setwd('~/Repositories/benchmarks/')

files = c('data/simdata1.txt', 'data/simdata2.txt', 'data/simdata3.txt')
max_nreplicates_per_group=8
samples = c(1,2,3)
sets = c('0a','0b','0c')
reps = c(8)
groups = c(2)

for (s in samples) {
  for (t in sets) {
    fin = files[s]
    raw = read.table(paste('set', t, '/', fin, sep=''))
    for (r in reps) {
      for (g in groups) {
        samples = rep(1:r, g) + rep(0:(g-1)*max_nreplicates_per_group, each = r)
        conds = rep(1:g,each = r)
        print(paste('set', t, ', ', fin, ', ', r, ' replicates, ', g, ' groups', sep=''))

        cds = newCountDataSet(raw[,samples], conds)
        cds = estimateSizeFactors(cds)
        if (r==1) {
          cds = estimateDispersions(cds, method='blind', sharingMode='fit-only')
        }
        else {
          cds = estimateDispersions(cds)
        }
        res = nbinomTest(cds, 1, 2)

        write.table(res, paste('set', t, '/', r, '/', g, '/',
                         'pvals', s, '_DESeq.txt', sep=''), quote=F, sep='\t')
      }
    }
  }
}

rm(list=ls())
setwd('~/Repositories/benchmarks/')

files = c('data/simdata1.txt', 'data/simdata2.txt', 'data/simdata3.txt')
max_nreplicates_per_group = 8
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
        conds = rep(1:g, each = r)
        print(paste('set', t, ', ', fin, ', ', r, ' replicates, ', g, ' groups', sep=''))

        dds = DGEList(counts=raw[, samples],group=conds)
        dds = calcNormFactors(dds)
        if (r==1) {
          res = exactTest(dds, dispersion = 0.3)
        } else {
          dds = estimateCommonDisp(dds)
          dds = estimateTagwiseDisp(dds)
          res = exactTest(dds)
        }
        top = topTags(res, n=1e6, sort.by='none')

        write.table(top, paste('set', t, '/', r, '/', g, '/',
                               'pvals', s, '_edgeR.txt', sep=''), quote=F, sep='\t')
      }
    }
  }
}

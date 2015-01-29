rm(list=ls())
setwd('~/Repositories/benchmarks/')

files = c('data/simdata1.txt', 'data/simdata2.txt', 'data/simdata3.txt')
max_nreplicates_per_group = 8
samples = c(1,2,3)
sets = c('0a','0b','0c')
reps = c(8)
groups = c(2)

cl <- makeCluster(8, "SOCK")

for (s in samples) {
  for (t in sets) {
    fin = files[s]
    raw = read.table(paste('set', t, '/', fin, sep=''))
    for (r in reps) {
      for (g in groups) {
        samples = rep(1:r, g) + rep(0:(g-1)*max_nreplicates_per_group, each = r)
        conds = rep(1:g, each = r)
        print(paste('set', t, ', ', fin, ', ', r, ' replicates, ', g, ' groups', sep=''))

        counts = raw[,samples]
        patt = list(NDE=rep(1, length(conds)), DE=conds)

        cds = new("countData", data = as.matrix(counts), replicates = conds, groups = patt)
        libsizes(cds) = getLibsizes(cds)
        cds = getPriors.NB(cds, samplesize = 5000, estimation = "QL", cl = cl)
        cds = getLikelihoods.NB(cds, pET = 'BIC', cl = cl)
        res = topCounts(cds, group = "DE", number=1e6)
        res = res[order(res$rowID),c('Likelihood','DE','FDR.DE','FWER.DE')]

        write.table(res, paste('set', t, '/', r, '/', g, '/',
                               'pvals', s, '_baySeq.txt', sep=''), quote=F, sep='\t')
      }
    }
  }
}

setwd('~/Repositories/simdata')

sets = c('set0a/','set0b/','set0c/')
files = c('simdata1.txt', 'simdata2.txt', 'simdata3.txt')
r = '4rep/'
samples = c(1,2,3,4, 9,10,11,12)
reps = c(1,1,1,1, 2,2,2,2)
groups = list(NDE=rep(1, length(reps)), DE=reps)

cl <- makeCluster(8, "SOCK")

for (s in sets) {
  for (i in 1:3) {
    fin = paste(s, files[i], sep='') 
    print(fin)
    
    raw = read.table(fin)[,samples]
    
    cds = new("countData", data = as.matrix(raw), replicates = reps, groups = groups)
    libsizes(cds) = getLibsizes(cds)
    cds = getPriors.NB(cds, samplesize = 1000, estimation = "QL", cl = cl)
    cds = getLikelihoods.NB(cds, pET = 'BIC', cl = cl)
    res = topCounts(cds, group = "DE", number=1e6)
    res = res[order(res$rowID),c('Likelihood','DE','FDR.DE','FWER.DE')]
        
    write.table(res, paste(s, r, 'pvals', i, '_baySeq.txt', sep=''), quote=F, sep='\t')
  }
}

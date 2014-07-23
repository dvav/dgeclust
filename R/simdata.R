for (i in 1:3) {
  simdata = generateSyntheticData(dataset='simdata', 
                                  n.diffexp=0, n.vars=10000, samples.per.cond=8, 
                                  fraction.upregulated=0.5, 
                                  fraction.non.overdispersed=0.5) 
  
  saveRDS(simdata, paste('simdata',i,'.rds', sep=''))
  write.table(simdata@count.matrix, paste('simdata',i,'.txt', sep=''), quote=F, sep='\t')
  write.table(simdata@variable.annotations$differential.expression, paste('ytrue',i,'.txt',sep=''), 
              quote=F, sep='\t', row.names=FALSE, col.names=FALSE)
}


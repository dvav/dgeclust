setwd('~/Repositories/simdata')

sets = c('set0a/','set0b/','set0c/')
files = c('simdata1.txt', 'simdata2.txt', 'simdata3.txt')
r = '4rep/'
samples = c(1,2,3,4, 9,10,11,12) 
conds = c(1,1,1,1, 2,2,2,2)
colData = data.frame(condition=factor(conds))

for (s in sets) {
  for (i in 1:3) {
    fin = paste(s, files[i], sep='') 
    print(fin)

    raw = read.table(fin)[,samples]
    
    dds = DESeqDataSetFromMatrix(raw, colData, design = ~ condition)
    dds = DESeq(dds)
    
    res = results(dds, contrast=c('condition','1','2'))  
    
    write.table(res, paste(s, r, 'pvals', i, '_DESeq2.txt', sep=''), quote=F, sep='\t')
  }
}

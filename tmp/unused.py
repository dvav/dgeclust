def compute_pvals(x, y):
    ## lambda function for computing p(y|x)
    computeP = lambda x, y, r: np.exp( y * np.log(r) + sp.gammaln(x + y + 1.) - sp.gammaln(x + 1.) -
sp.gammaln(y + 1.) - (x + y + 1.) * np.log(1. + r) )

    ## compute q values
    r  = y.sum() / x.sum()
    yy = [ np.arange(yi + 1)          for yi      in y          ]
    q  = [ computeP(xi, yyi, r).sum() for xi, yyi in zip(x, yy) ]
    q  = np.asarray(q)


    ## compute p values
    i    = q <= 0.5
    pval = np.zeros(q.size)

    pval[i]  = 2. * q[i]
    pval[~i] = 2. * ( 1. - q[~i] )

    ## return
    return pval

################################################################################

def adjust_pvals(p, method = 'Bonferonni'):
    padj = p * p.size
    padj[padj > 1] = 1

    return padj

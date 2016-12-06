import numpy as np


def getStats(listOfBinned):
    '''
    Returns statistics given an array-like object
    each row of which is an independent sample
    and each column holds possibly correlated
    binned values.

    The statistics are returned as a dictionary with
    keys:
    1. mean
    2. cov
    3. covmean
    4. err
    5. errmean
    6. corr
    '''
    # untested!
    
    
    arr = np.asarray(listOfBinned)
    N = arr.shape[0] # CHECK!!!
    ret = {}
    ret['mean'] = np.nanmean(arr,axis=0)
    ret['cov'] = np.cov(arr.transpose())
    ret['covmean'] = ret['covmat'] / N
    ret['err'] = np.sqrt(np.diagonal(ret['cov']))
    ret['errmean'] = ret['err'] / np.sqrt(N)

    # correlation matrix
    d = np.diag(ret['cov'])
    stddev = np.sqrt(d)
    ret['corr'] = ret['cov'] / stddev[:, None]
    ret['corr'] = ret['cov'] / stddev[None, :]
    np.clip(ret['corr'], -1, 1, out=ret['corr'])
    

        
    return ret
    
def bootstrapData(profs,Bmax=100000):
    # profs is a list of 1d arrays you want to bootstrap over
    print "Bootstrapping..."
    newprofs = []
    for i in range(Bmax):
        
        xn = np.random.choice(len(profs),size=(1,len(profs)))
        xn = list(xn.flatten())

        newprof = sum([profs[i] for i in xn])/len(profs)
        newprofs.append(newprof)


    return newprofs
                                                        
class coreBinner:
    '''
    * Takes data defined on x0 and produces values binned on x.
    * Assumes x0 is linearly spaced and continuous in a domain?
    * Assumes x is continuous in a subdomain of x0.
    * Should handle NaNs correctly.
    '''
    

    def __init__(self, bin_edges):

        self.updateBinEdges(bin_edges)


    def updateBinEdges(self,bin_edges):
        
        self.bin_edges = bin_edges
        self.numbins = len(bin_edges)-1


    def binned(self,x,y):


        # pretty sure this treats nans in y correctly, but should double-check!
        bin_means = binnedstat(x,y,bins=self.bin_edges,statistic=np.nanmean)[0]


        
        return bin_means

        

    def getBinCenters(self,mode="mean"):

        if mode=="mean":
            return (self.bin_edges[:-1]+self.bin_edges[1:])/2.
        else:
            raise ValueError


def loadBinFile(binfile,delimiter='\t',returnBinner=True):

    mat = np.loadtxt(binfile,delimiter=delimiter)

    left = mat[:,0]
    right = mat[:,1]
    try:
        center = mat[:,2]
    except:
        print "coreStats.py:loadBinFile says \"Third column absent in binfile. Using mean of left and right edges.\""
        center = (left+right)/2.

    if returnBinner:
        bin_edges = left.copy()
        bin_edges = np.append(bin_edges,right[-1])
        return coreBinner(bin_edges)
    else:
        return left,right,center


def binInAnnuli(data2d, modRMap, bin_edges):
    centers = (bin_edges[1:]+bin_edges[:-1])/2.
    digitized = np.digitize(np.ndarray.flatten(modRMap), bin_edges,right=True)
    data = np.ndarray.flatten(data2d)
    return centers,np.array([np.nanmean(data[digitized == i]) for i in range(1, len(bin_edges))])

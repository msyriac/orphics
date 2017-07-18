from __future__ import print_function
import numpy as np


def split_samples(in_samples,split_points):
    """ Calculate statistics on splits of a sample of data.
    If in_samples is a list containing measurements of (say) richness
    and split_points are the richness values corresponding to bin 
    edges of the splits, this function will return the "S/N", 
    average richness and number of objects in each bin defined 
    by the split_points bin edges.

    Here, S/N is average richness times square root of number
    objects in that bin. If the real S/N is linear in richness
    and sqrt(N), this returned quantity should be proportional
    to it.


    e.g. 
    >>> in_samples = [5,1,6,7,2,9,10]
    >>> split_points = [1,5,10]
    >>> sns, means, Ns = split_points(in_samples,split_points)
    >>> print (Ns)
    >>> [3,4]
    

    """
    assert np.all(np.diff(split_points))>0., "Split points should be monotonically increasing."
    A = np.asarray(in_samples)
    sns = []
    means = []
    Ns = []
        
    for a,b in zip(split_points[:-1],split_points[1:]):
        loc = np.where(np.logical_and(A>a,A<=b))
        split_mean = A[loc].mean()
        means.append(split_mean)
        split_N = len(A[loc])
        Ns.append(split_N)
        sns.append(split_mean*np.sqrt(split_N))
        
        
    return np.asarray(sns),np.asarray(means),np.asarray(Ns)


def optimize_splits(in_samples,in_splits):
    """ Given measurements in_samples and bin edges in_splits,
    this function solves for a new set of bin edges out_splits
    (keeping the leftmost and rightmost edge fixed) such
    that the "S/N" in each bin is as close to each other as
    possible. (Concretely, it minimizes the variance of the
    S/N in each bin).

    See split_samples for a definition of S/N.

    """

    def cost(*kwargs):
        in_array = np.asarray(kwargs).flatten()
        if np.any(np.diff(in_array))<0.: return np.inf
        out_splits = np.append(np.append(in_splits[0],in_array),in_splits[-1]).flatten()
        sns,means,Ns = split_samples(in_samples,out_splits)
        return np.var(sns)


    from scipy.optimize import fmin
    in_splits = np.asarray(in_splits)
    res = fmin(cost,in_splits[1:-1])
    out_splits = np.append(np.append(in_splits[0],res),in_splits[-1]).flatten()
    return out_splits
    


def generateRandomCatalogFromLiteMap(templateLiteMap,N=10000):

    x,y = (templateLiteMap.x0,templateLiteMap.x1)
    ra0 = min(x,y)
    ra1 = max(x,y)

    if ra1>180.:

        cra1 = ra0
        ra0 = ra1 - 360.
        ra1 = cra1

    dec0, dec1 = (templateLiteMap.y0,templateLiteMap.y1)
    rlist = np.random.uniform(ra0,ra1,size=(N,1))
    dlist = np.random.uniform(dec0,dec1,size=(N,1))
    return np.hstack((rlist,dlist))



def getCountMapFromCatalog(templateLiteMap,ras,decs,curved=False):

    from ..analysis.galaxyMapMaker import fitsMapFromCat as galMapper

    fmapper = galMapper(["ct"],[templateLiteMap],["temp"])
    fmapper.addCut('')

    for ra,dec in zip(ras,decs):

        fmapper.updateIndex(ra,dec)
        fmapper.increment("ct",'',1)

    return fmapper.getMap("ct","temp",'')


def getCountMapsFromCatalogFile(templateLiteMap,fitsFile,raMin,raMax,nsplits=1,Nmax=None,curved=False):

    from ..analysis.galaxyMapMaker import fitsMapFromCat as galMapper
    import astropy.io.fits as fitsio

    f = fitsio.open(fitsFile)

    suffixes = []
    for i in range(nsplits):
        suffixes.append("ct"+str(i))
    fmapper = galMapper(suffixes,[templateLiteMap],["temp"])
    fmapper.addCut('')
    if Nmax is None: Nmax = f[1].data['RA'].size

    i=0
    j=0
    while i<Nmax:
        try:
            ra = f[1].data['RA'][i]
            dec = f[1].data['DEC'][i]
        except:
            break
        if ra>raMin and ra<raMax: 
            fmapper.updateIndex(ra,dec)
            fmapper.increment("ct"+str(j%nsplits),'',1)
            j+=1
        i+=1
        if i%10000==0: print (i)

    f.close()
    return [fmapper.getMap("ct"+str(k),"temp",'') for k in range(nsplits)]

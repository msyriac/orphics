
import numpy as np

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
        if i%10000==0: print i

    f.close()
    return [fmapper.getMap("ct"+str(k),"temp",'') for k in range(nsplits)]

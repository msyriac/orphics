
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



def getCountMapFromCatalog(templateLiteMap,ras,decs):

    from ..analysis.galaxyMapMaker import fitsMapFromCat

    fmapper = fitsMapFromCat(["ct"],[templateLiteMap],["temp"])
    fmapper.addCut('')

    for ra,dec in zip(ras,decs):

        fmapper.updateIndex(ra,dec)
        fmapper.increment("ct",'',1)

    return fmapper.getMap("ct","temp",'')

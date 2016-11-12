import healpy as hp
import enmap as en
import numpy as np
from curvedsky import alm2map
import enlib


def quickMapView(hpMap,saveLoc=None,min=None,max=None,transform=True):
    '''
    Input map in galactic is shown in equatorial
    '''

    if transform:
        hp.mollview(hpMap,coord=['G','C'],min=min,max=max)
    else:    
        hp.mollview(hpMap,min=min,max=max,coord='C')
    if saveLoc==None: saveLoc="output/debug.png"
    matplotlib.pyplot.savefig(saveLoc)

    print bcolors.OKGREEN+"Saved healpix plot to", saveLoc+bcolors.ENDC


class healpixTools:

    def __init__(self):


        deg2pix = ctypes.CDLL('lib/deg2healpix.so')
        self._pix2radec = deg2pix.getRaDec
        self._pix2radec.argtypes = (ctypes.c_long, ctypes.c_long, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))

        self._rotate = deg2pix.RotateIndexGtoC
        self._rotate.argtypes = (ctypes.c_long, ctypes.c_long)
        

    def rotateIndexGToC(self,nside,pixind):
        ret = self._rotate(nside, pixind)
        return ret
        
    def pix2RaDec(self,nside,pixind):
        raret = ctypes.c_double()
        decret = ctypes.c_double()
        ret = self._pix2radec(nside, pixind, raret,decret)
        return raret.value,decret.value


    def rotateMapGToC(self,hpMap,nside,verbose=True):

        npix = hp.nside2npix(nside)
        rMap = hpMap.copy()
        if verbose: print "Rotating map... "
        for pixIndex in range(len(hpMap)):
    

            j = self.rotateIndexGToC(nside,pixIndex)
            rMap[pixIndex] = hpMap[j]


            if verbose and pixIndex%100000==0:
                print pixIndex*100./npix

        

        return rMap


def fullsky_geometry(res, dims=()):
    """Build an enmap covering the full sky, with the outermost pixel centers
    at the poles."""
    nx,ny = int(2*np.pi/res+0.5), int(np.pi/res+0.5)
    wcs   = enlib.wcs.WCS(naxis=2)
    wcs.wcs.crval = [0,0]
    wcs.wcs.cdelt = [360./nx,180./ny]
    wcs.wcs.crpix = [nx/2+1,ny/2+1]
    wcs.wcs.ctype = ["RA---CAR","DEC--CAR"]
    return dims+(ny+1,nx+0), wcs

def rotateHealpixFromEquToGal(hpmap):
    print "making empty enmap"
    res = 3.0 *np.pi/ 60./180.
    shape, wcs = fullsky_geometry(res, (1,))
    map = en.zeros(shape, wcs)
    print "rotating..."
    alms = hp.map2alm(hpmap)
    m1 = alm2map(alms,map)
    pos = np.array(hp.pix2ang(np.arange(len(hpmap))))
    pos[0] = np.pi/2. - pos[0]
    pos2 = pos
    #pos2 = eq2gal(pos)
    return m1.at(pos2)




print "loading planck"
#Planck
saveRoot = "/astro/astronfs01/workarea/msyriac/SkyData/"
planckRoot = saveRoot+'cmb/'
planckMaskPath = planckRoot+'planck2015_mask.fits'
mask15 = hp.read_map(planckMaskPath,verbose=True)

from hpTools import quickMapView

quickMapView(mask15,"galactic.png")

quickMapView(rotateHealpixFromEquToGal(mask15),"equ.png")


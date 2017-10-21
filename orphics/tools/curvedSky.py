import matplotlib
import healpy as hp
import numpy as np
from orphics.tools.io import bcolors
import os
import ctypes
import numpy.ctypeslib as npct


def flatFitsToHealpix(fitsFile,nside,downgrade=1):

    from enlib import enmap

    imap  = enmap.read_map(fitsFile)
    if downgrade>1:
        imap = enmap.downgrade(imap, args.downgrade)
    omap  = imap.to_healpix(nside=args.nside)
    return omap



def slowRotatorGtoC(hpMap,nside,verbose=True):

    if verbose: print "Upgrading map..."
    
    nsideUp = nside*2
    
    hpUp = hp.ud_grade(hpMap,nsideUp)

    array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')
    libcd = npct.load_library("/astro/u/msyriac/repos/orphics/orphics/tools/deg2hp.so", ".")

    libcd.RotateMapGtoC.restype = None
    libcd.RotateMapGtoC.argtypes = [array_1d_double,array_1d_double,ctypes.c_long]

    retMap = hpUp.copy()*0.
    if verbose: print "Rotating ..."
    ret = libcd.RotateMapGtoC(hpUp, retMap, nsideUp)
    
    if verbose: print "Downgrading map..."

    return hp.ud_grade(retMap,nside)
    

def quickMapView(hpMap,saveLoc=None,min=None,max=None,transform='C',**kwargs):
    '''
    Input map in galactic is shown in equatorial
    '''

    hp.mollview(hpMap,min=min,max=max,coord=transform,**kwargs)
    if saveLoc==None: saveLoc="output/debug.png"
    matplotlib.pyplot.savefig(saveLoc)

    print bcolors.OKGREEN+"Saved healpix plot to", saveLoc+bcolors.ENDC


class healpixTools:

    def __init__(self):


        libpath = os.path.dirname(__file__)
        deg2pix = ctypes.CDLL(libpath + '/deg2hp.so')        
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




import matplotlib
import healpy as hp
import numpy as np
from orphics.tools.io import bcolors
import os
import ctypes
import numpy.ctypeslib as npct



def cutout_gnomonic(map,rot=None,coord=None,
             xsize=200,ysize=None,reso=1.5,
             nest=False,remove_dip=False,
             remove_mono=False,gal_cut=0,
             flip='astro'):
    """Obtain a cutout from a healpix map (given as an array) in Gnomonic projection.

    Parameters
    ----------
    map : array-like
      The map to project, supports masked maps, see the `ma` function.
    rot : scalar or sequence, optional
      Describe the rotation to apply.
      In the form (lon, lat, psi) (unit: degrees) : the point at
      longitude *lon* and latitude *lat* will be at the center. An additional rotation
      of angle *psi* around this direction is applied.
    coord : sequence of character, optional
      Either one of 'G', 'E' or 'C' to describe the coordinate
      system of the map, or a sequence of 2 of these to rotate
      the map from the first to the second coordinate system.
    xsize : int, optional
      The size of the image. Default: 200
    ysize : None or int, optional
      The size of the image. Default: None= xsize
    reso : float, optional
      Resolution (in arcmin). Default: 1.5 arcmin
    nest : bool, optional
      If True, ordering scheme is NESTED. Default: False (RING)
    flip : {'astro', 'geo'}, optional
      Defines the convention of projection : 'astro' (default, east towards left, west towards right)
      or 'geo' (east towards roght, west towards left)
    remove_dip : bool, optional
      If :const:`True`, remove the dipole+monopole
    remove_mono : bool, optional
      If :const:`True`, remove the monopole
    gal_cut : float, scalar, optional
      Symmetric galactic cut for the dipole/monopole fit.
      Removes points in latitude range [-gal_cut, +gal_cut]
    

    See Also
    --------
    gnomview, mollview, cartview, orthview, azeqview
    """
    import pylab
    import healpy as hp
    import healpy.projaxes as PA

    margins = (0.075,0.05,0.075,0.05)
    extent = (0.0,0.0,1.0,1.0)
    extent = (extent[0]+margins[0],
              extent[1]+margins[1],
              extent[2]-margins[2]-margins[0],
              extent[3]-margins[3]-margins[1])
    f=pylab.figure(0,figsize=(5.5,6))
    map = hp.pixelfunc.ma_to_array(map)
    ax=PA.HpxGnomonicAxes(f,extent,coord=coord,rot=rot,
                          format="%.3g",flipconv=flip)
    if remove_dip:
        map=hp.pixelfunc.remove_dipole(map,gal_cut=gal_cut,nest=nest,copy=True)
    elif remove_mono:
        map=hp.pixelfunc.remove_monopole(map,gal_cut=gal_cut,nest=nest,copy=True)
    img = ax.projmap(map,nest=nest,coord=coord,
               xsize=xsize,ysize=ysize,reso=reso)

    pylab.close(f)
    return img


def flatFitsToHealpix(fitsFile,nside,downgrade=1):

    from enlib import enmap

    imap  = enmap.read_map(fitsFile)
    if downgrade>1:
        imap = enmap.downgrade(imap, args.downgrade)
    omap  = imap.to_healpix(nside=args.nside)
    return omap



def slowRotatorGtoC(hpMap,nside,verbose=True):

    if verbose: print("Upgrading map...")
    
    nsideUp = nside*2
    
    hpUp = hp.ud_grade(hpMap,nsideUp)

    array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='C_CONTIGUOUS')
    libcd = npct.load_library("/astro/u/msyriac/repos/orphics/orphics/tools/deg2hp.so", ".")

    libcd.RotateMapGtoC.restype = None
    libcd.RotateMapGtoC.argtypes = [array_1d_double,array_1d_double,ctypes.c_long]

    retMap = hpUp.copy()*0.
    if verbose: print("Rotating ...")
    ret = libcd.RotateMapGtoC(hpUp, retMap, nsideUp)
    
    if verbose: print("Downgrading map...")

    return hp.ud_grade(retMap,nside)
    



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
        if verbose: print("Rotating map... ")
        for pixIndex in range(len(hpMap)):
    

            j = self.rotateIndexGToC(nside,pixIndex)
            rMap[pixIndex] = hpMap[j]


            if verbose and pixIndex%100000==0:
                print((pixIndex*100./npix))

        

        return rMap


def rotateHealpixFromEquToGal(hpmap):
    print("making empty enmap")
    res = 3.0 *np.pi/ 60./180.
    shape, wcs = fullsky_geometry(res, (1,))
    map = en.zeros(shape, wcs)
    print("rotating...")
    alms = hp.map2alm(hpmap)
    m1 = alm2map(alms,map)
    pos = np.array(hp.pix2ang(np.arange(len(hpmap))))
    pos[0] = np.pi/2. - pos[0]
    pos2 = pos
    #pos2 = eq2gal(pos)
    return m1.at(pos2)




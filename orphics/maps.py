from enlib import enmap, utils
import numpy as np
from enlib.fft import fft,ifft

### ENMAP HELPER FUNCTIONS AND CLASSES


def rect_geometry(width_arcmin=None,width_deg=None,px_res_arcmin=0.5,proj="car",pol=False,height_deg=None,height_arcmin=None,xoffset_degree=0.,yoffset_degree=0.):
    """
    Get shape and wcs for a rectangular patch of specified size and coordinate center
    """

    if width_deg is not None:
        width_arcmin = 60.*width_deg
    if height_deg is not None:
        height_arcmin = 60.*height_deg
    
    hwidth = width_arcmin/2.
    if height_arcmin is None:
        vwidth = hwidth
    else:
        vwidth = height_arcmin/2.
    arcmin =  utils.arcmin
    degree =  utils.degree
    shape, wcs = enmap.geometry(pos=[[-vwidth*arcmin+yoffset_degree*degree,-hwidth*arcmin+xoffset_degree*degree],[vwidth*arcmin+yoffset_degree*degree,hwidth*arcmin+xoffset_degree*degree]], res=px_res_arcmin*arcmin, proj=proj)
    if pol: shape = (3,)+shape
    return shape, wcs


class MapGen(object):
        """
        Once you know the shape and wcs of an ndmap and the input power spectra, you can 
        pre-calculate some things to speed up random map generation.
        """
        
        def __init__(self,shape,wcs,cov,pixel_units=False):
                self.shape = shape
                self.wcs = wcs
                if cov.ndim==4:
                        if not(pixel_units): cov = cov * np.prod(shape[-2:])/enmap.area(shape,wcs )
                        self.covsqrt = enmap.multi_pow(cov, 0.5)
                else:
                        self.covsqrt = enmap.spec2flat(shape, wcs, cov, 0.5, mode="constant")

        def get_map(self,seed=None,scalar=False,iau_convention=True):
                if seed is not None: np.random.seed(seed)
                data = enmap.map_mul(self.covsqrt, enmap.rand_gauss_harm(self.shape, self.wcs))
                kmap = enmap.ndmap(data, self.wcs)
                if scalar:
                        return enmap.ifft(kmap).real
                else:
                        return enmap.harm2map(kmap,iau_convention=iau_convention)

        
        

class FourierCalc(object):
        """
        Once you know the shape and wcs of an ndmap, you can pre-calculate some things
        to speed up fourier transforms and power spectra.
        """

        def __init__(self,shape,wcs,iau_convention=True):
                self.shape = shape
                self.wcs = wcs
                self.normfact = enmap.area(self.shape,self.wcs )/ np.prod(self.shape[-2:])**2.         
                if len(shape) > 2 and shape[-3] > 1:
                        self.rot = enmap.queb_rotmat(enmap.lmap(shape,wcs),iau_convention=iau_convention)

        def iqu2teb(self,emap, nthread=0, normalize=True, rot=True):
                """Performs the 2d FFT of the enmap pixels, returning a complex enmap.
                Similar to harm2map, but uses a pre-calculated self.rot matrix.
                """
                emap = enmap.samewcs(enmap.fft(emap,nthread=nthread,normalize=normalize), emap)
                if emap.ndim > 2 and emap.shape[-3] > 1 and rot:
                        emap[...,-2:,:,:] = enmap.map_mul(self.rot, emap[...,-2:,:,:])
                return emap


        def f2power(self,kmap1,kmap2,pixel_units=False):
                norm = 1. if pixel_units else self.normfact
                return np.real(np.conjugate(kmap1)*kmap2)*norm

        def f1power(self,map1,kmap2,pixel_units=False,nthread=0):
                kmap1 = self.iqu2teb(map1,nthread,normalize=False)
                norm = 1. if pixel_units else self.normfact
                return np.real(np.conjugate(kmap1)*kmap2)*norm,kmap1

        def power2d(self,emap=None, emap2=None,nthread=0,pixel_units=False,skip_cross=False,rot=True, kmap=None, kmap2=None):
                """
                Calculate the power spectrum of emap crossed with emap2 (=emap if None)
                Returns in radians^2 by default unles pixel_units specified
                """

                if kmap is not None:
                        lteb1 = kmap
                        ndim = kmap.ndim
                        if ndim>2 : ncomp = kmap.shape[-3]
                else:
                        lteb1 = self.iqu2teb(emap,nthread,normalize=False,rot=rot)
                        ndim = emap.ndim
                        if ndim>2 : ncomp = emap.shape[-3]

                if kmap2 is not None:
                        lteb2 = kmap2
                else:
                        lteb2 = self.iqu2teb(emap2,nthread,normalize=False,rot=rot) if emap2 is not None else lteb1
                
                assert lteb1.shape==lteb2.shape
                
                if ndim > 2 and ncomp > 1:
                        retpow = np.empty((ncomp,ncomp,lteb1.shape[-2],lteb1.shape[-1]))
                        for i in range(ncomp):
                                retpow[i,i] = self.f2power(lteb1[i],lteb2[i],pixel_units)
                        if not(skip_cross):
                                for i in range(ncomp):
                                        for j in range(i+1,ncomp):
                                                retpow[i,j] = self.f2power(lteb1[i],lteb2[j],pixel_units)
                                                retpow[j,i] = retpow[i,j]
                        return retpow,lteb1,lteb2
                else:
                        if lteb1.ndim>2:
                                lteb1 = lteb1[0]
                        if lteb2.ndim>2:
                                lteb2 = lteb2[0]
                        p2d = self.f2power(lteb1,lteb2,pixel_units)
                        return p2d,lteb1,lteb2



### REAL AND FOURIER SPACE ATTRIBUTES

def get_ft_attributes(shape,wcs):
    Ny, Nx = shape[-2:]
    pixScaleY, pixScaleX = enmap.pixshape(shape,wcs)

    
        
    lx =  2*np.pi  * np.fft.fftfreq( Nx, d = pixScaleX )
    ly =  2*np.pi  * np.fft.fftfreq( Ny, d = pixScaleY )
    
    ix = np.mod(np.arange(Nx*Ny),Nx)
    iy = np.arange(Nx*Ny)//Nx
    
    modlmap = enmap.modlmap(shape,wcs)
    
    theta_map = np.zeros([Ny,Nx])
    theta_map[iy[:],ix[:]] = np.arctan2(ly[iy[:]],lx[ix[:]])


    lxMap, lyMap = np.meshgrid(lx, ly)  # is this the right order?

    return lxMap,lyMap,modlmap,theta_map,lx,ly


def get_real_attributes(shape,wcs):
    Ny, Nx = shape[-2:]
    pixScaleY, pixScaleX = enmap.pixshape(shape,wcs)
    
    xx =  (np.arange(Nx)-Nx/2.+0.5)*pixScaleX
    yy =  (np.arange(Ny)-Ny/2.+0.5)*pixScaleY
    
    ix = np.mod(np.arange(Nx*Ny),Nx)
    iy = np.arange(Nx*Ny)/Nx
    
    modRMap = np.zeros([Ny,Nx])
    modRMap[iy,ix] = np.sqrt(xx[ix]**2 + yy[iy]**2)
    

    xMap, yMap = np.meshgrid(xx, yy)  # is this the right order?

    return xMap,yMap,modRMap,xx,yy




## MAP OPERATIONS


def filter_map(imap,kfilter):
    return np.real(ifft(fft(imap,axes=[-2,-1])*kfilter,axes=[-2,-1],normalize=True)) 

def gauss_beam(ell,fwhm):
    tht_fwhm = np.deg2rad(fwhm / 60.)
    return np.exp(-(tht_fwhm**2.)*(ell**2.) / (16.*np.log(2.)))

def mask_kspace(shape,wcs, lxcut = None, lycut = None, lmin = None, lmax = None):
    output = np.ones(shape[-2:], dtype = int)
    if (lmin is not None) or (lmax is not None): modlmap = enmap.modlmap(shape, wcs)
    if (lxcut is not None) or (lycut is not None): ly, lx = enmap.laxes(shape, wcs, oversample=1)
    if lmin is not None:
        output[np.where(modlmap <= lmin)] = 0
    if lmax is not None:
        output[np.where(modlmap >= lmax)] = 0
    if lxcut is not None:
        output[:,np.where(np.abs(lx) < lxcut)] = 0
    if lycut is not None:
        output[np.where(np.abs(ly) < lycut),:] = 0
    return output

def ilc(kmaps,cinv,response=None):
    """Make an internal linear combination (ILC) of given fourier space maps at different frequencies
    and an inverse covariance matrix for its variance.
    
    Accepts
    -------

    kmaps -- (nfreq,Ny,Nx) array of beam-deconvolved fourier transforms at each frequency
    cinv -- (nfreq,nfreq,Ny,Nx) array of the inverted covariance matrix
    response -- (nfreq,) array of f_nu response factors. Defaults to unity for CMB estimate.

    Returns
    -------

    Fourier transform of ILC estimate, (Ny,Nx) array 
    """
    
    if response is None:
        # assume CMB
        nfreq = kmaps.shape[0]
        response = np.ones((nfreq,))

    # Get response^T Cinv kmaps
    weighted = np.einsum('k,kij->ij',response,np.einsum('klij,lij->kij',cinv,kmaps))
    # Get response^T Cinv response
    norm = np.einsum('l,lij->ij',response,np.einsum('k,klij->lij',response,cinv))
    return weighted/norm

## WORKING WITH DATA


class NoiseModel(object):

    def __init__(self,splits=None,wmap=None,mask=None,kmask=None,directory=None,spec_smooth_width=2.,skip_beam=True,skip_mask=True,skip_kmask=True,skip_cross=True,iau_convention=False):
        """
        shape, wcs for geometry
        unmasked splits
        hit counts wmap
        real-space mask that must include a taper. Can be 3-dimensional if Q/U different from I.
        k-space kmask
        """

        if directory is not None:
            self._load(directory,skip_beam,skip_mask,skip_kmask,skip_cross)
        else:
            
            shape = splits[0].shape
            wcs = splits[0].wcs

            if wmap is None: wmap = enmap.ones(shape[-2:],wcs)
            if mask is None: mask = np.ones(shape[-2:])
            if kmask is None: kmask = np.ones(shape[-2:])

            wmap = enmap.ndmap(wmap,wcs)

            osplits = [split*mask for split in splits]
            fc = enmap.FourierCalc(shape,wcs,iau_convention)
            n2d, p2d = noise_from_splits(osplits,fc)
            w2 = np.mean(mask**2.)
            n2d *= (1./w2)
            p2d *= (1./w2)

            n2d = enmap.smooth_spectrum(n2d, kernel="gauss", weight="mode", width=spec_smooth_width)
            self.spec_smooth_width = spec_smooth_width
            ncomp = shape[0] if len(shape)>2 else 1
            
            self.cross2d = p2d
            self.cross2d *= kmask
            n2d *= kmask
            self.noise2d = n2d.reshape((ncomp,ncomp,shape[-2],shape[-1]))
            self.mask = mask
            self.kmask = kmask
            self.wmap = wmap
            self.shape = shape
            self.wcs = wcs
        self.ngen = enmap.MapGen(self.shape,self.wcs,self.noise2d)
        #self.noise_modulation = 1./np.sqrt(self.wmap)/np.sqrt(np.mean((1./self.wmap)))
        wt = 1./np.sqrt(self.wmap)
        wtw2 = np.mean(1./wt**2.)
        self.noise_modulation = wt*np.sqrt(wtw2)


    def _load(self,directory,skip_beam=True,skip_mask=True,skip_kmask=True,skip_cross=True):

        self.wmap = enmap.read_hdf(directory+"/wmap.hdf")
        
        self.wcs = self.wmap.wcs
        self.noise2d = np.load(directory+"/noise2d.npy")
        if not(skip_mask): self.mask = enmap.read_hdf(directory+"/mask.hdf")
        if not(skip_cross): self.cross2d = np.load(directory+"/cross2d.npy")
        if not(skip_beam): self.kbeam2d = np.load(directory+"/kbeam2d.npy")
        if not(skip_kmask): self.kmask = np.load(directory+"/kmask.npy")
        loadnum = np.loadtxt(directory+"/spec_smooth_width.txt")
        assert loadnum.size==1
        self.spec_smooth_width = float(loadnum.ravel()[0])

        loadnum = np.loadtxt(directory+"/shape.txt")
        self.shape = tuple([int(x) for x in loadnum.ravel()])

    def save(self,directory):
        io.mkdir(directory)
        
        
        enmap.write_hdf(directory+"/wmap.hdf",self.wmap)
        enmap.write_hdf(directory+"/mask.hdf",self.mask)
        np.save(directory+"/noise2d.npy",self.noise2d)
        np.save(directory+"/cross2d.npy",self.cross2d)
        np.save(directory+"/kmask.npy",self.kmask)
        try: np.save(directory+"/kbeam2d.npy",self.kbeam2d)
        except: pass
        np.savetxt(directory+"/spec_smooth_width.txt",np.array(self.spec_smooth_width).reshape((1,1)))
        np.savetxt(directory+"/shape.txt",np.array(self.shape))


        
        

    def get_noise_sim(self,seed):
        return self.ngen.get_map(seed=seed,scalar=True) * self.noise_modulation
        
    def add_beam_1d(self,ells,beam_1d_transform):
        modlmap = enmap.modlmap(self.shape[-2:],self.wcs)
        self.kbeam2d = interp1d(ells,beam_1d_transform,bounds_error=False,fill_value=0.)(modlmap)
    def add_beam_2d(self,beam_2d_transform):
        assert self.shape[-2:]==beam_2d_transform.shape
        self.kbeam2d = beam_2d_transform
    

### FULL SKY


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

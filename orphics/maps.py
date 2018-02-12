from enlib import enmap, utils
import numpy as np
from enlib.fft import fft,ifft
from scipy.interpolate import interp1d
import yaml,six
from orphics import io
import math
from scipy.interpolate import RectBivariateSpline,interp2d,interp1d


### ENMAP HELPER FUNCTIONS AND CLASSES
def slice_from_box(shape, wcs, box, inclusive=False):
    """slice_from_box(shape, wcs, box, inclusive=False)
    Extract the part of the map inside the given box as a selection
    without returning the data. Does not work for boxes
    that straddle boundaries of maps. Use enmap.submap instead.
    Parameters
    ----------
    box : array_like
	    The [[fromy,fromx],[toy,tox]] bounding box to select.
	    The resulting map will have a bounding box as close
	    as possible to this, but will differ slightly due to
	    the finite pixel size.
    inclusive : boolean
		Whether to include pixels that are only partially
		inside the bounding box. Default: False."""
    ibox = enmap.subinds(shape, wcs, box, inclusive)
    print(shape,ibox)
    islice = utils.sbox2slice(ibox.T)
    return islice
    
def cutup(shape,numy,numx,pad=0):
    Ny,Nx = shape
    pixs_y = np.linspace(0,shape[-2],num=numy+1,endpoint=True)	
    pixs_x = np.linspace(0,shape[-1],num=numx+1,endpoint=True)
    num_boxes = numy*numx
    boxes = np.zeros((num_boxes,2,2))
    boxes[:,0,0] = np.tile(pixs_y[:-1],numx) - pad
    boxes[:,0,0][boxes[:,0,0]<0] = 0
    boxes[:,1,0] = np.tile(pixs_y[1:],numx) + pad
    boxes[:,1,0][boxes[:,1,0]>(Ny-1)] = Ny-1
    boxes[:,0,1] = np.repeat(pixs_x[:-1],numy) - pad
    boxes[:,0,1][boxes[:,0,1]<0] = 0
    boxes[:,1,1] = np.repeat(pixs_x[1:],numy) + pad
    boxes[:,1,1][boxes[:,1,1]>(Nx-1)] = Nx-1
    boxes = boxes.astype(np.int)

    return boxes


def bounds_from_list(blist):
    """Given blist = [dec0,ra0,dec1,ra1] in degrees
    return ndarray([[dec0,ra0],[dec1,ra1]]) in radians
    """
    return np.array(blist).reshape((2,2))*np.pi/180.
        

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
    pos = [[-vwidth*arcmin+yoffset_degree*degree,-hwidth*arcmin+xoffset_degree*degree],[vwidth*arcmin+yoffset_degree*degree,hwidth*arcmin+xoffset_degree*degree]]
    shape, wcs = enmap.geometry(pos=pos, res=px_res_arcmin*arcmin, proj=proj)
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

        def get_map(self,seed=None,scalar=False,iau=True):
                if seed is not None: np.random.seed(seed)
                data = enmap.map_mul(self.covsqrt, enmap.rand_gauss_harm(self.shape, self.wcs))
                kmap = enmap.ndmap(data, self.wcs)
                if scalar:
                        return enmap.ifft(kmap).real
                else:
                        return enmap.harm2map(kmap,iau=iau)

        
        

class FourierCalc(object):
        """
        Once you know the shape and wcs of an ndmap, you can pre-calculate some things
        to speed up fourier transforms and power spectra.
        """

        def __init__(self,shape,wcs,iau=True):
                self.shape = shape
                self.wcs = wcs
                self.normfact = enmap.area(self.shape,self.wcs )/ np.prod(self.shape[-2:])**2.         
                if len(shape) > 2 and shape[-3] > 1:
                        self.rot = enmap.queb_rotmat(enmap.lmap(shape,wcs),iau=iau)

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



class MapRotator(object):
    def __init__(self,shape_source,wcs_source,shape_target,wcs_target):
        self.pix_target = get_rotated_pixels(shape_source,wcs_source,shape_target,wcs_target)
    def rotate(self,imap,**kwargs):
        return rotate_map(imap,pix_target=self.pix_target,**kwargs)

class MapRotatorEquator(MapRotator):
    def __init__(self,shape_source,wcs_source,patch_width,patch_height,width_multiplier=1.,
                 height_multiplier=1.5,pix_target_override_arcmin=None,proj="car",verbose=False,
                 downsample=True,downsample_pix_arcmin=None):
        
        self.source_pix =  np.min(enmap.extent(shape_source,wcs_source)/shape_source[-2:])*60.*180./np.pi
        if pix_target_override_arcmin is None:
            input_dec = enmap.posmap(shape_source,wcs_source)[0]
            max_dec = np.max(np.abs(input_dec))
            del input_dec
            recommended_pix = self.source_pix*np.cos(max_dec)

            if verbose:
                print("INFO: Maximum declination in southern patch : ",max_dec*180./np.pi, " deg.")
                print("INFO: Recommended pixel size for northern patch : ",recommended_pix, " arcmin")

        else:
            recommended_pix = pix_target_override_arcmin
            
        shape_target,wcs_target = rect_geometry(width_arcmin=width_multiplier*patch_width*60.,
                                                      height_arcmin=height_multiplier*patch_height*60.,
                                                      px_res_arcmin=recommended_pix,yoffset_degree=0.,proj=proj)


        self.target_pix = recommended_pix
        self.wcs_target = wcs_target
        if verbose:
            print("INFO: Source pixel : ",self.source_pix, " arcmin")
        
        if downsample:
            dpix = downsample_pix_arcmin if downsample_pix_arcmin is not None else self.source_pix

            self.shape_final,self.wcs_final = rect_geometry(width_arcmin=width_multiplier*patch_width*60.,
                                              height_arcmin=height_multiplier*patch_height*60.,
                                              px_res_arcmin=dpix,yoffset_degree=0.,proj=proj)
        else:
            self.shape_final = shape_target
            self.wcs_final = wcs_target
        self.downsample = downsample
            
        MapRotator.__init__(self,shape_source,wcs_source,shape_target,wcs_target)

    def rotate(self,imap,**kwargs):
        rotated = MapRotator.rotate(self,imap,**kwargs)

        if self.downsample:
            from enlib import resample
            return enmap.ndmap(resample.resample_fft(rotated,self.shape_final),self.wcs_final)
        else:
            return rotated
    
def get_rotated_pixels(shape_source,wcs_source,shape_target,wcs_target,inverse=False):
    """ Given a source geometry (shape_source,wcs_source)
    return the pixel positions in the target geometry (shape_target,wcs_target)
    if the source geometry were rotated such that its center lies on the center
    of the target geometry.

    WARNING: Only currently tested for a rotation along declination from one CAR
    geometry to another CAR geometry.
    """

    from enlib import coordinates
    
    # what are the center coordinates of each geometries
    center_source = enmap.pix2sky(shape_source,wcs_source,(shape_source[0]/2.,shape_source[1]/2.))
    center_target= enmap.pix2sky(shape_target,wcs_target,(shape_target[0]/2.,shape_target[1]/2.))
    decs,ras = center_source
    dect,rat = center_target

    # what are the angle coordinates of each pixel in the target geometry
    pos_target = enmap.posmap(shape_target,wcs_target)
    lra = pos_target[1,:,:].ravel()
    ldec = pos_target[0,:,:].ravel()
    del pos_target

    # recenter the angle coordinates of the target from the target center to the source center
    if inverse:
        newcoord = coordinates.decenter((lra,ldec),(rat,dect,ras,decs))
    else:
        newcoord = coordinates.recenter((lra,ldec),(rat,dect,ras,decs))
    del lra
    del ldec

    # reshape these new coordinates into enmap-friendly form
    new_pos = np.empty((2,shape_target[0],shape_target[1]))
    new_pos[0,:,:] = newcoord[1,:].reshape(shape_target)
    new_pos[1,:,:] = newcoord[0,:].reshape(shape_target)
    del newcoord

    # translate these new coordinates to pixel positions in the target geometry based on the source's wcs
    pix_new = enmap.sky2pix(shape_source,wcs_source,new_pos)

    return pix_new

def rotate_map(imap,shape_target=None,wcs_target=None,pix_target=None,**kwargs):
    if pix_target is None:
        pix_target = get_rotated_pixels(shape_source,wcs_source,shape_target,wcs_target)
    else:
        assert (shape_target is None) and (wcs_target is None), "Both pix_target and shape_target,wcs_target must not be specified."

    rotmap = enmap.at(imap,pix_target,unit="pix",**kwargs)
    return rotmap
    
                    
### REAL AND FOURIER SPACE ATTRIBUTES

def angmap(shape,wcs,iau=False):
    sgn = -1 if iau else 1
    lmap = enmap.lmap(shape,wcs)
    return sgn*np.arctan2(-lmap[1], lmap[0])

def get_ft_attributes(shape,wcs):
    shape = shape[-2:]
    Ny, Nx = shape
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
    
    return get_real_attributes_generic(Ny,Nx,pixScaleY,pixScaleX)

def get_real_attributes_generic(Ny,Nx,pixScaleY,pixScaleX):
    
    xx =  (np.arange(Nx)-Nx/2.+0.5)*pixScaleX
    yy =  (np.arange(Ny)-Ny/2.+0.5)*pixScaleY
    
    ix = np.mod(np.arange(Nx*Ny),Nx)
    iy = np.arange(Nx*Ny)/Nx
    
    modRMap = np.zeros([Ny,Nx])
    modRMap[iy,ix] = np.sqrt(xx[ix]**2 + yy[iy]**2)
    

    xMap, yMap = np.meshgrid(xx, yy)  # is this the right order?

    return xMap,yMap,modRMap,xx,yy


## MAXLIKE

def diagonal_cov(power2d):
    ny,nx = power2d.shape[-2:]
    Cflat = np.zeros((nx*ny,nx*ny))
    np.fill_diagonal(Cflat,power2d.reshape(-1))
    return Cflat.reshape((ny,nx,ny,nx))


def ncov(shape,wcs,noise_uk_arcmin):
    noise_uK_rad = noise_uk_arcmin*np.pi/180./60.
    normfact = np.sqrt(np.prod(enmap.pixsize(shape,wcs)))
    noise_uK_pixel = noise_uK_rad/normfact
    return np.diag([(noise_uK_pixel)**2.]*np.prod(shape))

def pixcov(shape,wcs,fourierCov):
    fourierCov = fourierCov.astype(np.float32, copy=False)
    bny,bnx = shape
    from numpy.fft import fft2,ifft2

    pcov = fft2((ifft2(fourierCov,axes=(-4,-3))),axes=(-2,-1)).real
    return pcov*bnx*bny/enmap.area(shape,wcs)

def get_lnlike(covinv,instamp):
    Npix = instamp.size
    assert covinv.size==Npix**2
    vec = instamp.reshape((Npix,1))
    ans = np.dot(np.dot(vec.T,covinv),vec)
    assert ans.size==1
    return ans[0,0]

def pixcov_sim(shape,wcs,ps,Nsims,seed=None,mean_sub=True,pad=0):
    if pad>0:
        retmap = enmap.pad(enmap.zeros(shape,wcs), pad, return_slice=False, wrap=False)
        oshape,owcs = retmap.shape,retmap.wcs
    else:
        oshape,owcs = shape,wcs
        
    
    mg = MapGen(oshape,owcs,ps)
    np.random.seed(seed)
    umaps = []
    for i in range(Nsims):
        cmb = mg.get_map()
        if mean_sub: cmb -= cmb.mean()

        if pad>0:
            ocmb = enmap.extract(cmb, shape, wcs)
        else:
            ocmb = cmb
        umaps.append(ocmb.ravel())
        
    pixcov = np.cov(np.array(umaps).T)
    return pixcov


## MAP OPERATIONS



def butterworth(ells,ell0,n):
    return 1./(1.+(ells*1./ell0)**(2.*n))


def get_taper(shape,taper_percent = 12.0,pad_percent = 3.0,weight=None):
    Ny,Nx = shape[-2:]
    if weight is None: weight = np.ones(shape[-2:])
    taper = cosine_window(Ny,Nx,lenApodY=int(taper_percent*min(Ny,Nx)/100.),lenApodX=int(taper_percent*min(Ny,Nx)/100.),padY=int(pad_percent*min(Ny,Nx)/100.),padX=int(pad_percent*min(Ny,Nx)/100.))*weight
    w2 = np.mean(taper**2.)
    return taper,w2

def get_taper_deg(shape,wcs,taper_width_degrees = 1.0,pad_width_degrees = 0.,weight=None):
    Ny,Nx = shape[-2:]
    if weight is None: weight = np.ones(shape[-2:])
    res = resolution(shape,wcs)
    pix_apod = int(taper_width_degrees*np.pi/180./res)
    pix_pad = int(pad_width_degrees*np.pi/180./res)
    taper = cosine_window(Ny,Nx,lenApodY=pix_apod,lenApodX=pix_apod,padY=pix_pad,padX=pix_pad)*weight
    w2 = np.mean(taper**2.)
    return taper,w2


def cosine_window(Ny,Nx,lenApodY=30,lenApodX=30,padY=0,padX=0):
    # Based on a routine by Thibaut Louis
    win=np.ones((Ny,Nx))
    
    i = np.arange(Nx) 
    j = np.arange(Ny)
    ii,jj = np.meshgrid(i,j)

    # ii is array of x indices
    # jj is array of y indices
    # numpy indexes (j,i)

    # xdirection
    if lenApodX>0:
        r=ii.astype(float)-padX
        sel = np.where(ii<=(lenApodX+padX))
        win[sel] = 1./2*(1-np.cos(-np.pi*r[sel]/lenApodX))
        sel = np.where(ii>=((Nx-1)-lenApodX-padX))
        r=((Nx-1)-ii-padX).astype(float)
        win[sel] = 1./2*(1-np.cos(-np.pi*r[sel]/lenApodX))
    # ydirection
    if lenApodY>0:
        r=jj.astype(float)-padY
        sel = np.where(jj<=(lenApodY+padY))
        win[sel] *= 1./2*(1-np.cos(-np.pi*r[sel]/lenApodY))
        sel = np.where(jj>=((Ny-1)-lenApodY-padY))
        r=((Ny-1)-jj-padY).astype(float)
        win[sel] *= 1./2*(1-np.cos(-np.pi*r[sel]/lenApodY))

    win[0:padY,:]=0
    win[:,0:padX]=0
    win[Ny-padY:,:]=0
    win[:,Nx-padX:]=0
    return win

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

## ILC (MAP-LEVEL AND SPEC-LEVEL)

def silc(kmaps,cinv,response=None):
    """Make a simple internal linear combination (ILC) of given fourier space maps at different frequencies
    and an inverse covariance matrix for its variance.

    From Eq 4 of arXiv:1006.5599
    
    Accepts
    -------

    kmaps -- (nfreq,Ny,Nx) array of beam-deconvolved fourier transforms at each frequency
    cinv -- (nfreq,nfreq,Ny,Nx) array of the inverted covariance matrix
    response -- (nfreq,) array of f_nu response factors. Defaults to unity for CMB estimate.

    Returns
    -------

    Fourier transform of ILC estimate, (Ny,Nx) array 
    """

    response = ilc_def_response(response,cinv)
    
    # Get response^T Cinv kmaps
    weighted = ilc_map_term(kmaps,cinv,response)
    # Get response^T Cinv response
    norm = ilc_comb_a_b(response,response,cinv)
    return weighted/norm

def cilc(kmaps,cinv,response_a,response_b):
    """Constrained ILC -- Make a constrained internal linear combination (ILC) of given fourier space maps at different frequencies
    and an inverse covariance matrix for its variance. The component of interest is specified through its f_nu response vector
    response_a. The component to explicitly project out is specified through response_b.

    Derived from Eq 18 of arXiv:1006.5599

    Accepts
    -------

    kmaps -- (nfreq,Ny,Nx) array of beam-deconvolved fourier transforms at each frequency
    cinv -- (nfreq,nfreq,Ny,Nx) array of the inverted covariance matrix
    response_a -- (nfreq,) array of f_nu response factors for component of interest.
    response_b -- (nfreq,) array of f_nu response factors for component to project out.

    Returns
    -------

    Fourier transform of ILC estimate, (Ny,Nx) array 
    """
    
    brb = ilc_comb_a_b(response_b,response_b,cinv)
    arb = ilc_comb_a_b(response_a,response_b,cinv)
    arM = ilc_map_term(kmaps,cinv,response_a)
    brM = ilc_map_term(kmaps,cinv,response_b)
    ara = ilc_comb_a_b(response_a,response_a,cinv)

    numer = brb * arM - arb*brM
    norm = (ara*brb-arb**2.)
    return numer/norm

def ilc_def_response(response,cinv):
    """Default CMB response -- vector of ones"""
    if response is None:
        # assume CMB
        nfreq = cinv.shape[0]
        response = np.ones((nfreq,))
    return response

def ilc_index(ndim):
    """Returns einsum indexing given ndim of cinv.
    If covmat of 1d powers, return single index, else
    return 2 indices for 2D kspace matrix."""
    if ndim==3:
        return "p"
    elif ndim==4:
        return "ij"
    else:
        raise ValueError

def ilc_map_term(kmaps,cinv,response):
    """response^T . Cinv . kmaps """
    return np.einsum('k,kij->ij',response,np.einsum('klij,lij->kij',cinv,kmaps))
    
def silc_noise(cinv,response=None):
    """ Derived from Eq 4 of arXiv:1006.5599"""
    response = ilc_def_response(response,cinv)
    return (1./ilc_comb_a_b(response,response,cinv))

def cilc_noise(cinv,response_a,response_b):
    """ Derived from Eq 18 of arXiv:1006.5599 """
    
    brb = ilc_comb_a_b(response_b,response_b,cinv)
    ara = ilc_comb_a_b(response_a,response_a,cinv)
    arb = ilc_comb_a_b(response_a,response_b,cinv)
    bra = ilc_comb_a_b(response_b,response_a,cinv)

    numer = (brb)**2. * ara + (arb)**2.*brb - brb*arb*arb - arb*brb*bra
    denom = (ara*brb-arb**2.)**2.
    return numer/denom


def ilc_comb_a_b(response_a,response_b,cinv):
    """Return a^T cinv b"""
    pind = ilc_index(cinv.ndim) # either "p" or "ij" depending on whether we are dealing with 1d or 2d power
    return np.einsum('l,l'+pind+'->'+pind,response_a,np.einsum('k,kl'+pind+'->l'+pind,response_b,cinv))

def ilc_cinv(ells,cmb_ps,kbeams,freqs,noises,components,fnoise):
    """
    ells -- either 1D or 2D fourier wavenumbers
    cmb_ps -- Theory C_ell_TT in 1D or 2D fourier space
    kbeams -- 1d or 2d beam transforms
    freqs -- array of floats with frequency bandpasses
    noises -- 1d, 2d or float noise powers (in uK^2-radian^2)
    components -- list of strings representing foreground components recognized by fgGenerator
    fnoise -- A szar.foregrounds.fgNoises object (or derivative) containing foreground power definitions
    """

    nfreqs = len(noises)
    cshape = (nfreqs,nfreqs,1,1) if cmb_ps.ndim==2 else (nfreqs,nfreqs,1)
    Covmat = np.tile(cmb_ps,cshape)

    for i,(kbeam1,freq1,noise1) in enumerate(zip(kbeams,freqs,noises)):
        for j,(kbeam2,freq2,noise2) in enumerate(zip(kbeams,freqs,noises)):
            if i==j:
                Covmat[i,j,:] += noise1/kbeam1**2.
            for component in components:
                Covmat[i,j,:] += fnoise.get_noise(component,freq1,freq2,ells)


    cinv = np.linalg.inv(Covmat.T).T
    return cinv


def minimum_ell(shape,wcs):
    """
    Returns the lowest angular wavenumber of an ndmap
    rounded down to the nearest integer.
    """
    modlmap = enmap.modlmap(shape,wcs)
    min_ell = modlmap[modlmap>0].min()
    return int(min_ell)



def resolution(shape,wcs):
    res = np.min(np.abs(enmap.extent(shape,wcs))/shape[-2:])
    return res


def inpaint_cg(imap,rand_map,mask,power2d,eps=1.e-8):

    """
    by Thibaut Louis

    my_map  -- masked map
    my_map_2  -- random map with same power
    Mask -- mask
    power_spec -- 2d S+N power
    nxside
    nyside
    eps

    """

    assert imap.ndim==2
    nyside,nxside = imap.shape
    
    def apply_px_c_inv_px(my_map):
        
        my_map.shape=(nyside,nxside)
        #apply x_proj
        my_new_map=(1-mask)*my_map
        # change to Fourrier representation
        a_l = fft(my_new_map,axes=[-2,-1])
        # apply inverse power spectrum
        a_l=a_l*1/power2d
        # change back to pixel representation
        my_new_map = ifft(a_l,normalize=True,axes=[-2,-1])
        #Remove the imaginary part
        my_new_map=my_new_map.real
        # apply x_proj
        my_new_map=(1-mask)*my_new_map
        #Array to vector
        my_new_map.shape=(nxside*nyside,)
        my_map.shape=(nxside*nyside,)
        return(my_new_map)
    
    def apply_px_c_inv_py(my_map):
        # apply y_proj
        my_map.shape=(nyside,nxside)
        my_new_map=mask*my_map
        # change to Fourrier representation
        a_l = fft(my_new_map,axes=[-2,-1])
        # apply inverse power spectrum
        a_l=a_l*1/power2d
        # change back to pixel representation
        my_new_map = ifft(a_l,normalize=True,axes=[-2,-1])
        #Remove the imaginary part
        my_new_map=my_new_map.real
        # apply x_proj
        my_new_map=(1-mask)*my_new_map
        #Array to vector
        my_new_map.shape=(nxside*nyside,)
        return(my_new_map)
    
    b=-apply_px_c_inv_py(imap-rand_map)
    
    #Number of iterations
    i_max=2000
    
    #initial value of x
    x=b
    i=0
    
    r=b-apply_px_c_inv_px(x)
    d=r
    
    delta_new=np.inner(r,r)
    delta_o=delta_new
    
    delta_array=np.zeros(shape=(i_max))
    
    while i<i_max and delta_new > eps**2*delta_o:
        # print ""
        # print "number of iterations:", i
        # print ""
        # print "eps**2*delta_o=",eps**2*delta_o
        # print ""
        # print "delta new=",delta_new
        
        q=apply_px_c_inv_px(d)
        alpha=delta_new/(np.inner(d,q))
        x=x+alpha*d
        
        if i/50.<np.int(i/50):
            
            r=b-apply_px_c_inv_px(x)
        else:
            r=r-alpha*q
        
        delta_old=delta_new
        delta_new=np.inner(r,r)
        beta=delta_new/delta_old
        d=r+beta*d
        i=i+1
    
    #print "delta_o=", delta_o
    #print "delta_new=", delta_new
    
    x.shape=(nyside,nxside)
    x_old=x
    x=x+rand_map*(1-mask)
    complete=imap*mask
    rebuild_map=complete+x
    print("Num iterations : ",i)
    return rebuild_map


## WORKING WITH DATA


class NoiseModel(object):

    def __init__(self,splits=None,wmap=None,mask=None,kmask=None,directory=None,spec_smooth_width=2.,skip_beam=True,skip_mask=True,skip_kmask=True,skip_cross=True,iau=False):
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
            fc = FourierCalc(shape,wcs,iau)
            n2d, p2d = noise_from_splits(osplits,fc)
            del osplits
            del splits
            w2 = np.mean(mask**2.)
            n2d *= (1./w2)
            p2d *= (1./w2)

            n2d = np.fft.ifftshift(enmap.smooth_spectrum(np.fft.fftshift(n2d), kernel="gauss", weight="mode", width=spec_smooth_width))
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
        self.ngen = MapGen(self.shape,self.wcs,self.noise2d)
        # self.noise_modulation = 1./np.sqrt(self.wmap)/np.sqrt(np.mean((1./self.wmap)))
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


        
        

    def get_noise_sim(self,seed=None,noise_mod=True):
        nval = self.noise_modulation if noise_mod else 1.
        return self.ngen.get_map(seed=seed,scalar=True) * nval
        
    def add_beam_1d(self,ells,beam_1d_transform):
        modlmap = enmap.modlmap(self.shape[-2:],self.wcs)
        self.kbeam2d = interp1d(ells,beam_1d_transform,bounds_error=False,fill_value=0.)(modlmap)
    def add_beam_2d(self,beam_2d_transform):
        assert self.shape[-2:]==beam_2d_transform.shape
        self.kbeam2d = beam_2d_transform
    
def noise_from_splits(splits,fourier_calc=None,nthread=0):
    

    import itertools

    if fourier_calc is None:
        shape = splits[0].shape
        wcs = splits[0].wcs
        fourier_calc = FourierCalc(shape,wcs)
    
    Nsplits = len(splits)

    # Get fourier transforms of I,Q,U
    ksplits = [fourier_calc.iqu2teb(split, nthread=nthread, normalize=False, rot=False) for split in splits]

    kteb_splits = []
    # Rotate I,Q,U to T,E,B for cross power (not necssary for noise)
    for ksplit in ksplits:
        kteb_splits.append( ksplit.copy())
        if (splits[0].ndim==3 and splits[0].shape[0]==3):
            kteb_splits[-1][...,-2:,:,:] = enmap.map_mul(fourier_calc.rot, kteb_splits[-1][...,-2:,:,:])
            
    # get auto power of I,Q,U
    auto = sum([fourier_calc.power2d(kmap=ksplit)[0] for ksplit in ksplits])/Nsplits

    # do cross powers of I,Q,U
    cross_splits = [y for y in itertools.combinations(ksplits,2)]
    Ncrosses = len(cross_splits)
    assert Ncrosses==(Nsplits*(Nsplits-1)/2)
    cross = sum([fourier_calc.power2d(kmap=ksplit1,kmap2=ksplit2)[0] for (ksplit1,ksplit2) in cross_splits])/Ncrosses

    # do cross powers of T,E,B
    cross_teb_splits = [y for y in itertools.combinations(kteb_splits,2)]
    cross_teb = sum([fourier_calc.power2d(kmap=ksplit1,kmap2=ksplit2)[0] for (ksplit1,ksplit2) in cross_teb_splits])/Ncrosses

    # get noise model for I,Q,U
    noise = (auto-cross)/Nsplits

    # return I,Q,U noise model and T,E,B cross-power
    return noise,cross_teb


### FULL SKY

def enmap_from_healpix_alms(shape,wcs,hp_map_file=None,hp_map=None,ncomp=1,lmax=0,rot="gal,equ",rot_method="alm"):
        """Project a healpix map to an enmap of chosen shape and wcs. The wcs
        is assumed to be in equatorial (ra/dec) coordinates. If the healpix map
        is in galactic coordinates, this can be specified by hp_coords, and a
        slow conversion is done. No coordinate systems other than equatorial
        or galactic are currently supported. Only intensity maps are supported.
        If interpolate is True, bilinear interpolation using 4 nearest neighbours
        is done.

        shape -- 2-tuple (Ny,Nx)
        wcs -- enmap wcs object in equatorial coordinates
        hp_map -- array-like healpix map
        hp_coords -- "galactic" to perform a coordinate transform, "fk5","j2000" or "equatorial" otherwise
        interpolate -- boolean
        
        """
        
        import healpy
        from enlib import coordinates, curvedsky, sharp, utils

        # equatorial to galactic euler zyz angles
        euler = np.array([57.06793215,  62.87115487, -167.14056929])*utils.degree

        # If multiple templates are specified, the output file is
        # interpreted as an output directory.

        print("Loading map...")
        assert ncomp == 1 or ncomp == 3, "Only 1 or 3 components supported"
        dtype = np.float64
        ctype = np.result_type(dtype,0j)
        # Read the input maps
        if hp_map_file is not None:
            print(hp_map_file)
            m = np.atleast_2d(healpy.read_map(hp_map_file, field=tuple(range(0,ncomp)))).astype(dtype)
        else:
            assert hp_map is not None
            m = np.atleast_2d(hp_map).astype(dtype)

        # Prepare the transformation
        print("SHT prep...")

        nside = healpy.npix2nside(m.shape[1])
        lmax  = lmax or 3*nside
        minfo = sharp.map_info_healpix(nside)
        ainfo = sharp.alm_info(lmax)
        sht   = sharp.sht(minfo, ainfo)
        alm   = np.zeros((ncomp,ainfo.nelem), dtype=ctype)
        # Perform the actual transform
        print("SHT...")
        sht.map2alm(m[0], alm[0])
        
        if ncomp == 3:
                sht.map2alm(m[1:3],alm[1:3], spin=2)
        del m


        if rot and rot_method != "alm":
                print("rotate...")
                pmap = posmap(shape, wcs)
                s1,s2 = rot.split(",")
                opos = coordinates.transform(s2, s1, pmap[::-1], pol=ncomp==3)
                pmap[...] = opos[1::-1]
                if len(opos) == 3: psi = -opos[2].copy()
                del opos
                res  = curvedsky.alm2map_pos(alm, pmap)
                if ncomp==3:
                        res[1:3] = rotate_pol(res[1:3], psi)
        else:
                print(" alm rotate...")
                # We will project directly onto target map if possible
                if rot:
                        s1,s2 = rot.split(",")
                        if s1 != s2:
                                print("rotating alm...")
                                # Note: rotate_alm does not actually modify alm
                                # if it is single precision
                                if s1 == "gal" and (s2 == "equ" or s2 == "cel"):
                                        healpy.rotate_alm(alm, euler[0], euler[1], euler[2])
                                elif s2 == "gal" and (s1 == "equ" or s1 == "cel"):
                                        healpy.rotate_alm(alm,-euler[2],-euler[1],-euler[0])
                                else:
                                        raise NotImplementedError
                        print("done rotating alm...")
                res = enmap.zeros((len(alm),)+shape[-2:], wcs, dtype)
                res = curvedsky.alm2map(alm, res)
        return res


def enmap_from_healpix(shape,wcs,hp_map,hp_coords="galactic",interpolate=True):
        """Project a healpix map to an enmap of chosen shape and wcs. The wcs
        is assumed to be in equatorial (ra/dec) coordinates. If the healpix map
        is in galactic coordinates, this can be specified by hp_coords, and a
        slow conversion is done. No coordinate systems other than equatorial
        or galactic are currently supported. Only intensity maps are supported.
        If interpolate is True, bilinear interpolation using 4 nearest neighbours
        is done.

        shape -- 2-tuple (Ny,Nx)
        wcs -- enmap wcs object in equatorial coordinates
        hp_map -- array-like healpix map
        hp_coords -- "galactic" to perform a coordinate transform, "fk5","j2000" or "equatorial" otherwise
        interpolate -- boolean
        
        """
        
        import healpy as hp
        from astropy.coordinates import SkyCoord
        import astropy.units as u


        eq_coords = ['fk5','j2000','equatorial']
        gal_coords = ['galactic']
        
        imap = enmap.zeros(shape,wcs)
        Ny,Nx = shape

        pixmap = enmap.pixmap(shape,wcs)
        y = pixmap[0,...].T.ravel()
        x = pixmap[1,...].T.ravel()
        posmap = enmap.posmap(shape,wcs)

        ph = posmap[1,...].T.ravel()
        th = posmap[0,...].T.ravel()

        if hp_coords.lower() not in eq_coords:
                # This is still the slowest part. If there are faster coord transform libraries, let me know!
                assert hp_coords.lower() in gal_coords
                gc = SkyCoord(ra=ph*u.degree, dec=th*u.degree, frame='fk5')
                gc = gc.transform_to('galactic')
                phOut = gc.l.deg
                thOut = gc.b.deg
        else:
                thOut = th
                phOut = ph

        thOut = np.pi/2. - thOut #polar angle is 0 at north pole

        # Not as slow as you'd expect
        if interpolate:
                imap[y,x] = hp.get_interp_val(hp_map, thOut, phOut)
        else:
                ind = hp.ang2pix( hp.get_nside(hp_map), thOut, phOut )
                imap[:] = 0.
                imap[[y,x]]=hp_map[ind]

                
                
        return enmap.ndmap(imap,wcs)




def cutout_gnomonic(map,rot=None,coord=None,
             xsize=200,ysize=None,reso=1.5,
             nest=False,remove_dip=False,
             remove_mono=False,gal_cut=0,
             flip='astro'):
    """Obtain a cutout from a healpix map (given as an array) in Gnomonic projection.

    Derivative of healpy.visufunc.gnomonic

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


def whiteNoise2D(noiseLevels,beamArcmin,modLMap,TCMB = 2.7255e6,lknees=None,alphas=None,beamFile=None, \
                 noiseFuncs=None):
    # Returns 2d map noise in units of uK**0.
    # Despite the name of the function, there are options to add
    # a simplistic atmosphere noise model

    # If no atmosphere is specified, set lknee to zero and alpha to 1
    if lknees is None:
        lknees = (np.array(noiseLevels)*0.).tolist()
    if alphas is None:
        alphas = (np.array(noiseLevels)*0.+1.).tolist()

    # we'll loop over it, so make it a list if nothing is specified
    if noiseFuncs is None: noiseFuncs = [None]*len(noiseLevels)

        
    # if one of the noise files is not specified, we will need a beam
    if None in noiseFuncs:
        
        if beamFile is not None:
            ell, f_ell = np.transpose(np.loadtxt(beamFile))[0:2,:]
            filt = 1./(np.array(f_ell)**2.)
            bfunc = interp1d(ell,f_ell,bounds_error=False,fill_value=np.inf)
            filt2d = bfunc(modLMap)
        else:
            Sigma = beamArcmin *np.pi/60./180./ np.sqrt(8.*np.log(2.))  # radians
            filt2d = np.exp(-(modLMap**2.)*Sigma*Sigma)


    retList = []

    for noiseLevel,lknee,alpha,noiseFunc in zip(noiseLevels,lknees,alphas,noiseFuncs):
        if noiseFunc is not None:
            retList.append(nfunc(modLMap))
        else:
        
            noiseForFilter = (np.pi / (180. * 60))**2.  * noiseLevel**2. / TCMB**2.  

            if lknee>0.:
                atmFactor = (lknee*np.nan_to_num(1./modLMap))**(-alpha)
            else:
                atmFactor = 0.
                
            with np.errstate(divide='ignore'):
                retList.append(noiseForFilter*(atmFactor+1.)*np.nan_to_num(1./filt2d.copy()))

    return retList


### INTERFACES WITH EXPERIMENTS



class ACTMapReader(object):

    def __init__(self,config_yaml_path):

        with open(config_yaml_path, 'r') as ymlfile:
            self._cfg = yaml.load(ymlfile,Loader=yaml.BaseLoader)

        self.map_root = self._cfg['map_root']
        self.beam_root = self._cfg['beam_root']

        
        self.boxes = {}
        for key in self._cfg['patches']:
            self.boxes[key] = bounds_from_list(io.list_from_string(self._cfg['patches'][key]))


    def sel_from_region(self,region,shape=None,wcs=None):
        if shape is None: shape = self.shape
        if wcs is None: wcs = self.wcs
        if region is None:
            selection = None
        elif isinstance(region, six.string_types):
            selection = self.boxes[region] #enmap.slice_from_box(shape,wcs,self.boxes[region])
        else:
            selection = region #enmap.slice_from_box(shape,wcs,region)
        return selection
    
    def patch_bounds(self,patch):
        return (np.array([float(x) for x in self._cfg['patches'][patch].split(',')])*np.pi/180.).reshape((2,2))
        


class SigurdCoaddReader(ACTMapReader):
    
    def __init__(self,config_yaml_path):
        ACTMapReader.__init__(self,config_yaml_path)
        eg_file = self._fstring(split=-1,freq="150",day_night="daynight",planck=True)
        self.shape,self.wcs = enmap.read_fits_geometry(eg_file)


    def _config_tag(self,freq,day_night,planck):
        planckstr = "_planck" if planck else ""
        return freq+planckstr+"_"+day_night


    def get_ptsrc_mask(self,region=None):
        selection = self.sel_from_region(region)
        fstr = self.map_root+"s16/coadd/pointSourceMask_full_all.fits"
        fmap = enmap.read_fits(fstr,box=selection)
        return fmap
    
    def get_survey_mask(self,region=None):
        selection = self.sel_from_region(region)
        self.map_root+"s16/coadd/surveyMask_full_all.fits"
        fmap = enmap.read_fits(fstr,box=selection)
        return fmap
    
    def get_map(self,split,freq="150",day_night="daynight",planck=True,region=None,weight=False,get_identifier=False):

        
        fstr = self._fstring(split,freq,day_night,planck,weight)
        cal = float(self._cfg['coadd'][self._config_tag(freq,day_night,planck)]['cal']) if not(weight) else 1.

        selection = self.sel_from_region(region)
        fmap = enmap.read_fits(fstr,box=selection)*np.sqrt(cal)

        if get_identifier:
            identifier = '_'.join(map(str,[freq,day_night,"planck",planck]))
            return fmap,identifier
        else:
            return fmap
        

    def _fstring(self,split,freq="150",day_night="daynight",planck=True,weight=False):
        # Change this function if the map naming scheme changes
        splitstr = "" if split<0 or split>3 else "_2way_"+str(split)
        weightstr = "div" if weight else "map"
        return self.map_root+"s16/coadd/f"+freq+"_"+day_night+"_all"+splitstr+"_"+weightstr+"_mono.fits"


    def get_beam(self,freq="150",day_night="daynight",planck=True):
        beam_file = self.beam_root+self._cfg['coadd'][self._config_tag(freq,day_night,planck)]['beam']
        ls,bells = np.loadtxt(beam_file,usecols=[0,1],unpack=True)
        return ls, bells


class SigurdBNReader(ACTMapReader):
    
    def __init__(self,config_yaml_path):
        ACTMapReader.__init__(self,config_yaml_path)
        eg_file = self._fstring(split=-1,season="s15",array="pa1",freq="150",day_night="night")
        self.shape,self.wcs = enmap.read_fits_geometry(eg_file)

    def get_map(self,split,season,array,freq="150",day_night="night",region=None,weight=False,get_identifier=False):

        patch = "boss"
        fstr = self._fstring(split,season,array,freq,day_night,weight)
        cal = float(self._cfg[season][array][freq][patch][day_night]['cal']) if not(weight) else 1.
        selection = self.sel_from_region(region)
        fmap = enmap.read_fits(fstr,box=selection)*np.sqrt(cal)

        if get_identifier:
            identifier = '_'.join(map(str,[freq,day_night,"planck",planck]))
            return fmap,identifier
        else:
            return fmap
        
    def _fstring(self,split,season,array,freq="150",day_night="night",weight=False):
        # Change this function if the map naming scheme changes
        splitstr = "_4way_tot_" if split<0 or split>3 else "_4way_"+str(split)+"_"
        weightstr = "div" if weight else "map0500"
        return self.map_root+"mr2/"+season+"/boss_north/"+season+"_boss_"+array+"_f"+freq+"_"+day_night+"_nohwp"+splitstr+"sky_"+weightstr+"_mono.fits"


    def get_beam(self,season,array,freq="150",day_night="night"):
        patch = "boss"
        beam_file = self.beam_root+self._cfg[season][array][freq][patch][day_night]['beam']
        ls,bells = np.loadtxt(beam_file,usecols=[0,1],unpack=True)
        return ls, bells

    
class SimoneC7V5Reader(ACTMapReader):
    
    def __init__(self,config_yaml_path):
        ACTMapReader.__init__(self,config_yaml_path)
        
    def get_beam(self,season,patch,array,freq="150",day_night="night"):
        beam_file = self.beam_root+self._cfg[season][array][freq][patch][day_night]['beam']
        ls,bells = np.loadtxt(beam_file,usecols=[0,1],unpack=True)
        return ls, bells
    def get_map(self,split,season,patch,array,freq="150",day_night="night",full_map=False,weight=False,get_identifier=False,t_only=False):

        maps = []
        maplist = ['srcfree_I','Q','U'] if not(t_only) else ['srcfree_I']
        for pol in maplist if not(weight) else [None]:
            fstr = self._hstring(season,patch,array,freq,day_night) if weight else self._fstring(split,season,patch,array,freq,day_night,pol)
            cal = float(self._cfg[season][array][freq][patch][day_night]['cal']) if not(weight) else 1.
            fmap = enmap.read_map(fstr)*np.sqrt(cal) 
            if not(full_map):
                bounds = self.patch_bounds(patch) 
                retval = fmap.submap(bounds)
            else:
                retval = fmap
            maps.append(retval)
        retval = enmap.ndmap(np.stack(maps),maps[0].wcs) if not(weight) else maps[0]

        if get_identifier:
            identifier = '_'.join(map(str,[season,patch,array,freq,day_night]))
            return retval,identifier
        else:
            return retval
        

    def _fstring(self,split,season,patch,array,freq,day_night,pol):
        # Change this function if the map naming scheme changes
        splitstr = "set0123" if split<0 or split>3 else "set"+str(split)
        return self.map_root+"c7v5/"+season+"/"+patch+"/"+season+"_mr2_"+patch+"_"+array+"_f"+freq+"_"+day_night+"_"+splitstr+"_wpoly_500_"+pol+".fits"

    def _hstring(self,season,patch,array,freq,day_night):
        splitstr = "set0123"
        return self.map_root+"c7v5/"+season+"/"+patch+"/"+season+"_mr2_"+patch+"_"+array+"_f"+freq+"_"+day_night+"_"+splitstr+"_hits.fits"



### STACKING

class Stacker(object):

    def __init__(self,imap,arcmin_width):
        self.imap = imap
        res = np.min(imap.extent()/imap.shape[-2:])*180./np.pi*60.
        self.Npix = int(arcmin_width/res)*1.
        if self.Npix%2==0: self.Npix += 1
        self.shape,self.wcs = enmap.geometry(pos=(0.,0.),res=res/(180./np.pi*60.),shape=(self.Npix,self.Npix))

    def cutout(self,ra,dec):   
        iy,ix = self.imap.sky2pix(coords=(dec,ra))
        cutout = self.imap[int(iy-self.Npix/2):int(iy+self.Npix/2),int(ix-self.Npix/2):int(ix+self.Npix/2)]
        assert self.shape==cutout.shape
        return enmap.ndmap(cutout,self.wcs)


    
def cutout(imap,arcmin_width,ra=None,dec=None,iy=None,ix=None,pad=1,corner=False,preserve_wcs=False):
    Ny,Nx = imap.shape

    # see enmap.sky2pix for "corner" options
    if corner:
        fround = lambda x : int(math.floor(x))
    else:
        fround = lambda x : int(np.round(x))
    #fround = lambda x : int(x)

    if (iy is None) or (ix is None):
        iy,ix = imap.sky2pix(coords=(dec,ra),corner=corner)

    
    res = np.min(imap.extent()/imap.shape[-2:])*180./np.pi*60.
    Npix = int(arcmin_width/res)
    if fround(iy-Npix/2)<pad or fround(ix-Npix/2)<pad or fround(iy+Npix/2)>(Ny-pad) or fround(ix+Npix/2)>(Nx-pad): return None
    cutout = imap[fround(iy-Npix/2.+0.5):fround(iy+Npix/2.+0.5),fround(ix-Npix/2.+0.5):fround(ix+Npix/2.+0.5)]
    #cutout = imap[fround(iy-Npix/2):fround(iy+Npix/2),fround(ix-Npix/2):fround(ix+Npix/2)]
    #print(fround(iy-Npix/2.+0.5),fround(iy+Npix/2.+0.5),fround(ix-Npix/2.+0.5),fround(ix+Npix/2.+0.5))
    if preserve_wcs:
        return cutout
    else:
        shape,wcs = enmap.geometry(pos=(0.,0.),res=res/(180./np.pi*60.),shape=cutout.shape)
        return enmap.ndmap(cutout,wcs)


def aperture_photometry(instamp,aperture_radius,annulus_width,modrmap=None):
    # inputs in radians
    stamp = instamp.copy()
    if modrmap is None: modrmap = stamp.modrmap()
    mean = stamp[np.logical_and(modrmap>aperture_radius,modrmap<(aperture_radius+annulus_width))].mean()
    stamp -= mean
    pix_scale=resolution(stamp.shape,stamp.wcs)*(180*60)/np.pi
    flux = stamp[modrmap<aperture_radius].sum()*pix_scale**2
    return flux * enmap.area(stamp.shape,stamp.wcs )/ np.prod(stamp.shape[-2:])**2.




class InterpStack(object):

    def __init__(self,arc_width,px,proj="car"):
        if proj.upper()!="CAR": raise NotImplementedError
        
        self.shape_target,self.wcs_target = rect_geometry(width_arcmin=arc_width,
                                                      height_arcmin=arc_width,
                                                      px_res_arcmin=px,yoffset_degree=0.
                                                      ,xoffset_degree=0.,proj=proj)
                 
        center_target = enmap.pix2sky(self.shape_target,self.wcs_target,(self.shape_target[0]/2.,self.shape_target[1]/2.))
        self.dect,self.rat = center_target
        # what are the angle coordinates of each pixel in the target geometry
        pos_target = enmap.posmap(self.shape_target,self.wcs_target)
        self.lra = pos_target[1,:,:].ravel()
        self.ldec = pos_target[0,:,:].ravel()
        del pos_target

        self.arc_width = arc_width 


    def cutout(self,imap,ra,dec,**kwargs):
        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)

        box = self._box_from_ra_dec(ra_rad,dec_rad)
        submap = imap.submap(box,inclusive=True)
        return self._rot_cut(submap,ra_rad,dec_rad,**kwargs)
    
    def cutout_from_file(self,imap_file,shape,wcs,ra,dec,**kwargs):
        ra_rad = np.deg2rad(ra)
        dec_rad = np.deg2rad(dec)

        box = self._box_from_ra_dec(ra_rad,dec_rad)
        # print(ra_rad*180./np.pi,dec_rad*180./np.pi,box*180./np.pi)
        # selection = slice_from_box(shape,wcs,box)
        # print(selection)
        submap = enmap.read_fits(imap_file,box=box)#sel=selection)
        print(submap.shape)
        # sys.exit()
        # submap = enmap.read_fits(imap_file,sel=selection)
        # io.plot_img(submap,io.dout_dir+"scut.png",high_res=True)

        # try:
        #     self.count +=1
        # except:
        #     self.count = 0

        # print(ra,dec)
        # io.plot_img(submap,io.dout_dir+"stest_qwert_"+str(self.count)+".png")


        
        return self._rot_cut(submap,ra_rad,dec_rad,**kwargs)

    def _box_from_ra_dec(self,ra_rad,dec_rad):

        
        # CAR
        coord_width = np.deg2rad(self.arc_width/60.)#np.cos(dec_rad)/60.)
        coord_height = np.deg2rad(self.arc_width/60.)

        box = np.array([[dec_rad-coord_height/2.,ra_rad-coord_width/2.],[dec_rad+coord_height/2.,ra_rad+coord_width/2.]])

        return box

        
    def _rot_cut(self,submap,ra_rad,dec_rad,**kwargs):
        from enlib import coordinates
    
        if submap.shape[0]<1 or submap.shape[1]<1:
            return None
        
        
        newcoord = coordinates.recenter((self.lra,self.ldec),(self.rat,self.dect,ra_rad,dec_rad))

        # reshape these new coordinates into enmap-friendly form
        new_pos = np.empty((2,self.shape_target[0],self.shape_target[1]))
        new_pos[0,:,:] = newcoord[1,:].reshape(self.shape_target)
        new_pos[1,:,:] = newcoord[0,:].reshape(self.shape_target)
        del newcoord
        
        # translate these new coordinates to pixel positions in the target geometry based on the source's wcs
        pix_new = enmap.sky2pix(submap.shape,submap.wcs,new_pos)

        rotmap = enmap.at(submap,pix_new,unit="pix",**kwargs)
        assert rotmap.shape[-2:]==self.shape_target[-2:]

        
        return rotmap
        





def interpolate_grid(inGrid,inY,inX,outY,outX,regular=True,kind="cubic",kx=3,ky=3,**kwargs):
    '''
    if inGrid is [j,i]
    Assumes inY is along j axis
    Assumes inX is along i axis
    Similarly for outY/X
    '''

    if regular:
        interp_spline = RectBivariateSpline(inY,inX,inGrid,kx=kx,ky=ky,**kwargs)
        outGrid = interp_spline(outY,outX)
    else:
        interp_spline = interp2d(inX,inY,inGrid,kind=kind,**kwargs)
        outGrid = interp_spline(outX,outY)
    

    return outGrid
    


class MatchedFilter(object):

    def __init__(self,shape,wcs,template=None,noise_power=None):
        area = enmap.area(shape,wcs)
        self.normfact = area / (np.prod(shape))**2
        if noise_power is not None: self.n2d = noise_power
        if template is not None: self.ktemp = enmap.fft(template,normalize=False)

        

    def apply(self,imap=None,kmap=None,template=None,ktemplate=None,noise_power=None,kmask=None):
        if kmap is None:
            kmap = enmap.fft(imap,normalize=False)
        else:
            assert imap is None

        if kmask is None: kmask = kmap.copy()*0.+1.
        n2d = self.n2d if noise_power is None else noise_power
        if ktemplate is None:
            ktemp = self.ktemp if template is None else enmap.fft(template,normalize=False)
        else:
            ktemp = ktemplate
            
        phi_un = np.nansum(ktemp.conj()*kmap*self.normfact*kmask/n2d).real 
        phi_var = 1./np.nansum(ktemp.conj()*ktemp*self.normfact*kmask/n2d).real 
        return phi_un*phi_var, phi_var



def mask_center(imap):
    Ny,Nx = imap.shape
    assert Ny==Nx
    N = Ny
    if N%2==1:
        imap[N/2,N/2] = np.nan
    else:
        imap[N/2,N/2] = np.nan
        imap[N/2-1,N/2] = np.nan
        imap[N/2,N/2-1] = np.nan
        imap[N/2-1,N/2-1] = np.nan
    

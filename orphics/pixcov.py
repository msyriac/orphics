from __future__ import print_function
import numpy as np
from enlib import enmap,resample,bench

"""
Utilities that manipulate pixel-pixel covariance
matrices for small geometries.

Applications include:
1. fast brute force inpainting of circular holes
2. maximum-likelihood CMB halo lensing

"""

### ENMAP HELPER FUNCTIONS AND CLASSES
# These have been copied from enlib.jointmap since that module
# requires compilation (but these functions don't)
def map_ifft(x): return enmap.ifft(x).real
def corrfun_thumb(corr, n):
	tmp = np.roll(np.roll(corr, n, -1)[...,:2*n], n, -2)[...,:2*n,:]
	return np.roll(np.roll(tmp, -n, -1), -n, -2)

def corr_to_mat(corr, n):
	res = enmap.zeros([n,n,n,n],dtype=corr.dtype)
	for i in range(n):
		tmp = np.roll(corr, i, 0)[:n,:]
		for j in range(n):
			res[i,j] = np.roll(tmp, j, 1)[:,:n]
	return res
def ps2d_to_mat(ps2d, n):
	corrfun = map_ifft(ps2d+0j)/(ps2d.shape[-2]*ps2d.shape[-1])**0.5
	thumb   = corrfun_thumb(corrfun, n)
	mat     = corr_to_mat(thumb, n)
	return mat

###########

def rotate_pol_power(shape,wcs,cov,iau=False,inverse=False):
    """Rotate a 2D power spectrum from TQU to TEB (inverse=False) or
    back (inverse=True). cov is a (3,3,Ny,Nx) 2D power spectrum.
    WARNING: This function is duplicated from orphics.maps to make 
    this module independent. Ideally, it should be implemented in
    enlib.enmap.
    """
    rot = np.zeros((3,3,cov.shape[-2],cov.shape[-1]))
    rot[0,0,:,:] = 1
    prot = enmap.queb_rotmat(enmap.lmap(shape,wcs), inverse=inverse, iau=iau)
    rot[1:,1:,:,:] = prot
    Rt = np.transpose(rot, (1,0,2,3))
    tmp = np.einsum("ab...,bc...->ac...",rot,cov)
    rp2d = np.einsum("ab...,bc...->ac...",tmp,Rt)    
    return rp2d

def resolution(shape,wcs):
    res = np.min(np.abs(enmap.extent(shape,wcs))/shape[-2:])
    return res


#####
# General pixcov routines


def stamp_pixcov_from_theory(N,cmb2d_TEB,n2d_IQU=0.,beam2d=1.,iau=False):
    """Return the pixel covariance for a stamp N pixels across given the 2D IQU CMB power spectrum,
    2D beam template and 2D IQU noise power spectrum.
    """
    n2d = n2d_IQU
    assert n2d.ndim==4
    ncomp = n2d.shape[0]
    assert n2d.shape[1]==ncomp
    assert ncomp==3 or ncomp==1
    cmb2d = cmb2d_TEB
    
    wcs = n2d.wcs
    shape = n2d.shape[-2:]

    if ncomp==3: cmb2d = rotate_pol_power(shape,wcs,cmb2d_TEB,iau=iau,inverse=True)
    p2d = cmb2d*beam2d**2.+n2d
    return fcov_to_rcorr(shape,wcs,p2d,N)

def fcov_to_rcorr(shape,wcs,p2d,N):
    """Convert a 2D PS into a pix-pix covariance
    """
    ncomp = p2d.shape[0]
    p2d *= np.prod(shape[-2:])/enmap.area(shape,wcs)
    ocorr = enmap.zeros((ncomp,ncomp,N*N,N*N),wcs)
    for i in range(ncomp):
        for j in range(i,ncomp):
            dcorr = ps2d_to_mat(p2d[i,j].copy(), N).reshape((N*N,N*N))
            ocorr[i,j] = dcorr.copy()
            if i!=j: ocorr[j,i] = dcorr.copy()
    return ocorr


# Inpainting routines

def make_geometry(shape,wcs,hole_radius,cmb2d_TEB=None,n2d_IQU=None,context_width=None,n=None,beam2d=None,deproject=True,iau=False,res=None,tot_pow2d=None,store_pcov=False):

    """
    Make covariances for brute force maxlike inpainting of CMB maps.
    Eq 3 of arXiv:1109.0286

    shape,wcs -- enmap geometry of big map
    cmb2d_TEB -- (ncomp,ncomp,Ny,Nx) 2D CMB power in physical units
    n2d_IQU -- (ncomp,ncomp,Ny,Nx) 2D noise power in physical units
    hole_radius in radians
    context_width in radians or n as number of pixels
    beam2d -- (Ny,Nx) 2D beam template
    tot_pow2d -- (ncomp,ncomp,Ny,Nx) 2D total IQU power in physical units (includes CMB, beam and noise). If this argument is provided
                 cmb2d_TEB, n2d_IQU and beam2d will be ignored.
    deproject -- whether to deproject common mode
    iau -- whether to use IAU convention for polarization
    res -- specify resolution in radians instead of inferring from enmap geometry


    """

    # Make a flat stamp geometry
    if res is None: res = np.min(np.abs(enmap.extent(shape,wcs))/shape[-2:])
    if n is None: n = int(context_width/res)

    # Do we have polarization?
    n2d = n2d_IQU
    ncomp = n2d.shape[0]
    assert ncomp==1 or ncomp==2 or ncomp==3

    # Select the hole (m1) and context(m2) across all components
    m1,m2 = get_geometry_regions(ncomp,n,res,hole_radius)

    # Get the pix-pix covariance on the stamp geometry given CMB theory, beam and 2D noise on the big map
    if tot_pow2d is not None:
            pcov = fcov_to_rcorr(shape,wcs,tot_pow2d,n)
    else:
            pcov = stamp_pixcov_from_theory(n,cmb2d_TEB,n2d,beam2d=beam2d,iau=iau)
    # Make sure that the pcov is in the right order vector(I,Q,U)
    pcov = np.transpose(pcov,(0,2,1,3))
    pcov = pcov.reshape((ncomp*n**2,ncomp*n**2))

    # Invert
    Cinv = np.linalg.inv(pcov)
    
    # Woodbury deproject common mode
    if deproject:
        # Deproject I,Q,U common mode separately
        #u = (np.zeros((n*n,ncomp,ncomp))+np.eye(ncomp)).reshape(n*n*ncomp,ncomp)
        u = np.zeros((n*n*ncomp,ncomp))
        for i in range(ncomp):
            u[i*n*n:(i+1)*n*n,i] = 1
        #u = np.ones((ncomp*n**2,1)) # Deproject mode common to all of I,Q,U
        Cinvu = np.linalg.solve(pcov,u)
        precalc = np.dot(Cinvu,np.linalg.solve(np.dot(u.T,Cinvu),u.T))
        correction = np.dot(precalc,Cinv)
        Cinv -= correction
    
    # Get matrices for maxlike solution Eq 3 of arXiv:1109.0286
    cslice = Cinv[m1][:,m1]
    mul2 = Cinv[m1][:,m2]
    mean_mul = -np.linalg.solve(cslice,mul2)
    cov = np.linalg.inv(Cinv[m1][:,m1])
    cov_root = utils.eigpow(cov,0.5)

    geometry = {}
    geometry['covsqrt'] = cov_root
    geometry['meanmul'] = mean_mul
    geometry['n'] = n
    geometry['res'] = res
    geometry['ncomp'] = ncomp
    geometry['hole_radius'] = hole_radius
    if store_pcov: geometry['pcov'] = pcov

    return geometry

def paste(stamp,m,paste_this):
    '''Paste the result of an inpaint operation into a rectangular
    np array

    stamp  - a numpy array with the shape of your cutout stamp
    m               - a 1d boolean array that specifies where in the unraveled
                      cutout stamp the hole is
    paste_this - the result of an inpaint operation, say from fill_hole
    '''
    a = stamp.copy()
    a.reshape(-1)[m] = paste_this.copy()
    a.reshape(stamp.shape)
    return a


def mask_map(imap,iys,ixs,hole_arc,hole_frac=0.6):
    shape,wcs = imap.shape,imap.wcs
    Ny,Nx = shape[-2:]
    px = resolution(shape,wcs)*60.*180./np.pi
    hole_n = int(round(hole_arc/px))
    hole_ny = hole_nx = hole_n
    oshape,owcs = enmap.geometry(pos=(0.,0.),shape=(2*hole_n,2*hole_n),res=px*np.pi/180./60.)
    modrmap = enmap.modrmap(oshape,owcs)
    mask = enmap.ones(shape,wcs)
    
    for iy,ix in zip(iys,ixs):
        if iy<=hole_n or ix<=hole_n or iy>=(Ny-hole_n) or ix>=(Nx-hole_n): continue
        vslice = imap[np.int(iy-hole_ny):np.int(iy+hole_ny),np.int(ix-hole_nx):np.int(ix+hole_nx)]
        if np.any(vslice.shape!=oshape): continue
        vslice[modrmap<(hole_frac*hole_arc)*np.pi/180./60.] = np.nan # !!!! could cause a bias
        mask[np.int(iy-hole_ny):np.int(iy+hole_ny),np.int(ix-hole_nx):np.int(ix+hole_nx)][modrmap<hole_arc*np.pi/180./60.] = 0
        
    return mask


def inpaint(imap,coords_deg,do_pols,hole_radius_arcmin=5.,npix_context=60,resolution_arcmin=0.5,
            cmb2d_TEB=None,n2d_IQU=None,beam2d=None,deproject=True,iau=False,tot_pow2d=None,
            geometry_tags=None,geometry_dict=None):
    """Inpaint I, Q and U maps jointly accounting for their covariance using brute-force pre-calculated
    pixel covariance matrices.
    imap -- (ncomp,Ny,Nx) map to be filled, where ncomp is 1 or 3
    coords_deg -- (nobj,2) array where nobj is number of objects to inpaint, 
               and the columns are dec and ra in degrees
    do_pols -- (nobj,) array of booleans whether to inpaint Q/U map for those objects. 
               Should be true for polarized sources.

    # BASIC/DEBUG USAGE : only allows for single geometry for all objects, and will do slow calculation here.

    hole_radius_arcmin -- arcminute radius of hole for all objects
    npix_context       -- number of pixels for how wide the context region should be
    resolution_arcmin  -- arcminute pixel width. If None, will be determined from the map.
    cmb2d_TEB -- (ncomp,ncomp,Ny,Nx) 2D CMB power in physical units
    n2d_IQU -- (ncomp,ncomp,Ny,Nx) 2D noise power in physical units
    beam2d -- (Ny,Nx) 2D beam template
    tot_pow2d -- (ncomp,ncomp,Ny,Nx) 2D total IQU power in physical units (includes CMB, beam and noise). If this argument is provided,
                 cmb2d_TEB, n2d_IQU and beam2d will be ignored.
    deproject -- whether to deproject common mode (you almost definitely should)
    iau -- whether to use IAU convention for polarization

    # ADVANCED USAGE (recommended!) : allows for multiple pre-calculated geometries. In a full pipeline,
    these geometries should be pre-calculated once and used for all simulations.
    geometry_tags -- (nobj,) list of strings indicating key of dictionary 
                      geometry_dict that corresponds to the hole+context geometry to use.
    geometry_dicts -- dict mapping tags to geometry dicts. Use make_geometry to pre-calculate these.
                       e.g. if geometry_tags = [...,'ptsrc5arcmin','ptsrc5arcmin',...]
                       then geometry_dicts['ptsrc5arcmin'] should be a geometry dict such that
                       geometry_dicts['ptsrc5arcmin']['meanmul'] and geometry_dicts['ptsrc5arcmin']['covsqrt'] exist.

    """
    #for gtags in geometry_tags:
        

    
def inpaint_map(imap,ras,decs,radii_tags,radii_dict,tot_power_2d,seed=None):
    """
    Brute-force inpaints a map in circular regions.

    imap -- (Ny,Nx) enmap
    ras  -- list of RA of centers in degrees, length M
    decs -- list of DEC of centers in degrees, length M
    radii -- list of strings specifying radius tag for each object, length M
    radii_dict -- dict mapping radius tag string to float value of radius in arcminutes, length K
                  Most expensive operation scales with K.
    tot_power_2d -- total power in map in physical units (e.g. uK^2 radians)

    """


    class G:
        pass

    geometries = {}
    tags = radii_dict.keys()
    for key in tags:
        geometries[key] = calculate_circular_geometry(shape,wcs)





def get_geometry_regions(ncomp,n,res,hole_radius):
    tshape,twcs = enmap.geometry(pos=(0,0),shape=(n,n),res=res,proj='car')
    modrmap = enmap.modrmap(tshape,twcs)
    
    # Select the hole (m1) and context(m2) across all components
    amodrmap = np.repeat(modrmap.reshape((1,n,n)),ncomp,0)
    m1 = np.where(amodrmap.reshape(-1)<hole_radius)[0]
    m2 = np.where(amodrmap.reshape(-1)>=hole_radius)[0]
    
    return m1,m2




def fill_hole(masked_stamp,meanMatrix,holeArc,m1,m2,covRoot=None):
    '''Returns the result of an inpaint operation as a 1d unraveled vector

    Arguments
    ---------

    masked_liteMap_stamp - the cutout stamp that contains a masked hole and
                           unmasked context
    meanMatrix           - an (nh,nc) matrix. See docs for make_circular_geometry
    holeArc              - radius of masked hole in arcminutes
    m1                   - a 1d boolean selecting the hole region on an unraveled stamp
    m2                   - a 1d boolean selecting the context region on an unraveled stamp
    covRoot              - the square root of the covariance matrix inside the hole. See
                           docs for make_circular_geometry. If unspecified, the random
                           realization returned is zero.

    Returns
    -------

    mean  -  a 1d (nh) vector containing the mean inpainted value constrained according
             to the context
    rand  -  a 1d (nh) vector containing a random realization inside the hole
    sim   -  a 1d (nh) vector containing the sum of mean and rand

    '''

    mean = np.dot(meanMatrix,masked_stamp.reshape(-1)[m2])
    r = np.random.normal(0.,1.,size=(m1.size))
    if covRoot is not None:
        rand = np.dot(covRoot,r)
    else:
        rand = 0.
    sim = mean + rand
    return mean, rand, sim

                                                      



def fill_map(imap,iys,ixs,hole_arc,mean_mul,cov_root,m1,tshape,twcs,seed=None):
    Ny,Nx = imap.shape[-2:]
    sny,snx = tshape[-2:]
    modrmap = enmap.modrmap(tshape,twcs)
    ttemplate = enmap.empty(tshape,twcs)

    iys = iys.astype(np.int)
    ixs = ixs.astype(np.int)

    m1 = np.where(modrmap.reshape(-1)<hole_arc*np.pi/180./60.)[0]
    m2 = np.where(modrmap.reshape(-1)>=hole_arc*np.pi/180./60.)[0]    
    if seed is not None: np.random.seed(seed)

    # Further improvement possible by pre-calculating random vectors
    
    outside = 0
    j = 0
    for i,(iy,ix) in enumerate(zip(iys,ixs)):

        sy = iy-sny/2
        ey = iy+sny/2
        sx = ix-snx/2
        ex = ix+snx/2
        oslice = imap[sy:ey,sx:ex]

        if np.any(oslice.shape!=tshape) or sy<0 or sx<0 or ey>=Ny or ex>=Nx:
            outside+=1
            continue

        j += 1
        ttemplate = oslice.copy()
        
        masked, maskedMean = prepare_circular_mask(ttemplate,hole_arc)
                
        masked = np.nan_to_num(masked)
        mean, rand, sim = fill_hole(masked,mean_mul,hole_arc,m1,m2,cov_root)


        a = masked.reshape(-1)
        # a[m1] = sim+maskedMean
        a[m1] = mean+maskedMean
        a[m2] = oslice.reshape(-1)[m2]
        oslice[:,:] = a.reshape(masked.shape)

        
        
    if outside>0: print (outside, " pt source(s) at edge.")





    

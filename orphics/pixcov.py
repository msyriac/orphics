from __future__ import print_function
import numpy as np
from pixell import enmap,utils

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


def stamp_pixcov_from_theory(N,cmb2d_TEB,n2d_IQU=0.,beam2d=1.,iau=False,return_pow=False):
    """Return the pixel covariance for a stamp N pixels across given the 2D IQU CMB power spectrum,
    2D beam template and 2D IQU noise power spectrum.
    """
    n2d = n2d_IQU
    cmb2d = cmb2d_TEB
    assert cmb2d.ndim==4
    ncomp = cmb2d.shape[0]
    assert cmb2d.shape[1]==ncomp
    assert ncomp==3 or ncomp==1
    
    wcs = cmb2d.wcs
    shape = cmb2d.shape[-2:]

    if ncomp==3: cmb2d = rotate_pol_power(shape,wcs,cmb2d,iau=iau,inverse=True)
    p2d = cmb2d*beam2d**2.+n2d
    if not(return_pow): return fcov_to_rcorr(shape,wcs,p2d,N)
    return fcov_to_rcorr(shape,wcs,p2d,N), cmb2d

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

def ncov_from_ivar(ivar):
    n = ivar.shape[0]
    assert n==ivar.shape[1]
    var = 1./ivar
    var[~np.isfinite(var)] = 1./ivar[ivar>0].max() # this is wrong! but needed to prevent singular matrices?!
    ncov = np.diag(var.reshape(-1))
    ncov_IQU = np.zeros((3,3,n*n,n*n))
    ncov_IQU[0,0] = ncov.copy()
    ncov_IQU[1,1] = ncov.copy() * 2.
    ncov_IQU[2,2] = ncov.copy() * 2.
    return ncov_IQU

def scov_from_theory(modlmap,cmb_theory_fn,beam_fn,iau=False):
    """
    Get a pixel covariance matrix for a stamp around a given location
    from the white noise inverse variance map and theory and beam
    functions.
    """
    n = modlmap.shape[0]
    assert n==modlmap.shape[1]
    cmb2d_TEB = np.zeros((3,3,n,n))
    theory = cmb_theory_fn
    cmb2d_TEB[0,0] = theory('TT',modlmap)
    cmb2d_TEB[1,1] = theory('EE',modlmap)
    cmb2d_TEB[2,2] = theory('BB',modlmap)
    cmb2d_TEB[0,1] = theory('TE',modlmap)
    cmb2d_TEB[1,0] = theory('TE',modlmap)
    beam2d = beam_fn(modlmap)
    tcov = stamp_pixcov_from_theory(n,enmap.enmap(cmb2d_TEB,modlmap.wcs),n2d_IQU=0.,beam2d=beam2d,iau=iau,return_pow=False)    
    return tcov


def pcov_from_ivar(n,dec,ra,ivar,cmb_theory_fn,beam_fn,iau=False):
    """
    Get a pixel covariance matrix for a stamp around a given location
    from the white noise inverse variance map and theory and beam
    functions.
    """
    assert ivar.ndim==2
    py,px = ivar.sky2pix((dec,ra))
    py = int(py)
    px = int(px)
    sy = py - n//2
    sx = px - n//2
    ey = sy + n
    ex = sx + n
    sliced = 1./ivar[sy:ey,sx:ex]
    sliced[~np.isfinite(sliced)] = 1./ivar[ivar>0].max() # this is wrong! but needed to prevent singular matrices?!
    ncov = np.diag(sliced.reshape(-1))
    ncov_IQU = np.zeros((3,3,n*n,n*n))
    ncov_IQU[0,0] = ncov.copy()
    ncov_IQU[1,1] = ncov.copy() * 2.
    ncov_IQU[2,2] = ncov.copy() * 2.
    modlmap = sliced.modlmap()
    cmb2d_TEB = np.zeros((3,3,n,n))
    theory = cmb_theory_fn
    cmb2d_TEB[0,0] = theory('TT',modlmap)
    cmb2d_TEB[1,1] = theory('EE',modlmap)
    cmb2d_TEB[2,2] = theory('BB',modlmap)
    cmb2d_TEB[0,1] = theory('TE',modlmap)
    cmb2d_TEB[1,0] = theory('TE',modlmap)
    beam2d = beam_fn(modlmap)
    tcov = stamp_pixcov_from_theory(n,enmap.enmap(cmb2d_TEB,sliced.wcs),n2d_IQU=0.,beam2d=beam2d,iau=iau,return_pow=False)    
    return tcov + ncov_IQU

def make_geometry(shape=None,wcs=None,hole_radius=None,cmb2d_TEB=None,n2d_IQU=None,context_width=None,n=None,beam2d=None,deproject=True,iau=False,res=None,tot_pow2d=None,store_pcov=False,pcov=None):

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

    # Get the pix-pix covariance on the stamp geometry given CMB theory, beam and 2D noise on the big map
    if pcov is None:
        if tot_pow2d is not None:
                pcov = fcov_to_rcorr(shape,wcs,tot_pow2d,n)
        else:
                pcov = stamp_pixcov_from_theory(n,cmb2d_TEB,n2d_IQU,beam2d=beam2d,iau=iau)


    # Do we have polarization?
    ncomp = pcov.shape[0]
    assert ncomp==1 or ncomp==2 or ncomp==3

    # Select the hole (m1) and context(m2) across all components
    m1,m2 = get_geometry_regions(ncomp,n,res,hole_radius)

    # --- Make sure that the pcov is in the right order vector(I,Q,U) ---
    # It is currently in (ncomp,ncomp,n,n) order
    # We transpose it to (ncomp,n,ncomp,n) order
    # so that when it is reshaped into a 2D array, a row/column will correspond to an (I,Q,U) vector
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
    geometry['m1'] = m1
    geometry['m2'] = m2
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


def inpaint(imap,coords_deg,hole_radius_arcmin=5.,npix_context=60,resolution_arcmin=0.5,
            cmb2d_TEB=None,n2d_IQU=None,beam2d=None,deproject=True,iau=False,tot_pow2d=None,
            geometry_tags=None,geometry_dicts=None,verbose=True):
    """Inpaint I, Q and U maps jointly accounting for their covariance using brute-force pre-calculated
    pixel covariance matrices.
    imap -- (ncomp,Ny,Nx) map to be filled, where ncomp is 1 or 3
    coords_deg -- (2,nobj) array where nobj is number of objects to inpaint, 
               and the columns are dec and ra in degrees

    # BASIC/DEBUG MODE : only allows for single geometry for all objects, and will do slow calculation here.

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

    # ADVANCED MODE (recommended!) : allows for multiple pre-calculated geometries. In a full pipeline,
    these geometries should be pre-calculated once and used for all simulations.
    geometry_tags -- (nobj,) list of strings indicating key of dictionary 
                      geometry_dict that corresponds to the hole+context geometry to use.
    geometry_dicts -- dict mapping tags to geometry dicts. Use make_geometry to pre-calculate these.
                       e.g. if geometry_tags = [...,'ptsrc5arcmin','ptsrc5arcmin',...]
                       then geometry_dicts['ptsrc5arcmin'] should be a geometry dict such that
                       geometry_dicts['ptsrc5arcmin']['meanmul'] and geometry_dicts['ptsrc5arcmin']['covsqrt'] exist.

    # NOTE ON POLARIZATION AND ARRAY SHAPES

    You may include polarization maps in the leading dimension of imap which is of shape (ncomp,Ny,Nx)
    Whether or not polarization maps are inpainted depends on each geometry object.

    In basic/debug mode, you supply cmb2D and n2D power spectra of shape (ncomp,ncomp,Ny,Nx). The leading dimensions of these arrays
    determine whether (all of) the point sources are inpainted just in I (ncomp=1) or I/Q/U (ncomp=3).

    In advanced mode, you have a dictionary of pre-calculated geometries (calculated using make_geometry). You specify which 
    geometry to use for each point source by passing a list of dictionary keys in geometry_tags, which should be keys in
    geometry_dicts. (see e.g. above)
    Whether each point source is inpainted in I or I/Q/U is then determined by the shape of cmb2D and n2D you passed to make_geometry
    to make each unique geometry object.
    """

    shape,wcs = imap.shape,imap.wcs
    Nobj = coords_deg.shape[1]
    Ny,Nx = shape[-2:]
    assert coords_deg.ndim==2, "Wrong shape for coordinates."
    assert imap.ndim==3, "Input maps have to be of shape (ncomp,Ny,Nx)"
    
    if (geometry_tags is None) or (geometry_dicts is None):
        geometry_tags = ['basic']*Nobj
        geometry_dicts = {}
        geometry_dicts['basic'] = make_geometry(shape,wcs,np.deg2rad(hole_radius_arcmin/60.),cmb2d_TEB=cmb2d_TEB,n2d_IQU=n2d_IQU,context_width=None,n=npix_context,beam2d=beam2d,deproject=deproject,iau=iau,res=np.deg2rad(resolution_arcmin),tot_pow2d=tot_pow2d,store_pcov=False)

    omap = imap.copy()
    pixs = imap.sky2pix(np.deg2rad(coords_deg),corner=False)
    fround = lambda x : int(np.round(x))
    pad = 1

    skipped = 0
    for i in range(Nobj):

        # Get geometry for this object
        geotag = geometry_tags[i]
        geometry = geometry_dicts[geotag]
        cov_root = geometry['covsqrt']
        mean_mul = geometry['meanmul']
        Npix = geometry['n']
        m1 = geometry['m1']  # hole
        m2 = geometry['m2']  # context
        ncomp = geometry['ncomp']

        if ncomp==1 or ncomp==3:
            polslice = np.s_[:ncomp,...]
        elif ncomp==2:
            #polslice = np.s_[1:,...]
            raise NotImplementedError
        else:
            raise ValueError

        # Slice out stamp
        iy,ix = pixs[:,i]
        if fround(iy-Npix/2)<pad or fround(ix-Npix/2)<pad or fround(iy+Npix/2)>(Ny-pad) or fround(ix+Npix/2)>(Nx-pad):
            skipped += 1
            continue
        stamp = omap[polslice][:,fround(iy-Npix/2.+0.5):fround(iy+Npix/2.+0.5),fround(ix-Npix/2.+0.5):fround(ix+Npix/2.+0.5)]
        if stamp.shape!=(ncomp,Npix,Npix):
            skipped += 1
            continue

    
        # Set the masked region to be zero
        cstamp = stamp.copy().reshape(-1)
        cstamp[m1] = 0.

        # Get the mean infill
        mean = np.dot(mean_mul,cstamp[m2])
        # Get a random realization (this could be moved outside the loop)
        r = np.random.normal(0.,1.,size=(m1.size))
        rand = np.dot(cov_root,r)
        # Total
        sim = mean + rand

        # Paste into returned map
        rstamp = paste(stamp,m1,sim)
        stamp[:,:,:] = rstamp[:,:,:]

    if verbose: print("Objects skipped due to edges ", skipped , " / ",Nobj)
    return omap


def get_geometry_regions(ncomp,n,res,hole_radius):
    tshape,twcs = enmap.geometry(pos=(0,0),shape=(n,n),res=res,proj='car')
    modrmap = enmap.modrmap(tshape,twcs)
    
    # Select the hole (m1) and context(m2) across all components
    amodrmap = np.repeat(modrmap.reshape((1,n,n)),ncomp,0)
    m1 = np.where(amodrmap.reshape(-1)<hole_radius)[0]
    m2 = np.where(amodrmap.reshape(-1)>=hole_radius)[0]
    
    return m1,m2

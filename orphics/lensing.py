from __future__ import print_function
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from orphics import maps
from pixell import enmap, utils
from scipy.special import factorial
try:
    from pixell import lensing as enlensing
except:
    print("WARNING: Couldn't load pixell lensing. Some features will be unavailable.")

from scipy.integrate import simps
from scipy.interpolate import splrep,splev

from scipy.fftpack import fftshift,ifftshift,fftfreq
from scipy.interpolate import interp1d
from pixell.fft import fft,ifft

from orphics.stats import bin2D

import time
from six.moves import cPickle as pickle

from orphics import stats
import os,sys
from pyfisher import get_lensing_nl as get_nl

        

def validate_geometry(shape,wcs,verbose=False):
    area = enmap.area(shape,wcs)*(180./np.pi)**2.
    if verbose: print("Geometry area : ", area, " sq.deg.")
    if area>41252.:
        print("WARNING: Geometry has area larger than full-sky.")
        print(shape,wcs)
    if area<(1./60./60.):
        print("WARNING: Geometry has area less than 1 arcmin^2.")
        print(shape,wcs)
    res = np.rad2deg(maps.resolution(shape,wcs))
    if verbose: print("Geometry pixel width : ", res*60., " arcmin.")
    if res>30.0:
        print("WARNING: Geometry has pixel larger than 30 degrees.")
        print(shape,wcs)
    if res<(1./60./60.):
        print("WARNING: Geometry has pixel smaller than 1 arcsecond.")
        print(shape,wcs)




def binned_nfw(mass,z,conc,cc,shape,wcs,bin_edges_arcmin,lmax=None,lmin=None,overdensity=200.,
               critical=False,at_cluster_z=True,kmask=None,
               sigma_mis=None,improved=True,hm=None,exclude_2h=False):
    # mass in msolar/h
    # cc Cosmology object
    modrmap = enmap.modrmap(shape,wcs)
    binner = bin2D(modrmap,bin_edges_arcmin*np.pi/180./60.)

    if improved:
        thetas = np.linspace(bin_edges_arcmin.min(),bin_edges_arcmin.max(),101) * utils.arcmin
        Ms = [mass]
        concs = [conc]
        zsource = 1100
        sig = sigma_mis*utils.arcmin if sigma_mis is not None else None
        k1h = hm.kappa_1h_profiles(thetas,Ms,concs,zsource,sig_theta=sig,delta=overdensity,rho='critical' if critical else 'mean',rho_at_z=at_cluster_z)
        k2h = hm.kappa_2h_profiles(thetas,Ms,zsource,delta=overdensity,rho='critical' if critical else 'mean',rho_at_z=at_cluster_z,lmin=2,lmax=10000) if not(exclude_2h) else np.asarray(k1h).T*0
        k1h[~np.isfinite(k1h)] = 0
        k1h = np.asarray(k1h[0])
        k2h = k2h[:,0] 
        k = enmap.enmap(maps.interp(thetas,k1h+k2h)(modrmap),wcs)
    else:
        k = nfw_kappa(mass,modrmap,cc,zL=z,concentration=conc,overdensity=overdensity,critical=critical,atClusterZ=at_cluster_z)

    if kmask is None: kmask = maps.mask_kspace(shape,wcs,lmin=lmin,lmax=lmax)
    kf = maps.filter_map(k,kmask)
    cents,k1d = binner.bin(kf)
    return cents,k1d

def fit_nfw_profile(profile_data,profile_cov,masses,z,conc,cc,shape,wcs,bin_edges_arcmin,lmax,lmin=None,
                    overdensity=200.,critical=False,at_cluster_z=True,
                    mass_guess=2e14,sigma_guess=2e13,kmask=None,sigma_mis=None,improved=True):
    """
    Returns
    lnlikes - actual lnlike as function of masses
    like_fit - gaussian fit as function of masses
    fit_mass - fit mass
    mass_err - fit mass err
    """
    from orphics.stats import fit_gauss
    cinv = np.linalg.inv(profile_cov)
    lnlikes = []
    fprofiles = []

    if improved:
        import hmvec
        ks = np.geomspace(1e-2,10,200)
        ms = np.geomspace(1e8,4e15,102)
        zs = [z]
        hm = hmvec.HaloModel(zs,ks,ms=ms,params={},mass_function="tinker",
                             halofit=None,mdef='mean',nfw_numeric=False,skip_nfw=True)
    else:
        hm = None

    for mass in masses:
        cents,profile_theory = binned_nfw(mass,z,conc,cc,shape,wcs,bin_edges_arcmin,lmax,lmin,
                                          overdensity,critical,at_cluster_z,kmask=kmask,
                                          sigma_mis=sigma_mis,improved=improved,hm=hm)
        diff = profile_data - profile_theory
        fprofiles.append(profile_theory)
        lnlike = -0.5 * np.dot(np.dot(diff,cinv),diff)
        lnlikes.append(lnlike)

    fit_mass,mass_err,_,_ = fit_gauss(masses,np.exp(lnlikes),mu_guess=mass_guess,sigma_guess=sigma_guess)
    gaussian = lambda t,mu,sigma: np.exp(-(t-mu)**2./2./sigma**2.)/np.sqrt(2.*np.pi*sigma**2.)
    like_fit = gaussian(masses,fit_mass,mass_err)
    cents,fit_profile = binned_nfw(fit_mass,z,conc,cc,shape,wcs,bin_edges_arcmin,lmax,lmin,
                                   overdensity,critical,at_cluster_z,kmask=kmask,
                                   sigma_mis=sigma_mis,improved=improved,hm=hm)
    return np.array(lnlikes),np.array(like_fit),fit_mass,mass_err,fprofiles,fit_profile

def mass_estimate(kappa_recon,kappa_noise_2d,mass_guess,concentration,z):
    """Given a cutout kappa map centered on a cluster and a redshift,
    returns a mass estimate and variance of the estimate by applying a matched filter.

    Imagine that reliable richness estimates and redshifts exist for each cluster.
    We split the sample into n richness bins.
    We go through each bin. This bin has a mean richness and a mean redshift. We convert this to a fiducial mean mass and mean concentration.
    We choose a template with this mean mass and mean concentration.
    We do a reconstruction on each cluster for each array. We now have a 2D kappa measurement. We apply MF to this with the template.
    We get a relative amplitude for each cluster, which we convert to a mass for each cluster.

    We want a mean mass versus mean richness relationship.


    Step 1
    Loop through each cluster. 
    Cut out patches from each array split-coadd and split.
    Estimate noise spectrum from splits for each array.
    Use noise spectra to get optimal coadd of all arrays and splits of coadds.
    Use coadd for reconstruction, with coadd noise from splits in weights.
    For gradient, use Planck.
    Save reconstructions and 2d kappa noise to disk.
    Repeat above for 100x random locations and save only mean to disk.
    Postprocess by loading each reconstruction, subtract meanfield, crop out region where taper**2 < 1 with some threshold.
    Verify above by performing on simulations to check that cluster profile is recovered.

    Step 2
    For each richness bin, use fiducial mass and concentration and mean redshift to choose template.
    For each cluster in each richness bin, apply MF and get masses. Find mean mass in bin. Iterate above on mass until converged.
    This provides a mean mass, concentration for each richness bin.


    
    
    """
    shape,wcs = kappa_recon.shape,kappa_recon.wcs
    mf = maps.MatchedFilter(shape,wcs,template,kappa_noise_2d)
    mf.apply(kappa_recon,kmask=kmask)


def flat_taylens(phi,imap,taylor_order = 5):
    """
    Lens a map imap with lensing potential phi
    using the Taylens algorithm up to taylor_order.

    The original routine is from Thibaut Louis.
    It has been modified here to work with pixell.
    """
    f = lambda x: enmap.fft(x,normalize='phys')
    invf = lambda x: enmap.ifft(x,normalize='phys')
    def binomial(n,k):
        "Compute n factorial by a direct multiplicative method"
        if k > n-k: k = n-k  # Use symmetry of Pascal's triangle
        accum = 1
        for i in range(1,k+1):
            accum *= (n - (k - i))
            accum /= i
        return accum
    kmap = f(phi)
    Ny,Nx = phi.shape
    ly,lx = enmap.laxes(phi.shape,phi.wcs)

    ly_array,lx_array = phi.lmap()
    alphaX=np.real(invf(1j*lx_array*kmap))
    alphaY=np.real(invf(1j*ly_array*kmap))
    iy,ix = np.mgrid[0:Ny,0:Nx]
    py,px = enmap.pixshape(phi.shape,phi.wcs)
    alphaX0 = np.array(np.round(alphaX/ px),dtype='int64')
    alphaY0 = np.array(np.round(alphaY/ py),dtype='int64')

    delta_alphaX=alphaX-alphaX0*px
    delta_alphaY=alphaY-alphaY0*py

    lensed_T_Map = imap.copy()
    cont = imap.copy()
    lensed_T_Map = imap[(iy+alphaY0)%Ny, (ix+alphaX0)%Nx]
    kmap=f(imap)

    for n in range(1,taylor_order):
        cont[:]=0
        for k in range(n+1):
            fac=1j**n*binomial(n,k)*lx_array**(n-k)*ly_array**k/(factorial(n))
            T_add=np.real(invf(fac*kmap))[(iy+alphaY0)%Ny, (ix+alphaX0)%Nx]*delta_alphaX**(n-k)*delta_alphaY**k  
            lensed_T_Map[:] += T_add
            cont[:] += T_add
    return lensed_T_Map


def alpha_from_kappa(kappa=None,posmap=None,phi=None):
    if phi is None:
        phi,_ = kappa_to_phi(kappa,kappa.modlmap(),return_fphi=True)
        shape,wcs = phi.shape,phi.wcs
    else:
        shape,wcs = phi.shape,phi.wcs
    grad_phi = enmap.grad(phi)
    if posmap is None: posmap = enmap.posmap(shape,wcs)
    pos = posmap + grad_phi
    alpha_pix = enmap.sky2pix(shape,wcs,pos, safe=False)
    return enmap.enmap(alpha_pix,wcs)
    


class FlatLensingSims(object):
    def __init__(self,shape,wcs,theory,beam_arcmin,noise_uk_arcmin,noise_e_uk_arcmin=None,noise_b_uk_arcmin=None,pol=False,fixed_lens_kappa=None):
        # assumes theory in uK^2
        from orphics import cosmology
        if len(shape)<3 and pol: shape = (3,)+shape
        if noise_e_uk_arcmin is None: noise_e_uk_arcmin = np.sqrt(2.)*noise_uk_arcmin
        if noise_b_uk_arcmin is None: noise_b_uk_arcmin = noise_e_uk_arcmin
        self.modlmap = enmap.modlmap(shape,wcs)
        Ny,Nx = shape[-2:]
        lmax = self.modlmap.max()
        ells = np.arange(0,lmax,1)
        ps_cmb = cosmology.power_from_theory(ells,theory,lensed=False,pol=pol)
        self.mgen = maps.MapGen(shape,wcs,ps_cmb)
        if fixed_lens_kappa is not None:
            self._fixed = True
            self.kappa = fixed_lens_kappa
            self.alpha = alpha_from_kappa(self.kappa)
        else:
            self._fixed = False
            ps_kk = theory.gCl('kk',self.modlmap).reshape((1,1,Ny,Nx))
            self.kgen = maps.MapGen(shape[-2:],wcs,ps_kk)
            self.posmap = enmap.posmap(shape[-2:],wcs)
            self.ps_kk = ps_kk
        self.kbeam = maps.gauss_beam(self.modlmap,beam_arcmin)
        ncomp = 3 if pol else 1
        ps_noise = np.zeros((ncomp,ncomp,Ny,Nx))
        ps_noise[0,0] = (noise_uk_arcmin*np.pi/180./60.)**2.
        if pol:
            ps_noise[1,1] = (noise_e_uk_arcmin*np.pi/180./60.)**2.
            ps_noise[2,2] = (noise_b_uk_arcmin*np.pi/180./60.)**2.
        self.ngen = maps.MapGen(shape,wcs,ps_noise)
        self.ps_noise = ps_noise

    def update_kappa(self,kappa):
        self.kappa = kappa
        self.alpha = alpha_from_kappa(self.kappa)
        
    def get_unlensed(self,seed=None):
        return self.mgen.get_map(seed=seed)
    def get_kappa(self,seed=None):
        return self.kgen.get_map(seed=seed)
    def get_sim(self,seed_cmb=None,seed_kappa=None,seed_noise=None,lens_order=5,return_intermediate=False,skip_lensing=False,cfrac=None):
        unlensed = self.get_unlensed(seed_cmb)
        if skip_lensing:
            lensed = unlensed
            kappa = enmap.samewcs(lensed.copy()[0]*0,lensed)
        else:
            if not(self._fixed):
                kappa = self.get_kappa(seed_kappa)
                self.kappa = kappa
                self.alpha = alpha_from_kappa(kappa,posmap=self.posmap)
            else:
                kappa = None
                assert seed_kappa is None
            lensed = enlensing.displace_map(unlensed, self.alpha, order=lens_order)
        beamed = maps.filter_map(lensed,self.kbeam)
        noise_map = self.ngen.get_map(seed=seed_noise)
        
        observed = beamed + noise_map
        
        if return_intermediate:
            return [ maps.get_central(x,cfrac) for x in [unlensed,kappa,lensed,beamed,noise_map,observed] ]
        else:
            return maps.get_central(observed,cfrac)
        
        

def lens_cov_pol(shape,wcs,iucov,alpha_pix,lens_order=5,kbeam=None,npixout=None,comm=None):
    """Given the pix-pix covariance matrix for the unlensed CMB,
    returns the lensed covmat for a given pixel displacement model.

    ucov -- (ncomp,ncomp,Npix,Npix) array where Npix = Ny*Nx
    alpha_pix -- (2,Ny,Nx) array of lensing displacements in pixel units
    kbeam -- (Ny,Nx) array of 2d beam wavenumbers

    """
    from pixell import lensing as enlensing

    assert iucov.ndim==4
    ncomp = iucov.shape[0]
    assert ncomp==iucov.shape[1]
    assert 1 <= ncomp <= 3
    if len(shape)==2: shape = (1,)+shape
    n = shape[-2]
    assert n==shape[-1]

    ucov = iucov.copy()
    ucov = np.transpose(ucov,(0,2,1,3))
    ucov = ucov.reshape((ncomp*n**2,ncomp*n**2))

    npix = ncomp*n**2

    if comm is None:
        from orphics import mpi
        comm = mpi.MPI.COMM_WORLD

    def efunc(vec):
        unlensed = enmap.enmap(vec.reshape(shape),wcs)
        lensed = enlensing.displace_map(unlensed, alpha_pix, order=lens_order)
        if kbeam is not None: lensed = maps.filter_map(lensed,kbeam) # TODO: replace with convolution
        # because for ~(60x60) arrays, it is probably much faster. >1 threads means worse performance
        # with FFTs for these array sizes.
        return np.asarray(lensed).reshape(-1)

    
    Scov = np.zeros(ucov.shape,dtype=ucov.dtype)
    for i in range(comm.rank, npix, comm.size):
        Scov[i,:] = efunc(ucov[i,:])
    Scov2 = utils.allreduce(Scov, comm)

    Scov = np.zeros(ucov.shape,dtype=ucov.dtype)
    for i in range(comm.rank, npix, comm.size):
        Scov[:,i] = efunc(Scov2[:,i])
    Scov = utils.allreduce(Scov, comm)

    
    Scov = Scov.reshape((ncomp,n*n,ncomp,n*n))
    if (npixout is not None) and (npixout!=n):
        Scov = Scov.reshape((ncomp,n,n,ncomp,n,n))
        s = n//2-npixout//2
        e = s + npixout
        Scov = Scov[:,s:e,s:e,:,s:e,s:e].reshape((ncomp,npixout**2,ncomp,npixout**2)) 
    Scov = np.transpose(Scov,(0,2,1,3))
        
        
    return Scov


def lensing_noise(ells,ntt,nee,nbb,
                  ellmin_t,ellmin_e,ellmin_b,
                  ellmax_t,ellmax_e,ellmax_b,
                  bin_edges,
                  camb_theory_file_root=None,
                  estimators = ['TT'],
                  delens = False,
                  theory=None,
                  dimensionless=False,
                  unlensed_equals_lensed=True,
                  grad_cut=None,
                  ellmin_k = None,
                  ellmax_k = None,
                  y_ells=None,y_ntt=None,y_nee=None,y_nbb=None,
                  y_ellmin_t=None,y_ellmin_e=None,y_ellmin_b=None,
                  y_ellmax_t=None,y_ellmax_e=None,y_ellmax_b=None,
                  lxcut_t=None,lycut_t=None,y_lxcut_t=None,y_lycut_t=None,
                  lxcut_e=None,lycut_e=None,y_lxcut_e=None,y_lycut_e=None,
                  lxcut_b=None,lycut_b=None,y_lxcut_b=None,y_lycut_b=None,
                  width_deg=5.,px_res_arcmin=1.0,shape=None,wcs=None,bigell=9000):

    from orphics import cosmology, stats

    if theory is None: theory = cosmology.loadTheorySpectraFromCAMB(camb_theory_file_root,unlensedEqualsLensed=False,
                                                     useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)

            
    if y_ells is None: y_ells=ells
    if y_ntt is None: y_ntt=ntt
    if y_nee is None: y_nee=nee
    if y_nbb is None: y_nbb=nbb
    if y_ellmin_t is None: y_ellmin_t=ellmin_t
    if y_ellmin_e is None: y_ellmin_e=ellmin_e
    if y_ellmin_b is None: y_ellmin_b=ellmin_b
    if y_ellmax_t is None: y_ellmax_t=ellmax_t
    if y_ellmax_e is None: y_ellmax_e=ellmax_e
    if y_ellmax_b is None: y_ellmax_b=ellmax_b

    if ellmin_k is None: ellmin_k = bin_edges.min() #min(ellmin_t,ellmin_e,ellmin_b,y_ellmin_t,y_ellmin_e,y_ellmin_b)
    if ellmax_k is None: ellmax_k = bin_edges.max() #max(ellmax_t,ellmax_e,ellmax_b,y_ellmax_t,y_ellmax_e,y_ellmax_b)

    pol = False if estimators==['TT'] else True

    
    if ells.ndim==2:
        assert shape is None
        assert wcs is None
        modlmap = ells
        shape = modlmap.shape
        wcs = modlmap.wcs
        validate_geometry(shape,wcs,verbose=True)
        nTX = ntt
        nTY = y_ntt
        nEX = nee
        nEY = y_nee
        nBX = nbb
        nBY = y_nbb

    else:
        if (shape is None) or (wcs is None):
            shape,wcs = maps.rect_geometry(width_deg=width_deg,px_res_arcmin=px_res_arcmin)
        modlmap = enmap.modlmap(shape,wcs)
        nTX = maps.interp(ells,ntt)(modlmap)
        nTY = maps.interp(ells,y_ntt)(modlmap)
        nEX = maps.interp(ells,nee)(modlmap)
        nEY = maps.interp(ells,y_nee)(modlmap)
        nBX = maps.interp(ells,nbb)(modlmap)
        nBY = maps.interp(ells,y_nbb)(modlmap)

    kmask_TX = maps.mask_kspace(shape,wcs,lmin=ellmin_t,lmax=ellmax_t,lxcut=lxcut_t,lycut=lycut_t)
    kmask_TY = maps.mask_kspace(shape,wcs,lmin=y_ellmin_t,lmax=y_ellmax_t,lxcut=y_lxcut_t,lycut=y_lycut_t)
    kmask_EX = maps.mask_kspace(shape,wcs,lmin=ellmin_e,lmax=ellmax_e,lxcut=lxcut_e,lycut=lycut_e)
    kmask_EY = maps.mask_kspace(shape,wcs,lmin=y_ellmin_e,lmax=y_ellmax_e,lxcut=y_lxcut_e,lycut=y_lycut_e)
    kmask_BX = maps.mask_kspace(shape,wcs,lmin=ellmin_b,lmax=ellmax_b,lxcut=lxcut_b,lycut=lycut_b)
    kmask_BY = maps.mask_kspace(shape,wcs,lmin=y_ellmin_b,lmax=y_ellmax_b,lxcut=y_lxcut_b,lycut=y_lycut_b)
    kmask_K = maps.mask_kspace(shape,wcs,lmin=ellmin_k,lmax=ellmax_k)

    qest = Estimator(shape,wcs,
                     theory,
                     theorySpectraForNorm=theory,
                     noiseX2dTEB=[nTX,nEX,nBX],
                     noiseY2dTEB=[nTY,nEY,nBY],
                     noiseX_is_total = False,
                     noiseY_is_total = False,
                     fmaskX2dTEB=[kmask_TX,kmask_EX,kmask_BX],
                     fmaskY2dTEB=[kmask_TY,kmask_EY,kmask_BY],
                     fmaskKappa=kmask_K,
                     kBeamX = None,
                     kBeamY = None,
                     doCurl=False,
                     TOnly=not(pol),
                     halo=True,
                     gradCut=grad_cut,
                     verbose=False,
                     loadPickledNormAndFilters=None,
                     savePickledNormAndFilters=None,
                     uEqualsL=unlensed_equals_lensed,
                     bigell=bigell,
                     mpi_comm=None,
                     lEqualsU=False)

    nlkks = {}
    nsum = 0.
    for est in estimators:
        nlkk2d = qest.N.Nlkk[est]
        ls,nlkk = stats.bin_in_annuli(nlkk2d, modlmap, bin_edges)
        nlkks[est] = nlkk.copy()
        nsum += np.nan_to_num(kmask_K/nlkk2d)

    nmv = np.nan_to_num(kmask_K/nsum)
    nlkks['mv'] = stats.bin_in_annuli(nmv, modlmap, bin_edges)[1]
    
    return ls,nlkks,theory,qest
    
    

def lens_cov(shape,wcs,ucov,alpha_pix,lens_order=5,kbeam=None,bshape=None):
    """Given the pix-pix covariance matrix for the unlensed CMB,
    returns the lensed covmat for a given pixel displacement model.

    ucov -- (Npix,Npix) array where Npix = Ny*Nx
    alpha_pix -- (2,Ny,Nx) array of lensing displacements in pixel units
    kbeam -- (Ny,Nx) array of 2d beam wavenumbers

    """
    from pixell import lensing as enlensing

    Scov = ucov.copy()
    
    for i in range(ucov.shape[0]):
        unlensed = enmap.enmap(Scov[i,:].copy().reshape(shape),wcs)
        lensed = enlensing.displace_map(unlensed, alpha_pix, order=lens_order)
        if kbeam is not None: lensed = maps.filter_map(lensed,kbeam)
        Scov[i,:] = lensed.ravel()
    for j in range(ucov.shape[1]):
        unlensed = enmap.enmap(Scov[:,j].copy().reshape(shape),wcs)
        lensed = enlensing.displace_map(unlensed, alpha_pix, order=lens_order)
        if kbeam is not None: lensed = maps.filter_map(lensed,kbeam)
        Scov[:,j] = lensed.ravel()

    if (bshape is not None) and (bshape!=shape):
        ny,nx = shape
        Scov = Scov.reshape((ny,nx,ny,nx))
        bny,bnx = bshape
        sy = ny//2-bny//2
        ey = sy + bny
        sx = nx//2-bnx//2
        ex = sx + bnx
        Scov = Scov[sy:ey,sx:ex,sy:ey,sx:ex].reshape((np.prod(bshape),np.prod(bshape)))
    return Scov




def beam_cov(ucov,kbeam):
    """Given the pix-pix covariance matrix for the lensed CMB,
    returns the beamed covmat. The beam can be a ratio of beams to
    readjust the beam in a given matrix.

    ucov -- (Npix,Npix) array where Npix = Ny*Nx
    kbeam -- (Ny,Nx) array of 2d beam wavenumbers

    """
    Scov = ucov.copy()
    wcs = ucov.wcs
    shape = kbeam.shape[-2:]
    for i in range(Scov.shape[0]):
        lensed = enmap.enmap(Scov[i,:].copy().reshape(shape) ,wcs)
        lensed = maps.filter_map(lensed,kbeam)
        Scov[i,:] = lensed.ravel()
    for j in range(Scov.shape[1]):
        lensed = enmap.enmap(Scov[:,j].copy().reshape(shape),wcs)
        lensed = maps.filter_map(lensed,kbeam)
        Scov[:,j] = lensed.ravel()
    return Scov


def qest(shape,wcs,theory,noise2d=None,beam2d=None,kmask=None,noise2d_P=None,kmask_P=None,kmask_K=None,pol=False,grad_cut=None,unlensed_equals_lensed=False,bigell=9000,noise2d_B=None,noiseX_is_total=False,noiseY_is_total=False):
    # if beam2d is None, assumes input maps are beam deconvolved and noise2d is beam deconvolved
    # otherwise, it beam deconvolves itself
    if noise2d is None: noise2d = np.zeros(shape[-2:])
    if noise2d_P is None: noise2d_P = 2.*noise2d
    if noise2d_B is None: noise2d_B = noise2d_P
    if beam2d is None: beam2d = np.ones(shape[-2:])
    return Estimator(shape,wcs,
                     theory,
                     theorySpectraForNorm=theory,
                     noiseX2dTEB=[noise2d,noise2d_P,noise2d_B],
                     noiseY2dTEB=[noise2d,noise2d_P,noise2d_B],
                     noiseX_is_total = noiseX_is_total,
                     noiseY_is_total = noiseY_is_total,
                     fmaskX2dTEB=[kmask,kmask_P,kmask_P],
                     fmaskY2dTEB=[kmask,kmask_P,kmask_P],
                     fmaskKappa=kmask_K,
                     kBeamX = beam2d,
                     kBeamY = beam2d,
                     doCurl=False,
                     TOnly=not(pol),
                     halo=True,
                     gradCut=grad_cut,
                     verbose=False,
                     loadPickledNormAndFilters=None,
                     savePickledNormAndFilters=None,
                     uEqualsL=unlensed_equals_lensed,
                     bigell=bigell,
                     mpi_comm=None,
                     lEqualsU=False)


def kappa_to_phi(kappa,modlmap,return_fphi=False):
    fphi = enmap.samewcs(kappa_to_fphi(kappa,modlmap),kappa)
    phi =  enmap.samewcs(ifft(fphi,axes=[-2,-1],normalize=True).real, kappa) 
    if return_fphi:
        return phi, fphi
    else:
        return phi

def kappa_to_fphi(kappa,modlmap):
    return fkappa_to_fphi(fft(kappa,axes=[-2,-1]),modlmap)

def fkappa_to_fphi(fkappa,modlmap):
    kmap = np.nan_to_num(2.*fkappa/modlmap/(modlmap+1.))
    kmap[modlmap<2.] = 0.
    return kmap



def fillLowEll(ells,cls,ellmin):
    # Fill low ells with the same value
    low_index = np.where(ells>ellmin)[0][0]
    lowest_ell = ells[low_index]
    lowest_val = cls[low_index]
    fill_ells = np.arange(2,lowest_ell,1)
    new_ells = np.append(fill_ells,ells[low_index:])
    fill_cls = np.array([lowest_val]*len(fill_ells))
    new_cls = np.append(fill_cls,cls[low_index:])

    return new_ells,new_cls


def sanitizePower(Nlbinned):
    Nlbinned[Nlbinned<0.] = np.nan

    # fill nans with interp
    ok = ~np.isnan(Nlbinned)
    xp = ok.ravel().nonzero()[0]
    fp = Nlbinned[~np.isnan(Nlbinned)]
    x  = np.isnan(Nlbinned).ravel().nonzero()[0]
    Nlbinned[np.isnan(Nlbinned)] = np.interp(x, xp, fp)
    return Nlbinned


def getMax(polComb,tellmax,pellmax):
    if polComb=='TT':
        return tellmax
    elif polComb in ['EE','EB']:
        return pellmax
    else:
        return max(tellmax,pellmax)


class QuadNorm(object):

    
    def __init__(self,shape,wcs,gradCut=None,verbose=False,bigell=9000,kBeamX=None,kBeamY=None,fmask=None):
        
        self.shape = shape
        self.wcs = wcs
        self.verbose = verbose
        self.Ny,self.Nx = shape[-2:]
        self.lxMap,self.lyMap,self.modLMap,thetaMap,lx,ly = maps.get_ft_attributes(shape,wcs)
        self.lxHatMap = self.lxMap*np.nan_to_num(1. / self.modLMap)
        self.lyHatMap = self.lyMap*np.nan_to_num(1. / self.modLMap)

        self.fmask = fmask

        if kBeamX is not None:           
            self.kBeamX = kBeamX
        else:
            self.kBeamX = 1.
            
        if kBeamY is not None:           
            self.kBeamY = kBeamY
        else:
            self.kBeamY = 1.


        self.uClNow2d = {}
        self.uClFid2d = {}
        self.lClFid2d = {}
        self.noiseXX2d = {}
        self.noiseYY2d = {}
        self.fMaskXX = {}
        self.fMaskYY = {}

        self.lmax_T=bigell
        self.lmax_P=bigell
        self.defaultMaskT = maps.mask_kspace(self.shape,self.wcs,lmin=2,lmax=self.lmax_T)
        self.defaultMaskP = maps.mask_kspace(self.shape,self.wcs,lmin=2,lmax=self.lmax_P)
        #del lx
        #del ly
        self.thetaMap = thetaMap
        self.lx = lx
        self.ly = ly
        
        self.bigell=bigell #9000.
        if gradCut is not None: 
            self.gradCut = gradCut
        else:
            self.gradCut = bigell
        


        self.Nlkk = {}
        self.pixScaleY,self.pixScaleX = enmap.pixshape(shape,wcs)
        self.noiseX_is_total = False
        self.noiseY_is_total = False
        


    def fmask_func(self,arr,mask):        
        arr[mask<1.e-3] = 0.
        return arr

    def addUnlensedFilter2DPower(self,XY,power2dData):
        '''
        XY = TT, TE, EE, EB or TB
        power2d is a flipper power2d object
        These Cls belong in the Wiener filters, and will not
        be perturbed if/when calculating derivatives.
        '''
        self.uClFid2d[XY] = power2dData.copy()+0.j
    def addUnlensedNorm2DPower(self,XY,power2dData):
        '''
        XY = TT, TE, EE, EB or TB
        power2d is a flipper power2d object
        These Cls belong in the CMB normalization, and will
        be perturbed if/when calculating derivatives.
        '''
        self.uClNow2d[XY] = power2dData.copy()+0.j
    def addLensedFilter2DPower(self,XY,power2dData):
        '''
        XY = TT, TE, EE, EB or TB
        power2d is a flipper power2d object
        These Cls belong in the Wiener filters, and will not
        be perturbed if/when calculating derivatives.
        '''
        self.lClFid2d[XY] = power2dData.copy()+0.j
    def addNoise2DPowerXX(self,XX,power2dData,fourierMask=None,is_total=False):
        '''
        Noise power for the X leg of the quadratic estimator
        XX = TT, EE, BB
        power2d is a flipper power2d object
        fourierMask is an int array that is 0 where noise is
        infinite and 1 where not. It should be in the same
        fftshift state as power2d.powerMap
        '''
        # check if fourier mask is int!
        self.noiseX_is_total = is_total
        self.noiseXX2d[XX] = power2dData.copy()+0.j
        if fourierMask is not None:
            self.noiseXX2d[XX][fourierMask==0] = np.inf
            self.fMaskXX[XX] = fourierMask
        else:
            if XX=='TT':
                self.noiseXX2d[XX][self.defaultMaskT==0] = np.inf
            else:
                self.noiseXX2d[XX][self.defaultMaskP==0] = np.inf

    def addNoise2DPowerYY(self,YY,power2dData,fourierMask=None,is_total=False):
        '''
        Noise power for the Y leg of the quadratic estimator
        XX = TT, EE, BB
        power2d is a flipper power2d object
        fourierMask is an int array that is 0 where noise is
        infinite and 1 where not. It should be in the same
        fftshift state as power2d.powerMap
        '''
        # check if fourier mask is int!
        self.noiseY_is_total = is_total
        self.noiseYY2d[YY] = power2dData.copy()+0.j
        if fourierMask is not None:
            self.noiseYY2d[YY][fourierMask==0] = np.inf
            self.fMaskYY[YY] = fourierMask
        else:
            if YY=='TT':
                self.noiseYY2d[YY][self.defaultMaskT==0] = np.inf
            else:
                self.noiseYY2d[YY][self.defaultMaskP==0] = np.inf
        
    def addClkk2DPower(self,power2dData):
        '''
        Fiducial Clkk power
        Used if delensing
        power2d is a flipper power2d object            
        '''
        self.clkk2d = power2dData.copy()+0.j
        self.clpp2d = 0.j+np.nan_to_num(self.clkk2d.copy()*4./(self.modLMap**2.)/((self.modLMap+1.)**2.))


    def WXY(self,XY):
        X,Y = XY
        if Y=='B': Y='E'
        gradClXY = X+Y
        if XY=='ET': gradClXY = 'TE'
        if XY=='BE': gradClXY = 'EE'

        totnoise = self.noiseXX2d[X+X].copy() if self.noiseX_is_total else (self.lClFid2d[X+X].copy()*self.kBeamX**2.+self.noiseXX2d[X+X].copy())
        W = self.fmask_func(np.nan_to_num(self.uClFid2d[gradClXY].copy()/totnoise)*self.kBeamX,self.fMaskXX[X+X])
        W[self.modLMap>self.gradCut]=0.
        if X=='T':
            W[np.where(self.modLMap >= self.lmax_T)] = 0.
        else:
            W[np.where(self.modLMap >= self.lmax_P)] = 0.


        # debug_edges = np.arange(400,6000,50)
        # import orphics.tools.stats as stats
        # import orphics.tools.io as io
        # binner = stats.bin2D(self.modLMap,debug_edges)
        # cents,ws = binner.bin(W)
        # pl = io.Plotter()
        # pl.add(cents,ws)
        # pl.done("ws.png")
        # sys.exit()
        
            
        return W
        

    def WY(self,YY):
        assert YY[0]==YY[1]
        totnoise = self.noiseYY2d[YY].copy() if self.noiseY_is_total else (self.lClFid2d[YY].copy()*self.kBeamY**2.+self.noiseYY2d[YY].copy())
        W = self.fmask_func(np.nan_to_num(1./totnoise)*self.kBeamY,self.fMaskYY[YY]) #* self.modLMap  # !!!!!
        W[np.where(self.modLMap >= self.lmax_T)] = 0.
        if YY[0]=='T':
            W[np.where(self.modLMap >= self.lmax_T)] = 0.
        else:
            W[np.where(self.modLMap >= self.lmax_P)] = 0.


        # debug_edges = np.arange(400,6000,50)
        # import orphics.tools.stats as stats
        # import orphics.tools.io as io
        # io.quickPlot2d(np.fft.fftshift(W.real),"wy2d.png")
        # binner = stats.bin2D(self.modLMap,debug_edges)
        # cents,ws = binner.bin(W.real)
        # print cents
        # print ws
        # pl = io.Plotter()#scaleY='log')
        # pl.add(cents,ws)
        # pl._ax.set_xlim(2,6000)
        # pl.done("wy.png")
        # sys.exit()
            
        return W

    def getCurlNlkk2d(self,XY,halo=False):
        raise NotImplementedError

    def super_dumb_N0_TTTT(self,data_power_2d_TT):
        ratio = np.nan_to_num(data_power_2d_TT*self.WY("TT")/self.kBeamY)
        lmap = self.modLMap
        replaced = np.nan_to_num(self.getNlkk2d("TT",halo=True,l1Scale=self.fmask_func(ratio,self.fMaskXX["TT"]),l2Scale=self.fmask_func(ratio,self.fMaskYY["TT"]),setNl=False) / (2. * np.nan_to_num(1. / lmap/(lmap+1.))))
        unreplaced = self.Nlkk["TT"].copy()
        return np.nan_to_num(unreplaced**2./replaced)

    def super_dumb_N0_EEEE(self,data_power_2d_EE):
        ratio = np.nan_to_num(data_power_2d_EE*self.WY("EE")/self.kBeamY)
        lmap = self.modLMap
        replaced = np.nan_to_num(self.getNlkk2d("EE",halo=True,l1Scale=self.fmask_func(ratio,self.fMaskXX["EE"]),l2Scale=self.fmask_func(ratio,self.fMaskYY["EE"]),setNl=False) / (2. * np.nan_to_num(1. / lmap/(lmap+1.))))
        unreplaced = self.Nlkk["EE"].copy()
        return np.nan_to_num(unreplaced**2./replaced)
    
    def getNlkk2d(self,XY,halo=True,l1Scale=1.,l2Scale=1.,setNl=True):
        if not(halo): raise NotImplementedError
        
        lx,ly = self.lxMap,self.lyMap
        lmap = self.modLMap

        X,Y = XY
        XX = X+X
        YY = Y+Y

        if self.verbose: 
            print(("Calculating norm for ", XY))

            
        h=0.

        allTerms = []
            
        if XY == 'TT':
            
            clunlenTTArrNow = self.uClNow2d['TT'].copy()
                

            if halo:
            
                WXY = self.WXY('TT')*self.kBeamX*l1Scale
                WY = self.WY('TT')*self.kBeamY*l2Scale


                
                preG = WY
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTTArrNow*WXY
                    preFX = ell1*WXY
                    preGX = ell2*clunlenTTArrNow*WY
                    

                    calc = ell1*ell2*fft(ifft(preF,axes=[-2,-1],normalize=True)*ifft(preG,axes=[-2,-1],normalize=True)+ifft(preFX,axes=[-2,-1],normalize=True)*ifft(preGX,axes=[-2,-1],normalize=True),axes=[-2,-1])
                    allTerms += [calc]
                    

            else:

                clunlenTTArr = self.uClFid2d['TT'].copy()

                preG = self.WY('TT') #np.nan_to_num(1./cltotTTArrY)
                cltotTTArrX = np.nan_to_num(clunlenTTArr/self.WXY('TT'))
                cltotTTArrY = np.nan_to_num(1./self.WY('TT'))

                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTTArrNow*clunlenTTArr*np.nan_to_num(1./cltotTTArrX)/2.            
                    preFX = ell1*clunlenTTArrNow*np.nan_to_num(1./cltotTTArrX)
                    preGX = ell2*clunlenTTArr*np.nan_to_num(1./cltotTTArrY)


                    
                    calc = 2.*ell1*ell2*fft(ifft(preF,axes=[-2,-1],normalize=True)*ifft(preG,axes=[-2,-1],normalize=True)+ifft(preFX,axes=[-2,-1],normalize=True)*ifft(preGX,axes=[-2,-1],normalize=True)/2.,axes=[-2,-1])
                    allTerms += [calc]
          

        elif XY == 'EE':

            clunlenEEArrNow = self.uClNow2d['EE'].copy()


            sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

            lx = self.lxMap
            ly = self.lyMap


            lxhat = self.lxHatMap
            lyhat = self.lyHatMap

            sinf = sin2phi(lxhat,lyhat)
            sinsqf = sinf**2.
            cosf = cos2phi(lxhat,lyhat)
            cossqf = cosf**2.
                                
            if halo:
            

                WXY = self.WXY('EE')*self.kBeamX
                WY = self.WY('EE')*self.kBeamY
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                        preF = trigfact*ell1*ell2*clunlenEEArrNow*WXY
                        preG = trigfact*WY
                        allTerms += [ell1*ell2*fft(ifft(preF,axes=[-2,-1],normalize=True)*ifft(preG,axes=[-2,-1],normalize=True),axes=[-2,-1])]
                        
                        #allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
                        
                        preFX = trigfact*ell1*clunlenEEArrNow*WY
                        preGX = trigfact*ell2*WXY

                        allTerms += [ell1*ell2*fft(ifft(preFX,axes=[-2,-1],normalize=True)*ifft(preGX,axes=[-2,-1],normalize=True),axes=[-2,-1])]
                        #allTerms += [ell1*ell2*fft2(ifft2(preFX)*ifft2(preGX))]

                
            # else:


            #     rfact = 2.**0.25
            #     for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
            #         for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
            #             preF = trigfact*ell1*ell2*clunlenEEArrNow*clunlenEEArr*np.nan_to_num(1./cltotEEArr)/2.
            #             preG = trigfact*np.nan_to_num(1./cltotEEArr)
            #             preFX = trigfact*ell1*clunlenEEArrNow*np.nan_to_num(1./cltotEEArr)
            #             preGX = trigfact*ell2*clunlenEEArr*np.nan_to_num(1./cltotEEArr)

            #             allTerms += [2.*ell1*ell2*fft2(ifft2(preF)*ifft2(preG)+ifft2(preFX)*ifft2(preGX)/2.)]


            


        elif XY == 'EB':


            clunlenEEArrNow = self.uClNow2d['EE'].copy()
            clunlenBBArrNow = self.uClNow2d['BB'].copy()


            sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

            lx = self.lxMap
            ly = self.lyMap

            termsF = []
            termsF.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )

            termsG = []
            termsG.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )

            lxhat = self.lxHatMap
            lyhat = self.lyHatMap

            WXY = self.WXY('EB')*self.kBeamX
            WY = self.WY('BB')*self.kBeamY
            for ellsq in [lx*lx,ly*ly,np.sqrt(2.)*lx*ly]:
                preF = ellsq*clunlenEEArrNow*WXY
                preG = WY

                for termF,termG in zip(termsF,termsG):
                    allTerms += [ellsq*fft(ifft(termF(preF,lxhat,lyhat),axes=[-2,-1],normalize=True)*ifft(termG(preG,lxhat,lyhat),axes=[-2,-1],normalize=True),axes=[-2,-1])]

        elif XY == 'BE':


            clunlenEEArrNow = self.uClNow2d['EE'].copy()
            clunlenBBArrNow = self.uClNow2d['BB'].copy()


            sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

            lx = self.lxMap
            ly = self.lyMap

            termsF = []
            termsF.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )

            termsG = []
            termsG.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )

            lxhat = self.lxHatMap
            lyhat = self.lyHatMap

            WXY = self.WXY('BE')*self.kBeamX
            WY = self.WY('EE')*self.kBeamY
            for ellsq in [lx*lx,ly*ly,np.sqrt(2.)*lx*ly]:
                preF = WXY
                preG = ellsq*clunlenEEArrNow*WY

                for termF,termG in zip(termsF,termsG):
                    allTerms += [ellsq*fft(ifft(termF(preF,lxhat,lyhat),axes=[-2,-1],normalize=True)*ifft(termG(preG,lxhat,lyhat),axes=[-2,-1],normalize=True),axes=[-2,-1])]


        elif XY=='ET':

            clunlenTEArrNow = self.uClNow2d['TE'].copy()

            if halo:
                sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
                cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

                lx = self.lxMap
                ly = self.lyMap


                lxhat = self.lxHatMap
                lyhat = self.lyHatMap

                sinf = sin2phi(lxhat,lyhat)
                sinsqf = sinf**2.
                cosf = cos2phi(lxhat,lyhat)
                cossqf = cosf**2.


                WXY = self.WXY('ET')*self.kBeamX
                WY = self.WY('TT')*self.kBeamY

                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTEArrNow*WXY
                    preG = WY
                    allTerms += [ell1*ell2*fft(ifft(preF,axes=[-2,-1],normalize=True)*ifft(preG,axes=[-2,-1],normalize=True),axes=[-2,-1])]
                    for trigfact in [cosf,sinf]:

                        preFX = trigfact*ell1*clunlenTEArrNow*WY
                        preGX = trigfact*ell2*WXY

                        allTerms += [ell1*ell2*fft(ifft(preFX,axes=[-2,-1],normalize=True)*ifft(preGX,axes=[-2,-1],normalize=True),axes=[-2,-1])]


            # else:



            #     sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            #     cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

            #     lx = self.lxMap
            #     ly = self.lyMap

            
            #     lxhat = self.lxHatMap
            #     lyhat = self.lyHatMap

            #     sinf = sin2phi(lxhat,lyhat)
            #     sinsqf = sinf**2.
            #     cosf = cos2phi(lxhat,lyhat)
            #     cossqf = cosf**2.
                
                
            #     rfact = 2.**0.25
            #     for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
            #         preF = ell1*ell2*clunlenTEArrNow*clunlenTEArr*np.nan_to_num(1./cltotEEArr)
            #         preG = np.nan_to_num(1./cltotTTArr)
            #         allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
            #         for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
            #             preF = np.nan_to_num(1./cltotEEArr)
            #             preG = trigfact*ell1*ell2*clunlenTEArrNow*clunlenTEArr*np.nan_to_num(1./cltotTTArr)
            #             allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
            #         for trigfact in [cosf,sinf]:
                        
            #             preFX = trigfact*ell1*clunlenTEArrNow*np.nan_to_num(1./cltotEEArr)
            #             preGX = trigfact*ell2*clunlenTEArr*np.nan_to_num(1./cltotTTArr)

            #             allTerms += [2.*ell1*ell2*fft2(ifft2(preFX)*ifft2(preGX))]

                    

        elif XY=='TE':

            clunlenTEArrNow = self.uClNow2d['TE'].copy()

            if halo:
            
                sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
                cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

                lx = self.lxMap
                ly = self.lyMap

            
                lxhat = self.lxHatMap
                lyhat = self.lyHatMap

                sinf = sin2phi(lxhat,lyhat)
                sinsqf = sinf**2.
                cosf = cos2phi(lxhat,lyhat)
                cossqf = cosf**2.
                
                WXY = self.WXY('TE')*self.kBeamX
                WY = self.WY('EE')*self.kBeamY

                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                        preF = trigfact*ell1*ell2*clunlenTEArrNow*WXY
                        preG = trigfact*WY
                        allTerms += [ell1*ell2*fft(ifft(preF,axes=[-2,-1],normalize=True)*ifft(preG,axes=[-2,-1],normalize=True),axes=[-2,-1])]
                    for trigfact in [cosf,sinf]:
                        
                        preFX = trigfact*ell1*clunlenTEArrNow*WY
                        preGX = trigfact*ell2*WXY

                        allTerms += [ell1*ell2*fft(ifft(preFX,axes=[-2,-1],normalize=True)*ifft(preGX,axes=[-2,-1],normalize=True),axes=[-2,-1])]

                
            # else:



            #     sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            #     cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

            #     lx = self.lxMap
            #     ly = self.lyMap

            
            #     lxhat = self.lxHatMap
            #     lyhat = self.lyHatMap

            #     sinf = sin2phi(lxhat,lyhat)
            #     sinsqf = sinf**2.
            #     cosf = cos2phi(lxhat,lyhat)
            #     cossqf = cosf**2.
                
                
            #     rfact = 2.**0.25
            #     for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
            #         for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
            #             preF = trigfact*ell1*ell2*clunlenTEArrNow* self.WXY('TE')#clunlenTEArr*np.nan_to_num(1./cltotTTArr)
            #             preG = trigfact*self.WY('EE')#np.nan_to_num(1./cltotEEArr)
            #             allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
            #         preF = self.WY('TT')#np.nan_to_num(1./cltotTTArr)
            #         preG = ell1*ell2*clunlenTEArrNow* self.WXY('ET') #*clunlenTEArr*np.nan_to_num(1./cltotEEArr)
            #         allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
            #         for trigfact in [cosf,sinf]:
                        
            #             preFX = trigfact*ell1*clunlenTEArrNow*self.WY('TT')#np.nan_to_num(1./cltotTTArr)
            #             preGX = trigfact*ell2* self.WXY('ET')#*clunlenTEArr*np.nan_to_num(1./cltotEEArr)

            #             allTerms += [2.*ell1*ell2*fft2(ifft2(preFX)*ifft2(preGX))]


                

        elif XY == 'TB':

            clunlenTEArrNow = self.uClNow2d['TE'].copy()


                
            sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

            lx = self.lxMap
            ly = self.lyMap

            termsF = []
            termsF.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )

            termsG = []
            termsG.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )
            
            lxhat = self.lxHatMap
            lyhat = self.lyHatMap
            
            WXY = self.WXY('TB')*self.kBeamX
            WY = self.WY('BB')*self.kBeamY
            for ellsq in [lx*lx,ly*ly,np.sqrt(2.)*lx*ly]:
                preF = ellsq*clunlenTEArrNow*WXY
                preG = WY

                for termF,termG in zip(termsF,termsG):
                    allTerms += [ellsq*fft(ifft(termF(preF,lxhat,lyhat),axes=[-2,-1],normalize=True)*ifft(termG(preG,lxhat,lyhat),axes=[-2,-1],normalize=True),axes=[-2,-1])]
                    

            


        else:
            print("ERROR: Unrecognized polComb")
            sys.exit(1)    
        
                        
        ALinv = np.real(np.sum( allTerms, axis = 0))
        alval = np.nan_to_num(1. / ALinv)
        if self.fmask is not None: alval = self.fmask_func(alval,self.fmask)
        l4 = (lmap**2.) * ((lmap + 1.)**2.)
        NL = l4 *alval/ 4.
        NL[np.where(np.logical_or(lmap >= self.bigell, lmap <2.))] = 0.

        retval = np.nan_to_num(NL.real * self.pixScaleX*self.pixScaleY  )

        if setNl:
            self.Nlkk[XY] = retval.copy()
            #print "SETTING NL"


        # debug_edges = np.arange(400,6000,50)
        # import orphics.tools.stats as stats
        # import orphics.tools.io as io
        # io.quickPlot2d((np.fft.fftshift(retval)),"nl2d.png")
        # binner = stats.bin2D(self.modLMap,debug_edges)
        # cents,ws = binner.bin(retval.real)
        # pl = io.Plotter()#scaleY='log')
        # pl.add(cents,ws)
        # pl._ax.set_xlim(2,6000)
        # pl.done("nl.png")
        # sys.exit()


            
        return retval * 2. * np.nan_to_num(1. / lmap/(lmap+1.))
        
        
                  

        
      


    def delensClBB(self,Nlkk,fmask=None,halo=True):
        """
        Delens ClBB with input Nlkk curve
        """

        # Set the phi noise = Clpp + Nlpp
        Nlppnow = Nlkk*4./(self.modLMap**2.)/((self.modLMap+1.)**2.)
        clPPArr = self.clpp2d
        cltotPPArr = clPPArr + Nlppnow
        cltotPPArr[np.isnan(cltotPPArr)] = np.inf

        # Get uClEE
        clunlenEEArr = self.uClFid2d['EE'].copy()
        # Get lClEE + NEE
        clunlentotEEArr = (self.lClFid2d['EE'].copy()+self.noiseYY2d['EE'])

        # Mask
        clunlentotEEArr[self.fMaskYY['EE']==0] = np.inf
        if fmask is None:
            fmask = self.fMaskYY['EE']

        cltotPPArr[fmask==0] = np.inf

        # Trig required for responses
        sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
        cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)
        lx = self.lxMap
        ly = self.lyMap
        lxhat = self.lxHatMap
        lyhat = self.lyHatMap
        sinf = sin2phi(lxhat,lyhat)
        sinsqf = sinf**2.
        cosf = cos2phi(lxhat,lyhat)
        cossqf = cosf**2.

        # Use ffts to calculate each term instead of convolving 
        allTerms = []
        for ellsq in [lx*lx,ly*ly,np.sqrt(2.)*lx*ly]:
            for trigfactOut,trigfactIn in zip([sinsqf,cossqf,1.j*np.sqrt(2.)*sinf*cosf],[cossqf,sinsqf,1.j*np.sqrt(2.)*sinf*cosf]):
                preF1 = trigfactIn*ellsq*clunlenEEArr 
                preG1 = ellsq*clPPArr  

                preF2 = trigfactIn*ellsq*clunlenEEArr**2.*np.nan_to_num(1./clunlentotEEArr) * self.fMaskYY['EE']
                preG2 = ellsq*clPPArr**2.*np.nan_to_num(1./cltotPPArr) * fmask

                t1 = ifft(preF1,axes=[-2,-1],normalize=True)*ifft(preG1,axes=[-2,-1],normalize=True) # Orig B
                t2 = ifft(preF2,axes=[-2,-1],normalize=True)*ifft(preG2,axes=[-2,-1],normalize=True) # Delensed part
                
                allTerms += [trigfactOut*(fft(t1 - t2,axes=[-2,-1]))]


        # Sum all terms
        ClBBres = np.real(np.sum( allTerms, axis = 0))

        # Pixel factors
        ClBBres[np.where(np.logical_or(self.modLMap >= self.bigell, self.modLMap == 0.))] = 0.
        ClBBres *= self.Nx * self.Ny 
        area =self.Nx*self.Ny*self.pixScaleX*self.pixScaleY
        bbNoise2D = ((np.sqrt(ClBBres)/self.pixScaleX/self.pixScaleY)**2.)*(area/(self.Nx*self.Ny*1.0)**2)

        # Set lensed BB to delensed level
        self.lClFid2d['BB'] = bbNoise2D.copy()

        
        return bbNoise2D



class NlGenerator(object):
    def __init__(self,shape,wcs,theorySpectra,bin_edges=None,gradCut=None,TCMB=2.7255e6,bigell=9000,lensedEqualsUnlensed=False,unlensedEqualsLensed=True):
        self.shape = shape
        self.wcs = wcs
        self.N = QuadNorm(shape,wcs,gradCut=gradCut,bigell=bigell)
        self.TCMB = TCMB

        cmbList = ['TT','TE','EE','BB']
        
        self.theory = theorySpectra
        
        for cmb in cmbList:
            uClFilt = theorySpectra.uCl(cmb,self.N.modLMap)
            uClNorm = uClFilt
            lClFilt = theorySpectra.lCl(cmb,self.N.modLMap)
            if unlensedEqualsLensed:
                self.N.addUnlensedNorm2DPower(cmb,lClFilt.copy())
                self.N.addUnlensedFilter2DPower(cmb,lClFilt.copy())
            else:
                self.N.addUnlensedNorm2DPower(cmb,uClNorm.copy())
                self.N.addUnlensedFilter2DPower(cmb,uClFilt.copy())
            if lensedEqualsUnlensed:
                self.N.addLensedFilter2DPower(cmb,uClFilt.copy())
            else:
                self.N.addLensedFilter2DPower(cmb,lClFilt.copy())

        Clkk2d = theorySpectra.gCl("kk",self.N.modLMap)    
        self.N.addClkk2DPower(Clkk2d)
        self.N.bigell = bigell

        if bin_edges is not None:
            self.bin_edges = bin_edges
            self.binner = bin2D(self.N.modLMap, bin_edges)

    def updateBins(self,bin_edges):
        self.N.bigell = bin_edges[len(bin_edges)-1]
        self.binner = bin2D(self.N.modLMap, bin_edges)
        self.bin_edges = bin_edges

    def updateNoiseAdvanced(self,beamTX,noiseTX,beamPX,noisePX,tellminX,tellmaxX,pellminX,pellmaxX,beamTY,noiseTY,beamPY,noisePY,tellminY,tellmaxY,pellminY,pellmaxY,lkneesX=[0,0],alphasX=[1,1],lkneesY=[0,0],alphasY=[1,1],lxcutTX=None,lxcutTY=None,lycutTX=None,lycutTY=None,lxcutPX=None,lxcutPY=None,lycutPX=None,lycutPY=None,fgFuncX=None,fgFuncY=None,beamFileTX=None,beamFilePX=None,beamFileTY=None,beamFilePY=None,noiseFuncTX=None,noiseFuncTY=None,noiseFuncPX=None,noiseFuncPY=None):

        self.N.lmax_T = self.N.bigell
        self.N.lmax_P = self.N.bigell

        lkneeTX, lkneePX = lkneesX
        lkneeTY, lkneePY = lkneesY
        alphaTX, alphaPX = alphasX
        alphaTY, alphaPY = alphasY
        

        nTX = maps.whiteNoise2D([noiseTX],beamTX,self.N.modLMap, \
                                 TCMB=self.TCMB,lknees=[lkneeTX],alphas=[alphaTX],\
                                 beamFile=beamFileTX, \
                                 noiseFuncs = [noiseFuncTX])[0]
        nTY = maps.whiteNoise2D([noiseTY],beamTY,self.N.modLMap, \
                                 TCMB=self.TCMB,lknees=[lkneeTY],alphas=[alphaTY], \
                                 beamFile=beamFileTY, \
                                 noiseFuncs=[noiseFuncTY])[0]
        nPX = maps.whiteNoise2D([noisePX],beamPX,self.N.modLMap, \
                                 TCMB=self.TCMB,lknees=[lkneePX],alphas=[alphaPX],\
                                 beamFile=beamFilePX, \
                                 noiseFuncs = [noiseFuncPX])[0]
        nPY = maps.whiteNoise2D([noisePY],beamPY,self.N.modLMap, \
                                 TCMB=self.TCMB,lknees=[lkneePY],alphas=[alphaPY], \
                                 beamFile=beamFilePY, \
                                 noiseFuncs=[noiseFuncPY])[0]


        
        fMaskTX = maps.mask_kspace(self.shape,self.wcs,lmin=tellminX,lmax=tellmaxX,lxcut=lxcutTX,lycut=lycutTX)
        fMaskTY = maps.mask_kspace(self.shape,self.wcs,lmin=tellminY,lmax=tellmaxY,lxcut=lxcutTY,lycut=lycutTY)
        fMaskPX = maps.mask_kspace(self.shape,self.wcs,lmin=pellminX,lmax=pellmaxX,lxcut=lxcutPX,lycut=lycutPX)
        fMaskPY = maps.mask_kspace(self.shape,self.wcs,lmin=pellminY,lmax=pellmaxY,lxcut=lxcutPY,lycut=lycutPY)

        if fgFuncX is not None:
            fg2d = fgFuncX(self.N.modLMap) #/ self.TCMB**2.
            nTX += fg2d
        if fgFuncY is not None:
            fg2d = fgFuncY(self.N.modLMap) #/ self.TCMB**2.
            nTY += fg2d

            
        nList = ['TT','EE','BB']

        nListX = [nTX,nPX,nPX]
        nListY = [nTY,nPY,nPY]
        fListX = [fMaskTX,fMaskPX,fMaskPX]
        fListY = [fMaskTY,fMaskPY,fMaskPY]
        for i,noise in enumerate(nList):
            self.N.addNoise2DPowerXX(noise,nListX[i],fListX[i])
            self.N.addNoise2DPowerYY(noise,nListY[i],fListY[i])

        return nTX,nPX,nTY,nPY

        
    def updateNoise(self,beamX,noiseTX,noisePX,tellminX,tellmaxX,pellminX,pellmaxX,beamY=None,noiseTY=None,noisePY=None,tellminY=None,tellmaxY=None,pellminY=None,pellmaxY=None,lkneesX=[0.,0.],alphasX=[1.,1.],lkneesY=[0.,0.],alphasY=[1.,1.],lxcutTX=0,lxcutTY=0,lycutTX=0,lycutTY=0,lxcutPX=0,lxcutPY=0,lycutPX=0,lycutPY=0,fgFuncX=None,beamFileX=None,fgFuncY=None,beamFileY=None,noiseFuncTX=None,noiseFuncTY=None,noiseFuncPX=None,noiseFuncPY=None,bellminY=None,bellmaxY=None):

        def setDefault(A,B):
            if A is None:
                return B
            else:
                return A

        beamY = setDefault(beamY,beamX)
        noiseTY = setDefault(noiseTY,noiseTX)
        noisePY = setDefault(noisePY,noisePX)
        tellminY = setDefault(tellminY,tellminX)
        pellminY = setDefault(pellminY,pellminX)
        tellmaxY = setDefault(tellmaxY,tellmaxX)
        pellmaxY = setDefault(pellmaxY,pellmaxX)
        bellminY = setDefault(bellminY,pellminY)
        bellmaxY = setDefault(bellmaxY,pellmaxY)
        

        self.N.lmax_T = self.N.bigell
        self.N.lmax_P = self.N.bigell

        
        nTX,nPX = maps.whiteNoise2D([noiseTX,noisePX],beamX,self.N.modLMap, \
                                     TCMB=self.TCMB,lknees=lkneesX,alphas=alphasX,beamFile=beamFileX, \
                                     noiseFuncs = [noiseFuncTX,noiseFuncPX])
        nTY,nPY = maps.whiteNoise2D([noiseTY,noisePY],beamY,self.N.modLMap, \
                                     TCMB=self.TCMB,lknees=lkneesY,alphas=alphasY,beamFile=beamFileY, \
                                     noiseFuncs=[noiseFuncTY,noiseFuncPY])


        ### DEBUG
        # beam = 1.5
        # noise = 5.
        # from orphics import cosmology,io
        # import sys
        # nTX = cosmology.white_noise_with_atm_func(self.N.modLMap,noise,0,1,dimensionless=False,TCMB=2.7255e6)/maps.gauss_beam(self.N.modLMap,beam)**2.
        # nTY = nTX.copy()
        # nPX = nTX.copy()
        # nPY = nTX.copy()

        # # ells = np.arange(2,6000)
        # # nTX = cosmology.white_noise_with_atm_func(ells,noise,0,1,dimensionless=False,TCMB=2.7255e6)/maps.gauss_beam(ells,beam)**2.
        
        # # pl = io.Plotter(yscale='log')
        # # pl.add(ells,ells**2.*self.theory.lCl('TT',ells))
        # # pl.add(ells,nTX*ells**2.)
        # # pl.done()
        # # sys.exit()

        # print(tellminX,tellmaxX,tellminY,tellmaxY)

        ####
        
        
        fMaskTX = maps.mask_kspace(self.shape,self.wcs,lmin=tellminX,lmax=tellmaxX,lxcut=lxcutTX,lycut=lycutTX)
        fMaskTY = maps.mask_kspace(self.shape,self.wcs,lmin=tellminY,lmax=tellmaxY,lxcut=lxcutTY,lycut=lycutTY)
        fMaskPX = maps.mask_kspace(self.shape,self.wcs,lmin=pellminX,lmax=pellmaxX,lxcut=lxcutPX,lycut=lycutPX)
        fMaskPY = maps.mask_kspace(self.shape,self.wcs,lmin=pellminY,lmax=pellmaxY,lxcut=lxcutPY,lycut=lycutPY)
        fMaskBX = maps.mask_kspace(self.shape,self.wcs,lmin=pellminX,lmax=pellmaxX,lxcut=lxcutPX,lycut=lycutPX)
        fMaskBY = maps.mask_kspace(self.shape,self.wcs,lmin=bellminY,lmax=bellmaxY,lxcut=lxcutPY,lycut=lycutPY)
                

        if fgFuncX is not None:
            fg2d = fgFuncX(self.N.modLMap) #/ self.TCMB**2.
            nTX += fg2d
        if fgFuncY is not None:
            fg2d = fgFuncY(self.N.modLMap) #/ self.TCMB**2.
            nTY += fg2d

            
        nList = ['TT','EE','BB']

        nListX = [nTX,nPX,nPX]
        nListY = [nTY,nPY,nPY]
        fListX = [fMaskTX,fMaskPX,fMaskBX]
        fListY = [fMaskTY,fMaskPY,fMaskBY]
        for i,noise in enumerate(nList):
            self.N.addNoise2DPowerXX(noise,nListX[i],fListX[i])
            self.N.addNoise2DPowerYY(noise,nListY[i],fListY[i])

        return nTX,nPX,nTY,nPY

    def updateNoiseSimple(self,ells,nltt,nlee,lmin,lmax):

        nTX = interp1d(ells,nltt,bounds_error=False,fill_value=0.)(self.N.modLMap)
        nPX = interp1d(ells,nltt,bounds_error=False,fill_value=0.)(self.N.modLMap)
        nTY = nTX
        nPY = nPX
        
        fMaskTX = maps.mask_kspace(self.N.shape,self.N.wcs,lmin=lmin,lmax=lmax)
        fMaskTY = maps.mask_kspace(self.N.shape,self.N.wcs,lmin=lmin,lmax=lmax)
        fMaskPX = maps.mask_kspace(self.N.shape,self.N.wcs,lmin=lmin,lmax=lmax)
        fMaskPY = maps.mask_kspace(self.N.shape,self.N.wcs,lmin=lmin,lmax=lmax)

            
        nList = ['TT','EE','BB']

        nListX = [nTX,nPX,nPX]
        nListY = [nTY,nPY,nPY]
        fListX = [fMaskTX,fMaskPX,fMaskPX]
        fListY = [fMaskTY,fMaskPY,fMaskPY]
        for i,noise in enumerate(nList):
            self.N.addNoise2DPowerXX(noise,nListX[i],fListX[i])
            self.N.addNoise2DPowerYY(noise,nListY[i],fListY[i])

        return nTX,nPX,nTY,nPY
    
    def getNl(self,polComb='TT',halo=True):            

        AL = self.N.getNlkk2d(polComb,halo=halo)
        data2d = self.N.Nlkk[polComb]

        centers, Nlbinned = self.binner.bin(data2d)
        Nlbinned = sanitizePower(Nlbinned)
        
        return centers, Nlbinned

    def getNlIterative(self,polCombs,pellmin,pellmax,dell=20,halo=True,dTolPercentage=1.,verbose=True,plot=False,max_iterations=np.inf,eff_at=60,kappa_min=0,kappa_max=np.inf):

        kmax = max(pellmax,kappa_max)
        kmin = 2
        fmask = maps.mask_kspace(self.shape,self.wcs,lmin=kappa_min,lmax=kappa_max)
        Nleach = {}
        bin_edges = np.arange(2,kmax+dell/2.,dell)
        for polComb in polCombs:
            self.updateBins(bin_edges)
            AL = self.N.getNlkk2d(polComb,halo=halo)
            data2d = self.N.Nlkk[polComb]
            ls, Nls = self.binner.bin(data2d)
            Nls = sanitizePower(Nls)
            Nleach[polComb] = (ls,Nls)

        if ('EB' not in polCombs) and ('TB' not in polCombs):
            Nlret = Nlmv(Nleach,polCombs,None,None,bin_edges)
            return bin_edges,sanitizePower(Nlret),None,None,None

        origBB = self.N.lClFid2d['BB'].copy()
        delensBinner =  bin2D(self.N.modLMap, bin_edges)
        ellsOrig, oclbb = delensBinner.bin(origBB.real)
        oclbb = sanitizePower(oclbb)
        origclbb = oclbb.copy()

        if plot:
            from orphics.tools.io import Plotter
            pl = Plotter(scaleY='log',scaleX='log')
            pl.add(ellsOrig,oclbb*ellsOrig**2.,color='black',lw=2)

        ctol = np.inf
        inum = 0
        while ctol>dTolPercentage:
            if inum >= max_iterations: break
            bNlsinv = 0.
            polPass = list(polCombs)
            if verbose: print("Performing iteration ", inum+1)
            for pol in ['EB','TB']:
                if not(pol in polCombs): continue
                Al2d = self.N.getNlkk2d(pol,halo)
                centers, nlkkeach = delensBinner.bin(self.N.Nlkk[pol])
                nlkkeach = sanitizePower(nlkkeach)
                bNlsinv += 1./nlkkeach
                polPass.remove(pol)
            nlkk = 1./bNlsinv
            
            Nldelens = Nlmv(Nleach,polPass,centers,nlkk,bin_edges)
            Nldelens2d = interp1d(bin_edges,Nldelens,fill_value=0.,bounds_error=False)(self.N.modLMap)

            bbNoise2D = self.N.delensClBB(Nldelens2d,fmask=fmask,halo=halo)
            ells, dclbb = delensBinner.bin(bbNoise2D)
            dclbb = sanitizePower(dclbb)
            if inum>0:
                newLens = np.nanmean(nlkk)
                oldLens = np.nanmean(oldNl)
                new = np.nanmean(dclbb)
                old = np.nanmean(oclbb)
                ctol = np.abs(old-new)*100./new
                ctolLens = np.abs(oldLens-newLens)*100./newLens
                if verbose: print("Percentage difference between iterations is ",ctol, " compared to requested tolerance of ", dTolPercentage,". Diff of Nlkks is ",ctolLens)
            oldNl = nlkk.copy()
            oclbb = dclbb.copy()
            inum += 1
            if plot:
                pl.add(ells,dclbb*ells**2.,ls="--",alpha=0.5,color="black")

        if plot:
            import os
            pl.done(os.environ['WWW']+'delens.png')
        self.N.lClFid2d['BB'] = origBB.copy()

        def find_nearest(array,value):
            idx = (np.abs(array-value)).argmin()
            return idx

        new_ells,new_bb = ells,dclbb
        new_k_ells,new_nlkk = fillLowEll(bin_edges,sanitizePower(Nldelens),kmin)


        if eff_at is None:
            efficiency = ((origclbb-dclbb)*100./origclbb).max()
        else:
            id_ellO = find_nearest(ellsOrig,eff_at)
            id_ellD = find_nearest(new_ells,eff_at)
            efficiency = ((origclbb[id_ellO]-new_bb[id_ellD])*100./origclbb[id_ellO])

            
        
        
        return new_k_ells,new_nlkk,new_ells,new_bb,efficiency


    def iterativeDelens(self,xy,dTolPercentage=1.0,halo=True,verbose=True):
        assert xy=='EB' or xy=='TB'
        origBB = self.N.lClFid2d['BB'].copy()
        bin_edges = self.bin_edges #np.arange(100.,3000.,20.)
        delensBinner =  bin2D(self.N.modLMap, bin_edges)
        ells, oclbb = delensBinner.bin(origBB)
        oclbb = sanitizePower(oclbb)

        ctol = np.inf
        inum = 0


        
        #from orphics.tools.output import Plotter
        #pl = Plotter(scaleY='log',scaleX='log')
        #pl = Plotter(scaleY='log')
        while ctol>dTolPercentage:
            if verbose: print("Performing iteration ", inum+1)
            Al2d = self.N.getNlkk2d(xy,halo)
            centers, nlkk = delensBinner.bin(self.N.Nlkk[xy])
            nlkk = sanitizePower(nlkk)
            bbNoise2D = self.N.delensClBB(self.N.Nlkk[xy],halo)
            ells, dclbb = delensBinner.bin(bbNoise2D)
            dclbb = sanitizePower(dclbb)
            if inum>0:
                new = np.nanmean(nlkk)
                old = np.nanmean(oldNl)
                ctol = np.abs(old-new)*100./new
                if verbose: print("Percentage difference between iterations is ",ctol, " compared to requested tolerance of ", dTolPercentage)
            oldNl = nlkk.copy()
            inum += 1
            #pl.add(centers,nlkk)
            #pl.add(ells,dclbb*ells**2.)
        #pl.done('output/delens'+xy+'.png')
        self.N.lClFid2d['BB'] = origBB.copy()
        efficiency = (np.max(oclbb)-np.max(dclbb))*100./np.max(oclbb)
        return centers,nlkk,efficiency
    
class Estimator(object):
    '''
    Flat-sky lensing and Omega quadratic estimators
    Functionality includes:
    - small-scale lens estimation with gradient cutoff
    - combine maps from two different experiments


    NOTE: The TE estimator is not identical between large
    and small-scale estimators. Need to test this.
    '''


    def __init__(self,shape,wcs,
                 theorySpectraForFilters,
                 theorySpectraForNorm=None,
                 noiseX2dTEB=[None,None,None],
                 noiseY2dTEB=[None,None,None],
                 noiseX_is_total = False,
                 noiseY_is_total = False,
                 fmaskX2dTEB=[None,None,None],
                 fmaskY2dTEB=[None,None,None],
                 fmaskKappa=None,
                 kBeamX = None,
                 kBeamY = None,
                 doCurl=False,
                 TOnly=False,
                 halo=True,
                 gradCut=None,
                 verbose=False,
                 loadPickledNormAndFilters=None,
                 savePickledNormAndFilters=None,
                 uEqualsL=False,
                 bigell=9000,
                 mpi_comm=None,
                 lEqualsU=False):

        '''
        All the 2d fourier objects below are pre-fftshifting. They must be of the same dimension.

        shape,wcs: enmap geometry
        theorySpectraForFilters: an orphics.tools.cmb.TheorySpectra object with CMB Cls loaded
        theorySpectraForNorm=None: same as above but if you want to use a different cosmology in the expected value of the 2-pt
        noiseX2dTEB=[None,None,None]: a list of 2d arrays that corresponds to the noise power in T, E, B (same units as Cls above)
        noiseY2dTEB=[None,None,None]: the same as above but if you want to use a different experiment for the Y maps
        fmaskX2dTEB=[None,None,None]: a list of 2d integer arrays where 1 corresponds to modes included and 0 to those not included
        fmaskY2dTEB=[None,None,None]: same as above but for Y maps
        fmaskKappa=None: same as above but for output kappa map
        doCurl=False: return curl Omega estimates too? If yes, output of getKappa will be (kappa,curl)
        TOnly=False: do only TT? If yes, others will not be initialized and you'll get errors if you try to getKappa(XY) for XY!=TT
        halo=False: use the halo lensing estimators?
        gradCut=None: if using halo lensing estimators, specify an integer up to what L the X map will be retained
        verbose=False: print some occasional output?

        '''

        self.verbose = verbose

        # initialize norm and filters

        self.doCurl = doCurl



        if loadPickledNormAndFilters is not None:
            if verbose: print("Unpickling...")
            with open(loadPickledNormAndFilters,'rb') as fin:
                self.N,self.AL,self.OmAL,self.fmaskK,self.phaseY = pickle.load(fin)
            return



        self.halo = halo
        self.AL = {}
        if doCurl: self.OmAL = {}

        if kBeamX is not None:           
            self.kBeamX = kBeamX
        else:
            self.kBeamX = 1.
            
        if kBeamY is not None:           
            self.kBeamY = kBeamY
        else:
            self.kBeamY = 1.

        self.doCurl = doCurl
        self.halo = halo

        if fmaskKappa is None:
            ellMinK = 80
            ellMaxK = 3000
            print("WARNING: using default kappa mask of 80 < L < 3000")
            self.fmaskK = fmaps.fourierMask(self.N.lx,self.N.ly,self.N.modLMap,lmin=ellMinK,lmax=ellMaxK)
        else:
            self.fmaskK = fmaskKappa

        
        self.fmaskX2dTEB = fmaskX2dTEB
        self.fmaskY2dTEB = fmaskY2dTEB
        
        # Get MPI comm
        comm = mpi_comm
        if comm is not None:
            rank = comm.Get_rank()
            numcores = comm.Get_size()
        else:
            rank = 0
            numcores = 1

        self.wcs = wcs
        if rank==0:
            self.N = QuadNorm(shape,wcs,gradCut=gradCut,verbose=verbose,kBeamX=self.kBeamX,kBeamY=self.kBeamY,bigell=bigell,fmask=self.fmaskK)


            if TOnly: 
                nList = ['TT']
                cmbList = ['TT']
                estList = ['TT']
                self.phaseY = 1.
            else:
                self.phaseY = np.cos(2.*self.N.thetaMap)+1.j*np.sin(2.*self.N.thetaMap)
                nList = ['TT','EE','BB']
                cmbList = ['TT','TE','EE','BB']
                #estList = ['TT','TE','ET','EB','EE','TB']
                estList = ['TT','TE','ET','EB','EE','TB','BE']

            self.nList = nList

            if self.verbose: print("Initializing filters and normalization for quadratic estimators...")
            assert not(uEqualsL and lEqualsU)
            for cmb in cmbList:
                if uEqualsL:
                    uClFilt = theorySpectraForFilters.lCl(cmb,self.N.modLMap)
                else:
                    uClFilt = theorySpectraForFilters.uCl(cmb,self.N.modLMap)

                if theorySpectraForNorm is not None:
                    if uEqualsL:
                        uClNorm = theorySpectraForFilters.lCl(cmb,self.N.modLMap)
                    else:
                        uClNorm = theorySpectraForNorm.uCl(cmb,self.N.modLMap)
                else:
                    uClNorm = uClFilt

                if lEqualsU:
                    lClFilt = theorySpectraForFilters.uCl(cmb,self.N.modLMap)
                else:
                    lClFilt = theorySpectraForFilters.lCl(cmb,self.N.modLMap)
                    
                #lClFilt = theorySpectraForFilters.lCl(cmb,self.N.modLMap)
                self.N.addUnlensedFilter2DPower(cmb,uClFilt)
                self.N.addLensedFilter2DPower(cmb,lClFilt)
                self.N.addUnlensedNorm2DPower(cmb,uClNorm)
            for i,noise in enumerate(nList):
                self.N.addNoise2DPowerXX(noise,noiseX2dTEB[i],fmaskX2dTEB[i],is_total=noiseX_is_total)
                self.N.addNoise2DPowerYY(noise,noiseY2dTEB[i],fmaskY2dTEB[i],is_total=noiseY_is_total)
            try:
                self.N.addClkk2DPower(theorySpectraForFilters.gCl("kk",self.N.modLMap))
            except:
                print("Couldn't add Clkk2d power")

            self.estList = estList
            self.OmAL = None
            for est in estList:
                self.AL[est] = self.N.getNlkk2d(est,halo=halo)
                #if doCurl: self.OmAL[est] = self.N.getCurlNlkk2d(est,halo=halo)
                
                # send_dat = np.array(self.vectors[label]).astype(np.float64)
                # self.comm.Send(send_dat, dest=0, tag=self.tag_start+k)

        else:

            pass
        

    def updateNoise(self,nTX,nEX,nBX,nTY,nEY,nBY,noiseX_is_total=False,noiseY_is_total=False):
        noiseX2dTEB = [nTX,nEX,nBX]
        noiseY2dTEB = [nTY,nEY,nBY]
        for i,noise in enumerate(self.nList):
            self.N.addNoise2DPowerXX(noise,noiseX2dTEB[i],self.fmaskX2dTEB[i],is_total=noiseX_is_total)
            self.N.addNoise2DPowerYY(noise,noiseY2dTEB[i],self.fmaskY2dTEB[i],is_total=noiseY_is_total)

        for est in self.estList:
            self.AL[est] = self.N.getNlkk2d(est,halo=self.halo)
            if self.doCurl: self.OmAL[est] = self.N.getCurlNlkk2d(est,halo=self.halo)
            

    def updateTEB_X(self,T2DData,E2DData=None,B2DData=None,alreadyFTed=False):
        '''
        Masking and windowing and apodizing and beam deconvolution has to be done beforehand!

        Maps must have units corresponding to those of theory Cls and noise power
        '''
        self._hasX = True

        self.kGradx = {}
        self.kGrady = {}

        lx = self.N.lxMap
        ly = self.N.lyMap

        if alreadyFTed:
            self.kT = T2DData
        else:
            self.kT = fft(T2DData,axes=[-2,-1])
        self.kGradx['T'] = lx*self.kT.copy()*1j
        self.kGrady['T'] = ly*self.kT.copy()*1j

        if E2DData is not None:
            if alreadyFTed:
                self.kE = E2DData
            else:
                self.kE = fft(E2DData,axes=[-2,-1])
            self.kGradx['E'] = 1.j*lx*self.kE.copy()
            self.kGrady['E'] = 1.j*ly*self.kE.copy()
        if B2DData is not None:
            if alreadyFTed:
                self.kB = B2DData
            else:
                self.kB = fft(B2DData,axes=[-2,-1])
            self.kGradx['B'] = 1.j*lx*self.kB.copy()
            self.kGrady['B'] = 1.j*ly*self.kB.copy()
        
        

    def updateTEB_Y(self,T2DData=None,E2DData=None,B2DData=None,alreadyFTed=False):
        assert self._hasX, "Need to initialize gradient first."
        self._hasY = True
        
        self.kHigh = {}

        if T2DData is not None:
            if alreadyFTed:
                self.kHigh['T']=T2DData
            else:
                self.kHigh['T']=fft(T2DData,axes=[-2,-1])
        else:
            self.kHigh['T']=self.kT.copy()
        if E2DData is not None:
            if alreadyFTed:
                self.kHigh['E']=E2DData
            else:
                self.kHigh['E']=fft(E2DData,axes=[-2,-1])
        else:
            try:
                self.kHigh['E']=self.kE.copy()
            except:
                pass

        if B2DData is not None:
            if alreadyFTed:
                self.kHigh['B']=B2DData
            else:
                self.kHigh['B']=fft(B2DData,axes=[-2,-1])
        else:
            try:
                self.kHigh['B']=self.kB.copy()
            except:
                pass

    def kappa_from_map(self,XY,T2DData,E2DData=None,B2DData=None,T2DDataY=None,E2DDataY=None,B2DDataY=None,alreadyFTed=False,returnFt=False):
        self.updateTEB_X(T2DData,E2DData,B2DData,alreadyFTed)
        self.updateTEB_Y(T2DDataY,E2DDataY,B2DDataY,alreadyFTed)
        return self.get_kappa(XY,returnFt=returnFt)
        
        
    def fmask_func(self,arr):
        fMask = self.fmaskK
        arr[fMask<1.e-3] = 0.
        return arr

    def coadd_nlkk(self,ests):
        ninvtot = 0.
        for est in ests:
            ninvtot += self.fmask_func(np.nan_to_num(1./self.N.Nlkk[est]))
        return self.fmask_func(np.nan_to_num(1./ninvtot))
    
    def coadd_kappa(self,ests,returnFt=False):
        ktot = 0.
        for est in ests:
            rkappa = self.get_kappa(est,returnFt=True)
            ktot += self.fmask_func(np.nan_to_num(rkappa/self.N.Nlkk[est]))
        kft = ktot*self.coadd_nlkk(ests)
        if returnFt: return kft
        return ifft(kft,axes=[-2,-1],normalize=True).real
    
    def get_kappa(self,XY,returnFt=False):

        assert self._hasX and self._hasY
        assert XY in ['TT','TE','ET','EB','TB','EE','BE']
        X,Y = XY

        WXY = self.N.WXY(XY)
        WY = self.N.WY(Y+Y)



        lx = self.N.lxMap
        ly = self.N.lyMap

        if Y in ['E','B']:
            phaseY = self.phaseY
        else:
            phaseY = 1.

        phaseB = (int(Y=='B')*1.j)+(int(Y!='B'))
        
        fMask = self.fmaskK

        if self.verbose: startTime = time.time()

        HighMapStar = ifft((self.kHigh[Y]*WY*phaseY*phaseB),axes=[-2,-1],normalize=True).conjugate()
        kPx = fft(ifft(self.kGradx[X]*WXY*phaseY,axes=[-2,-1],normalize=True)*HighMapStar,axes=[-2,-1])
        kPy = fft(ifft(self.kGrady[X]*WXY*phaseY,axes=[-2,-1],normalize=True)*HighMapStar,axes=[-2,-1])        
        rawKappa = ifft((1.j*lx*kPx) + (1.j*ly*kPy),axes=[-2,-1],normalize=True).real

        AL = np.nan_to_num(self.AL[XY])


        assert not(np.any(np.isnan(rawKappa)))
        lmap = self.N.modLMap

        kappaft = -self.fmask_func(AL*fft(rawKappa,axes=[-2,-1]))
        
        if returnFt:
            return kappaft
        
        self.kappa = enmap.enmap(ifft(kappaft,axes=[-2,-1],normalize=True).real,self.wcs)
        try:
            assert not(np.any(np.isnan(self.kappa)))
        except:
            import orphics.tools.io as io
            import orphics.tools.stats as stats
            io.quickPlot2d(np.fft.fftshift(np.abs(kappaft)),"ftkappa.png")
            io.quickPlot2d(np.fft.fftshift(fMask),"fmask.png")
            io.quickPlot2d(self.kappa.real,"nankappa.png")
            debug_edges = np.arange(20,20000,100)
            dbinner = stats.bin2D(self.N.modLMap,debug_edges)
            cents, bclkk = dbinner.bin(self.N.clkk2d)
            cents, nlkktt = dbinner.bin(self.N.Nlkk['TT'])
            cents, alkktt = dbinner.bin(AL/2.*lmap*(lmap+1.))
            try:
                cents, nlkkeb = dbinner.bin(self.N.Nlkk['EB'])
            except:
                pass
            pl = io.Plotter(scaleY='log',scaleX='log')
            pl.add(cents,bclkk)
            pl.add(cents,nlkktt,label="TT")
            pl.add(cents,alkktt,label="TTnorm",ls="--")
            try:
                pl.add(cents,nlkkeb,label="EB")
            except:
                pass
            pl.legendOn()
            pl._ax.set_ylim(1.e-9,1.e-5)
            pl.done("clkk.png")

            sys.exit()
        
            
        # from orphics.tools.io import Plotter
        # pl = Plotter()
        # #pl.plot2d(np.nan_to_num(self.kappa))
        # pl.plot2d((self.kappa.real))
        # pl.done("output/nankappa.png")
        # sys.exit(0)
        # try:
        #     assert not(np.any(np.isnan(self.kappa)))
        # except:
        #     from orphics.tools.io import Plotter
        #     pl = Plotter()
        #     pl.plot2d(np.nan_to_num(self.kappa))
        #     pl.done("output/nankappa.png")
        #     sys.exit(0)

        # if self.verbose:
        #     elapTime = time.time() - startTime
        #     print(("Time for core kappa was ", elapTime ," seconds."))

        # if self.doCurl:
        #     OmAL = self.OmAL[XY]*fMask
        #     rawCurl = ifft(1.j*lx*kPy - 1.j*ly*kPx,axes=[-2,-1],normalize=True).real
        #     self.curl = -ifft(OmAL*fft(rawCurl,axes=[-2,-1]),axes=[-2,-1],normalize=True)
        #     return self.kappa, self.curl



        return self.kappa





def Nlmv(Nleach,pols,centers,nlkk,bin_edges):
    # Nleach: dict of (ls,Nls) for each polComb
    # pols: list of polCombs to include
    # centers,nlkk: additonal Nl to add
    
    Nlmvinv = 0.
    for polComb in pols:
        ls,Nls = Nleach[polComb]
        nlfunc = interp1d(ls,Nls,bounds_error=False,fill_value=np.inf)
        Nleval = nlfunc(bin_edges)
        Nlmvinv += np.nan_to_num(1./Nleval)
        
    if nlkk is not None:
        nlfunc = interp1d(centers,nlkk,bounds_error=False,fill_value=np.inf)
        Nleval = nlfunc(bin_edges)
        Nlmvinv += np.nan_to_num(1./Nleval)
        
    return np.nan_to_num(1./Nlmvinv)


## HALOS

# g(x) = g(theta/thetaS) HuDeDeoVale 2007
gnfw = lambda x: np.piecewise(x, [x>1., x<1., x==1.], \
                            [lambda y: (1./(y*y - 1.)) * \
                             ( 1. - ( (2./np.sqrt(y*y - 1.)) * np.arctan(np.sqrt((y-1.)/(y+1.))) ) ), \
                             lambda y: (1./(y*y - 1.)) * \
                            ( 1. - ( (2./np.sqrt(-(y*y - 1.))) * np.arctanh(np.sqrt(-((y-1.)/(y+1.)))) ) ), \
                        lambda y: (1./3.)])

f_c = lambda c: np.log(1.+c) - (c/(1.+c))


def nfw_kappa(massOverh,modrmap_radians,cc,zL=0.7,concentration=3.2,overdensity=180.,critical=False,atClusterZ=False):
    sgn = 1. if massOverh>0. else -1.
    comS = cc.results.comoving_radial_distance(cc.cmbZ)*cc.h
    comL = cc.results.comoving_radial_distance(zL)*cc.h
    winAtLens = (comS-comL)/comS
    kappa,r500 = NFWkappa(cc,np.abs(massOverh),concentration,zL,modrmap_radians* 180.*60./np.pi,winAtLens,
                          overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)

    return sgn*kappa



def NFWkappa(cc,massOverh,concentration,zL,thetaArc,winAtLens,overdensity=500.,critical=True,atClusterZ=True):

    comL  = (cc.results.comoving_radial_distance(zL) )*cc.h

    

    c = concentration
    M = massOverh

    zdensity = 0.
    if atClusterZ: zdensity = zL

    if critical:
        r500 = cc.rdel_c(M,zdensity,overdensity).flatten()[0] # R500 in Mpc/h
    else:
        r500 = cc.rdel_m(M,zdensity,overdensity) # R500 in Mpc/h

    conv=np.pi/(180.*60.)
    theta = thetaArc*conv # theta in radians

    rS = r500/c

    thetaS = rS/ comL 


    const12 = 9.571e-20 # 2G/c^2 in Mpc / solar mass 
    fc = np.log(1.+c) - (c/(1.+c))    
    #const3 = comL * comLS * (1.+zL) / comS #  Mpc
    const3 = comL *  (1.+zL) *winAtLens #  Mpc
    const4 = M / (rS*rS) #solar mass / MPc^2
    const5 = 1./fc
    
    kappaU = gnfw(theta/thetaS)+theta*0. # added for compatibility with enmap

    consts = const12 * const3 * const4 * const5
    kappa = consts * kappaU

    if thetaArc.shape[0]%2==1 and thetaArc.shape[1]%2==1:
        Ny,Nx = thetaArc.shape
        cx = int(Nx/2.)
        cy = int(Ny/2.)
        kappa[cy,cx] = kappa[cy-1,cx]
        
    assert np.all(np.isfinite(kappa))
    return kappa, r500



def NFWMatchedFilterSN(clusterCosmology,log10Moverh,c,z,ells,Nls,kellmax,overdensity=500.,critical=True,atClusterZ=True,arcStamp=100.,pxStamp=0.05,saveId=None,verbose=False,rayleighSigmaArcmin=None,returnKappa=False,winAtLens=None):
    if rayleighSigmaArcmin is not None: assert rayleighSigmaArcmin>=pxStamp
    M = 10.**log10Moverh

    
    shape,wcs = maps.rect_geometry(width_deg=arcStamp/60.,px_res_arcmin=pxStamp)
    kellmin = 2.*np.pi/arcStamp*np.pi/60./180.

    modLMap = enmap.modlmap(shape,wcs)
    xMap,yMap,modRMap,xx,yy  = maps.get_real_attributes(shape,wcs)
        
    cc = clusterCosmology

    cmb = False
    if winAtLens is None:
        cmb = True
        comS = cc.results.comoving_radial_distance(cc.cmbZ)*cc.h
        comL = cc.results.comoving_radial_distance(z)*cc.h
        winAtLens = (comS-comL)/comS

    kappaReal, r500 = NFWkappa(cc,M,c,z,modRMap*180.*60./np.pi,winAtLens,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)
    
    dAz = cc.results.angular_diameter_distance(z) * cc.h
    # print ("daz " , dAz , " mpc")
    # print ("r500 " , r500 , " mpc")
    th500 = r500/dAz
    #fiveth500 = 10.*np.pi/180./60. #5.*th500
    fiveth500 = 5.*th500
    # print ("5theta500 " , fiveth500*180.*60./np.pi , " arcminutes")
    # print ("maximum theta " , modRMap.max()*180.*60./np.pi, " arcminutes")

    kInt = kappaReal.copy()
    kInt[modRMap>fiveth500] = 0.
    # print "mean kappa inside theta500 " , kInt[modRMap<fiveth500].mean()
    # print "area of th500 disc " , np.pi*fiveth500**2.*(180.*60./np.pi)**2.
    # print "estimated integral " , kInt[modRMap<fiveth500].mean()*np.pi*fiveth500**2.
    k500 = simps(simps(kInt, yy), xx)
    
    if verbose: print(("integral of kappa inside disc ",k500))
    kappaReal[modRMap>fiveth500] = 0. #### !!!!!!!!! Might not be necessary!
    # if cmb: print z,fiveth500*180.*60./np.pi
    Ukappa = kappaReal/k500


    
    # pl = Plotter()
    # pl.plot2d(Ukappa)
    # pl.done("output/kappa.png")

    ellmax = kellmax
    ellmin = kellmin

    
    
    Uft = fft(Ukappa,axes=[-2,-1])

    if rayleighSigmaArcmin is not None:
        Prayleigh = rayleigh(modRMap*180.*60./np.pi,rayleighSigmaArcmin)
        outDir = "/gpfs01/astro/www/msyriac/plots/"
        # io.quickPlot2d(Prayleigh,outDir+"rayleigh.png")
        rayK = fft(ifftshift(Prayleigh),axes=[-2,-1])
        rayK /= rayK[modLMap<1.e-3]
        Uft = Uft.copy()*rayK
    
    Upower = np.real(Uft*Uft.conjugate())

    

    # pl = Plotter()
    # pl.plot2d(fftshift(Upower))
    # pl.done("output/upower.png")


    
    Nls[Nls<0.]=0.
    s = splrep(ells,Nls,k=3)
    Nl2d = splev(modLMap,s) 
    
    Nl2d[modLMap<ellmin]=np.inf
    Nl2d[modLMap>ellmax] = np.inf

    Ny,Nx = shape
    pixScaleY,pixScaleX = enmap.pixshape(shape,wcs)
    area = Nx*Ny*pixScaleX*pixScaleY
    Upower = Upower *area / (Nx*Ny)**2
        
    filter = np.nan_to_num(Upower/Nl2d)
    #filter = np.nan_to_num(1./Nl2d)
    filter[modLMap>ellmax] = 0.
    filter[modLMap<ellmin] = 0.
    # pl = Plotter()
    # pl.plot2d(fftshift(filter))
    # pl.done("output/filter.png")
    # if (cmb): print Upower.sum()
    # if not(cmb) and z>2.5:
    #     bin_edges = np.arange(500,ellmax,100)
    #     binner = bin2D(modLMap, bin_edges)
    #     centers, nl2dells = binner.bin(Nl2d)
    #     centers, upowerells = binner.bin(np.nan_to_num(Upower))
    #     centers, filterells = binner.bin(filter)
    #     from orphics.tools.io import Plotter
    #     pl = Plotter(scaleY='log')
    #     pl.add(centers,upowerells,label="upower")
    #     pl.add(centers,nl2dells,label="noise")
    #     pl.add(centers,filterells,label="filter")
    #     pl.add(ells,Nls,ls="--")
    #     pl.legendOn(loc='upper right')
    #     #pl._ax.set_ylim(0,1e-8)
    #     pl.done("output/filterells.png")
    #     sys.exit()
    
    varinv = filter.sum()
    std = np.sqrt(1./varinv)
    sn = k500/std
    if verbose: print(sn)

    if saveId is not None:
        np.savetxt("data/"+saveId+"_m"+str(log10Moverh)+"_z"+str(z)+".txt",np.array([log10Moverh,z,1./sn]))

    if returnKappa:
        return sn,ifft(Uft,axes=[-2,-1],normalize=True).real*k500
    return sn, k500, std



    

def rayleigh(theta,sigma):
    sigmasq = sigma*sigma
    #return np.exp(-0.5*theta*theta/sigmasq)
    return theta/sigmasq*np.exp(-0.5*theta*theta/sigmasq)
        




# NFW dimensionless form
fnfw = lambda x: 1./(x*((1.+x)**2.))
Gval = 4.517e-48 # Newton G in Mpc,seconds,Msun units
cval = 9.716e-15 # speed of light in Mpc,second units

# NFW density (M/L^3) as a function of distance from center of cluster
def rho_nfw(M,c,R):
    return lambda r: 1./(4.*np.pi)*((c/R)**3.)*M/f_c(c)*fnfw(c*r/R)

# NFW projected along line of sight (M/L^2) as a function of angle on the sky in radians
def proj_rho_nfw(theta,comL,M,c,R):
    thetaS = R/c/comL
    return 1./(4.*np.pi)*((c/R)**2.)*M/f_c(c)*(2.*gnfw(theta/thetaS))

# Generic profile projected along line of sight (M/L^2) as a function of angle on the sky in radians
# rhoFunc is density (M/L^3) as a function of distance from center of cluster
def projected_rho(thetas,comL,rhoFunc,pmaxN=2000,numps=500000):
    # default integration times are good to 0.01% for z=0.1 to 3
    # increase numps for lower z/theta and pmaxN for higher z/theta
    # g(x) = \int dl rho(sqrt(l**2+x**2)) = g(theta/thetaS)
    pzrange = np.linspace(-pmaxN,pmaxN,numps)
    g = np.array([np.trapz(rhoFunc(np.sqrt(pzrange**2.+(theta*comL)**2.)),pzrange) for theta in thetas])
    return g


def kappa_nfw_generic(theta,z,comLMpcOverh,M,c,R,windowAtLens):
    return 4.*np.pi*Gval*(1+z)*comLMpcOverh*windowAtLens*proj_rho_nfw(theta,comLMpcOverh,M,c,R)/cval**2.

def kappa_generic(theta,z,comLMpcOverh,rhoFunc,windowAtLens,pmaxN=2000,numps=500000):
    # default integration times are good to 0.01% for z=0.1 to 3
    # increase numps for lower z/theta and pmaxN for higher z/theta
    return 4.*np.pi*Gval*(1+z)*comLMpcOverh*windowAtLens*projected_rho(theta,comLMpcOverh,rhoFunc,pmaxN,numps)/cval**2.

def kappa_from_rhofunc(M,c,R,theta,cc,z,rhoFunc=None):
    if rhoFunc is None: rhoFunc = rho_nfw(M,c,R)
    sgn = 1. if M>0. else -1.
    comS = cc.results.comoving_radial_distance(cc.cmbZ)*cc.h
    comL = cc.results.comoving_radial_distance(z)*cc.h
    winAtLens = (comS-comL)/comS
    kappa = kappa_generic(theta,z,comL,rhoFunc,winAtLens)
    return sgn*kappa

def kappa_nfw(M,c,R,theta,cc,z):
    sgn = 1. if M>0. else -1.
    comS = cc.results.comoving_radial_distance(cc.cmbZ)*cc.h
    comL = cc.results.comoving_radial_distance(z)*cc.h
    winAtLens = (comS-comL)/comS
    kappa = kappa_nfw_generic(theta,z,comL,np.abs(M),c,R,winAtLens)
    return sgn*kappa


class SplitLensing(object):
    def __init__(self,shape,wcs,qest,XY="TT"):
        # PS calculator
        self.fc = maps.FourierCalc(shape,wcs)
        self.qest = qest
        self.est = XY

    def qpower(self,k1,k2):
        # PS func
        return self.fc.f2power(k1,k2)

    def qfrag(self,a,b):
        # kappa func (accepts fts, returns ft)
        if self.est=='TT':
            k1 = self.qest.kappa_from_map(self.est,T2DData=a.copy(),T2DDataY=b.copy(),alreadyFTed=True,returnFt=True)
        elif self.est=='EE': # wrong!
            k1 = self.qest.kappa_from_map(self.est,T2DData=a.copy(),E2DData=a.copy(),B2DData=a.copy(),
                                          T2DDataY=b.copy(),E2DDataY=b.copy(),B2DDataY=b.copy(),alreadyFTed=True,returnFt=True)
            
        return k1

    def cross_estimator(self,ksplits):
        # 4pt from splits
        splits = ksplits
        splits = np.asanyarray(ksplits)
        insplits = splits.shape[0]
        nsplits = float(insplits)
        s = np.mean(splits,axis=0)
        k = self.qfrag(s,s)
        kiisum = 0.
        psum = 0.
        psum2 = 0.
        for i in range(insplits):
            mi = splits[i]
            ki = (self.qfrag(mi,s)+self.qfrag(s,mi))/2.
            kii = self.qfrag(mi,mi)
            kiisum += kii
            kic = ki - (1./nsplits)*kii
            psum += self.qpower(kic,kic)
            for j in range(i+1,int(insplits)):
                mj = splits[j]
                kij = (self.qfrag(mi,mj)+self.qfrag(mj,mi))/2.
                psum2 += self.qpower(kij,kij)
        kc = k - (1./nsplits**2.)*kiisum
        return (nsplits**4.*self.qpower(kc,kc)-4.*nsplits**2.*psum+4.*psum2)/nsplits/(nsplits-1.)/(nsplits-2.)/(nsplits-3.)

    
class QE(object):
    def __init__(self,shape,wcs,cmb,xnoise,xbeam,ynoise=None,ybeam=None,ests=None,cmb_response=None):
        modlmap = enmap.modlmap(shape,wcs)
        self.modlmap = modlmap
        self.shape = shape
        self.wcs = wcs
        kbeamx = self._process_beam(xbeam)
        kbeamy = self._process_beam(ybeam) if ybeam is not None else kbeamx.copy()
        

    def _process_beam(self,beam):
        beam = np.asarray(beam)
        if beam.ndim==0:
            kbeam = maps.gauss_beam(beam,modlmap)
        elif beam.ndim==1:
            ells = np.arange(0,beam.size)
            kbeam = maps.interp(ells,maps.gauss_beam(beam,ells))(self.modlmap)
        elif beam.ndim==2:
            kbeam = beam
            assert kbeam.shape==self.shape
        return kbeam


    def WXY(self,XY):
        X,Y = XY
        if Y=='B': Y='E'
        gradClXY = X+Y
        if XY=='ET': gradClXY = 'TE'
        if XY=='BE': gradClXY = 'EE'
        totnoise = self.noiseXX2d[X+X].copy() if self.noiseX_is_total else (self.lClFid2d[X+X].copy()+self.noiseXX2d[X+X].copy())
        W = self.fmask_func(np.nan_to_num(self.uClFid2d[gradClXY].copy()/totnoise)*self.kBeamX,self.fMaskXX[X+X])
        W[self.modLMap>self.gradCut]=0.
        if X=='T':
            W[np.where(self.modLMap >= self.lmax_T)] = 0.
        else:
            W[np.where(self.modLMap >= self.lmax_P)] = 0.
        return W
        

    def WY(self,YY):
        assert YY[0]==YY[1]
        totnoise = self.noiseYY2d[YY].copy() if self.noiseY_is_total else (self.lClFid2d[YY].copy()*self.kBeamY**2.+self.noiseYY2d[YY].copy())
        W = self.fmask_func(np.nan_to_num(1./totnoise)*self.kBeamY,self.fMaskYY[YY]) #* self.modLMap  # !!!!!
        W[np.where(self.modLMap >= self.lmax_T)] = 0.
        if YY[0]=='T':
            W[np.where(self.modLMap >= self.lmax_T)] = 0.
        else:
            W[np.where(self.modLMap >= self.lmax_P)] = 0.
        return W


    def reconstruct_from_iqu(self,XYs,imapx,imapy=None,return_ft=True):
        pass
    
    def reconstruct(self,XYs,kmapx=None,kmapy=None,imapx=None,imapy=None,return_ft=True):
        pass
        
    def reconstruct_xy(self,XY,kmapx=None,kmapy=None,imapx=None,imapy=None,return_ft=True):
        X,Y = XY
        WXY = self.WXY(XY)
        WY = self.WY(Y+Y)
        lx = self.lxMap
        ly = self.lyMap
        if Y in ['E','B']:
            phaseY = self.phaseY
        else:
            phaseY = 1.
        phaseB = (int(Y=='B')*1.j)+(int(Y!='B'))
        fMask = self.fmaskK
        HighMapStar = ifft((self.kHigh[Y]*WY*phaseY*phaseB),axes=[-2,-1],normalize=True).conjugate()
        kPx = fft(ifft(self.kGradx[X]*WXY*phaseY,axes=[-2,-1],normalize=True)*HighMapStar,axes=[-2,-1])
        kPy = fft(ifft(self.kGrady[X]*WXY*phaseY,axes=[-2,-1],normalize=True)*HighMapStar,axes=[-2,-1])        
        rawKappa = ifft((1.j*lx*kPx) + (1.j*ly*kPy),axes=[-2,-1],normalize=True).real
        AL = np.nan_to_num(self.AL[XY])
        assert not(np.any(np.isnan(rawKappa)))
        lmap = self.N.modLMap
        kappaft = -self.fmask_func(AL*fft(rawKappa,axes=[-2,-1]))
        if return_ft:
            return kappaft
        else:
            kappa = ifft(kappaft,axes=[-2,-1],normalize=True).real
            return kappa,kappaft
    
    def norm(self,XY):
        kbeamx = self.kbeamx
        kbeamy = self.kbeamy
        allTerms = []
        if XY=='TT':
            clunlenTTArrNow = self.uClNow2d['TT'].copy()
            WXY = self.WXY('TT')*kbeamx*l1Scale
            WY = self.WY('TT')*kbeamy*l2Scale
            preG = WY
            rfact = 2.**0.25
            for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                preF = ell1*ell2*clunlenTTArrNow*WXY
                preFX = ell1*WXY
                preGX = ell2*clunlenTTArrNow*WY
                calc = ell1*ell2*fft(ifft(preF,axes=[-2,-1],normalize=True)*ifft(preG,axes=[-2,-1],normalize=True)
                                     +ifft(preFX,axes=[-2,-1],normalize=True)*ifft(preGX,axes=[-2,-1],normalize=True),axes=[-2,-1])
                allTerms += [calc]
        elif XY == 'EE':
            clunlenEEArrNow = self.uClNow2d['EE'].copy()
            sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)
            lx = self.lxMap
            ly = self.lyMap
            lxhat = self.lxHatMap
            lyhat = self.lyHatMap
            sinf = sin2phi(lxhat,lyhat)
            sinsqf = sinf**2.
            cosf = cos2phi(lxhat,lyhat)
            cossqf = cosf**2.
            WXY = self.WXY('EE')*kbeamx
            WY = self.WY('EE')*kbeamy
            rfact = 2.**0.25
            for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                    preF = trigfact*ell1*ell2*clunlenEEArrNow*WXY
                    preG = trigfact*WY
                    allTerms += [ell1*ell2*fft(ifft(preF,axes=[-2,-1],normalize=True)*ifft(preG,axes=[-2,-1],normalize=True),axes=[-2,-1])]
                    preFX = trigfact*ell1*clunlenEEArrNow*WY
                    preGX = trigfact*ell2*WXY
                    allTerms += [ell1*ell2*fft(ifft(preFX,axes=[-2,-1],normalize=True)*ifft(preGX,axes=[-2,-1],normalize=True),axes=[-2,-1])]
        elif XY == 'EB':
            clunlenEEArrNow = self.uClNow2d['EE'].copy()
            clunlenBBArrNow = self.uClNow2d['BB'].copy()
            sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)
            lx = self.lxMap
            ly = self.lyMap
            termsF = []
            termsF.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )
            termsG = []
            termsG.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )
            lxhat = self.lxHatMap
            lyhat = self.lyHatMap
            WXY = self.WXY('EB')*kbeamx
            WY = self.WY('BB')*kbeamy
            for ellsq in [lx*lx,ly*ly,np.sqrt(2.)*lx*ly]:
                preF = ellsq*clunlenEEArrNow*WXY
                preG = WY
                for termF,termG in zip(termsF,termsG):
                    allTerms += [ellsq*fft(ifft(termF(preF,lxhat,lyhat),axes=[-2,-1],normalize=True)
                                           *ifft(termG(preG,lxhat,lyhat),axes=[-2,-1],normalize=True),axes=[-2,-1])]
        elif XY == 'BE':
            clunlenEEArrNow = self.uClNow2d['EE'].copy()
            clunlenBBArrNow = self.uClNow2d['BB'].copy()
            sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)
            lx = self.lxMap
            ly = self.lyMap
            termsF = []
            termsF.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )
            termsG = []
            termsG.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )
            lxhat = self.lxHatMap
            lyhat = self.lyHatMap
            WXY = self.WXY('BE')*kbeamx
            WY = self.WY('EE')*kbeamy
            for ellsq in [lx*lx,ly*ly,np.sqrt(2.)*lx*ly]:
                preF = WXY
                preG = ellsq*clunlenEEArrNow*WY
                for termF,termG in zip(termsF,termsG):
                    allTerms += [ellsq*fft(ifft(termF(preF,lxhat,lyhat),axes=[-2,-1],normalize=True)
                                           *ifft(termG(preG,lxhat,lyhat),axes=[-2,-1],normalize=True),axes=[-2,-1])]
        elif XY=='ET':
            clunlenTEArrNow = self.uClNow2d['TE'].copy()
            sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

            lx = self.lxMap
            ly = self.lyMap


            lxhat = self.lxHatMap
            lyhat = self.lyHatMap

            sinf = sin2phi(lxhat,lyhat)
            sinsqf = sinf**2.
            cosf = cos2phi(lxhat,lyhat)
            cossqf = cosf**2.


            WXY = self.WXY('ET')*kbeamx
            WY = self.WY('TT')*kbeamy

            rfact = 2.**0.25
            for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                preF = ell1*ell2*clunlenTEArrNow*WXY
                preG = WY
                allTerms += [ell1*ell2*fft(ifft(preF,axes=[-2,-1],normalize=True)*ifft(preG,axes=[-2,-1],normalize=True),axes=[-2,-1])]
                for trigfact in [cosf,sinf]:

                    preFX = trigfact*ell1*clunlenTEArrNow*WY
                    preGX = trigfact*ell2*WXY

                    allTerms += [ell1*ell2*fft(ifft(preFX,axes=[-2,-1],normalize=True)*ifft(preGX,axes=[-2,-1],normalize=True),axes=[-2,-1])]
        elif XY=='TE':
            clunlenTEArrNow = self.uClNow2d['TE'].copy()
            sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)
            lx = self.lxMap
            ly = self.lyMap
            lxhat = self.lxHatMap
            lyhat = self.lyHatMap
            sinf = sin2phi(lxhat,lyhat)
            sinsqf = sinf**2.
            cosf = cos2phi(lxhat,lyhat)
            cossqf = cosf**2.
            WXY = self.WXY('TE')*kbeamx
            WY = self.WY('EE')*kbeamy
            rfact = 2.**0.25
            for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                    preF = trigfact*ell1*ell2*clunlenTEArrNow*WXY
                    preG = trigfact*WY
                    allTerms += [ell1*ell2*fft(ifft(preF,axes=[-2,-1],normalize=True)*ifft(preG,axes=[-2,-1],normalize=True),axes=[-2,-1])]
                for trigfact in [cosf,sinf]:
                    preFX = trigfact*ell1*clunlenTEArrNow*WY
                    preGX = trigfact*ell2*WXY
                    allTerms += [ell1*ell2*fft(ifft(preFX,axes=[-2,-1],normalize=True)*ifft(preGX,axes=[-2,-1],normalize=True),axes=[-2,-1])]
        elif XY == 'TB':
            clunlenTEArrNow = self.uClNow2d['TE'].copy()
            sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)
            lx = self.lxMap
            ly = self.lyMap
            termsF = []
            termsF.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )
            termsG = []
            termsG.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )
            lxhat = self.lxHatMap
            lyhat = self.lyHatMap
            WXY = self.WXY('TB')*kbeamx
            WY = self.WY('BB')*kbeamy
            for ellsq in [lx*lx,ly*ly,np.sqrt(2.)*lx*ly]:
                preF = ellsq*clunlenTEArrNow*WXY
                preG = WY
                for termF,termG in zip(termsF,termsG):
                    allTerms += [ellsq*fft(ifft(termF(preF,lxhat,lyhat),axes=[-2,-1],normalize=True)
                                           *ifft(termG(preG,lxhat,lyhat),axes=[-2,-1],normalize=True),axes=[-2,-1])]
        else:
            print("ERROR: Unrecognized polComb")
            sys.exit(1)    
        ALinv = np.real(np.sum( allTerms, axis = 0))
        alval = np.nan_to_num(1. / ALinv)
        if self.fmask is not None: alval = self.fmask_func(alval,self.fmask)
        l4 = (lmap**2.) * ((lmap + 1.)**2.)
        NL = l4 *alval/ 4.
        NL[np.where(np.logical_or(lmap >= self.bigell, lmap <2.))] = 0.
        retval = np.nan_to_num(NL.real * self.pixScaleX*self.pixScaleY  )
        if setNl:
            self.Nlkk[XY] = retval.copy()
        return retval * 2. * np.nan_to_num(1. / lmap/(lmap+1.))


class L1Integral(object):
    """
    Calculates I(L) = \int d^2l_1 f(l1,l2)
    on a grid.
    L is assumed to lie along the positive x-axis.
    This is ok for most integrals which are isotropic in L-space.
    The integrand has shape (num_Ls,Ny,Nx)
    """
    def __init__(self,Ls,degrees=None,pixarcmin=None,shape=None,wcs=None,pol=True):
        if (shape is None) or (wcs is None):
            if degrees is None: degrees = 10.
            if pixarcmin is None: pixarcmin = 2.0
            shape,wcs = maps.rgeo(degrees,pixarcmin)
        self.shape = shape
        self.wcs = wcs

        assert Ls.ndim==1
        Ls = Ls[:,None,None]
        ly,lx = enmap.lmap(shape,wcs)
        self.l1x = lx.copy()
        self.l1y = ly.copy()
        l1y = ly[None,...]
        l1x = lx[None,...]
        l1 = enmap.modlmap(shape,wcs)[None,...]
        l2y = -l1y
        l2x = Ls - l1x
        l2 = np.sqrt(l2x**2.+l2y**2.)
        self.Ldl1 = Ls*l1x
        self.Ldl2 = Ls*l2x
        self.l1 = l1
        self.l2 = l2

        print(self.Ldl1.shape,self.Ldl2.shape,self.l1.shape,self.l2.shape,self.l1x.shape,self.l1y.shape)
            
        if pol:
            from orphics import symcoupling as sc
            sl1x,sl1y,sl2x,sl2y,sl1,sl2 = sc.get_ells()
            scost2t12,ssint2t12 = sc.substitute_trig(sl1x,sl1y,sl2x,sl2y,sl1,sl2)
            feed_dict = {'l1x':l1x,'l1y':l1y,'l2x':l2x,'l2y':l2y,'l1':l1,'l2':l2}
            cost2t12 = sc.evaluate(scost2t12,feed_dict)
            sint2t12 = sc.evaluate(ssint2t12,feed_dict)
            self.cost2t12 = cost2t12
            self.sint2t12 = sint2t12

            
    def integrate(self,integrand):
        integral = np.trapz(y=integrand,x=self.l1x[0,:],axis=-1)
        integral = np.trapz(y=integral,x=self.l1y[:,0],axis=-1)
        return integral


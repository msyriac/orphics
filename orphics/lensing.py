from __future__ import print_function
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from orphics import maps
from pixell import enmap, utils, bench
from scipy.special import factorial
from scipy.optimize import curve_fit

try:
    from pixell import lensing as enlensing
except:
    print("WARNING: Couldn't load pixell lensing. Some features will be unavailable.")

from scipy.interpolate import splrep,splev

from contextlib import nullcontext

from scipy.fftpack import fftshift,ifftshift,fftfreq
from scipy.interpolate import interp1d
from pixell.fft import fft,ifft

from orphics.stats import bin2D

import time
from six.moves import cPickle as pickle

from orphics import stats, cosmology
import os,sys

class FixedLens(object):
    """
    A simulator for generating CMB maps lensed by a fixed radially symmetric
    convergence profile.

    Parameters
    ----------
    thetas : ndarray
        1D array of angular distances (in radians) corresponding to the convergence profile.
    kappa_1d : ndarray
        1D array representing the total convergence profile (e.g., sum of 1-halo and 2-halo terms), 
        corresponding to thetas.
    width_deg : float, optional
        Width of the simulated map in degrees. Used to define the stamp size. Default is 2.0.
    res_arcmin : float, optional
        Resolution of the map in arcminutes per pixel. Default is 0.5.
    pad_fact : int, optional
        Padding factor applied to the map width to simulate non-periodic boundary conditions. Default is 2.
        The simulation will be on a template that is pad_fact * width_deg wide.
    dfact : int, optional
        Upsampling factor by for performing lensing. Default is 3. The pixel size for the template 
        on which lensing is performed will be res_arcmin / dfact.
    skip_lensing : bool, optional
        Sets whether the generated map will have lensing. Default is False and will generate maps
        with convergence profiles.

    """
    def __init__(self,thetas, kappa_1d,
                 width_deg=2.0,res_arcmin=0.5, # for stamps
                 pad_fact=2,dfact=3, skip_lensing=False # for simulation
                 ):

        # Make the high-res geometry
        self.pad_fact = pad_fact
        self.dfact = dfact
        self.ushape,self.uwcs = maps.rect_geometry(width_deg=width_deg*pad_fact,px_res_arcmin=res_arcmin/dfact,proj="tan")
        self.dshape,self.dwcs = maps.rect_geometry(width_deg=width_deg,px_res_arcmin=res_arcmin,proj="tan")

        # Store the unlensed CMB theory
        theory =  cosmology.default_theory()
        self.ells = np.arange(10000)
        self.cltt = theory.uCl('TT',self.ells)

        # Make the deflection field from the kappa profile
        self.thetas = thetas
        self.kappa_1d = kappa_1d
        self.umodrmap = enmap.modrmap(self.ushape,self.uwcs)
        ukappa = enmap.enmap(maps.interp(thetas,kappa_1d)(self.umodrmap),self.uwcs)
        self.grad_phi = alpha_from_kappa(ukappa)

        # Save lensing setting
        self.skip_lensing = skip_lensing


    def generate_sim(self,seed):
        """
        Generate a simulated CMB map lensed by the fixed lens.

        Parameters
        ----------
        seed : int
            Seed for the random number generator to ensure reproducibility.

        Returns
        -------
        umap : enmap
            Unlensed high-resolution CMB map.
        lmap : enmap
            Lensed high-resolution CMB map using the precomputed deflection field.
        dmap : enmap
            Center-cropped and downsampled lensed map for analysis.
        """
        
        # Random unlensed
        umap = enmap.rand_map(self.ushape, self.uwcs, self.cltt, seed=seed)
        # Lensed
        if self.skip_lensing:
            lmap = umap.copy()
        else:
            lmap = enlensing.lens_map(umap, self.grad_phi)
        # Downgraded
        dmap = enmap.downgrade_fft(lmap, self.dfact)
        # Cropped
        return umap, lmap, (maps.get_central(dmap,1./self.pad_fact) if self.pad_fact!=1 else dmap)


def filter_bin_kappa1d(thetas,kappas,fls=None,lmin=200,lmax=6000,res=0.05*utils.arcmin,rstamp=30.*utils.arcmin,rmin=0.,rmax=15*utils.arcmin,rwidth=0.1*utils.arcmin):
    N = int(rstamp/res)
    shape,wcs = enmap.geometry(pos=(0,0),res=res,shape=(N,N),proj='tan')
    modrmap = enmap.modrmap(shape,wcs)
    omap = enmap.enmap(maps.interp(thetas,kappas)(modrmap),wcs)
    return filter_bin_kappa2d(omap,fls=fls,lmin=lmin,lmax=lmax,rmin=rmin,rmax=rmax,rwidth=rwidth)
    
def filter_bin_kappa2d(omap,fls=None,lmin=200,lmax=6000,rmin=0.,rmax=15*utils.arcmin,rwidth=0.1*utils.arcmin,taper_per=12.0):
    taper,_ = maps.get_taper(omap.shape[-2:],omap.wcs,taper_percent = taper_per)
    kmask = maps.mask_kspace(omap.shape,omap.wcs, lxcut = None, lycut = None, lmin = lmin, lmax = lmax).astype(bool)
    if fls is not None:
        modlmap = omap.modlmap()
        ells = np.arange(fls.size)
        kfilt = maps.interp(ells,fls)(modlmap)
    else:
        kfilt = omap*0+1.
    kfilt[~kmask] = 0
    fmap = maps.filter_map(omap*taper,kfilt)
    bin_edges = np.arange(rmin,rmax,rwidth)
    modrmap = enmap.modrmap(omap.shape,omap.wcs)
    binner = stats.bin2D(modrmap,bin_edges)
    cents,b1d = binner.bin(fmap)
    return cents,b1d


def kappa_nfw_profiley1d(thetas,mass=2e14,conc=None,z=0.7,z_s=1100.,
                         background='critical',delta=500,
                         R_off_Mpc = None,
                         R_off_Mpc_max = 1.0,
                         N_off = 50,
                         verbose=True,
                         h = 0.677,
                         Om = 0.3,
                         Ob = 0.045,
                         As = 2.1e-9,
                         ns = 0.96,
                         debug_time = False
                         ):
    from profiley.helpers.filtering import Filter
    from profiley.nfw import NFW
    from profiley.numeric import offset
    from astropy import units as u
    import pyccl as ccl
    from profiley.helpers.lss import power2xi, xi2sigma
    bshow = (lambda x: bench.show(x)) if debug_time else (lambda x: nullcontext())
    
    frame='comoving'

    if conc is None:
        from colossus.cosmology import cosmology
        from colossus.halo import concentration
        # Only this or WMAP7 can be used
        cosmology.setCosmology('planck13')
        # Define mass (in solar masses) and redshift
        # Compute concentration using Klypin et al. 2016 model
        conc = concentration.concentration(M=mass * h , z=z, mdef='500c', model='klypin16_m')

    nfw = NFW(mass, conc, z, overdensity=delta, background=background[0],
              frame=frame)
    
    if frame=='comoving':
        Rcon = nfw.cosmo.kpc_comoving_per_arcmin
    elif frame=='physical':
        Rcon = nfw.cosmo.kpc_proper_per_arcmin 
    R = Rcon(nfw.z) * thetas*u.radian
    kappa1 = nfw.convergence(R, z_s=z_s)
    
    # Miscentering
    if R_off_Mpc is not None:
        with bshow("miscenter"):
            Roff = np.linspace(0, R_off_Mpc_max, N_off) # Mpc
            weights = np.exp(-Roff**2 / (2*(R_off_Mpc)**2))
            # Some units gymnastics here because offset doesn't support units
            kappa_1h = offset((kappa1.T).to(u.Mpc).value, R.to(u.Mpc).value, Roff, weights=weights)[0] * u.Mpc
    else:
        kappa_1h = kappa1[:,0]
    
    # 2-halo
    with bshow("two-halo"):
        # This part can be sped up by caching the Pk
        cosmo = ccl.Cosmology(Omega_c=Om-Ob, Omega_b=Ob, h=h, A_s=As, n_s=ns)
        k = np.geomspace(1e-15, 1e15, 10000) # this wide range is needed for the Hankel transform
        kmin = 1e-4; kmax=20.0 # 1/Mpc; but only this k-range matters
        sel = np.logical_and(k>kmin,k<kmax)
        Pk = k*0
        Pk[sel] = ccl.linear_matter_power(cosmo, k[sel], 1/(1+z))
        mdef = ccl.halos.MassDef(delta, background)
        bias = ccl.halos.HaloBiasTinker10(mass_def=mdef)
        bh = bias(cosmo=cosmo,M=mass, a=1/(1+float(nfw.z)))
        if verbose: print("Halo bias : ", bh)
        Pgm = bh * Pk
        r_xi = np.geomspace(1e-3, 1e4, 100) # Mpc
        lnPgm_lnk = interp1d(np.log(k), np.log(Pgm))
        xi = power2xi(lnPgm_lnk, r_xi)
        rho_m = ccl.background.rho_x(cosmo, 1, 'matter')
        sigma_2h = xi2sigma(R.to(u.Mpc).value, r_xi, xi, rho_m).T
        kappa_2h = sigma_2h / nfw.sigma_crit(z_s)

    
    return kappa_1h, kappa_2h #.value


def kappa_nfw_profiley(mass=2e14,conc=None,z=0.7,z_s=1100.,
                       background='critical',delta=500,
                       thetamin=0.001*utils.arcmin,thetamax=240.*utils.arcmin,numthetas=500,
                       theta_extrap = 20*utils.arcmin,
                       R_off_Mpc = None,
                       R_off_Mpc_max = 5.0,
                       N_off = 50,
                       apply_filter = True,
                       fls = None,lmin=200,lmax=6000,
                       res=0.05*utils.arcmin,rstamp=30.*utils.arcmin,rmin=0.,rmax=15*utils.arcmin,rwidth=0.1*utils.arcmin,
                       verbose=True, h = 0.677,
                       Om = 0.3,
                       Ob = 0.045,
                       As = 2.1e-9,
                       ns = 0.96
                       ):

    # These are the radii at which we will do the expensive profiley calculation
    ithetas = np.linspace(thetamin, theta_extrap, numthetas)

    kappa_1h,kappa_2h = kappa_nfw_profiley1d(ithetas,mass=mass,conc=conc,z=z,z_s=z_s,
                                             background=background,delta=delta,
                                             R_off_Mpc = R_off_Mpc,
                                             R_off_Mpc_max = R_off_Mpc_max,
                                             N_off = N_off,
                                             verbose=verbose, h = h,
                                             Om = Om,
                                             Ob = Ob,
                                             As = As,
                                             ns = ns
                                             )

    # We extrapolate with a power law above that
    t_extra = np.linspace(theta_extrap, thetamax, numthetas)
    othetas, okappa_1h =  stats.extrapolate_power_law(ithetas, kappa_1h.value, t_extra, x_percentile=30.0)
    othetas, okappa_2h =  stats.extrapolate_power_law(ithetas, kappa_2h.value, t_extra, x_percentile=30.0)
    
    # And at zero radius fill with the initial value
    thetas = np.append([0.],othetas)
    kappa_1h = np.append([okappa_1h[0]],okappa_1h)
    kappa_2h = np.append([okappa_2h[0]],okappa_2h)
        
    tot_kappa = (kappa_1h+kappa_2h)

    if apply_filter:
        cents,b1d1h = filter_bin_kappa1d(thetas,kappa_1h,fls=fls,lmin=lmin,lmax=lmax,res=res,rstamp=rstamp,rmin=rmin,rmax=rmax,rwidth=rwidth)
        cents,b1d = filter_bin_kappa1d(thetas,tot_kappa,fls=fls,lmin=lmin,lmax=lmax,res=res,rstamp=rstamp,rmin=rmin,rmax=rmax,rwidth=rwidth)
        cents,b1d2h = filter_bin_kappa1d(thetas,kappa_2h,fls=fls,lmin=lmin,lmax=lmax,res=res,rstamp=rstamp,rmin=rmin,rmax=rmax,rwidth=rwidth)
    else:
        cents = b1d1h = b1d = b1d2h = None

    return thetas,kappa_1h,kappa_2h,tot_kappa,cents,b1d1h,b1d2h,b1d
        

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


def alpha_from_kappa(kappa=None,posmap=None,phi=None,grad=True):
    if phi is None:
        phi,_ = kappa_to_phi(kappa,kappa.modlmap(),return_fphi=True)
        shape,wcs = phi.shape,phi.wcs
    else:
        shape,wcs = phi.shape,phi.wcs
    grad_phi = enmap.grad(phi)
    if grad: return grad_phi
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




def kappa_to_phi(kappa,modlmap,return_fphi=False):
    fphi = enmap.samewcs(kappa_to_fphi(kappa,modlmap),kappa)
    phi =  enmap.samewcs(enmap.ifft(fphi,normalize='phys').real, kappa) 
    if return_fphi:
        return phi, fphi
    else:
        return phi

def kappa_to_fphi(kappa,modlmap):
    return fkappa_to_fphi(enmap.fft(kappa,normalize='phys'),modlmap)

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
    from scipy.integrate import simps
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

    


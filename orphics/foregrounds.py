"""
Utilities for ILC noise, source counts and associated power spectra, etc..

"""

import glob,os,sys,itertools,warnings
import numpy as np
from scipy.interpolate import interp1d
from orphics import maps, cosmology, io
from pixell import bench
from scipy.constants import h, k, c
from scipy.optimize import least_squares
from typing import Union, Callable, List

# For szar copies
default_constants = {'A_tsz': 5.6,
                     'TCMB': 2.726,
                     'nu0': 150.,
                     'TCMBmuk':2.726e6,
                     'Td': 24., #9.7,
                     'al_cib': 1.2, #2.2
                     'A_cibp': 6.9,
                     'A_cibc': 4.9,
                     'n_cib': 1.2,
                     'ell0sec': 3000.,
                     'A_ps': 3.1,
                     'al_ps': -0.5,
                     'zeta': 0.1}
H_CGS = 6.62608e-27
K_CGS = 1.3806488e-16
C_light = 2.99792e+10


# Helpers from tilec/fg.py [with J. Colin Hill]

######################################
# global constants
# MKS units, except electron rest mass-energy
######################################
TCMB = 2.726 #Kelvin
TCMB_uK = 2.726e6 #micro-Kelvin



hplanck=6.626068e-34 #MKS
kboltz=1.3806503e-23 #MKS
clight=299792458.0 #MKS
m_elec = 510.999 #keV


# function needed for Planck bandpass integration/conversion following approach in Sec. 3.2 of https://arxiv.org/pdf/1303.5070.pdf
# blackbody derivative
# units are 1e-26 Jy/sr/uK_CMB
def dBnudT(nu_ghz):
    nu = 1.e9*np.asarray(nu_ghz)
    X = hplanck*nu/(kboltz*TCMB)
    return (2.*hplanck*nu**3.)/clight**2. * (np.exp(X))/(np.exp(X)-1.)**2. * X/TCMB_uK

# conversion from specific intensity to Delta T units (i.e., 1/dBdT|T_CMB)
#   i.e., from W/m^2/Hz/sr (1e-26 Jy/sr) --> uK_CMB
#   i.e., you would multiply a map in 1e-26 Jy/sr by this factor to get an output map in uK_CMB
def ItoDeltaT(nu_ghz):
    return 1./dBnudT(nu_ghz)

def planck(nu_hz, T):
    """Planck intensity B_nu  (W m^{-2} Hz^{-1} sr^{-1})."""
    x = hplanck * nu_hz / (kboltz * T)
    return (2.0 * hplanck * nu_hz**3 / clight**2) / (np.expm1(x))


# tSZ frequency factor (dimensionless)
def g_tsz(nu_ghz, T_cmb=2.726):
    x = (h * nu_ghz * 1e9) / (k * T_cmb)
    return x * (np.exp(x) + 1.0) / (np.exp(x) - 1.0) - 4.0

def cltsz(atsz,nu1,nu2,clyy):
    return atsz * g_tsz(nu1) * g_tsz(nu2) * clyy * TCMB_uK**2.


# Copied from szar
def dl_filler(ells,ls,cls,fill_type="extrapolate",fill_positive=False,silence=False):
    ells = np.asarray(ells)
    if not(silence):
        if ells.max()>ls.max() and fill_type=="extrapolate":
            warnings.warn("Warning: Requested ells go higher than available." + \
                  " Extrapolating above highest ell.")
        elif ells.max()>ls.max() and fill_type=="constant_dl":
            warnings.warn("Warning: Requested ells go higher than available." + \
                  " Filling with constant ell^2C_ell above highest ell.")
    if fill_type=="constant_dl":
        fill_value = (0,cls[-1])
    elif fill_type=="extrapolate":
        fill_value = "extrapolate"
    elif fill_type=="zeros":
        fill_value = 0
    else:
        raise ValueError
    dls = interp1d(ls,cls,bounds_error=False,fill_value=fill_value)(ells)
    if fill_positive: dls[dls<0] = 0
    return dls


def power_y_template(ells,A_tsz=None,fill_type="extrapolate",silence=False):
    """
    fill_type can be "zeros" , "extrapolate" or "constant_dl"

    ptsz = Tcmb^2 gnu^2 * yy
    ptsz = Atsz * battaglia * gnu^2 / gnu150^2
    yy = Atsz * battaglia / gnu150^2 / Tcmb^2

    """
    if A_tsz is None: A_tsz = default_constants['A_tsz']
    ells = np.asarray(ells)
    assert np.all(ells>=0)
    root = os.path.dirname(__file__)+f"/../data/"
    ls,icls = np.loadtxt(root+"/foregrounds/sz_template_battaglia.csv",unpack=True,delimiter=',')
    dls = dl_filler(ells,ls,icls,fill_type=fill_type,fill_positive=True,silence=silence)
    nu0 = 150.0 ; tcmb = TCMB_uK
    with np.errstate(divide='ignore'): cls = A_tsz * dls*2.*np.pi*np.nan_to_num(1./ells/(ells+1.)) / ffunc(nu0)**2./tcmb**2.
    return cls


def compute_cl_yy(
        ell,                          # multipoles: array-like
        M_min=1e11,
        M_max=2e15,                   # mass cut in M_sun (M500c)
        zmin=0.001, zmax=5.0,          # redshift range
        nm=60, nz=50, nk=1050,               # grid resolution
        kmin=1e-4, kmax=60.0,          # h/Mpc
        include_2h=True               # toggle 2-halo term
):
    """
    Compute the thermal SZ cross-power spectrum C_ell^{yy} at given frequencies,
    with clusters above M_max masked out.

    Returns
    -------
    ell : array
        Input multipoles
    C_ell_tsz : array
        Thermal SZ power spectrum in µK^2 units (thermodynamic)
    """
    import hmvec as hm

    # grids
    zs = np.linspace(zmin, zmax, nz)
    ks = np.geomspace(kmin, kmax, nk)
    ms = np.geomspace(M_min, M_max, nm)  # upper bound = mass cut

    # halo model instance
    hcos = hm.HaloModel(zs, ks, ms=ms, accuracy='low')

    # add pressure profile
    hcos.add_battaglia_pres_profile('press')

    # get 1h and optional 2h terms
    P1h = hcos.get_power_1halo("press")
    Ppp = P1h
    if include_2h:
        P2h = hcos.get_power_2halo("press")
        Ppp += P2h

    # limber projection → C_ell^{yy}
    Cyy = hcos.C_yy(ell, zs, ks, Ppp)
    
    return Cyy

def compute_tsz_power(
        ell,                          # multipoles: array-like
        nu_i_ghz, nu_j_ghz,           # frequencies in GHz
        Cyy = None,
        M_max=2e15,                   # mass cut in M_sun (M500c)
        zmin=0.01, zmax=3.0,          # redshift range
        nz=40, nk=1050,               # grid resolution
        kmin=1e-4, kmax=50.,          # h/Mpc
        include_2h=True               # toggle 2-halo term
):
    # frequency scaling
    g_i, g_j = g_tsz(nu_i_ghz), g_tsz(nu_j_ghz)
    if Cyy is None:
        Cyy = compute_cl_yy(
            ell,                          # multipoles: array-like
            nu_i_ghz, nu_j_ghz,           # frequencies in GHz
            M_max,                   # mass cut in M_sun (M500c)
            zmin, zmax,          # redshift range
            nz, nk,               # grid resolution
            kmin, kmax,          # h/Mpc
            include_2h               # toggle 2-halo term
        )

    return Cyy * g_i * g_j * TCMB_uK**2.




# Lagache et al. (2019) source model

def get_radio_differential_source_counts(fluxes_mJy,freq_ghz):
    """
    Get differential source counts n(S) for an array of fluxes S
    at freq_ghz frequency.

    Parameters
    ----------
    fluxes_mJy : array-like
        Array of fluxes in mJy
    freq_ghz : float
        Frequency in GHz

    Returns
    -------
    ds_counts : array-like
        Array of differential source counts in units of 1/mJy/sr
    """
    rpath = os.path.dirname(__file__)+f"/../data/radio_counts/"
    files = glob.glob(rpath+"ns*_radio.dat")
    if len(files)==0: raise FileExistsError
    freqs = np.asarray(sorted([float(os.path.basename(f).split('_')[0][2:]) for f in files]))
    closest_freq = int(freqs[np.argmin(np.abs(freqs-freq_ghz))])
    fluxes_Jy, nS_Jy_sr = np.loadtxt(f"{rpath}ns{closest_freq}_radio.dat",unpack=True)
    return interp1d(fluxes_Jy*1000,nS_Jy_sr/1000,kind='cubic')(fluxes_mJy)


def get_radio_power(flux_limit_mJy,freq_ghz,
                    flux_limit_mJy_2=None,freq_ghz_2=None,
                    flux_min_mJy=1.6e-2,num_flux=10000,
                    prefit=True,units_Jy_sr=False,zero_above_ghz=200.):
    """
    Get the (cross-)power spectrum of radio sources for a 
    given pair of frequencies and flux limits.

    The power spectrum is in units of muK^2-sr

    Parameters
    ----------
    flux_limit_mJy : float
        Flux limit in mJy
    freq_ghz : float
        Frequency in GHz
    flux_limit_mJy : float, optional
        Flux limit in mJy for the second frequency
    freq_ghz_2 : float, optional
        Frequency in GHz for the second frequency
    flux_min_mJy : float, optional
        Minimum flux in mJy to use in the power spectrum integral.
        Default is 1.6e-2 mJy
    num_flux : int, optional
        Number of flux points to use in the power spectrum integral.
    prefit : bool, optional
        If True, use pre-fit power spectrum, otherwise calculate integral.


    Returns
    -------
    ps : array-like
        Power spectrum in muK^2-sr
    """

    if freq_ghz>zero_above_ghz or freq_ghz_2>zero_above_ghz: return 0.

    if flux_limit_mJy_2 is None:
        if not(freq_ghz_2) is None: raise ValueError
        cross = False
    else:
        cross = True
        if not(prefit): raise NotImplementedError

    if np.abs(freq_ghz-freq_ghz_2)<1e-3:
        if np.abs(flux_limit_mJy-flux_limit_mJy_2)>1e-3: raise ValueError
        cross = False


    if not(prefit) and not(cross):
        fluxes_mJy = np.geomspace(flux_min_mJy,flux_limit_mJy,num_flux)
        nS = get_radio_differential_source_counts(fluxes_mJy,freq_ghz)
        ps = np.trapz(nS*fluxes_mJy**2,fluxes_mJy) * (1e-3)**2 # (Jy/sr)^2 sr 
    else:
        rpath = os.path.dirname(__file__)+f"/../data/radio_counts/"
        if not(cross):
            freqs,logAs,logS0s,alphas,betas = np.loadtxt(f'{rpath}auto_fit_vals.dat',unpack=True,delimiter=',')
            idx = np.argmin(np.abs(freqs-freq_ghz))
            A = 10.**logAs[idx]
            S0 = 10.**logS0s[idx]
            alpha = alphas[idx]
            beta = betas[idx]
            Slim = flux_limit_mJy * 1e-3
            ps = Slim * 2 * A/((Slim/S0)**alpha + (Slim/S0)**beta)
        else:
            Kijs = parse_Kij_file()
            freqs = [30,44,70,100,143,217,353,545,857]
            freqs = np.asarray(freqs)
            cnu1 = int(freqs[np.argmin(np.abs(freqs-freq_ghz))])
            cnu2 = int(freqs[np.argmin(np.abs(freqs-freq_ghz_2))])
            try:
                Kij = Kijs[(cnu1,cnu2)]
            except:
                Kij = Kijs[(cnu2,cnu1)]
            t1 = (np.log10(flux_limit_mJy*1e-3)+3)/0.2
            t2 = (np.log10(flux_limit_mJy_2*1e-3)+3)/0.2
            logC = 0
            for i in range(7):
                for j in range(7):
                    # i,j order inferred from Fig 2 of Lagache et al 2019
                    logC += Kij[i,j]*(t1**j)*(t2**i)
            ps = 10**logC  # (Jy/sr)^2 sr
                    
    return ps * (1e-26)**2  * ItoDeltaT(freq_ghz)**2 if not(units_Jy_sr) else ps# (1e-26 Jy/sr)^2 sr -> muK^2 sr


def parse_Kij_file():
    """
    Parse the Kij file and return a dictionary of the form
    {(i,j):Kij}
    """
    rpath = os.path.dirname(__file__)+f"/../data/radio_counts/"
    filename = f'{rpath}Para_6degPol_XPS_Scut.dat'

    Kijs = {}
    with open(filename,'r') as f:
        for line in f:
            elems = line.split()
            nelems = len(elems)
            if nelems==2:
                i = int(elems[0])
                j = int(elems[1])
                Kijs[(i,j)] = []
            else:
                Kijs[(i,j)].append(np.asarray([float(e) for e in elems]))
    for key in Kijs:
        Kijs[key] = np.asarray(Kijs[key])
    return Kijs


def compton_y_cib_powers(freqs_ghz,flux_limits_mJy,lmin=2, lmax=4000, Mmin_msun = 1.e10,
                    Mmax_msun = 1e16,
                    Omega_M = 0.31,
                    Omega_B = 0.049,
                    Omega_L = 0.69,
                    h = 0.68,
                    sigma_8 = 0.81,
                    n_s = 0.965,
                    tau = 0.0543,
                    z_min = 0.0113,
                    z_max = 5.1433,
                    mfun='T08'):
    """
    This returns the following:
    yy (dimensionless)
    CIB-CIB (MJy/sr)^2
    y-CIB (MJy/sr)
    """
    # h/t Boris Bolliet for this code

    from classy_sz import Class
    #from tilec import fg

    common_settings = {
        'mass function' : mfun, 
    }

    cosmo = {
        'omega_b': Omega_B*h**2.,
        'omega_cdm': (Omega_M-Omega_B)*h**2.,
        'h': h,
        'tau_reio': tau,
        'sigma8': sigma_8,
        'n_s': n_s, 
        'use_websky_m200m_to_m200c_conversion': 1
    }

    M = Class()
    M.set(common_settings)
    M.set(cosmo)
    M.set({# class_sz parameters:
        'output':'tSZ_1h,tSZ_2h,cib_cib_1h,cib_cib_2h,tSZ_cib_1h,tSZ_cib_2h',
        'pressure profile': 'B12',  # check source/input.c for default parameter values of Battaglia et al profile (B12)
        'concentration parameter': 'D08',  # B13: Bhattacharya et al 2013  
        'ell_max' : lmax,
        'ell_min' : lmin,
        'dlogell': 0.1,
        'z_min': z_min,
        'z_max': z_max,
        'M_min':Mmin_msun*h, # all masses in Msun/h
        'M_max':Mmax_msun*h,
        'units for tSZ spectrum': 'dimensionless',
        'n_ell_pressure_profile' : 100,
        'n_m_pressure_profile' : 100,
        'n_z_pressure_profile' : 100,
        'x_outSZ': 4.,
        'truncate_wrt_rvir':0,
        'hm_consistency':0,
        'pressure_profile_epsrel':1e-3,
        'pressure_profile_epsabs':1e-40,
        'redshift_epsrel': 1e-4,
        'redshift_epsabs': 1e-100,
        'mass_epsrel':1e-4,
        'mass_epsabs':1e-100,
    })


    # ~ model 2 of https://arxiv.org/pdf/1208.5049.pdf (Table 5)
    # more exactly:
    # shang_zplat  = 2.0
    # shang_Td     = 20.7
    # shang_beta   = 1.6
    # shang_eta    = 1.28
    # shang_alpha  = 0.2
    # shang_Mpeak  = 10.**12.3
    # shang_sigmaM = 0.3

    # centrals is Ncen = 1 for all halos with mass bigger than websky's m_min
    # subhalo mass function is eq. 3.9 of the websky paper 
    # it is F. Jiang and F. C. van den Bosch, Generating merger trees for dark matter haloes: a comparison of
    # methods, MNRAS 440 (2014) 193 [1311.5225].

    L0_websky = 4.461102571695613e-07
    websky_cib_params = {

        'Redshift evolution of dust temperature' :  0.2,
        'Dust temperature today in Kelvins' : 20.7,
        'Emissivity index of sed' : 1.6,
        'Power law index of SED at high frequency' : 1.7, # not given in WebSky paper, actually not relevant since we dont use high freqs in websky.
        'Redshift evolution of L − M normalisation' : 1.28,
        'Most efficient halo mass in Msun' : 10.**12.3,
        'Normalisation of L − M relation in [Jy MPc2/Msun]' : L0_websky,  # not given in WebSky paper
        'Size of of halo masses sourcing CIB emission' : 0.3,
        'z_plateau_cib' : 2.,
        
        # M_min_HOD is the threshold above which nc = 1:
        # 'M_min_HOD' : 10.**10.1, # not used here
        'use_nc_1_for_all_halos_cib_HOD': 1,
        
        'sub_halo_mass_function' : 'JvdB14',
        'M_min_subhalo_in_Msun' : 1e11,
        'use_redshift_dependent_M_min': 1,
        # 'full_path_to_redshift_dependent_M_min':'/home/r/rbond/msyriac/repos/class_sz/sz_auxiliary_files/websky_halo_mass_completion_z_Mmin_in_Msun_over_h.txt',
        # 'full_path_to_redshift_dependent_M_min':'/home/r/rbond/msyriac/repos/class_sz/sz_auxiliary_files/websky_halo_mass_completion_z_Mmin_in_Msun_over_hwrong.txt',
        #'M_min' : 1e10*websky_cosmo['h'], # not used
        # 'M_max' : 1e16*websky_cosmo['h'],
        # 'z_min' : 5e-3,
        # 'z_max' : 4.6,
        'cib_frequency_list_num' : len(freqs_ghz),
        'cib_frequency_list_in_GHz' : ','.join([str(x) for x in freqs_ghz]),  
        #for the monopole computation:
        # 'freq_min': 2e1,
        # 'freq_max': 4e3,
        # 'dlogfreq' : 0.05,
        'cib_Snu_cutoff_list_in_mJy':','.join([str(x) for x in flux_limits_mJy]),
        'has_cib_flux_cut': 1
    }

    M.set(websky_cib_params)
    M.compute()
    cl_sz = M.cl_sz()
    cl_cib_cib = M.cl_cib_cib()
    cl_tsz_cib = M.cl_tSZ_cib()

    print(cl_sz.keys())
    print(cl_cib_cib.keys())
    print(cl_tsz_cib.keys())

    # print(cl_sz)
    # print(cl_cib_cib)
    # print(cl_tsz_cib)

    M.struct_cleanup()
    M.empty()

    ells = cl_sz['ell']
    ls = np.arange(lmin,max(ells))
    print(ls.shape)
    # cls = 
    finterp = lambda y: interp1d(ells,y,bounds_error=True)(ls)
    cl_sz['ell'] = np.asarray(cl_sz['ell'])
    cl_sz['1h'] = finterp(np.asarray(cl_sz['1h'])) * 1e-12 / ls / (ls+1.) * 2. * np.pi
    cl_sz['2h'] = finterp(np.asarray(cl_sz['2h'])) * 1e-12 / ls / (ls+1.) * 2. * np.pi
   

 
    cl_sz = cl_sz['1h']+cl_sz['2h']
    # f = fg.get_mix(freqs_ghz,'tSZ')
    # print(f**2)
    print(cl_cib_cib['90x90'].keys())
    
    return cl_sz




# Quick and dirty ILC noise

def ilc_power(beams,noises,freqs,flux_limits_mJy,inv_noise_weighting=False,total=False,include_fg=True):
    from szar import foregrounds as fg
    beams = np.asarray(beams)
    noises = (np.asarray(noises)*np.pi/180./60.)**2.
    freqs = np.asarray(freqs)
    flux_limits_mJy = np.asarray(flux_limits_mJy)

    ellmax = 25000
    ells = np.arange(0,ellmax,1)

    def flim(nu):
        return flux_limits_mJy[np.argmin(np.abs(freqs-nu))]

    fdict = {}
    fdict['tsz'] = lambda ells,nu1,nu2 : fg.power_tsz(ells,nu1,nu2,fill_type="extrapolate")
    fdict['cibc'] = lambda ells,nu1,nu2 : fg.power_cibc(ells,nu1,nu2)
    fdict['cibp'] = lambda ells,nu1,nu2 : fg.power_cibp(ells,nu1,nu2)
    fdict['radps'] = lambda ells,nu1,nu2 : get_radio_power(flim(nu1),nu1,
                        flux_limit_mJy_2=flim(nu2),freq_ghz_2=nu2,
                        flux_min_mJy=1.6e-2,num_flux=10000,
                        prefit=True,units_Jy_sr=False)+ells*0

    fdict['ksz'] = lambda ells,nu1,nu2 :  fg.power_ksz_reion(ells,fill_type="extrapolate") + fg.power_ksz_late(ells,fill_type="extrapolate")

    kbeams = []
    for beam in beams:
        kbeams.append(maps.gauss_beam(ells,beam))

    theory = cosmology.default_theory(lpad=ellmax)
    cltt = theory.lCl("TT",ells)

    components = ['cibc','tsz','ksz','radps' ,'cibp'] if include_fg else []
    
    cov = np.rollaxis(maps.ilc_cov(ells,cltt,kbeams,freqs,noises,components,fdict=fdict,ellmaxes=None,data=False,fgmax=None,narray=None,noise_only=False),2,0)
    if inv_noise_weighting:
        ncov = np.rollaxis(maps.ilc_cov(ells,cltt,kbeams,freqs,noises,components,fdict=fdict,ellmaxes=None,data=False,fgmax=None,narray=None,noise_only=True),2,0)
        ninv = np.linalg.inv(ncov)
        ntot = np.sum(ninv,axis=(-2,-1))
        nout = np.sum(np.einsum('lij,ljk->lik',np.einsum('lij,ljk->lik',ninv,cov),ninv),axis=(-2,-1))/ntot**2
    else:
        cinv = np.rollaxis(np.linalg.inv(cov),0,3)
        nout = maps.silc_noise(cinv,response=None)
    csub = 0 if total else cltt
    nell = np.nan_to_num(nout - csub)
    nell[ells<2] = 0

    return ells,nell
    

def get_official_ilc_noise(exp):
    rpath = os.path.dirname(__file__)+f"/../data/"
    if exp=='so':
        ells,nells = np.loadtxt(f'{rpath}SO_LAT_Nell_T_atmv1_baseline_fsky0p4_ILC_CMB.txt',unpack=True,usecols=[0,1])
    elif exp=='s4':
        ells,nells = np.loadtxt(f'{rpath}S4_190604d_2LAT_T_default_noisecurves_deproj0_SENS0_mask_16000_ell_TT_yy.txt',unpack=True,usecols=[0,1])
    return ells, nells


def get_ilc_noise(exp,scale_noise=1.0):

    beams = {}
    noises = {}
    fluxes = {}

    freqs = np.array([39.,93.,145.,225.,280.])
    beams['s4'] = np.array([5.1,2.2,1.4,1.0,0.9])
    beams['so'] = np.array([5.1,2.2,1.4,1.0,0.9])
    beams['hd'] = (10./60.)*145./freqs

    noises['so'] = np.array([36,8,10,22,54])
    noises['s4'] = np.array([12.4,2.0,2.0,6.9,16.7])
    noises['hd'] = noises['s4'].copy() * 0.5/1.8

    fluxes['so'] = [10.,7.,10.,10.,10.]
    fluxes['s4'] = [10.,7.,10.,10.,10.]
    fluxes['hd'] = [2.,1.,1.,1.,1.]


    beams = beams[exp]
    noises = noises[exp] * scale_noise

    flux_limits_mJy = fluxes[exp]

    return ilc_power(beams,noises,freqs,flux_limits_mJy)   


# Copied from szar


def dl_filler(ells,ls,cls,fill_type="extrapolate",fill_positive=False,silence=False):
    ells = np.asarray(ells)
    if not(silence):
        if ells.max()>ls.max() and fill_type=="extrapolate":
            warnings.warn("Warning: Requested ells go higher than available." + \
                  " Extrapolating above highest ell.")
        elif ells.max()>ls.max() and fill_type=="constant_dl":
            warnings.warn("Warning: Requested ells go higher than available." + \
                  " Filling with constant ell^2C_ell above highest ell.")
    if fill_type=="constant_dl":
        fill_value = (0,cls[-1])
    elif fill_type=="extrapolate":
        fill_value = "extrapolate"
    elif fill_type=="zeros":
        fill_value = 0
    else:
        raise ValueError
    dls = interp1d(ls,cls,bounds_error=False,fill_value=fill_value)(ells)
    if fill_positive: dls[dls<0] = 0
    return dls


def ffunc(nu,tcmb=None):
    """
    nu in GHz
    tcmb in Kelvin
    """
    if tcmb is None: tcmb = default_constants['TCMB']
    nu = np.asarray(nu)
    mu = H_CGS*(1e9*nu)/(K_CGS*tcmb)
    ans = mu/np.tanh(mu/2.0) - 4.0
    return ans

def power_ksz_reion(ells,A_rksz=1,fill_type="extrapolate",silence=False):
    root = os.path.dirname(__file__)+f"/../data/"
    ls,icls = np.loadtxt(root+f"foregrounds/early_ksz.txt",unpack=True)
    dls = dl_filler(ells,ls,icls,fill_type=fill_type,fill_positive=True,silence=silence)
    with np.errstate(divide='ignore',over='ignore'): cls = A_rksz * dls*2.*np.pi*np.nan_to_num(1./ells/(ells+1.))
    return cls

def power_ksz_late(ells,A_lksz=1,fill_type="extrapolate",silence=False):
    root = os.path.dirname(__file__)+f"/../data/"
    ls,icls = np.loadtxt(root+f"foregrounds/late_ksz.txt",unpack=True)
    dls = dl_filler(ells,ls,icls,fill_type=fill_type,fill_positive=True,silence=silence)
    with np.errstate(divide='ignore'): cls = A_lksz * dls*2.*np.pi*np.nan_to_num(1./ells/(ells+1.))
    return cls


def clyy_classy_sz(ells,
                   zmin = 0.001,
                   zmax = 5.,
                   mmin = 1e11, # msun /h
                   mmax = 5e15, # msun / h
                   ):


    from classy_sz import Class as Class_sz
    cosmo_params= {
    'omega_b': 0.02225,
    'omega_cdm':  0.1198,
    'H0': 67.3,
    'tau_reio': 0.0561,
    'ln10^{10}A_s': 3.091,
    'n_s': 0.9645,
    }

    

    precision_params = {
    'x_outSZ': 4., # truncate profile beyond x_outSZ*r_s
    'use_fft_for_profiles_transform' : 1, # use fft's or not.
    # only used if use_fft_for_profiles_transform set to 1
    'N_samp_fftw' : 512,
    'x_min_gas_pressure_fftw' : 1e-4,
    'x_max_gas_pressure_fftw' : 1e4,
    }


    classy_sz = Class_sz()
    classy_sz.set(cosmo_params)
    classy_sz.set(precision_params)
    classy_sz.set({

    'output': 'tSZ_tSZ_1h',

    "ell_min" : 2,
    "ell_max" : ells.max(),
    'dell': 0,
    'dlogell': 0.05,

    'z_min' : zmin,
    'z_max' : zmax,
    'M_min' : mmin,
    'M_max' : mmax,


    'mass_function' : 'T08M500c',



    'pressure_profile':'Battaglia', # can be Battaglia, Arnaud, etc

    # "P0GNFW": 8.130,
    # "c500": 1.156,
    # "gammaGNFW": 0.3292,
    # "alphaGNFW": 1.0620,
    # "betaGNFW":5.4807,

    "cosmo_model": 0, # lcdm with Mnu=0.06ev fixed

    })
    classy_sz.compute_class_szfast()
    l = np.asarray(classy_sz.cl_sz()['ell'])
    cl_yy_1h = np.asarray(classy_sz.cl_sz()['1h'])
    cl_yy_2h = np.asarray(classy_sz.cl_sz()['2h'])

    out = cl_yy_1h + cl_yy_2h
    cl = out/l/(l+1)*2.*np.pi/1e12
    return maps.interp(l,cl)(ells)
    

def wnoise_cl(sigma_uk_arcmin):
    return (sigma_uk_arcmin * np.pi / (180. * 60.)) ** 2

        

def fg_cl(ell, p, nu1, nu2, cl_tsz_tmpl, pivot_cib=150., pivot_dust=353.,
          ell0_dust=80.):
    """Foregrounds only (no CMB, no noise)."""
    ell0 = 3000.

    # Poisson point sources
    cl_ps = np.sqrt(p[f"Aps_{nu1}"] * p[f"Aps_{nu2}"])

    # Clustered CIB
    Acib150, alpha = p["Acib_150"], p["alpha_cib"] # change name to beta
    cl_cib = np.sqrt((Acib150 * (nu1/pivot_cib)**alpha) *
                     (Acib150 * (nu2/pivot_cib)**alpha)) * (ell/ell0)**(-1.2)

    # Thermal SZ
    cl_tsz = cltsz(p["Atsz"],nu1,nu2,cl_tsz_tmpl)

    
    # Galactic dust
    Ad   = p["Adust_353"]                 # amplitude at 353 GHz
    beta = p["beta_dust"]
    alpha_d = p["alpha_dust"]

    cl_dust = (Ad *
               (ell/ell0_dust)**(-alpha_d) *
               ((nu1*nu2)/(pivot_dust**2))**(beta/2.0))


    return cl_ps + cl_cib + cl_tsz + cl_dust


def get_noise(ell,i,j,sig_i,sig_j,lknees,alphas,atm_corr=0.):
    if i == j:
        if lknees[i]>0:
            return maps.rednoise(ell, sig_i, lknees[i],alpha=alphas[i])
        else:
            return wnoise_cl(sig_i)

    else:                                         # crosses
        sig_geom   = np.sqrt(sig_i * sig_j)
        lk_cross   = np.sqrt(lknees[i] * lknees[j])
        alpha_cross = 0.5 * (alphas[i] + alphas[j])

        # keep only the power-law term so it decays at high ell
        wnoise   = sig_geom*(np.pi/180./60.)**2
        corr_red = (lk_cross / np.maximum(ell, 1.0))**(-alpha_cross) * wnoise
        return atm_corr * corr_red

def model_vec(all_params, params, ell, freqs, dT_guess, beams, lknees, alphas, cl_cmb_tmpl, cl_tsz_tmpl):
    """CMB x A_cmb  +  foregrounds  +  noise-bias (autos only)."""
    p = dict(zip(all_params, params))
    Acmb   = p["A_cmb"]
    blocks = []

    for i, j in itertools.combinations_with_replacement(range(len(freqs)), 2):
        nu1, nu2 = freqs[i], freqs[j]
        b1, b2 = beams[i](ell), beams[j](ell)

        # signal + foregrounds
        mod = (Acmb * cl_cmb_tmpl + fg_cl(ell, p, nu1, nu2, cl_tsz_tmpl))*b1 * b2

        # add noise bias to autos
        sig_i  = dT_guess[i] * p[f"rN_{nu1}"]            # scaled RMS
        sig_j  = dT_guess[i] * p[f"rN_{nu1}"]            # scaled RMS

        mod = mod + get_noise(ell,i,j,sig_i,sig_j,lknees,alphas,p["Aatm_corr"])

        blocks.append(mod)

    return np.concatenate(blocks)

def quick_fit(
        ell: np.ndarray,
        cl_dict: dict,          # {(i,j): C_ell} with i <= j
        freqs: np.ndarray,      # [nu0, nu1, ...]  in GHz
        dT_guess: np.ndarray,   # guessed RMS per channel  (uK*arcmin)
        beams: Union[List[float], List[Callable[[float], float]]], # beam FWHM (arcmin per channel), or array of beam fn at each ells
        lknees: np.ndarray,
        alphas: np.ndarray,
        fsky: float,
        fixed_params: dict = { "alpha_cib": 3.5, "Aatm_corr": 0., "beta_dust": 1.6, "alpha_dust": 2.42, "Adust_353": 0. },
        priors: dict = {
            "A_cmb": (1.0, 0.03),        # mean, std dev
            "rN_150": (1.0, 0.3),
            "rN_90":  (1.0, 0.3),
            "A_tsz": (1.0, 0.4),        # mean, std dev
        },
        eval_ells: np.ndarray=None,
        verbose: bool = True,
        plot: bool=True,delta_ell: np.ndarray=20):

    fclyy = lambda x: power_y_template(x)
    theory = cosmology.default_theory()
    fcltt = lambda x: theory.lCl('TT',x) + power_ksz_reion(x) + power_ksz_late(x)
    

    return fg_fit(
        ell,
        cl_dict,   
        freqs,     
        dT_guess,  
        beams,
        lknees,
        alphas,
        fsky,
        fcltt,
        fclyy,
        fixed_params,
        priors,eval_ells,verbose,plot,delta_ell)

def _expand_beams(beams,ells,nfreqs):
    if len(beams)!=nfreqs: raise ValueError
    if all(callable(b) for b in beams):
        return beams
    elif all(isinstance(b, (float, int)) for b in beams):
        return [lambda x: maps.gauss_beam(x,b) for b in beams]
    else:
        raise TypeError("beams must be either a list of float FWHMs or a list of callables")

def fg_fit(
        ell: np.ndarray,
        cl_dict: dict,          # {(i,j): C_ell} with i <= j
        freqs: np.ndarray,      # [nu0, nu1, ...]  in GHz
        dT_guess: np.ndarray,   # guessed RMS per channel  (uK*arcmin)
        beams: Union[List[float], List[Callable[[float], float]]], # beam FWHM (arcmin per channel), or array of beam fn at each ells
        lknees: np.ndarray,
        alphas: np.ndarray,
        fsky: float,
        fcl_cmb_tmpl: callable,
        fcl_yy: callable,
        fixed_params: dict = None,
        priors: dict=None,
        eval_ells: np.ndarray=None,
        verbose: bool = True,
        plot: bool=True,
        delta_ell: np.ndarray=20,
        cl_masks=None,      #  {(i,j): bool-array | slice | 1-D index}
):
    """
    Fit all foreground / calibration / noise parameters **except**
    those supplied in `fixed_params`.

    Example:
        fixed_params = {
            "A_cmb": 1.0,          # hold CMB amplitude fixed
            "alpha_cib": 3.5,      # lock clustered-CIB spectral index
            "rN_150": 1.0          # keep 150-GHz noise scale at guess
        }
    """
    if fixed_params is None:
        fixed_params = {}

    if delta_ell is not None:
        edges     = np.arange(ell.min(), ell.max()+delta_ell, delta_ell)
        # list of arrays; idx_bins[k] gives the ℓ–indices in bin k
        idx_bins  = [np.where((ell >= lo) & (ell < hi))[0] for lo, hi in zip(edges[:-1], edges[1:])]
        nbin      = len(idx_bins)


    def bin_1d(arr):
        """Average over each ell-bin."""
        if delta_ell is not None:
            return np.array([arr[idx].mean() for idx in idx_bins])
        else:
            return arr

    if cl_masks is None:
        cl_masks = {}                                    # empty dict -> no special masks
    full_mask = np.ones_like(ell, dtype=bool)            # keep every multipole
    
    pairs     = list(itertools.combinations_with_replacement(range(len(freqs)), 2))
    # Masks for every pair  (same order as `pairs`)
    pair_masks = [cl_masks.get((i, j), full_mask) for i, j in pairs]
    
    data_vec = np.concatenate([bin_1d(cl_dict[(i, j)][m]) for (i, j), m in zip(pairs, pair_masks)])
    # data_vec = np.concatenate([bin_1d(cl_dict[(i, j)]) for i, j in pairs])


    # white-noise bias for every channel (μK²·rad²)
    N_guess = [wnoise_cl(dT) for dT in dT_guess]

    # CMB template *initial* amplitude (use 1.0; you can update later
    # with the best–fit Acmb if you want to iterate)
    cl_cmb_tmpl = fcl_cmb_tmpl(ell)        # already in μK²·rad²

    sigma_blk = []
    pref_auto  = np.sqrt(2.0 / ((2*ell + 1) * fsky))
    pref_cross = np.sqrt(1.0 / ((2*ell + 1) * fsky))

    # Pre-compute beams once (needed to beam the CMB term)
    beams = _expand_beams(beams,ell,len(freqs))

    for i, j in pairs:
        # beam the CMB template for each channel
        Cl_cmb_i = cl_cmb_tmpl * beams[i](ell)**2
        Cl_cmb_j = cl_cmb_tmpl * beams[j](ell)**2

        if i == j:
            # auto-spectrum variance: 2/(2ell+1) * (C_s + N)^2
            var = pref_auto * (Cl_cmb_i + N_guess[i])
        else:
            # cross-spectrum variance: 1/(2ell+1) * [(C_s_i+N_i)(C_s_j+N_j)]
            var = pref_cross * np.sqrt((Cl_cmb_i + N_guess[i]) *
                                       (Cl_cmb_j + N_guess[j]))

        sigma_blk.append(var)



    def bin_sigma(sig_ell):
        # var(mean) = Σ σ_i² / N²    → σ_bin = sqrt( Σ σ_i² ) / N
        if delta_ell is not None:
            return np.array([np.sqrt((sig_ell[idx]**2).sum()) / len(idx) for idx in idx_bins])
        else:
            return sig_ell

    # sigma_vec = np.concatenate([bin_sigma(s) for s in sigma_blk])
    sigma_vec = np.concatenate([bin_sigma(s[m]) for s,m in zip(sigma_blk, pair_masks)])
    
    # 3. Build the full PARAMS list (order matters for model_vec)
    all_params  = [f"Aps_{nu}" for nu in freqs]       # Poisson amps
    all_params += [f"rN_{nu}"  for nu in freqs]       # noise scales
    all_params += ["Acib_150", "alpha_cib", "Atsz", "A_cmb", "Aatm_corr",
                   "Adust_353", "beta_dust", "alpha_dust"]

    lbounds  = [0. for nu in freqs]       # Poisson amps
    lbounds += [0.  for nu in freqs]       # noise scales
    lbounds += [0, 0, 0, 0, 0,0,0,0]

    ubounds  = [np.inf for nu in freqs]       # Poisson amps
    ubounds += [np.inf  for nu in freqs]       # noise scales
    ubounds += [np.inf, np.inf, np.inf, np.inf, np.inf,np.inf,np.inf,np.inf]
    
    # default starting values (same order as all_params)
    p0_full  = [1e-5]*len(freqs)                 # Aps_nu
    p0_full += [1.0  ]*len(freqs)                # rN_nu
    p0_full += [1e-5, 3.5, 1, 1.,0.,
                10.,1.6,2.4]

    # map name -> default guess
    p0_dict  = dict(zip(all_params, p0_full))
    lb_dict  = dict(zip(all_params, lbounds))
    ub_dict  = dict(zip(all_params, ubounds))

    # Separate free vs. fixed parameters
    free_names = [n for n in all_params if n not in fixed_params]
    free_p0    = [p0_dict[n] for n in free_names]          # initial guesses
    free_lbounds    = [lb_dict[n] for n in free_names]          # initial guesses
    free_ubounds    = [ub_dict[n] for n in free_names]          # initial guesses

    # helper: merge free vector with fixed dict -> full ordered list
    def assemble_full(par_free):
        merged = []
        it = iter(par_free)
        for name in all_params:
            merged.append(fixed_params[name] if name in fixed_params else next(it))
        return merged

    cl_yy_temp = fcl_yy(ell)
    
    # Residuals function for least_squares
    def resid(par_free):
        full_par = assemble_full(par_free)
        model_full = model_vec(all_params, full_par, ell, freqs,
                          dT_guess, beams, lknees,alphas,
                          cl_cmb_tmpl, cl_yy_temp)

        # model = np.concatenate([bin_1d(model_full[k*len(ell):(k+1)*len(ell)])
        #                                for k in range(len(pairs))])
        model  = np.concatenate([bin_1d(model_full[k*len(ell):(k+1)*len(ell)][m])
                             for k, m in enumerate(pair_masks)])

        if not(np.all(np.isfinite(data_vec))): raise ValueError
        if not(np.all(np.isfinite(model))): raise ValueError
        residual = (data_vec - model) / sigma_vec

        # Optional Gaussian priors
        prior_terms = []
        param_dict = dict(zip(all_params, full_par))

        for pname, (mu, sigma) in (priors or {}).items():
            if pname in param_dict:
                penalty = (param_dict[pname] - mu) / sigma
                prior_terms.append(penalty)

        return np.concatenate([residual, np.array(prior_terms)])


    # Run the fit
    result   = least_squares(resid, free_p0, bounds=(free_lbounds,free_ubounds))
    best_fit = dict(zip(all_params, assemble_full(result.x)))
    chi2     = np.sum(result.fun**2)
    dof      = data_vec.size - len(result.x)      # DOF counts ONLY free params

    if eval_ells is None:
        eval_ells = ell
    cl_yy = fcl_yy(eval_ells)
    model_dict = evaluate_model_dict(
        eval_ells,
        best_fit,
        freqs,
        dT_guess,
        beams,
        lknees,
        alphas,
        fcl_cmb_tmpl(eval_ells),
        cl_yy,
    )
    if eval_ells is not None: # evaluate at same ells for residuals
        rmodel_dict = evaluate_model_dict(
            ell,
            best_fit,
            freqs,
            dT_guess,
            beams,
            lknees,
            alphas,
            cl_cmb_tmpl,
            cl_yy_temp,
        )
    else:
        rmodel_dict = model_dict
        

    best = best_fit
    if verbose:
        print("\nbest-fit parameters")
        for k,v in best.items():
            print(f"{k:10s} = {v: .3e}")
        print(f"\nχ² / dof = {chi2:.1f} / {dof}")
        
    if plot:

        import healpy as hp
        from pixell import curvedsky as cs
        nalm_090 = hp.read_alm("/data5/act/foregrounds/new_foregrounds/fg_nonoise_alms_0093_w2applied.fits")
        nalm_150 = hp.read_alm("/data5/act/foregrounds/new_foregrounds/fg_nonoise_alms_0145_w2applied.fits")
        print(nalm_090.shape)
        ncls = {}
        ncls['90_90'] = cs.alm2cl(nalm_090,nalm_090)
        ncls['150_150'] = cs.alm2cl(nalm_150,nalm_150)
        ncls['90_150'] = cs.alm2cl(nalm_090,nalm_150)
        nells = np.arange(ncls['90_90'].size)
        
        res = {}
        for i in range(len(freqs)):
            for j in range(i, len(freqs)):   # j >= i to cover autos + unique crosses
                fi = freqs[i]
                fj = freqs[j]
                print(f"Processing {fi} x {fj}")

                b1 = beams[i](eval_ells)
                b2 = beams[j](eval_ells)
                beamprod = b1*b2

                if not(np.all(np.isfinite(model_dict['total'][(i,j)]))):
                    print("Bad ells: ", eval_ells[~np.isfinite(model_dict['total'][(i,j)])])
                    raise ValueError

                pl = io.Plotter("Dell")
                pl.add(ell,cl_dict[(i,j)],label=f'observed {fi} x {fj}')
                pl.add(eval_ells,model_dict['total'][(i,j)],color='k',label='total')
                pl.add(eval_ells,beamprod*model_dict['cmb'][(i,j)],ls='--',label='cmb',alpha=0.5)
                pl.add(eval_ells,beamprod*model_dict['foreground'][(i,j)],ls=':',label='fg',alpha=0.5)
                pl.add(eval_ells,model_dict['noise'][(i,j)],ls='--',label='noise',alpha=0.5)

                # Galactic dust
                
                Ad   = best["Adust_353"]                 # amplitude at 353 GHz
                if Ad>0:
                    beta = best["beta_dust"]
                    alpha_d = best["alpha_dust"]
                    pivot_dust=353.
                    ell0_dust=80.
                    cl_dust = (Ad *
                               (eval_ells/ell0_dust)**(-alpha_d) *
                               ((fi*fj)/(pivot_dust**2))**(beta/2.0))

                    pl.add(eval_ells,cl_dust*beamprod,label='dust')


                cl_tsz = beamprod*cltsz(best["Atsz"],fi,fj,cl_yy) 
                pl.add(eval_ells,cl_tsz,label='tsz')

                nb1 = beams[i](nells)
                nb2 = beams[j](nells)
                
                nbeamprod = nb1*nb2    
                pl.add(nells,nbeamprod*ncls[f"{fi}_{fj}"],label='Niall FG')
                
                pl._ax.set_ylim(1e-2,1e4)
                pl.legend('outside')
                pl._ax.set_xlim(2,ell.max()+500)
                pl.done(f'cls_{fi}x{fj}.png')

                res[f'{fi}x{fj}'] = cl_dict[(i,j)] - rmodel_dict['total'][(i,j)]

        pl = io.Plotter("rCl")
        for i in range(len(freqs)):
            for j in range(i, len(freqs)):   # j >= i to cover autos + unique crosses
                fi = freqs[i]
                fj = freqs[j]

                pl.add(ell,res[f'{fi}x{fj}']/cl_dict[(i,j)],label=f'residual {fi} x {fj}',alpha=0.5)
        pl.hline(y=0)
        pl._ax.set_ylim(-1,1)
        pl._ax.set_xlim(2,ell.max()+500)
        pl.done(f'res.png')

        
    

    return best_fit, chi2, dof, model_dict

def evaluate_model_dict(
    ell: np.ndarray,
    best: dict,
    freqs: np.ndarray,
    dT_guess: np.ndarray,
    beams: np.ndarray,
    lknees: np.ndarray,
    alphas: np.ndarray,
    cl_cmb_tmpl: np.ndarray,
    cl_yy: np.ndarray,
) -> dict:
    """
    Return a dict of model spectra broken into components:
    {
        (i,j): {
            'total':     C_ell total,
            'cmb':       CMB only,
            'foreground':foregrounds only,
            'noise':     white noise (auto only, else 0)
        }
    }
    """
    model_dict = {}
    model_dict['total'] = {}
    model_dict['cmb'] = {}
    model_dict['foreground'] = {}
    model_dict['noise'] = {}

    def _clean(y):
        y[ell<2] = 0
        return y
    
    beams = _expand_beams(beams,ell,len(freqs))
    
    for i, j in itertools.combinations_with_replacement(range(len(freqs)), 2):
        nu1, nu2 = freqs[i], freqs[j]
        b1 = beams[i](ell)
        b2 = beams[j](ell)
        beamprod = b1*b2

        cmb   = best["A_cmb"] * cl_cmb_tmpl
        fg    = fg_cl(ell, best, nu1, nu2, cl_yy)

        sig_i = best[f"rN_{nu1}"] * dT_guess[i]
        sig_j = best[f"rN_{nu2}"] * dT_guess[j]
        noise = get_noise(ell,i,j,sig_i,sig_j,lknees,alphas,best["Aatm_corr"])

        total = (cmb + fg)*beamprod + noise

        
        model_dict['total'][(i, j)] = _clean(total).copy()
        model_dict['cmb'][(i, j)] = _clean(cmb).copy()
        model_dict['foreground'][(i, j)] = _clean(fg).copy()
        model_dict['noise'][(i, j)] = _clean(noise).copy()

    return model_dict

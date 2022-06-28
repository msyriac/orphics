"""
Utilities for ILC noise, source counts and associated power spectra, etc..

"""

import glob,os,sys
import numpy as np
from scipy.interpolate import interp1d
from orphics import maps, cosmology


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
        'cib_Snu_cutoff_list [mJy]':','.join([str(x) for x in flux_limits_mJy]),
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
    cl_sz['ell'] = np.asarray(cl_sz['ell'])
    ells = cl_sz['ell']
    cl_sz['1h'] = np.asarray(cl_sz['1h']) * 1e-12 / ells / (ells+1.) * 2. * np.pi
    cl_sz['2h'] = np.asarray(cl_sz['2h']) * 1e-12 / ells / (ells+1.) * 2. * np.pi
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

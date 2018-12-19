import healpy as hp
import numpy as np
import os,sys

path = os.environ['SEHGAL_SKY']
if path[-1]!='/': path = path + '/'

default_tcmb = 2.726
H_CGS = 6.62608e-27
K_CGS = 1.3806488e-16
C_light = 2.99792e+10

frequencies = [str(s).zfill(3) for s in [27,30,39,44,70,93,100,143,145,217,225,280,353]]

def fnu(nu,tcmb=default_tcmb):
    """
    nu in GHz
    tcmb in Kelvin
    """
    nu = np.asarray(nu)
    mu = H_CGS*(1e9*nu)/(K_CGS*tcmb)
    ans = mu/np.tanh(old_div(mu,2.0)) - 4.0
    return ans

def get_total_cmb(freq):
    filename = "%s_skymap_healpix_Nside4096_DeltaT_uK_SimLensCMB_tSZrescale0p75_CIBrescale0p75_Comm_synchffAME_rad_pts_fluxcut148_7mJy_lininterp.fits" % freq
    return hp.read_map(path+filename)

def get_lensed_cmb():
    filename = "Sehgalsimparams_healpix_4096_KappaeffLSStoCMBfullsky_phi_SimLens_Tsynfastnopell_fast_lmax8000_nside4096_interp2.5_method1_1_lensed_map.fits" 
    return hp.read_map(path+filename)

def get_ksz():
    filename = "148_ksz_healpix_nopell_Nside4096_DeltaT_uK.fits" 
    return hp.read_map(path+filename)

def get_kappa():
    filename = "healpix_4096_KappaeffLSStoCMBfullsky.fits" 
    return hp.read_map(path+filename)

def get_compton_y():
    filename = "tSZ_skymap_healpix_nopell_Nside4096_y_tSZrescale0p75.fits" 
    return hp.read_map(path+filename)

def get_tsz(freq,comptony=None,tcmb=default_tcmb):
    if comptony is None: comptony = get_compton_y()
    return fnu(freq) * comptony * tcmb * 1e6

def get_cib(freq):
    filename = "%s_ir_pts_healpix_nopell_Nside4096_DeltaT_uK_lininterp_CIBrescale0p75.fits" % freq
    return hp.read_map(path+filename)

def get_radio(freq):
    filename = "%s_rad_pts_healpix_nopell_Nside4096_DeltaT_uK_fluxcut148_7mJy_lininterp.fits" % freq
    return hp.read_map(path+filename)

def get_galactic_dust(freq):
    filename = "%s_dust_healpix_nopell_Nside4096_DeltaT_uK_lininterp.fits" % freq
    return hp.read_map(path+filename)

def get_galactic_lf(freq):
    filename = "commander_synch_freefree_AME_%s_Equ_uK_smooth17arcmin.fits" % freq
    return hp.read_map(path+filename)

def get_mask(area):
    area = str(int(area)).zfill(5)
    filename = "masks/mask_%s_Equ_bool_Nside4096.fits" % area
    return hp.read_map(path+filename)

def cmb_ps():
    ells,ps = np.loadtxt(path+"CMB_PS_healpix_Nside4096_DeltaT_uK_SimLensCMB.txt",unpack=True)
    return ells,ps

def ksz_ps():
    ells,ps = np.loadtxt(path+"kSZ_PS_Sehgal_healpix_Nside4096_DeltaT_uK.txt",unpack=True)
    return ells,ps

def compton_y_ps():
    ells,ps = np.loadtxt(path+"Sehgal_sim_tSZPS_unbinned_8192_y_rescale0p75.txt",unpack=True)
    return ells,ps

def tsz_ps(freq,comptony=None,tcmb=default_tcmb):
    ells,ps = np.loadtxt(path+"Sehgal_sim_tSZPS_unbinned_8192_y_rescale0p75.txt",unpack=True)
    return ells, ps * (fnu(freq)* tcmb * 1e6)**2.


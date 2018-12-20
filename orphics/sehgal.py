import healpy as hp
import numpy as np
import os,sys

default_tcmb = 2.726
H_CGS = 6.62608e-27
K_CGS = 1.3806488e-16
C_light = 2.99792e+10

def fnu(nu,tcmb=default_tcmb):
    """
    nu in GHz
    tcmb in Kelvin
    """
    nu = np.asarray(nu)
    mu = H_CGS*(1e9*nu)/(K_CGS*tcmb)
    ans = mu/np.tanh(old_div(mu,2.0)) - 4.0
    return ans

class SehgalSky(object):
    def __init__(self,path=None,healpix=True):
        if healpix:
            self.rfunc = hp.read_map
            self.nside = 4096
        else:
            self.rfunc = enmap.read_map
            self.shape,self.wcs = enmap.read_map_geometry
        if path is None: path = path = os.environ['SEHGAL_SKY']
        if path[-1]!='/': path = path + '/'
        self.path = path
        self.frequencies = [str(s).zfill(3) for s in [27,30,39,44,70,93,100,143,145,217,225,280,353]]
        self.areas = [4000,8000,16000]

    def get_total_cmb(self,freq,filename_only=False):
        filename = self.path + "%s_skymap_healpix_Nside4096_DeltaT_uK_SimLensCMB_tSZrescale0p75_CIBrescale0p75_Comm_synchffAME_rad_pts_fluxcut148_7mJy_lininterp.fits" % freq
        return filename if filename_only else self.rfunc(filename)

    def get_lensed_cmb(self,filename_only=False):
        filename = self.path + "Sehgalsimparams_healpix_4096_KappaeffLSStoCMBfullsky_phi_SimLens_Tsynfastnopell_fast_lmax8000_nside4096_interp2.5_method1_1_lensed_map.fits" 
        return filename if filename_only else self.rfunc(filename)

    def get_ksz(self,filename_only=False):
        filename = self.path + "148_ksz_healpix_nopell_Nside4096_DeltaT_uK.fits" 
        return filename if filename_only else self.rfunc(filename)

    def get_kappa(self,filename_only=False):
        filename = self.path + "healpix_4096_KappaeffLSStoCMBfullsky.fits" 
        return filename if filename_only else self.rfunc(filename)

    def get_compton_y(self,filename_only=False):
        filename = self.path + "tSZ_skymap_healpix_nopell_Nside4096_y_tSZrescale0p75.fits" 
        return filename if filename_only else self.rfunc(filename)

    def get_cib(self,freq,filename_only=False):
        filename = self.path + "%s_ir_pts_healpix_nopell_Nside4096_DeltaT_uK_lininterp_CIBrescale0p75.fits" % freq
        return filename if filename_only else self.rfunc(filename)

    def get_radio(self,freq,filename_only=False):
        filename = self.path + "%s_rad_pts_healpix_nopell_Nside4096_DeltaT_uK_fluxcut148_7mJy_lininterp.fits" % freq
        return filename if filename_only else self.rfunc(filename)

    def get_galactic_dust(self,freq,filename_only=False):
        filename = self.path + "%s_dust_healpix_nopell_Nside4096_DeltaT_uK_lininterp.fits" % freq
        return filename if filename_only else self.rfunc(filename)

    def get_galactic_lf(self,freq,filename_only=False):
        filename = self.path + "commander_synch_freefree_AME_%s_Equ_uK_smooth17arcmin.fits" % freq
        return filename if filename_only else self.rfunc(filename)

    def get_mask(self,area,filename_only=False):
        area = str(int(area)).zfill(5)
        filename = self.path + "masks/mask_%s_Equ_bool_Nside4096.fits" % area
        return filename if filename_only else self.rfunc(filename)

    def cmb_ps(self):
        ells,ps = np.loadtxt(self.path+"CMB_PS_healpix_Nside4096_DeltaT_uK_SimLensCMB.txt",unpack=True)
        return ells,ps

    def ksz_ps(self):
        ells,ps = np.loadtxt(self.path+"kSZ_PS_Sehgal_healpix_Nside4096_DeltaT_uK.txt",unpack=True)
        return ells,ps

    def compton_y_ps(self):
        ells,ps = np.loadtxt(self.path+"Sehgal_sim_tSZPS_unbinned_8192_y_rescale0p75.txt",unpack=True)
        return ells,ps

    def tsz_ps(self,freq,comptony=None,tcmb=default_tcmb):
        ells,ps = np.loadtxt(self.path+"Sehgal_sim_tSZPS_unbinned_8192_y_rescale0p75.txt",unpack=True)
        return ells, ps * (fnu(freq)* tcmb * 1e6)**2.

    def get_tsz(self,freq,comptony=None,tcmb=default_tcmb):
        if comptony is None: comptony = self.get_compton_y()
        return fnu(freq) * comptony * tcmb * 1e6

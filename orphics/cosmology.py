from __future__ import print_function
import camb
import warnings
from math import pi
from camb import model, initialpower
import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import quad
import itertools
try:
    import cPickle as pickle
except:
    import pickle

import time, re, os
from scipy.integrate import odeint

defaultConstants = {'TCMB': 2.7255
                    ,'G_CGS': 6.67259e-08
                    ,'MSUN_CGS': 1.98900e+33
                    ,'MPC2CM': 3.085678e+24
                    ,'ERRTOL': 1e-12
                    ,'K_CGS': 1.3806488e-16
                    ,'H_CGS': 6.62608e-27
                    ,'C': 2.99792e+10
                    ,'A_ps': 3.1
                    ,'A_g': 0.9
                    ,'nu0': 150.
                    ,'n_g': -0.7
                    ,'al_g': 3.8
                    ,'al_ps': -0.5
                    ,'Td': 9.7
                    ,'al_cib': 2.2
                    ,'A_cibp': 6.9
                    ,'A_cibc': 4.9
                    ,'n_cib': 1.2
                    ,'A_tsz': 5.6
                    ,'ell0sec': 3000.
}

# Planck TT,TE,EE+lowP 2015 cosmology but with updated tau and minimal neutrino mass
defaultCosmology = {'omch2': 0.1198
                    ,'ombh2': 0.02225
                    ,'H0': 67.3
                    ,'ns': 0.9645
                    ,'As': 2.2e-9
                    ,'mnu': 0.06
                    ,'w0': -1.0
                    ,'tau':0.06
                    ,'nnu':3.046
                    ,'wa': 0.
}


class Cosmology(object):
    '''
    A wrapper around CAMB that tries to pre-calculate as much as possible
    Intended to be inherited by other classes like LimberCosmology and 
    ClusterCosmology

    Many member functions were copied/adapted from Cosmicpy:
    http://cosmicpy.github.io/
    '''
    def __init__(self,paramDict=defaultCosmology,constDict=defaultConstants,lmax=2000,clTTFixFile=None,skipCls=False,pickling=False,fill_zero=True,dimensionless=True,verbose=True,skipPower=True,pkgrid_override=None,kmax=10.,skip_growth=True,nonlinear=True,zmax=10.,low_acc=False,z_growth=None,camb_var=None):

        self.camb_var = camb_var
        self.dimensionless = dimensionless
        cosmo = paramDict
        self.paramDict = paramDict
        c = constDict
        self.c = c
        self.cosmo = paramDict


        self.zmax = zmax

        self.c['TCMBmuK'] = self.c['TCMB'] * 1.0e6

        try:
            self.nnu = cosmo['nnu']
        except:
            self.nnu = defaultCosmology['nnu']
            
        self.H0 = cosmo['H0']
        self.h = self.H0/100.
        try:
            self.omch2 = cosmo['omch2']
            self.om = (cosmo['omch2']+cosmo['ombh2'])/self.h**2.
        except:
            self.omch2 = (cosmo['om']-cosmo['ob'])*self.H0*self.H0/100./100.
            self.om = cosmo['om']
            
        try:
            self.ombh2 = cosmo['ombh2']
            self.ob = cosmo['ombh2']/self.h**2.
        except:
            self.ombh2 = cosmo['ob']*self.H0*self.H0/100./100.
            self.ob = cosmo['ob']


        try:
            self.tau = cosmo['tau']
        except:
            self.tau = defaultCosmology['tau']
            warnings.warn("No tau specified; assuming default of "+str(self.tau))
            
            
        self.mnu = cosmo['mnu']
        self.w0 = cosmo['w0']
        self.wa = cosmo['wa']
        self.pars = camb.CAMBparams()
        self.pars.Reion.Reionization = 0
        #print("WARNING: theta fixed!!!")
        self.pars.set_cosmology(H0=self.H0, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, tau=self.tau,nnu=self.nnu,num_massive_neutrinos=3)
        #self.pars.set_cosmology(ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, tau=self.tau,num_massive_neutrinos=3,nnu=self.nnu,H0=None,cosmomc_theta=1.04e-2)
        self.pars.Reion.Reionization = 0
        try:
            self.pars.set_dark_energy(w=self.w0,wa=self.wa,dark_energy_model='ppf')
        except:
            assert np.abs(self.wa)<1e-3, "Non-zero wa requires PPF, which requires devel version of pycamb to be installed."
            print("WARNING: Could not use PPF dark energy model with pycamb. Falling back to non-PPF. Please install the devel branch of pycamb.")
            self.pars.set_dark_energy(w=self.w0)
                  
        self.pars.InitPower.set_params(ns=cosmo['ns'],As=cosmo['As'])

        self.nonlinear = nonlinear
        if nonlinear:
            self.pars.NonLinear = model.NonLinear_both
        else:
            self.pars.NonLinear = model.NonLinear_none
        

        self.results= camb.get_background(self.pars)
        self.omnuh2 = self.pars.omegan * ((self.H0 / 100.0) ** 2.)
        self.chistar = self.results.conformal_time(0)- model.tau_maxvis.value
        self.zstar = self.results.redshift_at_comoving_radial_distance(self.chistar)
        

        # self.rho_crit0 = 3. / (8. * pi) * (self.h*100 * 1.e5)**2. / c['G_CGS'] * c['MPC2CM'] / c['MSUN_CGS']
        self.rho_crit0H100 = 3. / (8. * pi) * (100 * 1.e5)**2. / c['G_CGS'] * c['MPC2CM'] / c['MSUN_CGS']
        self.cmbZ = 1100.
        self.lmax = lmax

        if (clTTFixFile is not None) and not(skipCls):
            ells,cltts = np.loadtxt(clTTFixFile,unpack=True)
            from scipy.interpolate import interp1d
            self.clttfunc = interp1d(ells,cltts,bounds_error=False,fill_value=0.)

        if not(low_acc):
            self.pars.set_accuracy(AccuracyBoost=2.0, lSampleBoost=4.0, lAccuracyBoost=4.0)
            if nonlinear:
                self.pars.NonLinear = model.NonLinear_both
            else:
                self.pars.NonLinear = model.NonLinear_none
        if not(skipCls) and (clTTFixFile is None):
            if verbose: print("Generating theory Cls...")
            if not(low_acc):
                self.pars.set_for_lmax(lmax=(lmax+500), lens_potential_accuracy=3, max_eta_k=2*(lmax+500))
            if nonlinear:
                self.pars.NonLinear = model.NonLinear_both
            else:
                self.pars.NonLinear = model.NonLinear_none
            theory = loadTheorySpectraFromPycambResults(self.results,self.pars,lmax,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=lmax,pickling=pickling,fill_zero=fill_zero,get_dimensionless=dimensionless,verbose=verbose,prefix="_low_acc_"+str(low_acc))
            self.clttfunc = lambda ell: theory.lCl('TT',ell)
            self.theory = theory

            # ells = np.arange(2,lmax,1)
            # cltts = self.clttfunc(ells)
            # np.savetxt("data/cltt_lensed_Feb18.txt",np.vstack((ells,cltts)).transpose())

            
        self.kmax = kmax
        if not(skipPower):
            self.results.calc_transfers(self.pars)
            self._initPower(pkgrid_override)

        self.deltac = 1.42
        self.Omega_m = (self.ombh2+self.omch2)/self.h**2.  # DOESN'T INCLUDE NEUTRINOS
        self.Omega_m_all = (self.ombh2+self.omch2+self.omnuh2)/self.h**2.
        self.Omega_de = 1.-self.Omega_m_all
        self.fnu = (self.omnuh2)/(self.ombh2+self.omch2+self.omnuh2)
        self.kfs_approx_func = lambda a : 0.04 * (a**2.) * np.sqrt(self.Omega_m_all*(a**(-3.))+self.Omega_de) * (self.mnu/0.05) * self.h  # Eq 3.20 S4 science book, in Mpc no h factor
        self.Omega_k = 0.
        self.wa = 0.

        # some useful numbers
        self._cSpeedKmPerSec = 299792.458
        self.G_SI = 6.674e-11
        self.mProton_SI = 1.673e-27
        self.H100_SI = 3.241e-18
        self.thompson_SI = 6.6524e-29
        self.meterToMegaparsec = 3.241e-23

        self.tcmb = self.pars.TCMB #2.726

        T_2_7_sqr = (self.tcmb/2.7)**2
        h2 = self.h**2
        w_m = self.omch2 + self.ombh2
        w_b = self.ombh2

        self._k_eq = 7.46e-2*w_m/T_2_7_sqr / self.h     # Eq. (3) [h/Mpc]
        self._z_eq = 2.50e4*w_m/(T_2_7_sqr)**2          # Eq. (2)

        # z drag from Eq. (4)
        b1 = 0.313*pow(w_m, -0.419)*(1.0+0.607*pow(w_m, 0.674))
        b2 = 0.238*pow(w_m, 0.223)
        self._z_d = 1291.0*pow(w_m, 0.251)/(1.0+0.659*pow(w_m, 0.828)) * \
            (1.0 + b1*pow(w_b, b2))

        # Ratio of the baryon to photon momentum density at z_d  Eq. (5)
        self._R_d = 31.5 * w_b / (T_2_7_sqr)**2 * (1.e3/self._z_d)
        # Ratio of the baryon to photon momentum density at z_eq Eq. (5)
        self._R_eq = 31.5 * w_b / (T_2_7_sqr)**2 * (1.e3/self._z_eq)
        # Sound horizon at drag epoch in h^-1 Mpc Eq. (6)
        self.sh_d = 2.0/(3.0*self._k_eq) * np.sqrt(6.0/self._R_eq) * \
            np.log((np.sqrt(1.0 + self._R_d) + np.sqrt(self._R_eq + self._R_d)) /
                (1.0 + np.sqrt(self._R_eq)))
        # Eq. (7) but in [hMpc^{-1}]
        self._k_silk = 1.6 * pow(w_b, 0.52) * pow(w_m, 0.73) * \
            (1.0 + pow(10.4*w_m, -0.95)) / self.h

        self._da_interp = None

        self._amin = 0.001    # minimum scale factor
        self._amax = 1.0      # maximum scale factor
        self._na = 512        # number of points in interpolation arrays
        self.atab = np.linspace(self._amin,
                             self._amax,
                             self._na)
        

        if not(skip_growth): self._init_growth_rate()

    def growth_scale_dependent(self,ks,z,comp):
        # f(k) exact
        growthfn = self.results.get_redshift_evolution(ks, z, [comp])
        gcomp = growthfn
        return gcomp

    def growth_scale_independent(self,z,zmax=None,numzs=100000):
        # f = dlnD/dlna
        if zmax is None: zmax = z + 0.1
        a = np.linspace(self.z2a(zmax),1,numzs)
        Ds = self.D_growth(a) 
        fapprox = np.gradient(np.log(Ds),np.log(a))
        az = self.z2a(z)
        return interp1d(a,fapprox)(az)

    def growth_approximate(self,z):
        # Approximate growth rate f calculation from Dodelson Eq 9.67
        hfactor = self.H0**2./self.results.hubble_parameter(z)**2.
        omegam = (self.pars.omegab+self.pars.omegac+self.pars.omegan) * ((1.+z)**3.) *hfactor
        omegav = self.pars.omegav * hfactor
        fapprox = omegam**0.6 + omegav*(1.+omegam/2.)/70.
        return fapprox
    

        
    def _initPower(self,pkgrid_override=None):
        print("initializing power...")
        if pkgrid_override is None:
            self.pars.Transfer.accurate_massive_neutrinos = True
            self.PK = camb.get_matter_power_interpolator(self.pars, nonlinear=self.nonlinear,hubble_units=False, k_hunit=False, kmax=self.kmax, zmax=self.zmax,var1=self.camb_var,var2=self.camb_var)
        else:
            class Ptemp:
                def __init__(self,pkgrid):
                    self.pk = pkgrid
                def P(self,zs,ks,grid=True):
                    ks = np.asarray(ks)
                    zs = np.asarray(zs)                            
                    return self.pk(ks,zs,grid=grid).T
            self.PK = Ptemp(pkgrid_override)
            

    def _init_growth_rate(self):
        self.Ds = []
        for a in self.atab:
            self.Ds.append( self.D_growth(a) )
        self.Ds = np.array(self.Ds)
        self.fs = np.gradient(self.Ds,np.diff(self.atab)[0]) * self.atab/self.Ds
        self.Dfunc = interp1d(self.atab,self.Ds)
        self.fsfunc = interp1d(self.atab,self.fs)
        

        

    def Fstar(self,z,xe=1,shaw=True):
        '''
        Get the norm of the kSZ temperature at redshift z
        '''

        TcmbMuK = self.pars.TCMB*1.e6

        ne0 = self.ne0z(z,shaw=shaw)
        return TcmbMuK*self.thompson_SI*ne0*(1.+z)**2./self.meterToMegaparsec  *xe  #*np.exp(-self.tau)


    def ne0z(self,z,shaw=True):
        '''
        Average electron density today but with
        Helium II reionization at z<3
        Units: 1/meter**3
        '''

        if not(shaw):
        
            if z>3.: 
                NHe=1.
            else:
                NHe=2.

            ne0_SI = (1.-(4.-NHe)*self.pars.YHe/4.)*self.ombh2 * 3.*(self.H100_SI**2.)/self.mProton_SI/8./np.pi/self.G_SI

        else:
            chi = 0.86
            me = 1.14
            gasfrac = 0.9
            omgh2 = gasfrac* self.ombh2
            ne0_SI = chi*omgh2 * 3.*(self.H100_SI**2.)/self.mProton_SI/8./np.pi/self.G_SI/me
            
            
        return ne0_SI

    
    #cdm transfer functions. normalized to be 1 at large scales.
    def transfer(self, k, type='camb'):
        if (type == 'camb'):
            transferdata = self.results.get_matter_transfer_data()
            Tk_camb_matter = transferdata.transfer_z('delta_cdm', z_index=0)
            Tk_camb_matter = Tk_camb_matter/Tk_camb_matter[0] 
            Tk_camb_matter_k = transferdata.q/self.h
            #interpolate to required sampling
            interpolation = interp1d(Tk_camb_matter_k,Tk_camb_matter,bounds_error=False,fill_value=0.)
            return interpolation(k)

        if (type == 'eisenhu')or(type == 'eisenhu_osc'):
            w_m = self.omch2 + self.ombh2 #self.Omega_m * self.h**2
            w_b = self.ombh2 #self.Omega_b * self.h**2
            fb = self.ombh2 / (self.omch2+self.ombh2) # self.Omega_b / self.Omega_m
            fc = self.omch2 / (self.omch2+self.ombh2) # self.ombh2 #(self.Omega_m - self.Omega_b) / self.Omega_m
            alpha_gamma = 1.-0.328*np.log(431.*w_m)*w_b/w_m + \
                0.38*np.log(22.3*w_m)*(fb)**2
            gamma_eff = self.Omega_m*self.h * \
                (alpha_gamma + (1.-alpha_gamma)/(1.+(0.43*k*self.sh_d)**4))

            res = np.zeros_like(k)

            if(type == 'eisenhu'):

                q = k * pow(self.tcmb/2.7, 2)/gamma_eff

                # EH98 (29) #
                L = np.log(2.*np.exp(1.0) + 1.8*q)
                C = 14.2 + 731.0/(1.0 + 62.5*q)
                res = L/(L + C*q*q)

            elif(type == 'eisenhu_osc'):
                # Cold dark matter transfer function

                # EH98 (11, 12)
                a1 = pow(46.9*w_m, 0.670) * (1.0 + pow(32.1*w_m, -0.532))
                a2 = pow(12.0*w_m, 0.424) * (1.0 + pow(45.0*w_m, -0.582))
                alpha_c = pow(a1, -fb) * pow(a2, -fb**3)
                b1 = 0.944 / (1.0 + pow(458.0*w_m, -0.708))
                b2 = pow(0.395*w_m, -0.0266)
                beta_c = 1.0 + b1*(pow(fc, b2) - 1.0)
                beta_c = 1.0 / beta_c

                # EH98 (19). [k] = h/Mpc
                def T_tilde(k1, alpha, beta):
                    # EH98 (10); [q] = 1 BUT [k] = h/Mpc
                    q = k1 / (13.41 * self._k_eq)
                    L = np.log(np.exp(1.0) + 1.8 * beta * q)
                    C = 14.2 / alpha + 386.0 / (1.0 + 69.9 * pow(q, 1.08))
                    T0 = L/(L + C*q*q)
                    return T0

                # EH98 (17, 18)
                f = 1.0 / (1.0 + (k * self.sh_d / 5.4)**4)
                Tc = f * T_tilde(k, 1.0, beta_c) + \
                    (1.0 - f) * T_tilde(k, alpha_c, beta_c)

                # Baryon transfer function
                # EH98 (19, 14, 21)
                y = (1.0 + self._z_eq) / (1.0 + self._z_d)
                x = np.sqrt(1.0 + y)
                G_EH98 = y * (-6.0 * x +
                              (2.0 + 3.0*y) * np.log((x + 1.0) / (x - 1.0)))
                alpha_b = 2.07 * self._k_eq * self.sh_d * \
                    pow(1.0 + self._R_d, -0.75) * G_EH98

                beta_node = 8.41 * pow(w_m, 0.435)
                tilde_s = self.sh_d / pow(1.0 + (beta_node /
                                                 (k * self.sh_d))**3, 1.0/3.0)

                beta_b = 0.5 + fb + (3.0 - 2.0 * fb) * np.sqrt((17.2 * w_m)**2 + 1.0)

                # [tilde_s] = Mpc/h
                Tb = (T_tilde(k, 1.0, 1.0) / (1.0 + (k * self.sh_d / 5.2)**2) +
                      alpha_b / (1.0 + (beta_b/(k * self.sh_d))**3) *
                      np.exp(-pow(k / self._k_silk, 1.4))) * np.sinc(k*tilde_s/np.pi)

                # Total transfer function
                res = fb * Tb + fc * Tc
            return res

    def D_growth(self, a, type="camb_z0norm"):
        # D(a)
        if (type=="camb_z0norm")or(type=="camb_anorm"):
            if (self._da_interp is None) or (self._da_interp_type == "cosmicpy"):
                ks = np.logspace(np.log10(1e-5),np.log10(1.),num=100) 
                zs = self.a2z(self.atab)
                deltakz = self.results.get_redshift_evolution(ks, zs, ['delta_cdm']) #index: k,z,0
                D_camb = deltakz[0,:,0]/deltakz[0,0,0]
                self._da_interp = interp1d(self.atab, D_camb, kind='linear')
                self._da_interp_type = "camb"
            if (type=="camb_z0norm"):  #normed so that D(a=1)=1
                return self._da_interp(a)/self._da_interp(1.0)
            if (type=="camb_anorm"):  #normed so that D(a)=a in matter domination
                return self._da_interp(a)/self._da_interp(1.0)*0.76

        elif (type=="cosmicpy"): #also z0 norm
            if (self._da_interp is None) or (self._da_interp_type == "camb"):
                def D_derivs(y, x):
                    q = (2.0 - 0.5 * (self.Omega_m_a(x) +
                                      (1.0 + 3.0 * self.w(x))
                                      * self.Omega_de_a(x)))/x
                    r = 1.5*self.Omega_m_a(x)/x/x
                    return [y[1], -q * y[1] + r * y[0]]
                y0 = [self._amin, 1]

                y = odeint(D_derivs, y0, self.atab)
                self._da_interp = interp1d(self.atab, y[:, 0], kind='linear')
                self._da_interp_type = "cosmicpy"
            return self._da_interp(a)/self._da_interp(1.0)
        
    def Omega_m_a(self, a):
        return self.Omega_m * pow(a, -3) / self.Esqr(a)

    def Omega_de_a(self, a):
        return self.Omega_de*pow(a, self.f_de(a))/self.Esqr(a)

    def Esqr(self, a):
        return self.Omega_m*pow(a, -3) + self.Omega_k*pow(a, -2) + \
            self.Omega_de*pow(a, self.f_de(a))

    def f_de(self, a):
        # Just to make sure we are not diving by 0
        epsilon = 0.000000001
        return -3.0*(1.0+self.w0) + 3.0*self.wa*((a-1.0)/np.log(a-epsilon) - 1.0)

    def w(self, a):
        return self.w0 + (1.0 - a) * self.wa

    def z2a(self,z):
        return 1.0/(1.0 + z)

    def a2z(self,a):
        return (1.0/a)-1.0
    


class LimberCosmology(Cosmology):
    '''Partially based on Anthony Lewis' CAMB Python Notebook
    To do:
    - Add support for curvature
    - Test magnification bias for counts
    - Test that delta function and step function dndz(z)s are equivalent as step function width -> 0

    How To Use:
    1. Initialize with cosmology, constants, lmax, kmax, number of z points
    2. Add a delta function, step-function or generic dndz with addDeltaNz, addStepNz or addNz. A delta function window tagged "cmb" is automatically generated on initialization.
    3. If you want, access the following objects (tag is the name of the window you specified in step 2):
       a) LimberCosmologyObject.zs (the zrange on which the following things are evaluated)
       b) LimberCosmologyObject.kernels[tag]['W'] (the lensing window function)
       c) LimberCosmologyObject.kernels[tag]['window_z'] (only the (chi-chi)/chi part -- or equivalent integral for non-delta-function windows -- of the lensing windo returned as a callable function win(z))
       d) LimberCosmologyObject.kernels[tag]['dndz'] (a callable function of z that gives the normalized redshift distribution of the sources)


    pkgrid_override can be a RectBivariateSpline object such that camb.PK.P(z,k,grid=True) returns the same as pkgrid_override(k,z)
    '''
    def __init__(self,paramDict=defaultCosmology,constDict=defaultConstants,lmax=2000,clTTFixFile=None,skipCls=False,pickling=False,numz=1000,kmax=42.47,nonlinear=True,fill_zero=True,skipPower=False,pkgrid_override=None,zmax=1100.,low_acc=False,skip_growth=True,dimensionless=True,camb_var=None):
        Cosmology.__init__(self,paramDict,constDict,lmax=lmax,clTTFixFile=clTTFixFile,skipCls=skipCls,pickling=pickling,fill_zero=fill_zero,pkgrid_override=pkgrid_override,skipPower=skipPower,kmax=kmax,nonlinear=nonlinear,zmax=zmax,low_acc=low_acc,skip_growth=skip_growth,dimensionless=dimensionless,camb_var=camb_var)


        self.kmax = kmax
        self.chis = np.linspace(0,self.chistar,numz)
        self.zs=self.results.redshift_at_comoving_radial_distance(self.chis)
        self.dchis = (self.chis[2:]-self.chis[:-2])/2
        self.chis = self.chis[1:-1]
        self.zs = self.zs[1:-1]
        self.Hzs = np.array([self.results.hubble_parameter(z) for z in self.zs])
        self.kernels = {}
        self._initWkappaCMB()

        self.skipPower = skipPower

        self.precalcFactor = self.Hzs**2. /self.chis/self.chis/self._cSpeedKmPerSec**2.


    def volume(self,zmin,zmax,fsky=1.):
        """ Return the comoving volume of the universe
        contained within redshifts zmin and zmax, in Mpc^3"""
        return fsky * 4.*np.pi * np.trapz(self.chis[np.logical_and(self.zs>zmin,self.zs<zmax)]**2.*self._cSpeedKmPerSec/self.Hzs[np.logical_and(self.zs>zmin,self.zs<zmax)],self.zs[np.logical_and(self.zs>zmin,self.zs<zmax)])


    def generateCls(self,ellrange,autoOnly=False,zmin=0.):

        if self.skipPower: self._initPower()


        w = np.ones(self.chis.shape)

        retList = {}
        if autoOnly:
            listKeys = list(zip(list(self.kernels.keys()),list(self.kernels.keys())))
        else:
            listKeys = list(itertools.combinations_with_replacement(list(self.kernels.keys()),2))
            
        for key1,key2 in listKeys:
            retList[key1+","+key2] = []
        for ell in ellrange:
            k=(ell+0.5)/self.chis
            w[:]=1
            w[k<1e-4]=0
            w[k>=self.kmax]=0
            pkin = self.PK.P(self.zs, k, grid=False)
            common = ((w*pkin)*self.precalcFactor)[self.zs>=zmin]
    
            for key1,key2 in listKeys:
                estCl = np.dot(self.dchis[self.zs>=zmin], common*(self.kernels[key1]['W']*self.kernels[key2]['W'])[self.zs>=zmin])
                retList[key1+","+key2].append(estCl)
                
        for key1,key2 in listKeys:
            retList[key1+","+key2] = np.array(retList[key1+","+key2])

            
        self.Clmatrix = retList
        self.ellrange = ellrange

    def getCl(self,key1,key2):

        try:
            return self.Clmatrix[key1+","+key2]
        except KeyError:
            return self.Clmatrix[key2+","+key1]
        except:
            print("Key combination not found")
            raise
            
            
    def _lensWindow(self,kernel,numzIntegral):
        '''
        Calculates the following integral
        W(z) = \int dz'  p(z') (chi(z')-chi(z))/chi(z')
        where p(z) is the dndz/pdf of spectra
        and kernel must contain ['dndz']
        '''

        if kernel['dndz']=="delta":
            zthis = kernel['zdelta']
            retvals = ((1.-self.chis/(self.results.comoving_radial_distance(zthis))))
            retvals[self.zs>zthis] = 0.
            return retvals
        else:
            

            retvals = []
            for chinow,znow in zip(self.chis,self.zs):

                if znow>kernel['zmax']:
                    retvals.append(0.) # this could be sped up
                else:

                    _galFunc = lambda z: (kernel['dndz'](z)*((1.-chinow/(self.results.comoving_radial_distance(z))))) 
                    zStart = max(znow,kernel['zmin'])
                    zs=np.linspace(zStart,kernel['zmax'],numzIntegral)
                    dzs = (zs[2:]-zs[:-2])/2
                    zs = zs[1:-1]
                    _galInt = np.dot(dzs,np.array([_galFunc(zp) for zp in zs]))            

                    retvals.append((_galInt))
            return np.array(retvals)
    
    def addDeltaNz(self,tag,zsource,bias=None,magbias=None,ignore_exists=False):

        if not(ignore_exists): assert not(tag in list(self.kernels.keys())), "Tag already exists."
        assert tag!="cmb", "cmb is a tag reserved for cosmic microwave background. Use a different tag."
        
        
        self.kernels[tag] = {}
        self.kernels[tag]['dndz'] = "delta" 
        self.kernels[tag]['zdelta'] = zsource

        self._generateWindow(tag,bias,magbias,numzIntegral=None)
          
            
    def addStepNz(self,tag,zmin,zmax,bias=None,magbias=None,numzIntegral=300,ignore_exists=False):
        if not(ignore_exists): assert not(tag in list(self.kernels.keys())), "Tag already exists."
        assert tag!="cmb", "cmb is a tag reserved for cosmic microwave background. Use a different tag."
        
        self.kernels[tag] = {}
        self.kernels[tag]['zmin'] = zmin
        self.kernels[tag]['zmax'] = zmax
        normStep = (self.kernels[tag]['zmax']-self.kernels[tag]['zmin'])
        self.kernels[tag]['dndz'] = lambda z: 1./normStep
        
        self._generateWindow(tag,bias,magbias,numzIntegral)
        
    def addNz(self,tag,zedges,nz,bias=None,magbias=None,numzIntegral=300,ignore_exists=False):

        '''
        Assumes equally spaced bins
        If bias, then assumes counts, else assumes lensing
        If magbias provided, applies it as magnification bias assuming it is 's' in Eq 7 of Omuri Holder. Bias must be provided too.
        '''

        if not(ignore_exists): assert not(tag in list(self.kernels.keys())), "Tag already exists."
        assert tag!="cmb", "cmb is a tag reserved for cosmic microwave background. Use a different tag."
        
            
        dzs = (zedges[1:]-zedges[:-1])
        norm = np.dot(dzs,nz)
        zmids = (zedges[1:]+zedges[:-1])/2.
        self.kernels[tag] = {}
        self.kernels[tag]['dndz'] = interp1d(zmids,nz/norm,bounds_error=False,fill_value=0.)
        self.kernels[tag]['zmin'] = zedges.min()
        self.kernels[tag]['zmax'] = zedges.max()

        self._generateWindow(tag,bias,magbias,numzIntegral)

    def _generateWindow(self,tag,bias,magbias,numzIntegral):
        print(("Initializing galaxy window for ", tag , " ..."))
        if bias==None:

            retvals = self._lensWindow(self.kernels[tag],numzIntegral)
            self.kernels[tag]['window_z'] = interp1d(self.zs,retvals.copy())
            self.kernels[tag]['W'] =  retvals *1.5*(self.omch2+self.ombh2+self.omnuh2)*100.*100.*(1.+self.zs)*self.chis/self.Hzs/self._cSpeedKmPerSec
            self.kernels[tag]['type'] = 'lensing'
        else:
            # FAILS FOR STEP !!!!
            assert self.kernels[tag]['dndz']!="delta"
            self.kernels[tag]['W'] = self.zs*0.+bias*self.kernels[tag]['dndz'](self.zs)
            self.kernels[tag]['W'][self.zs<self.kernels[tag]['zmin']] = 0.
            self.kernels[tag]['W'][self.zs>self.kernels[tag]['zmax']] = 0.
            self.kernels[tag]['type'] = 'counts'
            if magbias!=None:
                retvals = self._lensWindow(self.kernels[tag],numzIntegral)
                magcorrection = retvals*1.5*(self.omch2+self.ombh2+self.omnuh2)*100.*100.*(1.+self.zs)*self.chis*(5.*magbias-2.)/self.Hzs**2./self._cSpeedKmPerSec # this needs to be checked again
                self.kernels[tag]['W'] += magcorrection
                print(("Lensing bias max percent correction in counts ", np.max((np.nan_to_num(magcorrection *100./ self.kernels[tag]['W'])))))
                print(("Lensing bias min percent correction in counts ", np.min((np.nan_to_num(magcorrection *100./ self.kernels[tag]['W'])))))


            
        


    def _initWkappaCMB(self):#,numz):

        print("Initializing CMB window..")
        chirange = self.chis
        
        iwcmb =  1.5*(self.omch2+self.ombh2+self.omnuh2)*100.*100.*(1.+self.zs)*self.chis*((self.chistar - self.chis)/self.chistar)/self.Hzs/self._cSpeedKmPerSec
        self.kernels['cmb']={}
        self.kernels['cmb']['W'] = iwcmb
        self.kernels['cmb']['window_z'] = interp1d(self.zs,(self.chistar - self.chis)/self.chistar)
                
                

def unpack_cmb_theory(theory,ells,lensed=False):
    
    if lensed:
        cltt = theory.lCl('TT',ells)
        clee = theory.lCl('EE',ells)
        clte = theory.lCl('TE',ells)
        clbb = theory.lCl('BB',ells)    
    else:
        cltt = theory.uCl('TT',ells)
        clee = theory.uCl('EE',ells)
        clte = theory.uCl('TE',ells)
        clbb = theory.uCl('BB',ells)

    return cltt, clee, clte, clbb

def enmap_power_from_orphics_theory(theory,lmax=None,ells=None,lensed=False,dimensionless=True,orphics_dimensionless=True,TCMB=2.7255e6):
    if orphics_dimensionless and dimensionless: tmul = 1.
    if orphics_dimensionless and not(dimensionless): tmul = TCMB**2.
    if not(orphics_dimensionless) and not(dimensionless): tmul = 1.
    if not(orphics_dimensionless) and dimensionless: tmul = 1./TCMB**2.
    
    oned = False
    if ells is None:
        ells = np.arange(0,lmax,1)
    if ells.ndim==1: oned = True
    cltt, clee, clte, clbb = unpack_cmb_theory(theory,ells,lensed=lensed)
    ps = np.zeros((3,3,fine_ells.size)) if oned else np.zeros((3,3,ells.shape[0],ells.shape[1]))
    ps[0,0] = cltt
    ps[1,1] = clee
    ps[0,1] = clte
    ps[1,0] = clte
    ps[2,2] = clbb

    return ps*tmul

        
def loadTheorySpectraFromPycambResults(results,pars,kellmax,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000,pickling=False,fill_zero=False,get_dimensionless=True,verbose=True,prefix=""):
    '''

    The spectra are stored in dimensionless form, so TCMB has to be specified. They should 
    be used with dimensionless noise spectra and dimensionless maps.

    All ell and 2pi factors are also stripped off.

 
    '''

    
    if get_dimensionless:
        tmul = 1.
    else:
        tmul = TCMB**2.
        
    if useTotal:
        uSuffix = "unlensed_total"
        lSuffix = "total"
    else:
        uSuffix = "unlensed_scalar"
        lSuffix = "lensed_scalar"

    try:
        assert pickling
        clfile = "output/clsAll"+prefix+"_"+str(kellmax)+"_"+time.strftime('%Y%m%d') +".pkl"
        cmbmat = pickle.load(open(clfile,'rb'))
        if verbose: print("Loaded cached Cls from ", clfile)
    except:
        cmbmat = results.get_cmb_power_spectra(pars)
        if pickling:
            import os
            directory = "output/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            pickle.dump(cmbmat,open("output/clsAll"+prefix+"_"+str(kellmax)+"_"+time.strftime('%Y%m%d') +".pkl",'wb'))

    theory = TheorySpectra()
    for i,pol in enumerate(['TT','EE','BB','TE']):
        cls =cmbmat[lSuffix][2:,i]

        ells = np.arange(2,len(cls)+2,1)
        cls *= 2.*np.pi/ells/(ells+1.)*tmul
        theory.loadCls(ells,cls,pol,lensed=True,interporder="linear",lpad=lpad,fill_zero=fill_zero)

        if unlensedEqualsLensed:
            theory.loadCls(ells,cls,pol,lensed=False,interporder="linear",lpad=lpad,fill_zero=fill_zero)            
        else:
            cls = cmbmat[uSuffix][2:,i]
            ells = np.arange(2,len(cls)+2,1)
            cls *= 2.*np.pi/ells/(ells+1.)*tmul
            theory.loadCls(ells,cls,pol,lensed=False,interporder="linear",lpad=lpad,fill_zero=fill_zero)

    try:
        assert pickling
        clfile = "output/clphi"+prefix+"_"+str(kellmax)+"_"+time.strftime('%Y%m%d') +".txt"
        clphi = np.loadtxt(clfile)
        if verbose: print("Loaded cached Cls from ", clfile)
    except:
        lensArr = results.get_lens_potential_cls(lmax=kellmax)
        clphi = lensArr[2:,0]
        if pickling:
            import os
            directory = "output/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            np.savetxt("output/clphi"+prefix+"_"+str(kellmax)+"_"+time.strftime('%Y%m%d') +".txt",clphi)

    clkk = clphi* (2.*np.pi/4.)
    ells = np.arange(2,len(clkk)+2,1)
    theory.loadGenericCls(ells,clkk,"kk",lpad=lpad,fill_zero=fill_zero)


    theory.dimensionless = get_dimensionless
    return theory



class TheorySpectra:
    '''
    Essentially just an interpolator that takes a CAMB-like
    set of discrete Cls and provides lensed and unlensed Cl functions
    for use in integrals
    '''
    

    def __init__(self):


        self._uCl={}
        self._lCl={}
        self._gCl = {}


    def loadGenericCls(self,ells,Cls,keyName,lpad=9000,fill_zero=True):
        if not(fill_zero):
            fillval = Cls[ells<lpad][-1]
            self._gCl[keyName] = lambda x: np.piecewise(x, [x<=lpad,x>lpad], [lambda y: interp1d(ells[ells<lpad],Cls[ells<lpad],bounds_error=False,fill_value=0.)(y),lambda y: fillval*(lpad/y)**4.])

        else:
            fillval = 0.            
            self._gCl[keyName] = interp1d(ells[ells<lpad],Cls[ells<lpad],bounds_error=False,fill_value=fillval)
        

        

    def gCl(self,keyName,ell):

        if len(keyName)==3:
            # assume uTT, lTT, etc.
            ultype = keyName[0].lower()
            if ultype=="u":
                return self.uCl(keyName[1:],ell)
            elif ultype=="l":
                return self.lCl(keyName[1:],ell)
            else:
                raise ValueError
        
        try:
            return self._gCl[keyName](ell)
        except:
            return self._gCl[keyName[::-1]](ell)
        
    def loadCls(self,ell,Cl,XYType="TT",lensed=False,interporder="linear",lpad=9000,fill_zero=True):

        # Implement ellnorm

        mapXYType = XYType.upper()
        validateMapType(mapXYType)


        if not(fill_zero):
            fillval = Cl[ell<lpad][-1]
            f = lambda x: np.piecewise(x, [x<=lpad,x>lpad], [lambda y: interp1d(ell[ell<lpad],Cl[ell<lpad],bounds_error=False,fill_value=0.)(y),lambda y: fillval*(lpad/y)**4.])

        else:
            fillval = 0.            
            f = interp1d(ell[ell<lpad],Cl[ell<lpad],bounds_error=False,fill_value=fillval)
                    
        if lensed:
            self._lCl[XYType]=f
        else:
            self._uCl[XYType]=f

    def _Cl(self,XYType,ell,lensed=False):

            
        mapXYType = XYType.upper()
        validateMapType(mapXYType)

        if mapXYType=="ET": mapXYType="TE"
        ell = np.array(ell)

        try:
            if lensed:    
                retlist = np.array(self._lCl[mapXYType](ell))
                return retlist
            else:
                retlist = np.array(self._uCl[mapXYType](ell))
                return retlist

        except:
            zspecs = ['EB','TB']
            if (XYType in zspecs) or (XYType[::-1] in zspecs):
                return ell*0.
            else:
                raise

    def uCl(self,XYType,ell):
        return self._Cl(XYType,ell,lensed=False)
    def lCl(self,XYType,ell):
        return self._Cl(XYType,ell,lensed=True)
    


def validateMapType(mapXYType):
    assert not(re.search('[^TEB]', mapXYType)) and (len(mapXYType)==2), \
      bcolors.FAIL+"\""+mapXYType+"\" is an invalid map type. XY must be a two" + \
      " letter combination of T, E and B. e.g TT or TE."+bcolors.ENDC


def default_theory(lpad=9000):
    cambRoot = os.path.dirname(__file__)+"/../data/Aug6_highAcc_CDM"
    return loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=lpad,get_dimensionless=False)
    
def loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=True):
    '''
    Given a CAMB path+output_root, reads CMB and lensing Cls into 
    an orphics.theory.gaussianCov.TheorySpectra object.

    The spectra are stored in dimensionless form, so TCMB has to be specified. They should 
    be used with dimensionless noise spectra and dimensionless maps.

    All ell and 2pi factors are also stripped off.

 
    '''
    if not(get_dimensionless): TCMB = 1.
    if useTotal:
        uSuffix = "_totCls.dat"
        lSuffix = "_lensedtotCls.dat"
    else:
        uSuffix = "_scalCls.dat"
        lSuffix = "_lensedCls.dat"

    uFile = cambRoot+uSuffix
    lFile = cambRoot+lSuffix

    theory = TheorySpectra()

    ell, lcltt, lclee, lclbb, lclte = np.loadtxt(lFile,unpack=True,usecols=[0,1,2,3,4])
    mult = 2.*np.pi/ell/(ell+1.)/TCMB**2.
    lcltt *= mult
    lclee *= mult
    lclte *= mult
    lclbb *= mult
    theory.loadCls(ell,lcltt,'TT',lensed=True,interporder="linear",lpad=lpad)
    theory.loadCls(ell,lclte,'TE',lensed=True,interporder="linear",lpad=lpad)
    theory.loadCls(ell,lclee,'EE',lensed=True,interporder="linear",lpad=lpad)
    theory.loadCls(ell,lclbb,'BB',lensed=True,interporder="linear",lpad=lpad)

    try:
        elldd, cldd = np.loadtxt(cambRoot+"_lenspotentialCls.dat",unpack=True,usecols=[0,5])
        clkk = 2.*np.pi*cldd/4.
    except:
        elldd, cldd = np.loadtxt(cambRoot+"_scalCls.dat",unpack=True,usecols=[0,4])
        clkk = cldd*(elldd+1.)**2./elldd**2./4./TCMB**2.
        
    theory.loadGenericCls(elldd,clkk,"kk",lpad=lpad)


    if unlensedEqualsLensed:

        theory.loadCls(ell,lcltt,'TT',lensed=False,interporder="linear",lpad=lpad)
        theory.loadCls(ell,lclte,'TE',lensed=False,interporder="linear",lpad=lpad)
        theory.loadCls(ell,lclee,'EE',lensed=False,interporder="linear",lpad=lpad)
        theory.loadCls(ell,lclbb,'BB',lensed=False,interporder="linear",lpad=lpad)

    else:
        ell, cltt, clee, clte = np.loadtxt(uFile,unpack=True,usecols=[0,1,2,3])
        mult = 2.*np.pi/ell/(ell+1.)/TCMB**2.
        cltt *= mult
        clee *= mult
        clte *= mult
        clbb = clee*0.

        theory.loadCls(ell,cltt,'TT',lensed=False,interporder="linear",lpad=lpad)
        theory.loadCls(ell,clte,'TE',lensed=False,interporder="linear",lpad=lpad)
        theory.loadCls(ell,clee,'EE',lensed=False,interporder="linear",lpad=lpad)
        theory.loadCls(ell,clbb,'BB',lensed=False,interporder="linear",lpad=lpad)

    theory.dimensionless = get_dimensionless
    return theory



#### FISHER FORECASTS

class LensForecast:

    def __init__(self,theory=None):
        '''
        Make S/N projections for CMB and OWL auto and cross-correlations.

        K refers to the CMB (source) kappa
        S refers to the shear/kappa of an optical background galaxy sample
        G refers to the number density of an optical foreground galaxy sample

        '''
        self._haveKK = False
        self._haveKG = False
        self._haveGG = False
        
        self._haveSS = False
        self._haveSG = False
        
        self._haveKS = False

        if theory is None:
            self.theory = TheorySpectra()
        else:
            self.theory = theory
            
        self.Nls = {}
        

    def loadKK(self,ellsCls,Cls,ellsNls,Nls):
        self.Nls['kk'] = interp1d(ellsNls,Nls,bounds_error=False,fill_value=np.inf)
        self.theory.loadGenericCls(ellsCls,Cls,'kk')
    
        self._haveKK = True
        

    def loadGG(self,ellsCls,Cls,ngal):
        self.ngalForeground = ngal
        self.Nls['gg'] = lambda x: x*0.+1./(self.ngalForeground*1.18e7)
        self.theory.loadGenericCls(ellsCls,Cls,'gg')
    
        self._haveGG = True
        
        

    def loadSS(self,ellsCls,Cls,ngal,shapeNoise=0.3):
        if shapeNoise==None or shapeNoise<1.e-9:
            print("No/negligible shape noise given. Using default = 0.3.")
            self.shapeNoise=0.3

        else:             
            self.shapeNoise = shapeNoise
        self.ngalBackground = ngal
        self.Nls['ss'] = lambda x: x*0.+self.shapeNoise*self.shapeNoise/(2.*self.ngalBackground*1.18e7)


        self.theory.loadGenericCls(ellsCls,Cls,'ss')
        
        self._haveSS = True


    def loadSG(self,ellsCls,Cls):
        self.theory.loadGenericCls(ellsCls,Cls,'sg')
        
        self._haveSG = True


    def loadKG(self,ellsCls,Cls):
        self.theory.loadGenericCls(ellsCls,Cls,'kg')
        self._haveKG = True
                

    def loadKS(self,ellsCls,Cls):
        self.theory.loadGenericCls(ellsCls,Cls,'ks')

        self._haveKS = True

    def loadGenericCls(self,specType,ellsCls,Cls,ellsNls=None,Nls=None):
        if Nls is not None: self.Nls[specType] = interp1d(ellsNls,Nls,bounds_error=False,fill_value=np.inf)
        self.theory.loadGenericCls(ellsCls,Cls,specType)
        
    def _bin_cls(self,spec,ell_left,ell_right,noise=True):
        a,b = spec
        ells = np.arange(ell_left,ell_right+1,1)
        cls = self.theory.gCl(spec,ells)
        Noise = 0.
        if noise:
            if a==b:
                Noise = self.Nls[spec](ells)
            else:
                Noise = 0.
        tot = cls+Noise
        return np.sum(ells*tot)/np.sum(ells)

    def KnoxCov(self,specTypeXY,specTypeWZ,ellBinEdges,fsky):
        '''
        returns cov(Cl_XY,Cl_WZ),signalToNoise(Cl_XY)^2, signalToNoise(Cl_WZ)^2
        '''
        def ClTot(spec,ell1,ell2):
            return self._bin_cls(spec,ell1,ell2,noise=True)
        
        X, Y = specTypeXY
        W, Z = specTypeWZ

        ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1]) / 2
        ellWidths = np.diff(ellBinEdges)

        covs = []
        sigs1 = []
        sigs2 = []

        for ell_left,ell_right in zip(ellBinEdges[:-1],ellBinEdges[1:]):
            ClSum = ClTot(X+W,ell_left,ell_right)*ClTot(Y+Z,ell_left,ell_right)+ClTot(X+Z,ell_left,ell_right)*ClTot(Y+W,ell_left,ell_right)
            ellMid = (ell_right+ell_left)/2.
            ellWidth = ell_right-ell_left
            var = ClSum/(2.*ellMid+1.)/ellWidth/fsky
            covs.append(var)
            sigs1.append(self._bin_cls(specTypeXY,ell_left,ell_right,noise=False)**2.*np.nan_to_num(1./var))
            sigs2.append(self._bin_cls(specTypeWZ,ell_left,ell_right,noise=False)**2.*np.nan_to_num(1./var))
        

        return np.array(covs), np.array(sigs1), np.array(sigs2)

    def sigmaClSquared(self,specType,ellBinEdges,fsky):
        return self.KnoxCov(specType,specType,ellBinEdges,fsky)[0]

    def sn(self,ellBinEdges,fsky,specType):
        
        var, sigs1, sigs2 = self.KnoxCov(specType,specType,ellBinEdges,fsky)

        signoise = np.sqrt(sigs1.sum())
        errs = np.sqrt(var)

        return signoise, errs
            

    def snRatio(self,ellBinEdges,fsky):
        
        ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1]) / 2
        ellWidths = np.diff(ellBinEdges)

        sumchisq = 0.
        signum = 0.
        sigden = 0.
        
        for ellMid,ellWidth in zip(ellMids,ellWidths):
            Clkk = self.theory.gCl('kk',ellMid)
            Nlkk = self.Nls['kk'](ellMid)
            Nlgg = self.Nls['gg'](ellMid)
            Nlss = self.Nls['ss'](ellMid)
            Clkg = self.theory.gCl('kg',ellMid)
            Clgg = self.theory.gCl('gg',ellMid)
            Clks = self.theory.gCl('ks',ellMid)
            Clss = self.theory.gCl('ss',ellMid)
            Clsg = self.theory.gCl('sg',ellMid)
    
            r0 = Clkg / Clsg
            pref = 1./(fsky*(2.*ellMid+1.)*ellWidth) # added ellWidth

            sigmaZsq = ((Clkk+Nlkk)*(Clgg+Nlgg))+(Clkg**2.)+((r0**2.)*((Clss+Nlss)*(Clgg+Nlgg)+Clsg**2.))-(2*r0*(Clks*(Clgg+Nlgg)+Clkg*Clsg))

            sigmaZsq = sigmaZsq * pref

            numer = (Clsg**2.)
            denom = sigmaZsq

            signum += (Clkg*Clsg/sigmaZsq)
            sigden += ((Clsg**2.)/sigmaZsq)


            chisq = numer/denom

            sumchisq += chisq
    
        maxlike = signum/sigden

        sigmaR = 1./np.sqrt(sumchisq)
        percentR = sigmaR*100./maxlike
        snR = maxlike/sigmaR

        return percentR,snR,maxlike
           
def noise_func(ell,fwhm,rms_noise,lknee=0.,alpha=0.,dimensionless=False,TCMB=2.7255e6):
    '''Beam deconvolved noise in whatever units rms_noise is in.                         
    e.g. If rms_noise is in uK-arcmin, returns noise in uK**2.                           
    '''
    atmFactor = atm_factor(ell,lknee,alpha)
    rms = rms_noise * (1./60.)*(np.pi/180.)
    tht_fwhm = np.deg2rad(fwhm / 60.)

    nfact = white_noise_with_atm_func(ell,rms_noise,lknee,alpha,dimensionless,TCMB)

    ans = nfact * np.exp((tht_fwhm**2.)*(ell**2.) / (8.*np.log(2.)))
    return ans


def atm_factor(ell,lknee,alpha):
    if lknee>1.e-3:
        atmFactor = (lknee*np.nan_to_num(1./ell))**(-alpha)
    else:
        atmFactor = 0.
    return np.nan_to_num(atmFactor)

def white_noise_with_atm_func(ell,uk_arcmin,lknee,alpha,dimensionless,TCMB=2.7255e6):
    atmFactor = atm_factor(ell,lknee,alpha)
    noiseWhite = ell*0.+(uk_arcmin*np.pi / (180. * 60))**2.
    dfact = (1./TCMB**2.) if dimensionless else 1.
    return (atmFactor+1.)*noiseWhite*dfact

def noise_pad_infinity(Nlfunc,ellmin,ellmax):
    return lambda x: np.piecewise(np.asarray(x).astype(float), [np.asarray(x)<ellmin,np.logical_and(np.asarray(x)>=ellmin,np.asarray(x)<=ellmax),np.asarray(x)>ellmax], [lambda y: np.inf, lambda y: Nlfunc(y), lambda y: np.inf])
    
def getAtmosphere(beamFWHMArcmin=None,returnFunctions=False):
    '''Get TT-lknee, TT-alpha, PP-lknee, PP-alpha  
    Returns either as functions of beam FWHM (arcmin) or for specified beam FWHM (arcmin)
    '''

    if beamFWHMArcmin is None: assert returnFunctions
    if not(returnFunctions): assert beamFWHMArcmin is not None

    # best fits from M.Hasselfield                                                                                            
    ttalpha = -4.7
    ttlknee = np.array([350.,3400.,4900.])
    pplknee = np.array([60,330,460])
    ppalpha = np.array([-2.6,-3.8,-3.9])
    size = np.array([0.5,5.,7.]) # size in meters                                                                             

    freq = 150.e9
    cspeed = 299792458.
    wavelength = cspeed/freq
    resin = 1.22*wavelength/size*60.*180./np.pi
    from scipy.interpolate import interp1d,splrep,splev

    ttlkneeFunc = interp1d(resin,ttlknee,fill_value="extrapolate",kind="linear")
    ttalphaFunc = lambda x: ttalpha
    pplkneeFunc = interp1d(resin,pplknee,fill_value="extrapolate",kind="linear")
    ppalphaFunc = interp1d(resin,ppalpha,fill_value="extrapolate",kind="linear")

    if returnFunctions:
        return ttlkneeFunc,ttalphaFunc,pplkneeFunc,ppalphaFunc
    else:
        b = beamFWHMArcmin
        return ttlkneeFunc(b),ttalphaFunc(b),pplkneeFunc(b),ppalphaFunc(b)


def get_lensed_cls(theory,ells,clkk,lmax):
    import camb.correlations as corr
    
    ellrange = np.arange(0,lmax+2000,1)
    mulfact = ellrange*(ellrange+1.)/2./np.pi
    ucltt = theory.uCl('TT',ellrange)*mulfact
    uclee = theory.uCl('EE',ellrange)*mulfact
    uclbb = theory.uCl('BB',ellrange)*mulfact
    uclte = theory.uCl('TE',ellrange)*mulfact
    from scipy.interpolate import interp1d
    clkkfunc = interp1d(ells,clkk)
    clpp = clkkfunc(ellrange)*4./2./np.pi

    cmbarr = np.vstack((ucltt,uclee,uclbb,uclte)).T
    #print "Calculating lensed cls..."
    lcls = corr.lensed_cls(cmbarr,clpp)

    lmax = lmax+2000
    
    cellrange = ellrange[:lmax].reshape((ellrange[:lmax].size,1)) #cellrange.ravel()[:lmax]
    lclall = lcls[:lmax,:]
    with np.errstate(divide='ignore', invalid='ignore'):
        lclall = np.nan_to_num(lclall/cellrange/(cellrange+1.)*2.*np.pi)
    cellrange = cellrange.ravel()
    #clcltt = lcls[:lmax,0]
    #clcltt = np.nan_to_num(clcltt/cellrange/(cellrange+1.)*2.*np.pi)
    #print clcltt
    lpad = lmax
    
    dtheory = TheorySpectra()
    with np.errstate(divide='ignore', invalid='ignore'):
        mult = np.nan_to_num(1./mulfact)
    ucltt *= mult
    uclee *= mult
    uclte *= mult
    uclbb *= mult
    #print cellrange.shape
    #print ucltt.shape
    dtheory.loadCls(cellrange,ucltt[:lmax],'TT',lensed=False,interporder="linear",lpad=lpad)
    dtheory.loadCls(cellrange,uclte[:lmax],'TE',lensed=False,interporder="linear",lpad=lpad)
    dtheory.loadCls(cellrange,uclee[:lmax],'EE',lensed=False,interporder="linear",lpad=lpad)
    dtheory.loadCls(cellrange,uclbb[:lmax],'BB',lensed=False,interporder="linear",lpad=lpad)
    dtheory.loadGenericCls(ells,clkk,"kk",lpad=lpad)

    lcltt = lclall[:,0]
    lclee = lclall[:,1]
    lclbb = lclall[:,2]
    lclte = lclall[:,3]
    #lcltt *= mult
    #lclee *= mult
    #lclte *= mult
    #lclbb *= mult
    dtheory.loadCls(cellrange,lcltt,'TT',lensed=True,interporder="linear",lpad=lpad)
    dtheory.loadCls(cellrange,lclte,'TE',lensed=True,interporder="linear",lpad=lpad)
    dtheory.loadCls(cellrange,lclee,'EE',lensed=True,interporder="linear",lpad=lpad)
    dtheory.loadCls(cellrange,lclbb,'BB',lensed=True,interporder="linear",lpad=lpad)


    return dtheory




def power_from_theory(ells,theory,lensed=True,pol=False):
    ncomp = 3 if pol else 1
    cfunc = theory.lCl if lensed else theory.uCl
    ps = np.zeros((ncomp,ncomp,)+ells.shape)
    ps[0,0] = cfunc('TT',ells)
    if pol:
        ps[1,1] = cfunc('EE',ells)
        ps[2,2] = cfunc('BB',ells)
        ps[0,1] = cfunc('TE',ells)
        ps[1,0] = cfunc('TE',ells)
    return ps



def fk_comparison(param,z,val1,val2,oparams=None):

    params = defaultCosmology
    params[param] = val1

    if oparams is not None:
        for key in oparams.keys():
            params[key] = oparams[key]

    cc = Cosmology(params,skipCls=True,zmax=z+1,kmax=10,low_acc=True)
    ks = np.logspace(np.log10(1e-4),np.log10(0.3),500)
    comp = 'growth'

    gfunc = lambda cci: cci.results.get_redshift_evolution(ks, z, [comp])
    g1 = gfunc(cc)
    
    g1approx2 = cc.growth_scale_independent(z)
    params = defaultCosmology
    params[param] = val2
    if oparams is not None:
        for key in oparams.keys():
            params[key] = oparams[key]

    
    cc = Cosmology(params,skipCls=True,zmax=z+1,kmax=10,low_acc=True)
    g2 = gfunc(cc)
    
    g2approx2 = cc.growth_scale_independent(z)
    from orphics import io
    pl = io.Plotter(xlabel='k',ylabel='$f(k)$',xscale='log')
    pl.add(ks,g1.ravel(),label=param+'='+str(val1),color="C0")
    pl.add(ks,g2.ravel(),label=param+'='+str(val2),color="C1")
    pl.hline(y=g1approx2,color="C0")
    pl.hline(y=g2approx2,color="C1")
    pl.legend(loc = 'upper right')
    pl.done()

def pk_comparison(param,z,val1,val2,oparams=None):

    params = defaultCosmology
    params[param] = val1

    if oparams is not None:
        for key in oparams.keys():
            params[key] = oparams[key]

    cc = Cosmology(params,skipCls=True,zmax=z+1,kmax=10,low_acc=True,skipPower=False)
    ks = np.logspace(np.log10(1e-4),np.log10(0.3),500)

    pk1 = cc.PK.P(z, ks, grid=False)
    
    params = defaultCosmology
    params[param] = val2
    if oparams is not None:
        for key in oparams.keys():
            params[key] = oparams[key]

    
    cc = Cosmology(params,skipCls=True,zmax=z+1,kmax=10,low_acc=True,skipPower=False)
    pk2 = cc.PK.P(z, ks, grid=False)

    
    from orphics import io
    pl = io.Plotter(xlabel='k',ylabel='$P(k)$',xscale='log',yscale='log')
    pl.add(ks,pk1.ravel(),label=param+'='+str(val1),color="C0")
    pl.add(ks,pk2.ravel(),label=param+'='+str(val2),color="C1")
    pl.legend(loc = 'upper right')
    pl.done()
    
    pl = io.Plotter(xlabel='k',ylabel='$\Delta P(k) / P$',xscale='log')
    pl.add(ks,(pk2.ravel()-pk1.ravel())/pk2.ravel(),label=param+'='+str(val1),color="C0")
    pl.legend(loc = 'upper right')
    pl.done()




def class_cls(lmax,params=None,cosmo=None,zmin=None,zmax=None,bias=None,dndz_file=None):
    from classy import Class
    smean = (zmin+zmax)/2.
    shalfwidth = (zmax-zmin)/2.
    print(smean,shalfwidth)

    # Define your cosmology (what is not specified will be set to CLASS default parameters)
    oparams = {
        'output': 'tCl lCl dCl',
        'l_max_scalars': lmax,
        'lensing': 'yes',
        'A_s': 2.3e-9,
        'n_s': 0.9624, 
        'h': 0.6711,
        'omega_b': 0.022068,
        'omega_cdm': 0.12029,
        'selection':'tophat',
        'selection_mean':'%f'%smean,
        'selection_width': '%f'%shalfwidth,
        'selection_bias':'%f'%bias,
        'number count contributions' : 'density, rsd, lensing, gr',
        'dNdz_selection':'%s'%dndz_file,'l_max_lss':lmax,'l_max_scalars':lmax}

    if params is not None:
        for key in params.keys():
            oparams[key] = params[key]

    if cosmo is None:
        cosmo = Class()
        cosmo.set(oparams)
        cosmo.compute()

    cls = cosmo.density_cl(lmax)
    cls2 = cosmo.lensed_cl(lmax)

    clpg = cls['pd'][0]
    clgg = cls['dd'][0]
    clpp = cls2['pp']
    ells = cls['ell']
    assert np.all(np.isclose(ells,cls2['ell']))

    retcls = {}
    retcls['kg'] = clpg * ells * (ells+1.)/2.
    retcls['kk'] = clpp * (ells * (ells+1.)/2.)**2.
    retcls['gg'] = clgg
    retcls['ells'] = ells

    return retcls,cosmo,params
    



class ClassCosmology(object):

    def __init__(self,params,pol=True,gal=True):

        oparams = {
            'output': 'tCl lCl',
            'l_max_scalars': lmax,
            'lensing': 'yes',
            'A_s': 2.3e-9,
            'n_s': 0.9624, 
            'h': 0.6711,
            'omega_b': 0.022068,
            'omega_cdm': 0.12029,
            'selection':'tophat',
            'selection_mean':'%f'%smean,
            'selection_width': '%f'%shalfwidth,
            'selection_bias':'%f'%bias,
            'number count contributions' : 'density, rsd, lensing, gr',
            'dNdz_selection':'%s'%dndz_file,'l_max_lss':lmax,'l_max_scalars':lmax}

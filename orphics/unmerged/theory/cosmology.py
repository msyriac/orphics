from __future__ import print_function
import warnings
warnings.warn("WARNING: This module is deprecated. Most of its contents have moved to orphics.cosmology. If you do not find the function you require there, please raise an issue.")
import camb
from math import pi
from orphics.tools.cmb import loadTheorySpectraFromPycambResults
from camb import model, initialpower
import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import quad
import itertools

defaultConstants = {}
defaultConstants['TCMB'] = 2.7255
defaultConstants['G_CGS'] = 6.67259e-08
defaultConstants['MSUN_CGS'] = 1.98900e+33
defaultConstants['MPC2CM'] = 3.085678e+24
defaultConstants['ERRTOL'] = 1e-12


defaultCosmology = {}
defaultCosmology['omch2'] = 0.12470
defaultCosmology['ombh2'] = 0.02230
defaultCosmology['H0'] = 67.0
defaultCosmology['ns'] = 0.96
defaultCosmology['As'] = 2.2e-9
defaultCosmology['mnu'] = 0.0
defaultCosmology['w0'] = -1.0



class Cosmology(object):
    '''
    A wrapper around CAMB that tries to pre-calculate as much as possible
    Intended to be inherited by other classes like LimberCosmology and 
    ClusterCosmology
    '''
    def __init__(self,paramDict=defaultCosmology,constDict=defaultConstants,lmax=2000,clTTFixFile=None,skipCls=False,pickling=False,fill_zero=True,dimensionless=True):

        cosmo = paramDict
        self.paramDict = paramDict
        c = constDict
        self.c = c
        self.cosmo = paramDict


        self.c['TCMBmuK'] = self.c['TCMB'] * 1.0e6
            
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
        self.pars = camb.CAMBparams()
        self.pars.set_cosmology(H0=self.H0, ombh2=self.ombh2, omch2=self.omch2, mnu=self.mnu, tau=self.tau) # add tau
        self.pars.set_dark_energy(w=self.w0)
        self.pars.InitPower.set_params(ns=cosmo['ns'],As=cosmo['As'])

        self.results= camb.get_background(self.pars)
        self.omnuh2 = self.pars.omegan * ((self.H0 / 100.0) ** 2.)
        

        # self.rho_crit0 = 3. / (8. * pi) * (self.h*100 * 1.e5)**2. / c['G_CGS'] * c['MPC2CM'] / c['MSUN_CGS']
        self.rho_crit0H100 = 3. / (8. * pi) * (100 * 1.e5)**2. / c['G_CGS'] * c['MPC2CM'] / c['MSUN_CGS']
        self.cmbZ = 1100.
        self.lmax = lmax

        if (clTTFixFile is not None) and not(skipCls):
            import numpy as np
            ells,cltts = np.loadtxt(clTTFixFile,unpack=True)
            from scipy.interpolate import interp1d
            self.clttfunc = interp1d(ells,cltts,bounds_error=False,fill_value=0.)

        elif not(skipCls):
            print("Generating theory Cls...")
            self.pars.set_accuracy(AccuracyBoost=2.0, lSampleBoost=4.0, lAccuracyBoost=4.0)
            self.pars.set_for_lmax(lmax=(lmax+500), lens_potential_accuracy=3, max_eta_k=2*(lmax+500))
            theory = loadTheorySpectraFromPycambResults(self.results,self.pars,lmax,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=lmax,pickling=pickling,fill_zero=fill_zero,get_dimensionless=dimensionless)
            self.clttfunc = lambda ell: theory.lCl('TT',ell)
            self.theory = theory

            # import numpy as np
            # ells = np.arange(2,lmax,1)
            # cltts = self.clttfunc(ells)
            # np.savetxt("data/cltt_lensed_Feb18.txt",np.vstack((ells,cltts)).transpose())

            


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
    def __init__(self,paramDict=defaultCosmology,constDict=defaultConstants,lmax=2000,clTTFixFile=None,skipCls=False,pickling=False,numz=100,kmax=42.47,nonlinear=True,skipPower=False,fill_zero=True,pkgrid_override=None):
        Cosmology.__init__(self,paramDict,constDict,lmax,clTTFixFile,skipCls,pickling,fill_zero)

        
        self.chistar = self.results.conformal_time(0)- model.tau_maxvis.value
        self.zstar = self.results.redshift_at_comoving_radial_distance(self.chistar)

        self.kmax = kmax
        if nonlinear:
            self.pars.NonLinear = model.NonLinear_both
        else:
            self.pars.NonLinear = model.NonLinear_none
        self.chis = np.linspace(0,self.chistar,numz)
        self.zs=self.results.redshift_at_comoving_radial_distance(self.chis)
        self.dchis = (self.chis[2:]-self.chis[:-2])/2
        self.chis = self.chis[1:-1]
        self.zs = self.zs[1:-1]
        self.Hzs = np.array([self.results.hubble_parameter(z) for z in self.zs])
        self._cSpeedKmPerSec = 299792.458
        self.kernels = {}
        self._initWkappaCMB()

        self.nonlinear = nonlinear
        self.skipPower = skipPower

        if not(skipPower): self._initPower(pkgrid_override)


    def _initPower(self,pkgrid_override=None):
        print("initializing power...")
        if pkgrid_override is None:
            self.PK = camb.get_matter_power_interpolator(self.pars, nonlinear=self.nonlinear,hubble_units=False, k_hunit=False, kmax=self.kmax, zmax=self.zs[-1])
        else:
            class Ptemp:
                def __init__(self,pkgrid):
                    self.pk = pkgrid
                def P(self,zs,ks,grid=True):
                    ks = np.asarray(ks)
                    zs = np.asarray(zs)                            
                    return self.pk(ks,zs,grid=grid).T
            self.PK = Ptemp(pkgrid_override)
            
        self.precalcFactor = self.Hzs**2. /self.chis/self.chis/self._cSpeedKmPerSec**2.


        


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
    
    def addDeltaNz(self,tag,zsource,bias=None,magbias=None):

        assert not(tag in list(self.kernels.keys())), "Tag already exists."
        assert tag!="cmb", "cmb is a tag reserved for cosmic microwave background. Use a different tag."
        
        
        self.kernels[tag] = {}
        self.kernels[tag]['dndz'] = "delta" 
        self.kernels[tag]['zdelta'] = zsource

        self._generateWindow(tag,bias,magbias,numzIntegral=None)
          
            
    def addStepNz(self,tag,zmin,zmax,bias=None,magbias=None,numzIntegral=300):
        assert not(tag in list(self.kernels.keys())), "Tag already exists."
        assert tag!="cmb", "cmb is a tag reserved for cosmic microwave background. Use a different tag."
        
        self.kernels[tag] = {}
        self.kernels[tag]['zmin'] = zmin
        self.kernels[tag]['zmax'] = zmax
        normStep = (self.kernels[tag]['zmax']-self.kernels[tag]['zmin'])
        self.kernels[tag]['dndz'] = lambda z: 1./normStep
        
        self._generateWindow(tag,bias,magbias,numzIntegral)
        
    def addNz(self,tag,zedges,nz,bias=None,magbias=None,numzIntegral=300):

        '''
        Assumes equally spaced bins
        If bias, then assumes counts, else assumes lensing
        If magbias provided, applies it as magnification bias assuming it is 's' in Eq 7 of Omuri Holder. Bias must be provided too.
        '''

        assert not(tag in list(self.kernels.keys())), "Tag already exists."
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
                
                


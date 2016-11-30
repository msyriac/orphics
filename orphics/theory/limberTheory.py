import itertools
import camb
from camb import model, initialpower

import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import quad

# Partially based on Anthony Lewis' CAMB Python Notebook
# To do:
# - Add support for curvature
# - Test magnification bias for counts
# - Test that delta function and step function dndz(z)s are equivalent as step function width -> 0


class XCorrIntegrator(object):



    def __init__(self,cosmo,numz=100,kmax=42.47,nonlinear=True):

        self.H0 = cosmo['H0']
        self.omch2 = cosmo['omch2']
        self.ombh2 = cosmo['ombh2']
        
        self.pars = camb.CAMBparams()
        self.pars.set_cosmology(H0=self.H0, ombh2=cosmo['ombh2'], omch2=self.omch2)
        self.pars.InitPower.set_params(ns=cosmo['ns'],As=cosmo['As'])

        self.results= camb.get_background(self.pars)
        self.omnuh2 = self.pars.omegan * ((self.H0 / 100.0) ** 2.)
        
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

        
        print "initializing power..."
        self.PK = camb.get_matter_power_interpolator(self.pars, nonlinear=nonlinear,hubble_units=False, k_hunit=False, kmax=self.kmax, zmax=self.zs[-1])
        self.precalcFactor = self.Hzs**2. /self.chis/self.chis/self._cSpeedKmPerSec**2.


        


    def generateCls(self,ellrange,autoOnly=False):



        w = np.ones(self.chis.shape)

        retList = {}
        if autoOnly:
            listKeys = zip(self.kernels.keys(),self.kernels.keys())
        else:
            listKeys = list(itertools.combinations_with_replacement(self.kernels.keys(),2))
            
        for key1,key2 in listKeys:
            retList[key1+","+key2] = []
        for ell in ellrange:
            k=(ell+0.5)/self.chis
            w[:]=1
            w[k<1e-4]=0
            w[k>=self.kmax]=0
            pkin = self.PK.P(self.zs, k, grid=False)
            common = (w*pkin)*self.precalcFactor
    
            for key1,key2 in listKeys:
                estCl = np.dot(self.dchis, common*self.kernels[key1]['W']*self.kernels[key2]['W'])
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
            print "Key combination not found"
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

        assert not(tag in self.kernels.keys()), "Tag already exists."
        assert tag!="cmb", "cmb is a tag reserved for cosmic microwave background. Use a different tag."
        
        
        self.kernels[tag] = {}
        self.kernels[tag]['dndz'] = "delta" 
        self.kernels[tag]['zdelta'] = zsource

        self._generateWindow(tag,bias,magbias,numzIntegral=None)
          
            
    def addStepNz(self,tag,zmin,zmax,bias=None,magbias=None,numzIntegral=300):
        assert not(tag in self.kernels.keys()), "Tag already exists."
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

        assert not(tag in self.kernels.keys()), "Tag already exists."
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
        print "Initializing galaxy window for ", tag , " ..."
        if bias==None:

            retvals = self._lensWindow(self.kernels[tag],numzIntegral)

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
                print "Lensing bias max percent correction in counts ", np.max(np.abs(np.nan_to_num(magcorrection *100./ self.kernels[tag]['W'])))


            
        


    def _initWkappaCMB(self):#,numz):

        print "Initializing CMB window.."
        chirange = self.chis
        
        iwcmb =  1.5*(self.omch2+self.ombh2+self.omnuh2)*100.*100.*(1.+self.zs)*self.chis*((self.chistar - self.chis)/self.chistar)/self.Hzs/self._cSpeedKmPerSec
        self.kernels['cmb']={}
        self.kernels['cmb']['W'] = iwcmb
                
                


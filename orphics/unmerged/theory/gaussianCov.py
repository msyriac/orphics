from orphics.tools.cmb import validateMapType
import numpy as np
from orphics.tools.cmb import TheorySpectra
from scipy.interpolate import interp1d
        

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
        self.Nls['gg'] = lambda x: 1./(self.ngalForeground*1.18e7)
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
           

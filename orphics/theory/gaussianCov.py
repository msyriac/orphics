from orphics.tools.cmb import validateMapType
import numpy as np

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

        self._lowWarn = False
        self._highWarn = False
        self._negWarn=False
        self._ellMaxes = {}

    def loadGenericCls(self,ells,Cls,keyName):
        self._gCl[keyName] = interp1d(ells,Cls,bounds_error=False,fill_value=0.)
        

    def gCl(self,keyName,ell):
        try:
            return self._gCl[keyName](ell)
        except:
            return self._gCl[keyName[::-1]](ell)
        
    def loadCls(self,ell,Cl,XYType="TT",lensed=False,interporder="linear",lpad=9000):
        from scipy.interpolate import interp1d

        # Implement ellnorm

        mapXYType = XYType.upper()
        validateMapType(mapXYType)


            
        #print bcolors.OKBLUE+"Interpolating", XYType, "spectrum to", interporder, "order..."+bcolors.ENDC
        f=interp1d(ell,Cl,kind=interporder,bounds_error=False,fill_value=0.)
        if lensed:
            self._lCl[XYType]=f
            self._ellMaxes['l'+XYType]=lpad
        else:
            self._uCl[XYType]=f
            self._ellMaxes['u'+XYType]=lpad

    def _Cl(self,XYType,ell,lensed=False):

            
        mapXYType = XYType.upper()
        validateMapType(mapXYType)

        if mapXYType=="ET": mapXYType="TE"
        ell = np.array(ell)

        if lensed:    
            retlist = np.array(self._lCl[mapXYType](ell))
            retlist[ell>self._ellMaxes['l'+mapXYType]]=0.
            return retlist
        else:
            retlist = np.array(self._uCl[mapXYType](ell))
            retlist[ell>self._ellMaxes['u'+mapXYType]]=0.
            return retlist

    def uCl(self,XYType,ell):
        return self._Cl(XYType,ell,lensed=False)
    def lCl(self,XYType,ell):
        return self._Cl(XYType,ell,lensed=True)
    

        

class LensForecast:

    def __init__(self):
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

        self.theory = TheorySpectra()
        self.Nls = {}
        

    def loadKKDirect(self,ellsCls,Cls,ellsNls,Nls):
        self.Nls['kk'] = interp1d(ellsNls,Nls,bounds_error=False,fill_value=np.inf)
        self.theory.loadGenericCls(ellsCls,Cls,'kk')
    
        self._haveKK = True
        

    def loadGGDirect(self,ellsCls,Cls,ngal):
        self.ngalForeground = ngal
        self.Nls['gg'] = lambda x: 1./(self.ngalForeground*1.18e7)
        self.theory.loadGenericCls(ellsCls,Cls,'gg')
    
        self._haveGG = True
        
        

    def loadSSDirect(self,ellsCls,Cls,ngal,shapeNoise=0.3):
        if shapeNoise==None or shapeNoise<1.e-9:
            print "No/negligible shape noise given. Using default = 0.3."
            self.shapeNoise=0.3

        else:             
            self.shapeNoise = shapeNoise
        self.ngalBackground = ngal
        self.Nls['ss'] = lambda x: self.shapeNoise*self.shapeNoise/(2.*self.ngalBackground*1.18e7)


        self.theory.loadGenericCls(ellsCls,Cls,'ss')
        
        self._haveSS = True


    def loadSGDirect(self,ellsCls,Cls):
        self.theory.loadGenericCls(ellsCls,Cls,'sg')
        
        self._haveSG = True


    def loadKGDirect(self,ellsCls,Cls):
        self.theory.loadGenericCls(ellsCls,Cls,'kg')
        self._haveKG = True
                

    def loadKSDirect(self,ellsCls,Cls):
        self.theory.loadGenericCls(ellsCls,Cls,'ks')

        self._haveKS = True

    def KnoxCov(self,specTypeXY,specTypeWZ,ellBinEdges,fsky):
        '''
        returns cov(Cl_XY,Cl_WZ),signalToNoise(Cl_XY)^2, signalToNoise(Cl_WZ)^2
        '''
        def ClTot(spec,ell):
            a,b = spec
            if a==b:
                Noise = self.Nls[spec](ell)
            else:
                Noise = 0.
            return self.theory.gCl(spec,ell)+Noise        
        
        X, Y = specTypeXY
        W, Z = specTypeWZ

        ellMids  =  (ellBinEdges[1:] + ellBinEdges[:-1]) / 2
        ellWidths = np.diff(ellBinEdges)

        covs = []
        sigs1 = []
        sigs2 = []
        for ellMid,ellWidth in zip(ellMids,ellWidths):
            ClSum = ClTot(X+W,ellMid)*ClTot(Y+Z,ellMid)+ClTot(X+Z,ellMid)*ClTot(Y+W,ellMid)
            var = ClSum/(2.*ellMid+1.)/ellWidth/fsky
            covs.append(var)
            sigs1.append(self.theory.gCl(specTypeXY,ellMid)**2./var)
            sigs2.append(self.theory.gCl(specTypeWZ,ellMid)**2./var)

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
           

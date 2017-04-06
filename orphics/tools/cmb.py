import re
from orphics.tools.io import bcolors
import numpy as np
from scipy.interpolate import interp1d
import time
import cPickle as pickle


def noise_func(ell,fwhm,rms_noise,lknee=0.,alpha=0.):
    '''Beam deconvolved noise in whatever units rms_noise is in.
    e.g. If rms_noise is in uK-arcmin, returns noise in uK**2.    
    '''
    if lknee>1.e-3:
        atmFactor = (lknee/ell)**(-alpha)
    else:
        atmFactor = 0.
    rms = rms_noise * (1./60.)*(np.pi/180.)
    tht_fwhm = np.deg2rad(fwhm / 60.)
    ans = (atmFactor+1.) * (rms**2.) * np.exp((tht_fwhm**2.)*(ell**2.) / (8.*np.log(2.)))
    return ans


def pad_1d_power(ell,Cl,ellmax):
    if ell[-1]<ellmax:
        appendArr = np.arange(ell[-1]+1,ellmax,1)
        ell = np.append(np.asarray(ell),appendArr)
        Cl = np.append(np.asarray(Cl),np.asarray([0.]*len(appendArr)))
    return ell,Cl

def get_noise_func(beamArcmin,noiseMukArcmin,ellmin=-np.inf,ellmax=np.inf,TCMB=2.7255e6):
    Sigma = beamArcmin *np.pi/60./180./ np.sqrt(8.*np.log(2.))  # radians
    noiseWhite = (np.pi / (180. * 60))**2.  * noiseMukArcmin**2. / TCMB**2.
    return lambda x: np.piecewise(np.asarray(x).astype(float), [np.asarray(x)<ellmin,np.logical_and(np.asarray(x)>=ellmin,np.asarray(x)<=ellmax),np.asarray(x)>ellmax], [lambda y: np.inf, lambda y: noiseWhite*np.exp((np.asarray(y)**2.)*Sigma*Sigma), lambda y: np.inf])

def total_1d_power(ell,Cl,ellmax,beamArcmin,noiseMukArcmin,TCMB=2.7255e6,deconvolve=False):
    if ell[-1]<ellmax:
        appendArr = np.arange(ell[-1]+1,ellmax,1)
        ell = np.append(np.asarray(ell),appendArr)
        Cl = np.append(np.asarray(Cl),np.asarray([0.]*len(appendArr)))
    Sigma = beamArcmin *np.pi/60./180./ np.sqrt(8.*np.log(2.))  # radians
    beamFac = np.exp(-(ell**2.)*Sigma*Sigma)
    Cl *= beamFac
    noiseWhite = (np.pi / (180. * 60))**2.  * noiseMukArcmin**2. / TCMB**2.  
    Cl += noiseWhite
    if deconvolve: Cl /= beamFac
    return ell, Cl

def loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000):
    '''
    Given a CAMB path+output_root, reads CMB and lensing Cls into 
    an orphics.theory.gaussianCov.TheorySpectra object.

    The spectra are stored in dimensionless form, so TCMB has to be specified. They should 
    be used with dimensionless noise spectra and dimensionless maps.

    All ell and 2pi factors are also stripped off.

 
    '''
    
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

    ell, cldd = np.loadtxt(cambRoot+"_lenspotentialCls.dat",unpack=True,usecols=[0,5])
    clkk = 2.*np.pi*cldd/4. #/ell/(ell+1.)
    theory.loadGenericCls(ell,clkk,"kk",lpad=lpad)


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

        theory.loadCls(ell,cltt,'TT',lensed=False,interporder="linear",lpad=9000)
        theory.loadCls(ell,clte,'TE',lensed=False,interporder="linear",lpad=9000)
        theory.loadCls(ell,clee,'EE',lensed=False,interporder="linear",lpad=9000)
        theory.loadCls(ell,clbb,'BB',lensed=False,interporder="linear",lpad=9000)


    return theory


def loadTheorySpectraFromPycambResults(results,pars,kellmax,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000,pickling=False):
    '''

    The spectra are stored in dimensionless form, so TCMB has to be specified. They should 
    be used with dimensionless noise spectra and dimensionless maps.

    All ell and 2pi factors are also stripped off.

 
    '''
    
    if useTotal:
        uSuffix = "unlensed_total"
        lSuffix = "total"
    else:
        uSuffix = "unlensed_scalar"
        lSuffix = "lensed_scalar"

    try:
        assert pickling
        clfile = "output/clsAll_"+time.strftime('%Y%m%d') +".pkl"
        cmbmat = pickle.load(open(clfile,'rb'))
        print "Loaded cached Cls from ", clfile
    except:
        cmbmat = results.get_cmb_power_spectra(pars)
        if pickling: pickle.dump(cmbmat,open("output/clsAll_"+time.strftime('%Y%m%d') +".pkl",'wb'))

    theory = TheorySpectra()
    for i,pol in enumerate(['TT','EE','BB','TE']):
        cls =cmbmat[lSuffix][2:,i]

        ells = np.arange(2,len(cls)+2,1)
        cls *= 2.*np.pi/ells/(ells+1.)
        theory.loadCls(ells,cls,pol,lensed=True,interporder="linear",lpad=lpad)

        if unlensedEqualsLensed:
            theory.loadCls(ells,cls,pol,lensed=False,interporder="linear",lpad=lpad)            
        else:
            cls = cmbmat[uSuffix][2:,i]
            ells = np.arange(2,len(cls)+2,1)
            cls *= 2.*np.pi/ells/(ells+1.)
            theory.loadCls(ells,cls,pol,lensed=False,interporder="linear",lpad=lpad)

    try:
        assert pickling
        clfile = "output/clphi_"+time.strftime('%Y%m%d') +".txt"
        clphi = np.loadtxt(clfile)
        print "Loaded cached Cls from ", clfile
    except:
        lensArr = results.get_lens_potential_cls(lmax=kellmax)
        clphi = lensArr[2:,0]
        if pickling: np.savetxt("output/clphi_"+time.strftime('%Y%m%d') +".txt",clphi)

    clkk = clphi* (2.*np.pi/4.)
    ells = np.arange(2,len(clkk)+2,1)
    theory.loadGenericCls(ells,clkk,"kk",lpad=lpad)


    return theory


def validateMapType(mapXYType):
    assert not(re.search('[^TEB]', mapXYType)) and (len(mapXYType)==2), \
      bcolors.FAIL+"\""+mapXYType+"\" is an invalid map type. XY must be a two" + \
      " letter combination of T, E and B. e.g TT or TE."+bcolors.ENDC



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


    def loadGenericCls(self,ells,Cls,keyName,lpad=9000):
        self._gCl[keyName] = interp1d(ells[ells<lpad],Cls[ells<lpad],bounds_error=False,fill_value=0.)
        

    def gCl(self,keyName,ell):
        try:
            return self._gCl[keyName](ell)
        except:
            return self._gCl[keyName[::-1]](ell)
        
    def loadCls(self,ell,Cl,XYType="TT",lensed=False,interporder="linear",lpad=9000):

        # Implement ellnorm

        mapXYType = XYType.upper()
        validateMapType(mapXYType)


            
        #print bcolors.OKBLUE+"Interpolating", XYType, "spectrum to", interporder, "order..."+bcolors.ENDC
        f=interp1d(ell[ell<lpad],Cl[ell<lpad],kind=interporder,bounds_error=False,fill_value=0.)
        if lensed:
            self._lCl[XYType]=f
        else:
            self._uCl[XYType]=f

    def _Cl(self,XYType,ell,lensed=False):

            
        mapXYType = XYType.upper()
        validateMapType(mapXYType)

        if mapXYType=="ET": mapXYType="TE"
        ell = np.array(ell)

        if lensed:    
            retlist = np.array(self._lCl[mapXYType](ell))
            return retlist
        else:
            retlist = np.array(self._uCl[mapXYType](ell))
            return retlist

    def uCl(self,XYType,ell):
        return self._Cl(XYType,ell,lensed=False)
    def lCl(self,XYType,ell):
        return self._Cl(XYType,ell,lensed=True)
    
    def __getstate__(self):
        # Clkk2d is not pickled yet!!!
        return self.verbose, self.lxMap,self.lyMap,self.modLMap,self.thetaMap,self.lx,self.ly, self.lxHatMap, self.lyHatMap,self.uClNow2d, self.uClFid2d, self.lClFid2d, self.noiseXX2d, self.noiseYY2d, self.fMaskXX, self.fMaskYY, self.lmax_T, self.lmax_P, self.defaultMaskT, self.defaultMaskP, self.bigell, self.gradCut,self.Nlkk,self.pixScaleX,self.pixScaleY



    def __setstate__(self, state):
        self.verbose, self.lxMap,self.lyMap,self.modLMap,self.thetaMap,self.lx,self.ly, self.lxHatMap, self.lyHatMap,self.uClNow2d, self.uClFid2d, self.lClFid2d, self.noiseXX2d, self.noiseYY2d, self.fMaskXX, self.fMaskYY, self.lmax_T, self.lmax_P, self.defaultMaskT, self.defaultMaskP, self.bigell, self.gradCut,self.Nlkk,self.pixScaleX,self.pixScaleY = state


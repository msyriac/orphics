from __future__ import print_function
import re
from orphics.io import bcolors
import numpy as np
from scipy.interpolate import interp1d
import time
import pickle as pickle

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
    import orphics.tools.cmb as cmb
    dtheory = cmb.TheorySpectra()
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

def enmap_power_from_orphics_theory(theory,lmax,lensed=False,dimensionless=True,orphics_dimensionless=True,TCMB=2.7255e6):
    if orphics_dimensionless and dimensionless: tmul = 1.
    if orphics_dimensionless and not(dimensionless): tmul = TCMB**2.
    if not(orphics_dimensionless) and not(dimensionless): tmul = 1.
    if not(orphics_dimensionless) and dimensionless: tmul = 1./TCMB**2.
    
    
    fine_ells = np.arange(0,lmax,1)
    cltt, clee, clte, clbb = unpack_cmb_theory(theory,fine_ells,lensed=lensed)
    ps = np.zeros((3,3,fine_ells.size))
    ps[0,0] = cltt
    ps[1,1] = clee
    ps[0,1] = clte
    ps[1,0] = clte
    ps[2,2] = clbb

    return ps*tmul


def fit_noise_power(ells,nls,ell_fit=5000.,lknee_guess=2000.,alpha_guess=-4.0):
    ''' Fit beam-convolved (i.e. does not know about beam) noise power (uK^2 units) to
    an atmosphere+white noise model parameterized by rms_noise, lknee, alpha

    ell_fit is the ell above which an average of the nls is taken to estimate
    the rms white noise
    '''
    from scipy.optimize import curve_fit as cfit
    
    noise_guess = np.sqrt(np.nanmean(nls[ells>ell_fit]))*(180.*60./np.pi)
    nlfitfunc = lambda ell,l,a: noise_func(ell,0.,noise_guess,l,a,dimensionless=False)
    popt,pcov = cfit(nlfitfunc,ells,nls,p0=[lknee_guess,alpha_guess])
    lknee_fit,alpha_fit = popt
    return noise_guess,lknee_fit,alpha_fit



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


def gauss_beam(ell,fwhm):
    tht_fwhm = np.deg2rad(fwhm / 60.)
    return np.exp(-(tht_fwhm**2.)*(ell**2.) / (16.*np.log(2.)))
    
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
        atmFactor = (lknee/ell)**(-alpha)
    else:
        atmFactor = 0.
    return atmFactor

def pad_1d_power(ell,Cl,ellmax):
    if ell[-1]<ellmax:
        appendArr = np.arange(ell[-1]+1,ellmax,1)
        ell = np.append(np.asarray(ell),appendArr)
        Cl = np.append(np.asarray(Cl),np.asarray([0.]*len(appendArr)))
    return ell,Cl


def white_noise_with_atm_func(ell,uk_arcmin,lknee,alpha,dimensionless,TCMB=2.7255e6):
    atmFactor = atm_factor(ell,lknee,alpha)
    noiseWhite = ell*0.+(uk_arcmin*np.pi / (180. * 60))**2.  
    dfact = (1./TCMB**2.) if dimensionless else 1.
    return (atmFactor+1.)*noiseWhite*dfact

def noise_pad_infinity(Nlfunc,ellmin,ellmax):
    return lambda x: np.piecewise(np.asarray(x).astype(float), [np.asarray(x)<ellmin,np.logical_and(np.asarray(x)>=ellmin,np.asarray(x)<=ellmax),np.asarray(x)>ellmax], [lambda y: np.inf, lambda y: Nlfunc(y), lambda y: np.inf])

def fwhm_arcmin_to_sigma_radians(fwhm):
    return fwhm *np.pi/60./180./ np.sqrt(8.*np.log(2.))  # radians

def get_noise_func(beamArcmin,noiseMukArcmin,dimensionless,ellmin=-np.inf,ellmax=np.inf,TCMB=2.7255e6,lknee=0.,alpha=0.):
    Sigma = fwhm_arcmin_to_sigma_radians(beamArcmin)
    Nlfunc = lambda y: white_noise_with_atm(y,noiseMukArcmin,lknee,alpha,dimensionless=dimensionless,TCMB=TCMB)
    #lambda x: np.piecewise(np.asarray(x).astype(float), [np.asarray(x)<ellmin,np.logical_and(np.asarray(x)>=ellmin,np.asarray(x)<=ellmax),np.asarray(x)>ellmax], [lambda y: np.inf, lambda y: white_noise_with_atm(y,noiseMukArcmin,lknee,alpha,dimensionless=dimensionless,TCMB=TCMB)*np.exp((np.asarray(y)**2.)*Sigma*Sigma), lambda y: np.inf])
    return noise_pad_infinity(Nlfunc,ellmin,ellmax)

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


def load_theory_spectra_from_enlib(file_root,TCMB = 2.7255e6,lpad=9000,get_dimensionless=True):
    if get_dimensionless:
        tmul = TCMB**2.
    else:
        tmul = 1.

    theory = TheorySpectra()
    uFile = file_root+"_lensinput.dat"
    lFile = file_root+"_lensed.dat"

    ell, ucltt, uclee, uclbb, uclte, cldd = np.loadtxt(uFile,unpack=True,usecols=[0,1,2,3,4,5])
    mult = 2.*np.pi/ell/(ell+1.)/tmul
    ucltt *= mult
    uclee *= mult
    uclte *= mult
    uclbb *= mult
    clkk = 2.*np.pi*cldd/4.
    theory.loadCls(ell,ucltt,'TT',lensed=False,interporder="linear",lpad=lpad)
    theory.loadCls(ell,uclte,'TE',lensed=False,interporder="linear",lpad=lpad)
    theory.loadCls(ell,uclee,'EE',lensed=False,interporder="linear",lpad=lpad)
    theory.loadCls(ell,uclbb,'BB',lensed=False,interporder="linear",lpad=lpad)
    theory.loadGenericCls(ell,clkk,"kk",lpad=lpad)

    ell, lcltt, lclee, lclbb, lclte = np.loadtxt(lFile,unpack=True,usecols=[0,1,2,3,4])
    mult = 2.*np.pi/ell/(ell+1.)/tmul
    lcltt *= mult
    lclee *= mult
    lclte *= mult
    lclbb *= mult
    theory.loadCls(ell,lcltt,'TT',lensed=True,interporder="linear",lpad=lpad)
    theory.loadCls(ell,lclte,'TE',lensed=True,interporder="linear",lpad=lpad)
    theory.loadCls(ell,lclee,'EE',lensed=True,interporder="linear",lpad=lpad)
    theory.loadCls(ell,lclbb,'BB',lensed=True,interporder="linear",lpad=lpad)

    return theory
    

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


    return theory


def loadTheorySpectraFromPycambResults(results,pars,kellmax,unlensedEqualsLensed=False,useTotal=False,TCMB = 2.7255e6,lpad=9000,pickling=False,fill_zero=False,get_dimensionless=True):
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
        clfile = "output/clsAll_"+str(kellmax)+"_"+time.strftime('%Y%m%d') +".pkl"
        cmbmat = pickle.load(open(clfile,'rb'))
        print("Loaded cached Cls from ", clfile)
    except:
        cmbmat = results.get_cmb_power_spectra(pars)
        if pickling:
            import os
            directory = "output/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            pickle.dump(cmbmat,open("output/clsAll_"+str(kellmax)+"_"+time.strftime('%Y%m%d') +".pkl",'wb'))

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
        clfile = "output/clphi_"+str(kellmax)+"_"+time.strftime('%Y%m%d') +".txt"
        clphi = np.loadtxt(clfile)
        print("Loaded cached Cls from ", clfile)
    except:
        lensArr = results.get_lens_potential_cls(lmax=kellmax)
        clphi = lensArr[2:,0]
        if pickling:
            import os
            directory = "output/"
            if not os.path.exists(directory):
                os.makedirs(directory)
            np.savetxt("output/clphi_"+str(kellmax)+"_"+time.strftime('%Y%m%d') +".txt",clphi)

    clkk = clphi* (2.*np.pi/4.)
    ells = np.arange(2,len(clkk)+2,1)
    theory.loadGenericCls(ells,clkk,"kk",lpad=lpad,fill_zero=fill_zero)


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
                    
        # if not(fill_zero):
        #     fillval = Cl[ell<lpad][-1]
        # else:
        #     fillval = 0.
            
        # f=interp1d(ell[ell<lpad],Cl[ell<lpad],kind=interporder,bounds_error=False,fill_value=fillval)
        
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
    
    def __getstate__(self):
        # Clkk2d is not pickled yet!!!
        return self.verbose, self.lxMap,self.lyMap,self.modLMap,self.thetaMap,self.lx,self.ly, self.lxHatMap, self.lyHatMap,self.uClNow2d, self.uClFid2d, self.lClFid2d, self.noiseXX2d, self.noiseYY2d, self.fMaskXX, self.fMaskYY, self.lmax_T, self.lmax_P, self.defaultMaskT, self.defaultMaskP, self.bigell, self.gradCut,self.Nlkk,self.pixScaleX,self.pixScaleY



    def __setstate__(self, state):
        self.verbose, self.lxMap,self.lyMap,self.modLMap,self.thetaMap,self.lx,self.ly, self.lxHatMap, self.lyHatMap,self.uClNow2d, self.uClFid2d, self.lClFid2d, self.noiseXX2d, self.noiseYY2d, self.fMaskXX, self.fMaskYY, self.lmax_T, self.lmax_P, self.defaultMaskT, self.defaultMaskP, self.bigell, self.gradCut,self.Nlkk,self.pixScaleX,self.pixScaleY = state


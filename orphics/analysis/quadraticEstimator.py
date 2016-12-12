import numpy as np
from orphics.theory.quadEstTheory import QuadNorm
import orphics.analysis.flatMaps as fmaps 

'''
This module relies heavily on FFTs, so the keywords
fft2 and ifft2 are reserved for the chosen implementation.
'''
from scipy.fftpack import fft2,ifft2,fftshift,ifftshift,fftfreq
#from numpy.fft import fft2,ifft2,fftshift,ifftshift,fftfreq






class Estimator(object):
    '''
    Flat-sky lensing and Omega quadratic estimators
    Functionality includes:
    - small-scale lens estimation with gradient cutoff
    - combine maps from two different experiments


    NOTE: The TE estimator is not identical between large
    and small-scale estimators. Need to test this.
    '''


    def __init__(self,templateLiteMap,
                 theorySpectraForFilters,
                 theorySpectraForNorm=None,
                 noiseX2dTEB=[None,None,None],
                 noiseY2dTEB=[None,None,None],
                 fmaskX2dTEB=[None,None,None],
                 fmaskY2dTEB=[None,None,None],
                 fmaskKappa=None,
                 doCurl=False,
                 TOnly=False,
                 halo=False,
                 gradCut=None,
                 verbose=False):

        '''
        All the 2d fourier objects below are pre-fftshifting. They must be of the same dimension.

        templateLiteMap: any object that contains the attributes Nx, Ny, pixScaleX, pixScaleY specifying map dimensions
        theorySpectraForFilters: a orphics.theory.gaussianCov.TheorySpectra object with CMB Cls loaded
        theorySpectraForNorm=None: same as above but if you want to use a different cosmology in the expected value of the 2-pt
        noiseX2dTEB=[None,None,None]: a list of 2d arrays that corresponds to the noise power in T, E, B (same units as Cls above)
        noiseY2dTEB=[None,None,None]: the same as above but if you want to use a different experiment for the Y maps
        fmaskX2dTEB=[None,None,None]: a list of 2d integer arrays where 1 corresponds to modes included and 0 to those not included
        fmaskY2dTEB=[None,None,None]: same as above but for Y maps
        fmaskKappa=None: same as above but for output kappa map
        doCurl=False: return curl Omega estimates too? If yes, output of getKappa will be (kappa,curl)
        TOnly=False: do only TT? If yes, others will not be initialized and you'll get errors if you try to getKappa(XY) for XY!=TT
        halo=False: use the halo lensing estimators?
        gradCut=None: if using halo lensing estimators, specify an integer up to what L the X map will be retained
        verbose=False: print some occasional output?

        '''

        self.doCurl = doCurl
        self.halo = halo
        self.verbose = verbose

        # initialize norm and filters

        self.AL = {}
        if doCurl: self.OmAL = {}

        self.N = QuadNorm(templateLiteMap,gradCut=gradCut)
        if fmaskKappa is None:
            ellMinK = 80
            ellMaxK = 3000
            print "WARNING: using default kappa mask of 80 < L < 3000"
            self.fmaskK = fmaps.fourierMask(self.N.lx,self.N.ly,self.N.modLMap,lmin=ellMinK,lmax=ellMaxK)
        else:
            self.fmaskK = fmaskKappa

        if TOnly: 
            nList = ['TT']
            cmbList = ['TT']
            estList = ['TT']
        else:
            self.phaseY = np.cos(2.*self.N.thetaMap)+1.j*np.sin(2.*self.N.thetaMap)
            nList = ['TT','EE','BB']
            cmbList = ['TT','TE','EE','BB']
            estList = ['TT','TE','ET','EB','EE','TB']

        
        if self.verbose: print "Initializing filters and normalization for quadratic estimators..."
        for cmb in cmbList:
            uClFilt = theorySpectraForFilters.uCl(cmb,self.N.modLMap)

            if theorySpectraForNorm is not None:
                uClNorm = theorySpectraForNorm.uCl(cmb,self.N.modLMap)
            else:
                uClNorm = uClFilt
            lClFilt = theorySpectraForFilters.lCl(cmb,self.N.modLMap)
            self.N.addUnlensedFilter2DPower(cmb,uClFilt)
            self.N.addLensedFilter2DPower(cmb,lClFilt)
            self.N.addUnlensedNorm2DPower(cmb,uClNorm)
        for i,noise in enumerate(nList):
            self.N.addNoise2DPowerXX(noise,noiseX2dTEB[i],fmaskX2dTEB[i])
            self.N.addNoise2DPowerYY(noise,noiseY2dTEB[i],fmaskY2dTEB[i])



        for est in estList:
            self.AL[est] = self.N.getNlkk2d(est,halo=halo)
            if doCurl: self.OmAL[est] = self.N.getCurlNlkk2d(est,halo=halo)

        

    def updateTEB_X(self,T2DData,E2DData=None,B2DData=None,alreadyFTed=False):
        '''
        Masking and windowing and apodizing and beam deconvolution has to be done beforehand!

        Maps must have units corresponding to those of theory Cls and noise power
        '''
        self._hasX = True

        self.kGradx = {}
        self.kGrady = {}

        lx = self.N.lxMap
        ly = self.N.lyMap
        #ly = self.N.lxMap
        #lx = self.N.lyMap
        #lx = self.N.lx
        #ly = self.N.ly.reshape((self.N.template.Ny,1))
        
        if alreadyFTed:
            self.kT = T2DData
        else:
            self.kT = fft2(T2DData)
        self.kGradx['T'] = lx*self.kT.copy()*1j
        self.kGrady['T'] = ly*self.kT.copy()*1j

        if E2DData is not None:
            if alreadyFTed:
                self.kE = E2DData
            else:
                self.kE = fft2(E2DData)
            self.kGradx['E'] = 1.j*lx*self.kE.copy()
            self.kGrady['E'] = 1.j*ly*self.kE.copy()
        if B2DData is not None:
            if alreadyFTed:
                self.kB = B2DData
            else:
                self.kB = fft2(B2DData)
            self.kGradx['B'] = 1.j*lx*self.kB.copy()
            self.kGrady['B'] = 1.j*ly*self.kB.copy()
        
        

    def updateTEB_Y(self,T2DData=None,E2DData=None,B2DData=None,alreadyFTed=False):
        assert self._hasX, "Need to initialize gradient first."
        self._hasY = True
        
        self.kHigh = {}

        if T2DData is not None:
            if alreadyFTed:
                self.kHigh['T']=T2DData
            else:
                self.kHigh['T']=fft2(T2DData)
        else:
            self.kHigh['T']=self.kT.copy()
        if E2DData is not None:
            if alreadyFTed:
                self.kHigh['E']=E2DData
            else:
                self.kHigh['E']=fft2(E2DData)
        else:
            try:
                self.kHigh['E']=self.kE.copy()
            except:
                pass

        if B2DData is not None:
            if alreadyFTed:
                self.kHigh['B']=B2DData
            else:
                self.kHigh['B']=fft2(B2DData)
        else:
            try:
                self.kHigh['B']=self.kB.copy()
            except:
                pass

    def getKappa(self,XY):

        assert self._hasX and self._hasY
        assert XY in ['TT','TE','ET','EB','TB','EE']
        X,Y = XY

        WXY = self.N.WXY(XY)
        WY = self.N.WY(Y+Y)

        lx = self.N.lxMap
        ly = self.N.lyMap
        #ly = self.N.lxMap
        #lx = self.N.lyMap


        # import sys
        # from orphics.tools.output import Plotter
        # X = lx
        # pl = Plotter()
        # pl.plot2d(fftshift(X))
        # #pl.plot2d(np.log10(fftshift(X)))
        # pl.done("lx.png")
        # X = ly
        # pl = Plotter()
        # pl.plot2d(fftshift(X))
        # #pl.plot2d(np.log10(fftshift(X)))
        # pl.done("ly.png")
        # from orphics.tools.stats import binInAnnuli
        # bin_edges = np.arange(2,8000,10)
        # centers, Nlbinned = binInAnnuli(X, self.N.modLMap, bin_edges)
        # pl = Plotter()
        # pl.add(centers,Nlbinned)
        # pl.done("debug.png")
        #sys.exit()




        #lx = self.N.lx
        #ly = self.N.ly.reshape((self.N.template.Ny,1))

        if Y in ['E','B']:
            phaseY = self.phaseY
        else:
            phaseY = 1.


        fMask = self.fmaskK


        # HighMapStar = ifft2(self.kHigh[Y]*WY*fMask).real
        # kPx = fft2(ifft2(self.kGradx[X]*WXY).real*HighMapStar)
        # kPy = fft2(ifft2(self.kGrady[X]*WXY).real*HighMapStar)
        # rawKappa = ifft2(lx*kPx*fMask*1j + ly*kPy*fMask*1j).real
        # #rawKappa = ifft2(ly*kPx*fMask*1j + lx*kPy*fMask*1j).real
        # AL = self.AL[XY]*fMask
        # self.kappa = -ifft2(AL*fft2(rawKappa))


        HighMapStar = ifft2(self.kHigh[Y]*WY*phaseY*fMask).conjugate()
        kPx = fft2(ifft2(self.kGradx[X]*WXY*phaseY)*HighMapStar)
        kPy = fft2(ifft2(self.kGrady[X]*WXY*phaseY)*HighMapStar)
        rawKappa = ifft2(1.j*lx*kPx*fMask + 1.j*ly*kPy*fMask).real
        AL = self.AL[XY]*fMask
        self.kappa = -ifft2(AL*fft2(rawKappa))

        if self.doCurl:
            OmAL = self.OmAL[XY]*fMask
            rawCurl = ifft2(1.j*lx*kPy - 1.j*ly*kPx).real
            self.curl = -ifft2(OmAL*fft2(rawCurl))
            return self.kappa, self.curl


        # import sys
        # from orphics.tools.output import Plotter
        # X = WY
        # pl = Plotter()
        # #pl.plot2d(X)
        # pl.plot2d(np.log10(fftshift(X)))
        # pl.done("debug2d.png")
        # from orphics.tools.stats import binInAnnuli
        # bin_edges = np.arange(2,8000,10)
        # centers, Nlbinned = binInAnnuli(X, self.N.modLMap, bin_edges)
        # pl = Plotter()
        # pl.add(centers,Nlbinned)
        # pl.done("debug.png")
        # #sys.exit()


            
        return self.kappa






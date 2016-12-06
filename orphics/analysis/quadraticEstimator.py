import numpy as np
from ..theory.quadEstTheory import quadNorm

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


    def __init__(self,templateLiteMap,theorySpectraForFilters,theorySpectraForNorm=None,noiseX2d=None,noiseY2d=None,doCurl=False,TOnly=False):

        self.lx,self.ly,self.modLMap,self.thetaMap = getFTAttributesFromLiteMap(templateLiteMap)
        self.thetaMap *= np.pi/180.
        self.Nx = templateLiteMap.Nx
        self.Ny = templateLiteMap.Ny
        self.doCurl = doCurl

        # initialize norm and filters
        # For T
        
        # For P
        if not(TOnly):
            
            self.phaseY = np.cos(2.*self.thetaMap)+1.j*np.sin(2.*self.thetaMap)


    def updateTEB_X(self,T2DData,E2DData=None,B2DData=None):
        '''
        Masking and windowing and apodizing and beam deconvolution has to be done beforehand!
        '''
        self._hasX = True
        assert B2DData is None, "It's"

        self.kGradx = {}
        self.kGrady = {}
        
        self.kT = fft2(T2DData)
        self.kGradx['T'] = 1.j*self.lx*self.kT
        self.kGrady['T'] = 1.j*self.ly*self.kT

        if E2Data is not None:
            self.kE = fft2(E2DData)
            self.kGradx['E'] = 1.j*self.lx*self.kE
            self.kGrady['E'] = 1.j*self.ly*self.kE
        if B2Data is not None:
            self.kB = fft2(B2DData)
            self.kGradx['B'] = 1.j*self.lx*self.kB
            self.kGrady['B'] = 1.j*self.ly*self.kB
        
        

    def updateTEB_Y(self,T2DData=None,E2DData=None,B2DData=None):
        assert self._hasX, "Need to initialize gradient first."
        self._hasY = True
        
        self.kHigh = {}

        if T2DData is not None:
            self.kHigh['T']=fft2(TLiteMap)
        else:
            self.kHigh['T']=self.kT.copy()
        if E2DData is not None:
            self.kHigh['E']=fft2(ELiteMap)
        else:
            try:
                self.kHigh['E']=self.kE.copy()
            except:
                pass

        if B2DData is not None:
            self.kHigh['B']=fft2(BLiteMap)
        else:
            try:
                self.kHigh['B']=self.kB.copy()
            except:
                pass

    def getKappa(self,XY):

        assert self._hasX and self._hasY
        assert XY in ['TT','TE','ET','EB','TB','EE']
        X,Y = XY
        WXY = self.getGradFilter(XY)
        WY = self.getHighFilter(Y)

        if Y in ['E','B']:
            phaseY = self.phaseY
        else:
            phaseY = 1.

        HighMapStar = ifft2(self.kHigh[Y]*WY*phaseY).conjugate()
        kPx = fft2(ifft2(self.kGradx[X]*WXY*phaseY)*HighMapStar)
        kPy = fft2(ifft2(self.kGrady[X]*WXY*phaseY)*HighMapStar)
        
        rawKappa = ifft2(1.j*self.lx*kPx + 1.j*self.ly*kPy).real()
        self.kappa = -ifft2(self.norm[XY]*fft2(rawKappa))

        if self.doCurl:
            rawCurl = ifft2(1.j*self.lx*kPy - 1.j*self.ly*kPx).real()
            self.curl = -ifft2(self.normCurl[XY]*fft2(rawCurl))
            return self.kappa, self.curl
            
        return self.kappa





def getFTAttributesFromLiteMap(templateLM):
    '''
    Given a liteMap, return the fourier frequencies,
    magnitudes and phases.
    '''

        
    Nx = templateLM.Nx
    Ny = templateLM.Ny
    pixScaleX = templateLM.pixScaleX 
    pixScaleY = templateLM.pixScaleY
    
    
    lx =  2*np.pi  * fftfreq( Nx, d = pixScaleX )
    ly =  2*np.pi  * fftfreq( Ny, d = pixScaleY )
    
    ix = np.mod(np.arange(Nx*Ny),Nx)
    iy = np.arange(Nx*Ny)/Nx
    
    modLMap = np.zeros([Ny,Nx])
    modLMap[iy,ix] = np.sqrt(lx[ix]**2 + ly[iy]**2)
    
    thetaMap = np.zeros([Ny,Nx])
    thetaMap[iy[:],ix[:]] = np.arctan2(ly[iy[:]],lx[ix[:]])
    thetaMap *=180./np.pi

    return lx,ly,modLMap,thetaMap

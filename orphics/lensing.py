from __future__ import print_function
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from orphics import maps
from enlib import enmap

from scipy.integrate import simps
from scipy.interpolate import splrep,splev

from scipy.fftpack import fftshift,ifftshift,fftfreq
from scipy.interpolate import interp1d
from enlib.fft import fft,ifft

from orphics.stats import bin2D

import time
import cPickle as pickle


    

def lens_cov(ucov,alpha_pix,lens_order=5,kbeam=None,bshape=None):
    """Given the pix-pix covariance matrix for the unlensed CMB,
    returns the lensed covmat for a given pixel displacement model.

    ucov -- (Npix,Npix) array where Npix = Ny*Nx
    alpha_pix -- (2,Ny,Nx) array of lensing displacements in pixel units
    kbeam -- (Ny,Nx) array of 2d beam wavenumbers

    """
    from enlib import lensing as enlensing

    shape = alpha_pix.shape[-2:]
    Scov = ucov.copy()
    wcs = ucov.wcs
    for i in range(ucov.shape[0]):
        unlensed = enmap.enmap(Scov[i,:].copy().reshape(shape),wcs)
        lensed = enlensing.displace_map(unlensed, alpha_pix, order=lens_order)
        if kbeam is not None: lensed = maps.filter_map(lensed,kbeam)
        Scov[i,:] = lensed.ravel()
    for j in range(ucov.shape[1]):
        unlensed = enmap.enmap(Scov[:,j].copy().reshape(shape),wcs)
        lensed = enlensing.displace_map(unlensed, alpha_pix, order=lens_order)
        if kbeam is not None: lensed = maps.filter_map(lensed,kbeam)
        Scov[:,j] = lensed.ravel()

    if (bshape is not None) and (bshape!=shape):
        ny,nx = shape
        Scov = Scov.reshape((ny,nx,ny,nx))
        bny,bnx = bshape
        sy = int(ny/2.-bny/2.)
        ey = int(ny/2.+bny/2.)
        sx = int(nx/2.-bnx/2.)
        ex = int(nx/2.+bnx/2.)
        Scov = Scov[sy:ey,sx:ex,sy:ey,sx:ex].reshape((np.prod(bshape),np.prod(bshape)))
    return Scov




def beam_cov(ucov,kbeam):
    """Given the pix-pix covariance matrix for the lensed CMB,
    returns the beamed covmat. The beam can be a ratio of beams to
    readjust the beam in a given matrix.

    ucov -- (Npix,Npix) array where Npix = Ny*Nx
    kbeam -- (Ny,Nx) array of 2d beam wavenumbers

    """
    Scov = ucov.copy()
    shape = alpha_pix.shape[-2:]
    for i in range(Scov.shape[0]):
        lensed = Scov[i,:].copy().reshape(shape) 
        lensed = maps.filter_map(lensed,kbeam)
        Scov[i,:] = lensed.ravel()
    for j in range(Scov.shape[1]):
        lensed = Scov[:,j].copy().reshape(shape)
        lensed = maps.filter_map(lensed,kbeam)
        Scov[:,j] = lensed.ravel()
    return Scov


def qest(shape,wcs,theory,noise2d=None,beam2d=None,kmask=None,noise2d_P=0.,kmask_P=None,kmask_K=None,pol=False,grad_cut=None,unlensed_equals_lensed=False):
    if noise2d is None: noise2d = np.zeros(shape[-2:])
    if beam2d is None: beam2d = np.ones(shape[-2:])
    return Estimator(shape,wcs,
                     theory,
                     theorySpectraForNorm=theory,
                     noiseX2dTEB=[noise2d,noise2d_P,noise2d_P],
                     noiseY2dTEB=[noise2d,noise2d_P,noise2d_P],
                     noiseX_is_total = False,
                     noiseY_is_total = False,
                     fmaskX2dTEB=[kmask,kmask_P,kmask_P],
                     fmaskY2dTEB=[kmask,kmask_P,kmask_P],
                     fmaskKappa=kmask_K,
                     kBeamX = beam2d,
                     kBeamY = beam2d,
                     doCurl=False,
                     TOnly=not(pol),
                     halo=True,
                     gradCut=grad_cut,
                     verbose=False,
                     loadPickledNormAndFilters=None,
                     savePickledNormAndFilters=None,
                     uEqualsL=unlensed_equals_lensed,
                     bigell=9000,
                     mpi_comm=None,
                     lEqualsU=False)


def kappa_to_phi(kappa,modlmap,return_fphi=False):
    fphi = enmap.samewcs(kappa_to_fphi(kappa,modlmap),kappa)
    phi =  enmap.samewcs(ifft(fphi,axes=[-2,-1],normalize=True).real, kappa) 
    if return_fphi:
        return phi, fphi
    else:
        return phi

def kappa_to_fphi(kappa,modlmap):
    return fkappa_to_fphi(fft(kappa,axes=[-2,-1]),modlmap)

def fkappa_to_fphi(fkappa,modlmap):
    kmap = np.nan_to_num(2.*fkappa/modlmap/(modlmap+1.))
    kmap[modlmap<2.] = 0.
    return kmap



def fillLowEll(ells,cls,ellmin):
    # Fill low ells with the same value
    low_index = np.where(ells>ellmin)[0][0]
    lowest_ell = ells[low_index]
    lowest_val = cls[low_index]
    fill_ells = np.arange(2,lowest_ell,1)
    new_ells = np.append(fill_ells,ells[low_index:])
    fill_cls = np.array([lowest_val]*len(fill_ells))
    new_cls = np.append(fill_cls,cls[low_index:])

    return new_ells,new_cls


def sanitizePower(Nlbinned):
    Nlbinned[Nlbinned<0.] = np.nan

    # fill nans with interp
    ok = ~np.isnan(Nlbinned)
    xp = ok.ravel().nonzero()[0]
    fp = Nlbinned[~np.isnan(Nlbinned)]
    x  = np.isnan(Nlbinned).ravel().nonzero()[0]
    Nlbinned[np.isnan(Nlbinned)] = np.interp(x, xp, fp)
    return Nlbinned


def getMax(polComb,tellmax,pellmax):
    if polComb=='TT':
        return tellmax
    elif polComb in ['EE','EB']:
        return pellmax
    else:
        return max(tellmax,pellmax)


class QuadNorm(object):

    
    def __init__(self,shape,wcs,gradCut=None,verbose=False,bigell=9000,kBeamX=None,kBeamY=None,fmask=None):
        
        self.shape = shape
        self.wcs = wcs
        self.verbose = verbose
        self.Ny,self.Nx = shape[-2:]
        self.lxMap,self.lyMap,self.modLMap,thetaMap,lx,ly = maps.get_ft_attributes(shape,wcs)
        self.lxHatMap = self.lxMap*np.nan_to_num(1. / self.modLMap)
        self.lyHatMap = self.lyMap*np.nan_to_num(1. / self.modLMap)

        self.fmask = fmask

        if kBeamX is not None:           
            self.kBeamX = kBeamX
        else:
            self.kBeamX = 1.
            
        if kBeamY is not None:           
            self.kBeamY = kBeamY
        else:
            self.kBeamY = 1.


        self.uClNow2d = {}
        self.uClFid2d = {}
        self.lClFid2d = {}
        self.noiseXX2d = {}
        self.noiseYY2d = {}
        self.fMaskXX = {}
        self.fMaskYY = {}

        self.lmax_T=bigell
        self.lmax_P=bigell
        self.defaultMaskT = maps.mask_kspace(self.shape,self.wcs,lmin=2,lmax=self.lmax_T)
        self.defaultMaskP = maps.mask_kspace(self.shape,self.wcs,lmin=2,lmax=self.lmax_P)
        #del lx
        #del ly
        self.thetaMap = thetaMap
        self.lx = lx
        self.ly = ly
        
        self.bigell=bigell #9000.
        if gradCut is not None: 
            self.gradCut = gradCut
        else:
            self.gradCut = bigell
        


        self.Nlkk = {}
        self.pixScaleY,self.pixScaleX = enmap.pixshape(shape,wcs)
        self.noiseX_is_total = False
        self.noiseY_is_total = False
        


    def fmask_func(self,arr,mask):        
        arr[mask<1.e-3] = 0.
        return arr

    def addUnlensedFilter2DPower(self,XY,power2dData):
        '''
        XY = TT, TE, EE, EB or TB
        power2d is a flipper power2d object
        These Cls belong in the Wiener filters, and will not
        be perturbed if/when calculating derivatives.
        '''
        self.uClFid2d[XY] = power2dData.copy()+0.j
    def addUnlensedNorm2DPower(self,XY,power2dData):
        '''
        XY = TT, TE, EE, EB or TB
        power2d is a flipper power2d object
        These Cls belong in the CMB normalization, and will
        be perturbed if/when calculating derivatives.
        '''
        self.uClNow2d[XY] = power2dData.copy()+0.j
    def addLensedFilter2DPower(self,XY,power2dData):
        '''
        XY = TT, TE, EE, EB or TB
        power2d is a flipper power2d object
        These Cls belong in the Wiener filters, and will not
        be perturbed if/when calculating derivatives.
        '''
        self.lClFid2d[XY] = power2dData.copy()+0.j
    def addNoise2DPowerXX(self,XX,power2dData,fourierMask=None,is_total=False):
        '''
        Noise power for the X leg of the quadratic estimator
        XX = TT, EE, BB
        power2d is a flipper power2d object
        fourierMask is an int array that is 0 where noise is
        infinite and 1 where not. It should be in the same
        fftshift state as power2d.powerMap
        '''
        # check if fourier mask is int!
        self.noiseX_is_total = is_total
        self.noiseXX2d[XX] = power2dData.copy()+0.j
        if fourierMask is not None:
            self.noiseXX2d[XX][fourierMask==0] = np.inf
            self.fMaskXX[XX] = fourierMask
        else:
            if XX=='TT':
                self.noiseXX2d[XX][self.defaultMaskT==0] = np.inf
            else:
                self.noiseXX2d[XX][self.defaultMaskP==0] = np.inf

    def addNoise2DPowerYY(self,YY,power2dData,fourierMask=None,is_total=False):
        '''
        Noise power for the Y leg of the quadratic estimator
        XX = TT, EE, BB
        power2d is a flipper power2d object
        fourierMask is an int array that is 0 where noise is
        infinite and 1 where not. It should be in the same
        fftshift state as power2d.powerMap
        '''
        # check if fourier mask is int!
        self.noiseY_is_total = is_total
        self.noiseYY2d[YY] = power2dData.copy()+0.j
        if fourierMask is not None:
            self.noiseYY2d[YY][fourierMask==0] = np.inf
            self.fMaskYY[YY] = fourierMask
        else:
            if YY=='TT':
                self.noiseYY2d[YY][self.defaultMaskT==0] = np.inf
            else:
                self.noiseYY2d[YY][self.defaultMaskP==0] = np.inf
        
    def addClkk2DPower(self,power2dData):
        '''
        Fiducial Clkk power
        Used if delensing
        power2d is a flipper power2d object            
        '''
        self.clkk2d = power2dData.copy()+0.j
        self.clpp2d = 0.j+self.clkk2d.copy()*4./(self.modLMap**2.)/((self.modLMap+1.)**2.)


    def WXY(self,XY):
        X,Y = XY
        if Y=='B': Y='E'
        gradClXY = X+Y
        if XY=='ET': gradClXY = 'TE'

        totnoise = self.noiseXX2d[X+X].copy() if self.noiseX_is_total else (self.lClFid2d[X+X].copy()*self.kBeamX**2.+self.noiseXX2d[X+X].copy())
        W = self.fmask_func(np.nan_to_num(self.uClFid2d[gradClXY].copy()/totnoise)*self.kBeamX,self.fMaskXX[X+X])
        W[self.modLMap>self.gradCut]=0.
        if X=='T':
            W[np.where(self.modLMap >= self.lmax_T)] = 0.
        else:
            W[np.where(self.modLMap >= self.lmax_P)] = 0.


        # debug_edges = np.arange(400,6000,50)
        # import orphics.tools.stats as stats
        # import orphics.tools.io as io
        # binner = stats.bin2D(self.modLMap,debug_edges)
        # cents,ws = binner.bin(W)
        # pl = io.Plotter()
        # pl.add(cents,ws)
        # pl.done("ws.png")
        # sys.exit()
        
            
        return W
        

    def WY(self,YY):
        assert YY[0]==YY[1]
        totnoise = self.noiseYY2d[YY].copy() if self.noiseY_is_total else (self.lClFid2d[YY].copy()*self.kBeamY**2.+self.noiseYY2d[YY].copy())
        W = self.fmask_func(np.nan_to_num(1./totnoise)*self.kBeamY,self.fMaskYY[YY]) #* self.modLMap  # !!!!!
        W[np.where(self.modLMap >= self.lmax_T)] = 0.
        if YY[0]=='T':
            W[np.where(self.modLMap >= self.lmax_T)] = 0.
        else:
            W[np.where(self.modLMap >= self.lmax_P)] = 0.


        # debug_edges = np.arange(400,6000,50)
        # import orphics.tools.stats as stats
        # import orphics.tools.io as io
        # io.quickPlot2d(np.fft.fftshift(W.real),"wy2d.png")
        # binner = stats.bin2D(self.modLMap,debug_edges)
        # cents,ws = binner.bin(W.real)
        # print cents
        # print ws
        # pl = io.Plotter()#scaleY='log')
        # pl.add(cents,ws)
        # pl._ax.set_xlim(2,6000)
        # pl.done("wy.png")
        # sys.exit()
            
        return W

    def getCurlNlkk2d(self,XY,halo=False):
        raise NotImplementedError

    def super_dumb_N0_TTTT(self,data_power_2d_TT):
        ratio = np.nan_to_num(data_power_2d_TT*self.WY("TT")/self.kBeamY)
        lmap = self.modLMap
        replaced = np.nan_to_num(self.getNlkk2d("TT",halo=True,l1Scale=self.fmask_func(ratio,self.fMaskXX["TT"]),l2Scale=self.fmask_func(ratio,self.fMaskYY["TT"]),setNl=False) / (2. * np.nan_to_num(1. / lmap/(lmap+1.))))
        unreplaced = self.Nlkk["TT"].copy()
        return np.nan_to_num(unreplaced**2./replaced)
    
    def getNlkk2d(self,XY,halo=True,l1Scale=1.,l2Scale=1.,setNl=True):
        #if not(halo): raise NotImplementedError
        lx,ly = self.lxMap,self.lyMap
        lmap = self.modLMap

        X,Y = XY
        XX = X+X
        YY = Y+Y

        if self.verbose: 
            print(("Calculating norm for ", XY))

            
        h=0.

        allTerms = []
            
        if XY == 'TT':
            
            clunlenTTArrNow = self.uClNow2d['TT'].copy()
                

            if halo:
            
                WXY = self.WXY('TT')*self.kBeamX*l1Scale
                WY = self.WY('TT')*self.kBeamY*l2Scale


                
                preG = WY
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTTArrNow*WXY
                    preFX = ell1*WXY
                    preGX = ell2*clunlenTTArrNow*WY
                    

                    calc = ell1*ell2*fft(ifft(preF,axes=[-2,-1],normalize=True)*ifft(preG,axes=[-2,-1],normalize=True)+ifft(preFX,axes=[-2,-1],normalize=True)*ifft(preGX,axes=[-2,-1],normalize=True),axes=[-2,-1])
                    allTerms += [calc]
                    

            else:

                clunlenTTArr = self.uClFid2d['TT'].copy()

                preG = self.WY('TT') #np.nan_to_num(1./cltotTTArrY)
                cltotTTArrX = np.nan_to_num(clunlenTTArr/self.WXY('TT'))
                cltotTTArrY = np.nan_to_num(1./self.WY('TT'))

                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTTArrNow*clunlenTTArr*np.nan_to_num(1./cltotTTArrX)/2.            
                    preFX = ell1*clunlenTTArrNow*np.nan_to_num(1./cltotTTArrX)
                    preGX = ell2*clunlenTTArr*np.nan_to_num(1./cltotTTArrY)


                    
                    calc = 2.*ell1*ell2*fft(ifft(preF,axes=[-2,-1])*ifft(preG,axes=[-2,-1])+ifft(preFX,axes=[-2,-1])*ifft(preGX,axes=[-2,-1])/2.,axes=[-2,-1])
                    allTerms += [calc]
          

        elif XY == 'EE':

            clunlenEEArrNow = self.uClNow2d['EE'].copy()


            sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

            lx = self.lxMap
            ly = self.lyMap


            lxhat = self.lxHatMap
            lyhat = self.lyHatMap

            sinf = sin2phi(lxhat,lyhat)
            sinsqf = sinf**2.
            cosf = cos2phi(lxhat,lyhat)
            cossqf = cosf**2.
                                
            if halo:
            

                WXY = self.WXY('EE')*self.kBeamX
                WY = self.WY('EE')*self.kBeamY
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                        preF = trigfact*ell1*ell2*clunlenEEArrNow*WXY
                        preG = trigfact*WY
                        allTerms += [ell1*ell2*fft(ifft(preF,axes=[-2,-1],normalize=True)*ifft(preG,axes=[-2,-1],normalize=True),axes=[-2,-1])]
                        
                        #allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
                        
                        preFX = trigfact*ell1*clunlenEEArrNow*WY
                        preGX = trigfact*ell2*WXY

                        allTerms += [ell1*ell2*fft(ifft(preFX,axes=[-2,-1],normalize=True)*ifft(preGX,axes=[-2,-1],normalize=True),axes=[-2,-1])]
                        #allTerms += [ell1*ell2*fft2(ifft2(preFX)*ifft2(preGX))]

                
            # else:


            #     rfact = 2.**0.25
            #     for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
            #         for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
            #             preF = trigfact*ell1*ell2*clunlenEEArrNow*clunlenEEArr*np.nan_to_num(1./cltotEEArr)/2.
            #             preG = trigfact*np.nan_to_num(1./cltotEEArr)
            #             preFX = trigfact*ell1*clunlenEEArrNow*np.nan_to_num(1./cltotEEArr)
            #             preGX = trigfact*ell2*clunlenEEArr*np.nan_to_num(1./cltotEEArr)

            #             allTerms += [2.*ell1*ell2*fft2(ifft2(preF)*ifft2(preG)+ifft2(preFX)*ifft2(preGX)/2.)]


            


        elif XY == 'EB':


            clunlenEEArrNow = self.uClNow2d['EE'].copy()
            clunlenBBArrNow = self.uClNow2d['BB'].copy()


            sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

            lx = self.lxMap
            ly = self.lyMap

            termsF = []
            termsF.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )

            termsG = []
            termsG.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )

            lxhat = self.lxHatMap
            lyhat = self.lyHatMap

            WXY = self.WXY('EB')*self.kBeamX
            WY = self.WY('BB')*self.kBeamY
            for ellsq in [lx*lx,ly*ly,np.sqrt(2.)*lx*ly]:
                preF = ellsq*clunlenEEArrNow*WXY
                preG = WY

                for termF,termG in zip(termsF,termsG):
                    allTerms += [ellsq*fft(ifft(termF(preF,lxhat,lyhat),axes=[-2,-1],normalize=True)*ifft(termG(preG,lxhat,lyhat),axes=[-2,-1],normalize=True),axes=[-2,-1])]


        elif XY=='ET':

            clunlenTEArrNow = self.uClNow2d['TE'].copy()

            if halo:
                sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
                cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

                lx = self.lxMap
                ly = self.lyMap


                lxhat = self.lxHatMap
                lyhat = self.lyHatMap

                sinf = sin2phi(lxhat,lyhat)
                sinsqf = sinf**2.
                cosf = cos2phi(lxhat,lyhat)
                cossqf = cosf**2.


                WXY = self.WXY('ET')*self.kBeamX
                WY = self.WY('TT')*self.kBeamY

                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTEArrNow*WXY
                    preG = WY
                    allTerms += [ell1*ell2*fft(ifft(preF,axes=[-2,-1],normalize=True)*ifft(preG,axes=[-2,-1],normalize=True),axes=[-2,-1])]
                    for trigfact in [cosf,sinf]:

                        preFX = trigfact*ell1*clunlenTEArrNow*WY
                        preGX = trigfact*ell2*WXY

                        allTerms += [ell1*ell2*fft(ifft(preFX,axes=[-2,-1],normalize=True)*ifft(preGX,axes=[-2,-1],normalize=True),axes=[-2,-1])]


            # else:



            #     sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            #     cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

            #     lx = self.lxMap
            #     ly = self.lyMap

            
            #     lxhat = self.lxHatMap
            #     lyhat = self.lyHatMap

            #     sinf = sin2phi(lxhat,lyhat)
            #     sinsqf = sinf**2.
            #     cosf = cos2phi(lxhat,lyhat)
            #     cossqf = cosf**2.
                
                
            #     rfact = 2.**0.25
            #     for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
            #         preF = ell1*ell2*clunlenTEArrNow*clunlenTEArr*np.nan_to_num(1./cltotEEArr)
            #         preG = np.nan_to_num(1./cltotTTArr)
            #         allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
            #         for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
            #             preF = np.nan_to_num(1./cltotEEArr)
            #             preG = trigfact*ell1*ell2*clunlenTEArrNow*clunlenTEArr*np.nan_to_num(1./cltotTTArr)
            #             allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
            #         for trigfact in [cosf,sinf]:
                        
            #             preFX = trigfact*ell1*clunlenTEArrNow*np.nan_to_num(1./cltotEEArr)
            #             preGX = trigfact*ell2*clunlenTEArr*np.nan_to_num(1./cltotTTArr)

            #             allTerms += [2.*ell1*ell2*fft2(ifft2(preFX)*ifft2(preGX))]

                    

        elif XY=='TE':

            clunlenTEArrNow = self.uClNow2d['TE'].copy()

            if halo:
            
                sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
                cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

                lx = self.lxMap
                ly = self.lyMap

            
                lxhat = self.lxHatMap
                lyhat = self.lyHatMap

                sinf = sin2phi(lxhat,lyhat)
                sinsqf = sinf**2.
                cosf = cos2phi(lxhat,lyhat)
                cossqf = cosf**2.
                
                WXY = self.WXY('TE')*self.kBeamX
                WY = self.WY('EE')*self.kBeamY

                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                        preF = trigfact*ell1*ell2*clunlenTEArrNow*WXY
                        preG = trigfact*WY
                        allTerms += [ell1*ell2*fft(ifft(preF,axes=[-2,-1],normalize=True)*ifft(preG,axes=[-2,-1],normalize=True),axes=[-2,-1])]
                    for trigfact in [cosf,sinf]:
                        
                        preFX = trigfact*ell1*clunlenTEArrNow*WY
                        preGX = trigfact*ell2*WXY

                        allTerms += [ell1*ell2*fft(ifft(preFX,axes=[-2,-1],normalize=True)*ifft(preGX,axes=[-2,-1],normalize=True),axes=[-2,-1])]

                
            # else:



            #     sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            #     cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

            #     lx = self.lxMap
            #     ly = self.lyMap

            
            #     lxhat = self.lxHatMap
            #     lyhat = self.lyHatMap

            #     sinf = sin2phi(lxhat,lyhat)
            #     sinsqf = sinf**2.
            #     cosf = cos2phi(lxhat,lyhat)
            #     cossqf = cosf**2.
                
                
            #     rfact = 2.**0.25
            #     for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
            #         for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
            #             preF = trigfact*ell1*ell2*clunlenTEArrNow* self.WXY('TE')#clunlenTEArr*np.nan_to_num(1./cltotTTArr)
            #             preG = trigfact*self.WY('EE')#np.nan_to_num(1./cltotEEArr)
            #             allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
            #         preF = self.WY('TT')#np.nan_to_num(1./cltotTTArr)
            #         preG = ell1*ell2*clunlenTEArrNow* self.WXY('ET') #*clunlenTEArr*np.nan_to_num(1./cltotEEArr)
            #         allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
            #         for trigfact in [cosf,sinf]:
                        
            #             preFX = trigfact*ell1*clunlenTEArrNow*self.WY('TT')#np.nan_to_num(1./cltotTTArr)
            #             preGX = trigfact*ell2* self.WXY('ET')#*clunlenTEArr*np.nan_to_num(1./cltotEEArr)

            #             allTerms += [2.*ell1*ell2*fft2(ifft2(preFX)*ifft2(preGX))]


                

        elif XY == 'TB':

            clunlenTEArrNow = self.uClNow2d['TE'].copy()


                
            sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

            lx = self.lxMap
            ly = self.lyMap

            termsF = []
            termsF.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )

            termsG = []
            termsG.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )
            
            lxhat = self.lxHatMap
            lyhat = self.lyHatMap
            
            WXY = self.WXY('TB')*self.kBeamX
            WY = self.WY('BB')*self.kBeamY
            for ellsq in [lx*lx,ly*ly,np.sqrt(2.)*lx*ly]:
                preF = ellsq*clunlenTEArrNow*WXY
                preG = WY

                for termF,termG in zip(termsF,termsG):
                    allTerms += [ellsq*fft(ifft(termF(preF,lxhat,lyhat),axes=[-2,-1],normalize=True)*ifft(termG(preG,lxhat,lyhat),axes=[-2,-1],normalize=True),axes=[-2,-1])]
                    

            


        else:
            print("ERROR: Unrecognized polComb")
            sys.exit(1)    
        
                        
        ALinv = np.real(np.sum( allTerms, axis = 0))
        alval = np.nan_to_num(1. / ALinv)
        if self.fmask is not None: alval = self.fmask_func(alval,self.fmask)
        l4 = (lmap**2.) * ((lmap + 1.)**2.)
        NL = l4 *alval/ 4.
        NL[np.where(np.logical_or(lmap >= self.bigell, lmap <2.))] = 0.

        retval = np.nan_to_num(NL.real * self.pixScaleX*self.pixScaleY  )

        if setNl:
            self.Nlkk[XY] = retval.copy()
            #print "SETTING NL"


        # debug_edges = np.arange(400,6000,50)
        # import orphics.tools.stats as stats
        # import orphics.tools.io as io
        # io.quickPlot2d((np.fft.fftshift(retval)),"nl2d.png")
        # binner = stats.bin2D(self.modLMap,debug_edges)
        # cents,ws = binner.bin(retval.real)
        # pl = io.Plotter()#scaleY='log')
        # pl.add(cents,ws)
        # pl._ax.set_xlim(2,6000)
        # pl.done("nl.png")
        # sys.exit()



            
        return retval * 2. * np.nan_to_num(1. / lmap/(lmap+1.))
        
        
                  

        
      


    def delensClBB(self,Nlkk,halo=True):
        self.Nlppnow = Nlkk*4./(self.modLMap**2.)/((self.modLMap+1.)**2.)
        clPPArr = self.clpp2d
        cltotPPArr = clPPArr + self.Nlppnow
        cltotPPArr[np.isnan(cltotPPArr)] = np.inf
        
        clunlenEEArr = self.uClFid2d['EE'].copy()
        clunlentotEEArr = (self.lClFid2d['EE'].copy()+self.noiseYY2d['EE'])
        clunlentotEEArr[self.fMaskYY['EE']==0] = np.inf
        clunlenEEArr[self.fMaskYY['EE']==0] = 0.
        clPPArr[self.fMaskYY['EE']==0] = 0.
        cltotPPArr[self.fMaskYY['EE']==0] = np.inf
        

        #if halo: clunlenEEArr[np.where(self.modLMap >= self.gradCut)] = 0.
                
        sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
        cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

        lx = self.lxMap
        ly = self.lyMap

            
        lxhat = self.lxHatMap
        lyhat = self.lyHatMap

        sinf = sin2phi(lxhat,lyhat)
        sinsqf = sinf**2.
        cosf = cos2phi(lxhat,lyhat)
        cossqf = cosf**2.

        
        allTerms = []
        for ellsq in [lx*lx,ly*ly,np.sqrt(2.)*lx*ly]:
            for trigfactOut,trigfactIn in zip([sinsqf,cossqf,1.j*np.sqrt(2.)*sinf*cosf],[cossqf,sinsqf,1.j*np.sqrt(2.)*sinf*cosf]):
                preF1 = trigfactIn*ellsq*clunlenEEArr
                preG1 = ellsq*clPPArr

                preF2 = trigfactIn*ellsq*clunlenEEArr**2.*np.nan_to_num(1./clunlentotEEArr)
                preG2 = ellsq*clPPArr**2.*np.nan_to_num(1./cltotPPArr)

                allTerms += [trigfactOut*(fft(ifft(preF1,axes=[-2,-1],normalize=True)*ifft(preG1,axes=[-2,-1],normalize=True) - ifft(preF2,axes=[-2,-1],normalize=True)*ifft(preG2,axes=[-2,-1],normalize=True),axes=[-2,-1]))]


        
        ClBBres = np.real(np.sum( allTerms, axis = 0))

        
        ClBBres[np.where(np.logical_or(self.modLMap >= self.bigell, self.modLMap == 0.))] = 0.
        ClBBres *= self.Nx * self.Ny 
        ClBBres[self.fMaskYY['EE']==0] = 0.
                
        
        area =self.Nx*self.Ny*self.pixScaleX*self.pixScaleY
        bbNoise2D = ((np.sqrt(ClBBres)/self.pixScaleX/self.pixScaleY)**2.)*(area/(self.Nx*self.Ny*1.0)**2)

        self.lClFid2d['BB'] = bbNoise2D.copy()

        
        return bbNoise2D



class NlGenerator(object):
    def __init__(self,shape,wcs,theorySpectra,bin_edges=None,gradCut=None,TCMB=2.7255e6,bigell=9000,lensedEqualsUnlensed=False,unlensedEqualsLensed=True):
        self.shape = shape
        self.wcs = wcs
        self.N = QuadNorm(shape,wcs,gradCut=gradCut,bigell=bigell)
        self.TCMB = TCMB

        cmbList = ['TT','TE','EE','BB']
        
        self.theory = theorySpectra
        
        for cmb in cmbList:
            uClFilt = theorySpectra.uCl(cmb,self.N.modLMap)
            uClNorm = uClFilt
            lClFilt = theorySpectra.lCl(cmb,self.N.modLMap)
            if unlensedEqualsLensed:
                self.N.addUnlensedNorm2DPower(cmb,lClFilt.copy())
                self.N.addUnlensedFilter2DPower(cmb,lClFilt.copy())
            else:
                self.N.addUnlensedNorm2DPower(cmb,uClNorm.copy())
                self.N.addUnlensedFilter2DPower(cmb,uClFilt.copy())
            if lensedEqualsUnlensed:
                self.N.addLensedFilter2DPower(cmb,uClFilt.copy())
            else:
                self.N.addLensedFilter2DPower(cmb,lClFilt.copy())

        Clkk2d = theorySpectra.gCl("kk",self.N.modLMap)    
        self.N.addClkk2DPower(Clkk2d)
        self.N.bigell = bigell

        if bin_edges is not None:
            self.bin_edges = bin_edges
            self.binner = bin2D(self.N.modLMap, bin_edges)

    def updateBins(self,bin_edges):
        self.N.bigell = bin_edges[len(bin_edges)-1]
        self.binner = bin2D(self.N.modLMap, bin_edges)
        self.bin_edges = bin_edges

    def updateNoiseAdvanced(self,beamTX,noiseTX,beamPX,noisePX,tellminX,tellmaxX,pellminX,pellmaxX,beamTY,noiseTY,beamPY,noisePY,tellminY,tellmaxY,pellminY,pellmaxY,lkneesX=[0,0],alphasX=[1,1],lkneesY=[0,0],alphasY=[1,1],lxcutTX=None,lxcutTY=None,lycutTX=None,lycutTY=None,lxcutPX=None,lxcutPY=None,lycutPX=None,lycutPY=None,fgFuncX=None,fgFuncY=None,beamFileTX=None,beamFilePX=None,beamFileTY=None,beamFilePY=None,noiseFuncTX=None,noiseFuncTY=None,noiseFuncPX=None,noiseFuncPY=None):

        self.N.lmax_T = self.N.bigell
        self.N.lmax_P = self.N.bigell

        lkneeTX, lkneePX = lkneesX
        lkneeTY, lkneePY = lkneesY
        alphaTX, alphaPX = alphasX
        alphaTY, alphaPY = alphasY
        

        nTX = maps.whiteNoise2D([noiseTX],beamTX,self.N.modLMap, \
                                 TCMB=self.TCMB,lknees=[lkneeTX],alphas=[alphaTX],\
                                 beamFile=beamFileTX, \
                                 noiseFuncs = [noiseFuncTX])[0]
        nTY = maps.whiteNoise2D([noiseTY],beamTY,self.N.modLMap, \
                                 TCMB=self.TCMB,lknees=[lkneeTY],alphas=[alphaTY], \
                                 beamFile=beamFileTY, \
                                 noiseFuncs=[noiseFuncTY])[0]
        nPX = maps.whiteNoise2D([noisePX],beamPX,self.N.modLMap, \
                                 TCMB=self.TCMB,lknees=[lkneePX],alphas=[alphaPX],\
                                 beamFile=beamFilePX, \
                                 noiseFuncs = [noiseFuncPX])[0]
        nPY = maps.whiteNoise2D([noisePY],beamPY,self.N.modLMap, \
                                 TCMB=self.TCMB,lknees=[lkneePY],alphas=[alphaPY], \
                                 beamFile=beamFilePY, \
                                 noiseFuncs=[noiseFuncPY])[0]


        
        fMaskTX = maps.mask_kspace(self.shape,self.wcs,lmin=tellminX,lmax=tellmaxX,lxcut=lxcutTX,lycut=lycutTX)
        fMaskTY = maps.mask_kspace(self.shape,self.wcs,lmin=tellminY,lmax=tellmaxY,lxcut=lxcutTY,lycut=lycutTY)
        fMaskPX = maps.mask_kspace(self.shape,self.wcs,lmin=pellminX,lmax=pellmaxX,lxcut=lxcutPX,lycut=lycutPX)
        fMaskPY = maps.mask_kspace(self.shape,self.wcs,lmin=pellminY,lmax=pellmaxY,lxcut=lxcutPY,lycut=lycutPY)

        if fgFuncX is not None:
            fg2d = fgFuncX(self.N.modLMap) #/ self.TCMB**2.
            nTX += fg2d
        if fgFuncY is not None:
            fg2d = fgFuncY(self.N.modLMap) #/ self.TCMB**2.
            nTY += fg2d

            
        nList = ['TT','EE','BB']

        nListX = [nTX,nPX,nPX]
        nListY = [nTY,nPY,nPY]
        fListX = [fMaskTX,fMaskPX,fMaskPX]
        fListY = [fMaskTY,fMaskPY,fMaskPY]
        for i,noise in enumerate(nList):
            self.N.addNoise2DPowerXX(noise,nListX[i],fListX[i])
            self.N.addNoise2DPowerYY(noise,nListY[i],fListY[i])

        return nTX,nPX,nTY,nPY

        
    def updateNoise(self,beamX,noiseTX,noisePX,tellminX,tellmaxX,pellminX,pellmaxX,beamY=None,noiseTY=None,noisePY=None,tellminY=None,tellmaxY=None,pellminY=None,pellmaxY=None,lkneesX=[0.,0.],alphasX=[1.,1.],lkneesY=[0.,0.],alphasY=[1.,1.],lxcutTX=0,lxcutTY=0,lycutTX=0,lycutTY=0,lxcutPX=0,lxcutPY=0,lycutPX=0,lycutPY=0,fgFuncX=None,beamFileX=None,fgFuncY=None,beamFileY=None,noiseFuncTX=None,noiseFuncTY=None,noiseFuncPX=None,noiseFuncPY=None):

        def setDefault(A,B):
            if A is None:
                return B
            else:
                return A

        beamY = setDefault(beamY,beamX)
        noiseTY = setDefault(noiseTY,noiseTX)
        noisePY = setDefault(noisePY,noisePX)
        tellminY = setDefault(tellminY,tellminX)
        pellminY = setDefault(pellminY,pellminX)
        tellmaxY = setDefault(tellmaxY,tellmaxX)
        pellmaxY = setDefault(pellmaxY,pellmaxX)

        self.N.lmax_T = self.N.bigell
        self.N.lmax_P = self.N.bigell

        
        nTX,nPX = maps.whiteNoise2D([noiseTX,noisePX],beamX,self.N.modLMap, \
                                     TCMB=self.TCMB,lknees=lkneesX,alphas=alphasX,beamFile=beamFileX, \
                                     noiseFuncs = [noiseFuncTX,noiseFuncPX])
        nTY,nPY = maps.whiteNoise2D([noiseTY,noisePY],beamY,self.N.modLMap, \
                                     TCMB=self.TCMB,lknees=lkneesY,alphas=alphasY,beamFile=beamFileY, \
                                     noiseFuncs=[noiseFuncTY,noiseFuncPY])


        ### DEBUG
        # beam = 1.5
        # noise = 5.
        # from orphics import cosmology,io
        # import sys
        # nTX = cosmology.white_noise_with_atm_func(self.N.modLMap,noise,0,1,dimensionless=False,TCMB=2.7255e6)/maps.gauss_beam(self.N.modLMap,beam)**2.
        # nTY = nTX.copy()
        # nPX = nTX.copy()
        # nPY = nTX.copy()

        # # ells = np.arange(2,6000)
        # # nTX = cosmology.white_noise_with_atm_func(ells,noise,0,1,dimensionless=False,TCMB=2.7255e6)/maps.gauss_beam(ells,beam)**2.
        
        # # pl = io.Plotter(yscale='log')
        # # pl.add(ells,ells**2.*self.theory.lCl('TT',ells))
        # # pl.add(ells,nTX*ells**2.)
        # # pl.done()
        # # sys.exit()

        # print(tellminX,tellmaxX,tellminY,tellmaxY)

        ####
        
        
        fMaskTX = maps.mask_kspace(self.shape,self.wcs,lmin=tellminX,lmax=tellmaxX,lxcut=lxcutTX,lycut=lycutTX)
        fMaskTY = maps.mask_kspace(self.shape,self.wcs,lmin=tellminY,lmax=tellmaxY,lxcut=lxcutTY,lycut=lycutTY)
        fMaskPX = maps.mask_kspace(self.shape,self.wcs,lmin=pellminX,lmax=pellmaxX,lxcut=lxcutPX,lycut=lycutPX)
        fMaskPY = maps.mask_kspace(self.shape,self.wcs,lmin=pellminY,lmax=pellmaxY,lxcut=lxcutPY,lycut=lycutPY)

        if fgFuncX is not None:
            fg2d = fgFuncX(self.N.modLMap) #/ self.TCMB**2.
            nTX += fg2d
        if fgFuncY is not None:
            fg2d = fgFuncY(self.N.modLMap) #/ self.TCMB**2.
            nTY += fg2d

            
        nList = ['TT','EE','BB']

        nListX = [nTX,nPX,nPX]
        nListY = [nTY,nPY,nPY]
        fListX = [fMaskTX,fMaskPX,fMaskPX]
        fListY = [fMaskTY,fMaskPY,fMaskPY]
        for i,noise in enumerate(nList):
            self.N.addNoise2DPowerXX(noise,nListX[i],fListX[i])
            self.N.addNoise2DPowerYY(noise,nListY[i],fListY[i])

        return nTX,nPX,nTY,nPY

    def updateNoiseSimple(self,ells,nltt,nlee,lmin,lmax):

        nTX = interp1d(ells,nltt,bounds_error=False,fill_value=0.)(self.N.modLMap)
        nPX = interp1d(ells,nltt,bounds_error=False,fill_value=0.)(self.N.modLMap)
        nTY = nTX
        nPY = nPX
        
        fMaskTX = maps.mask_kspace(self.N.shape,self.N.wcs,lmin=lmin,lmax=lmax)
        fMaskTY = maps.mask_kspace(self.N.shape,self.N.wcs,lmin=lmin,lmax=lmax)
        fMaskPX = maps.mask_kspace(self.N.shape,self.N.wcs,lmin=lmin,lmax=lmax)
        fMaskPY = maps.mask_kspace(self.N.shape,self.N.wcs,lmin=lmin,lmax=lmax)

            
        nList = ['TT','EE','BB']

        nListX = [nTX,nPX,nPX]
        nListY = [nTY,nPY,nPY]
        fListX = [fMaskTX,fMaskPX,fMaskPX]
        fListY = [fMaskTY,fMaskPY,fMaskPY]
        for i,noise in enumerate(nList):
            self.N.addNoise2DPowerXX(noise,nListX[i],fListX[i])
            self.N.addNoise2DPowerYY(noise,nListY[i],fListY[i])

        return nTX,nPX,nTY,nPY
    
    def getNl(self,polComb='TT',halo=True):            

        AL = self.N.getNlkk2d(polComb,halo=halo)
        data2d = self.N.Nlkk[polComb]

        centers, Nlbinned = self.binner.bin(data2d)
        Nlbinned = sanitizePower(Nlbinned)
        
        return centers, Nlbinned

    def getNlIterative(self,polCombs,kmin,kmax,tellmax,pellmin,pellmax,dell=20,halo=True,dTolPercentage=1.,verbose=True,plot=False):
        
        Nleach = {}
        bin_edges = np.arange(kmin-dell/2.,kmax+dell/2.,dell)#+dell
        for polComb in polCombs:
            self.updateBins(bin_edges)
            AL = self.N.getNlkk2d(polComb,halo=halo)
            data2d = self.N.Nlkk[polComb]
            ls, Nls = self.binner.bin(data2d)
            Nls = sanitizePower(Nls)
            Nleach[polComb] = (ls,Nls)

        if ('EB' not in polCombs) and ('TB' not in polCombs):
            Nlret = Nlmv(Nleach,polCombs,None,None,bin_edges)
            return bin_edges,sanitizePower(Nlret),None,None,None

        origBB = self.N.lClFid2d['BB'].copy()
        delensBinner =  bin2D(self.N.modLMap, bin_edges)
        ellsOrig, oclbb = delensBinner.bin(origBB.real)
        oclbb = sanitizePower(oclbb)
        origclbb = oclbb.copy()

        if plot:
            from orphics.tools.io import Plotter
            pl = Plotter(scaleY='log',scaleX='log')
            pl.add(ellsOrig,oclbb*ellsOrig**2.,color='black',lw=2)
            
        ctol = np.inf
        inum = 0
        while ctol>dTolPercentage:
            bNlsinv = 0.
            polPass = list(polCombs)
            if verbose: print("Performing iteration ", inum+1)
            for pol in ['EB','TB']:
                if not(pol in polCombs): continue
                Al2d = self.N.getNlkk2d(pol,halo)
                centers, nlkkeach = delensBinner.bin(self.N.Nlkk[pol])
                nlkkeach = sanitizePower(nlkkeach)
                bNlsinv += 1./nlkkeach
                polPass.remove(pol)
            nlkk = 1./bNlsinv
            
            Nldelens = Nlmv(Nleach,polPass,centers,nlkk,bin_edges)
            Nldelens2d = interp1d(bin_edges,Nldelens,fill_value=0.,bounds_error=False)(self.N.modLMap)

            bbNoise2D = self.N.delensClBB(Nldelens2d,halo)
            ells, dclbb = delensBinner.bin(bbNoise2D)
            dclbb = sanitizePower(dclbb)
            dclbb[ells<pellmin] = oclbb[ellsOrig<pellmin].copy()
            if inum>0:
                newLens = np.nanmean(nlkk)
                oldLens = np.nanmean(oldNl)
                new = np.nanmean(dclbb)
                old = np.nanmean(oclbb)
                ctol = np.abs(old-new)*100./new
                ctolLens = np.abs(oldLens-newLens)*100./newLens
                if verbose: print("Percentage difference between iterations is ",ctol, " compared to requested tolerance of ", dTolPercentage,". Diff of Nlkks is ",ctolLens)
            oldNl = nlkk.copy()
            oclbb = dclbb.copy()
            inum += 1
            if plot:
                pl.add(ells,dclbb*ells**2.,ls="--",alpha=0.5,color="black")

        if plot:
            import os
            pl.done(os.environ['WWW']+'delens.png')
        self.N.lClFid2d['BB'] = origBB.copy()
        efficiency = ((origclbb-dclbb)*100./origclbb).max()


        new_ells,new_bb = fillLowEll(ells,dclbb,pellmin)
        new_k_ells,new_nlkk = fillLowEll(bin_edges,sanitizePower(Nldelens),kmin)
        
        return new_k_ells,new_nlkk,new_ells,new_bb,efficiency


    def iterativeDelens(self,xy,dTolPercentage=1.0,halo=True,verbose=True):
        assert xy=='EB' or xy=='TB'
        origBB = self.N.lClFid2d['BB'].copy()
        bin_edges = self.bin_edges #np.arange(100.,3000.,20.)
        delensBinner =  bin2D(self.N.modLMap, bin_edges)
        ells, oclbb = delensBinner.bin(origBB)
        oclbb = sanitizePower(oclbb)

        ctol = np.inf
        inum = 0


        
        #from orphics.tools.output import Plotter
        #pl = Plotter(scaleY='log',scaleX='log')
        #pl = Plotter(scaleY='log')
        while ctol>dTolPercentage:
            if verbose: print("Performing iteration ", inum+1)
            Al2d = self.N.getNlkk2d(xy,halo)
            centers, nlkk = delensBinner.bin(self.N.Nlkk[xy])
            nlkk = sanitizePower(nlkk)
            bbNoise2D = self.N.delensClBB(self.N.Nlkk[xy],halo)
            ells, dclbb = delensBinner.bin(bbNoise2D)
            dclbb = sanitizePower(dclbb)
            if inum>0:
                new = np.nanmean(nlkk)
                old = np.nanmean(oldNl)
                ctol = np.abs(old-new)*100./new
                if verbose: print("Percentage difference between iterations is ",ctol, " compared to requested tolerance of ", dTolPercentage)
            oldNl = nlkk.copy()
            inum += 1
            #pl.add(centers,nlkk)
            #pl.add(ells,dclbb*ells**2.)
        #pl.done('output/delens'+xy+'.png')
        self.N.lClFid2d['BB'] = origBB.copy()
        efficiency = (np.max(oclbb)-np.max(dclbb))*100./np.max(oclbb)
        return centers,nlkk,efficiency
    
class Estimator(object):
    '''
    Flat-sky lensing and Omega quadratic estimators
    Functionality includes:
    - small-scale lens estimation with gradient cutoff
    - combine maps from two different experiments


    NOTE: The TE estimator is not identical between large
    and small-scale estimators. Need to test this.
    '''


    def __init__(self,shape,wcs,
                 theorySpectraForFilters,
                 theorySpectraForNorm=None,
                 noiseX2dTEB=[None,None,None],
                 noiseY2dTEB=[None,None,None],
                 noiseX_is_total = False,
                 noiseY_is_total = False,
                 fmaskX2dTEB=[None,None,None],
                 fmaskY2dTEB=[None,None,None],
                 fmaskKappa=None,
                 kBeamX = None,
                 kBeamY = None,
                 doCurl=False,
                 TOnly=False,
                 halo=True,
                 gradCut=None,
                 verbose=False,
                 loadPickledNormAndFilters=None,
                 savePickledNormAndFilters=None,
                 uEqualsL=False,
                 bigell=9000,
                 mpi_comm=None,
                 lEqualsU=False):

        '''
        All the 2d fourier objects below are pre-fftshifting. They must be of the same dimension.

        shape,wcs: enmap geometry
        theorySpectraForFilters: an orphics.tools.cmb.TheorySpectra object with CMB Cls loaded
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

        self.verbose = verbose

        # initialize norm and filters

        self.doCurl = doCurl



        if loadPickledNormAndFilters is not None:
            if verbose: print("Unpickling...")
            with open(loadPickledNormAndFilters,'rb') as fin:
                self.N,self.AL,self.OmAL,self.fmaskK,self.phaseY = pickle.load(fin)
            return



        self.halo = halo
        self.AL = {}
        if doCurl: self.OmAL = {}

        if kBeamX is not None:           
            self.kBeamX = kBeamX
        else:
            self.kBeamX = 1.
            
        if kBeamY is not None:           
            self.kBeamY = kBeamY
        else:
            self.kBeamY = 1.

        self.doCurl = doCurl
        self.halo = halo

        if fmaskKappa is None:
            ellMinK = 80
            ellMaxK = 3000
            print("WARNING: using default kappa mask of 80 < L < 3000")
            self.fmaskK = fmaps.fourierMask(self.N.lx,self.N.ly,self.N.modLMap,lmin=ellMinK,lmax=ellMaxK)
        else:
            self.fmaskK = fmaskKappa

        
        self.fmaskX2dTEB = fmaskX2dTEB
        self.fmaskY2dTEB = fmaskY2dTEB
        
        # Get MPI comm
        comm = mpi_comm
        if comm is not None:
            rank = comm.Get_rank()
            numcores = comm.Get_size()
        else:
            rank = 0
            numcores = 1

        if rank==0:
            self.N = QuadNorm(shape,wcs,gradCut=gradCut,verbose=verbose,kBeamX=self.kBeamX,kBeamY=self.kBeamY,bigell=bigell,fmask=self.fmaskK)


            if TOnly: 
                nList = ['TT']
                cmbList = ['TT']
                estList = ['TT']
                self.phaseY = 1.
            else:
                self.phaseY = np.cos(2.*self.N.thetaMap)+1.j*np.sin(2.*self.N.thetaMap)
                nList = ['TT','EE','BB']
                cmbList = ['TT','TE','EE','BB']
                estList = ['TT','TE','ET','EB','EE','TB']

            self.nList = nList

            if self.verbose: print("Initializing filters and normalization for quadratic estimators...")
            assert not(uEqualsL and lEqualsU)
            for cmb in cmbList:
                if uEqualsL:
                    uClFilt = theorySpectraForFilters.lCl(cmb,self.N.modLMap)
                else:
                    uClFilt = theorySpectraForFilters.uCl(cmb,self.N.modLMap)

                if theorySpectraForNorm is not None:
                    if uEqualsL:
                        uClNorm = theorySpectraForFilters.lCl(cmb,self.N.modLMap)
                    else:
                        uClNorm = theorySpectraForNorm.uCl(cmb,self.N.modLMap)
                else:
                    uClNorm = uClFilt

                if lEqualsU:
                    lClFilt = theorySpectraForFilters.uCl(cmb,self.N.modLMap)
                else:
                    lClFilt = theorySpectraForFilters.lCl(cmb,self.N.modLMap)
                    
                #lClFilt = theorySpectraForFilters.lCl(cmb,self.N.modLMap)
                self.N.addUnlensedFilter2DPower(cmb,uClFilt)
                self.N.addLensedFilter2DPower(cmb,lClFilt)
                self.N.addUnlensedNorm2DPower(cmb,uClNorm)
            for i,noise in enumerate(nList):
                self.N.addNoise2DPowerXX(noise,noiseX2dTEB[i],fmaskX2dTEB[i],is_total=noiseX_is_total)
                self.N.addNoise2DPowerYY(noise,noiseY2dTEB[i],fmaskY2dTEB[i],is_total=noiseY_is_total)
            try:
                self.N.addClkk2DPower(theorySpectraForFilters.gCl("kk",self.N.modLMap))
            except:
                print("Couldn't add Clkk2d power")

            self.estList = estList
            self.OmAL = None
            for est in estList:
                self.AL[est] = self.N.getNlkk2d(est,halo=halo)
                #if doCurl: self.OmAL[est] = self.N.getCurlNlkk2d(est,halo=halo)
                
                # send_dat = np.array(self.vectors[label]).astype(np.float64)
                # self.comm.Send(send_dat, dest=0, tag=self.tag_start+k)

        else:

            pass
        

    def updateNoise(self,nTX,nEX,nBX,nTY,nEY,nBY,noiseX_is_total=False,noiseY_is_total=False):
        noiseX2dTEB = [nTX,nEX,nBX]
        noiseY2dTEB = [nTY,nEY,nBY]
        for i,noise in enumerate(self.nList):
            self.N.addNoise2DPowerXX(noise,noiseX2dTEB[i],self.fmaskX2dTEB[i],is_total=noiseX_is_total)
            self.N.addNoise2DPowerYY(noise,noiseY2dTEB[i],self.fmaskY2dTEB[i],is_total=noiseY_is_total)

        for est in self.estList:
            self.AL[est] = self.N.getNlkk2d(est,halo=self.halo)
            if self.doCurl: self.OmAL[est] = self.N.getCurlNlkk2d(est,halo=self.halo)
            

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

        if alreadyFTed:
            self.kT = T2DData
        else:
            self.kT = fft(T2DData,axes=[-2,-1])
        self.kGradx['T'] = lx*self.kT.copy()*1j
        self.kGrady['T'] = ly*self.kT.copy()*1j

        if E2DData is not None:
            if alreadyFTed:
                self.kE = E2DData
            else:
                self.kE = fft(E2DData,axes=[-2,-1])
            self.kGradx['E'] = 1.j*lx*self.kE.copy()
            self.kGrady['E'] = 1.j*ly*self.kE.copy()
        if B2DData is not None:
            if alreadyFTed:
                self.kB = B2DData
            else:
                self.kB = fft(B2DData,axes=[-2,-1])
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
                self.kHigh['T']=fft(T2DData,axes=[-2,-1])
        else:
            self.kHigh['T']=self.kT.copy()
        if E2DData is not None:
            if alreadyFTed:
                self.kHigh['E']=E2DData
            else:
                self.kHigh['E']=fft(E2DData,axes=[-2,-1])
        else:
            try:
                self.kHigh['E']=self.kE.copy()
            except:
                pass

        if B2DData is not None:
            if alreadyFTed:
                self.kHigh['B']=B2DData
            else:
                self.kHigh['B']=fft(B2DData,axes=[-2,-1])
        else:
            try:
                self.kHigh['B']=self.kB.copy()
            except:
                pass

    def kappa_from_map(self,XY,T2DData,E2DData=None,B2DData=None,T2DDataY=None,E2DDataY=None,B2DDataY=None,alreadyFTed=False,returnFt=False):
        self.updateTEB_X(T2DData,E2DData,B2DData,alreadyFTed)
        self.updateTEB_Y(T2DDataY,E2DDataY,B2DDataY,alreadyFTed)
        return self.get_kappa(XY,returnFt=returnFt)
        
        
    def fmask_func(self,arr):
        fMask = self.fmaskK
        arr[fMask<1.e-3] = 0.
        return arr
        
    def get_kappa(self,XY,returnFt=False):

        assert self._hasX and self._hasY
        assert XY in ['TT','TE','ET','EB','TB','EE']
        X,Y = XY

        WXY = self.N.WXY(XY)
        WY = self.N.WY(Y+Y)



        lx = self.N.lxMap
        ly = self.N.lyMap

        if Y in ['E','B']:
            phaseY = self.phaseY
        else:
            phaseY = 1.

        phaseB = (int(Y=='B')*1.j)+(int(Y!='B'))
        
        fMask = self.fmaskK

        if self.verbose: startTime = time.time()

        # HighMapStar = ifft(self.fmask_func(self.kHigh[Y]*WY*phaseY*phaseB),axes=[-2,-1],normalize=True).conjugate()
        # kPx = fft(ifft(self.kGradx[X]*WXY*phaseY,axes=[-2,-1],normalize=True)*HighMapStar,axes=[-2,-1])
        # kPy = fft(ifft(self.kGrady[X]*WXY*phaseY,axes=[-2,-1],normalize=True)*HighMapStar,axes=[-2,-1])        
        # rawKappa = ifft(self.fmask_func(1.j*lx*kPx) + self.fmask_func(1.j*ly*kPy),axes=[-2,-1],normalize=True).real

        HighMapStar = ifft((self.kHigh[Y]*WY*phaseY*phaseB),axes=[-2,-1],normalize=True).conjugate()
        kPx = fft(ifft(self.kGradx[X]*WXY*phaseY,axes=[-2,-1],normalize=True)*HighMapStar,axes=[-2,-1])
        kPy = fft(ifft(self.kGrady[X]*WXY*phaseY,axes=[-2,-1],normalize=True)*HighMapStar,axes=[-2,-1])        
        rawKappa = ifft((1.j*lx*kPx) + (1.j*ly*kPy),axes=[-2,-1],normalize=True).real

        AL = np.nan_to_num(self.AL[XY])


        assert not(np.any(np.isnan(rawKappa)))
        # debug_edges = np.arange(400,6000,50)
        # import orphics.tools.stats as stats
        # import orphics.tools.io as io
        # io.quickPlot2d(rawKappa,"kappa.png")
        # binner = stats.bin2D(self.N.modLMap,debug_edges)
        # cents,ws = binner.bin(rawKappa.real)
        # pl = io.Plotter()#scaleY='log')
        # pl.add(cents,ws)
        # pl._ax.set_xlim(2,6000)
        # pl.done("rawkappa1d.png")
        #sys.exit()


        lmap = self.N.modLMap
        
        kappaft = -self.fmask_func(AL*fft(rawKappa,axes=[-2,-1]))
        #kappaft = np.nan_to_num(-AL*fft(rawKappa,axes=[-2,-1])) # added after beam convolved change
        self.kappa = ifft(kappaft,axes=[-2,-1],normalize=True).real
        try:
            #raise
            assert not(np.any(np.isnan(self.kappa)))
        except:
            import orphics.tools.io as io
            import orphics.tools.stats as stats
            io.quickPlot2d(np.fft.fftshift(np.abs(kappaft)),"ftkappa.png")
            io.quickPlot2d(np.fft.fftshift(fMask),"fmask.png")
            io.quickPlot2d(self.kappa.real,"nankappa.png")
            debug_edges = np.arange(20,20000,100)
            dbinner = stats.bin2D(self.N.modLMap,debug_edges)
            cents, bclkk = dbinner.bin(self.N.clkk2d)
            cents, nlkktt = dbinner.bin(self.N.Nlkk['TT'])
            cents, alkktt = dbinner.bin(AL/2.*lmap*(lmap+1.))
            try:
                cents, nlkkeb = dbinner.bin(self.N.Nlkk['EB'])
            except:
                pass
            pl = io.Plotter(scaleY='log',scaleX='log')
            pl.add(cents,bclkk)
            pl.add(cents,nlkktt,label="TT")
            pl.add(cents,alkktt,label="TTnorm",ls="--")
            try:
                pl.add(cents,nlkkeb,label="EB")
            except:
                pass
            pl.legendOn()
            pl._ax.set_ylim(1.e-9,1.e-5)
            pl.done("clkk.png")

            sys.exit()
        
            
        # from orphics.tools.io import Plotter
        # pl = Plotter()
        # #pl.plot2d(np.nan_to_num(self.kappa))
        # pl.plot2d((self.kappa.real))
        # pl.done("output/nankappa.png")
        # sys.exit(0)
        # try:
        #     assert not(np.any(np.isnan(self.kappa)))
        # except:
        #     from orphics.tools.io import Plotter
        #     pl = Plotter()
        #     pl.plot2d(np.nan_to_num(self.kappa))
        #     pl.done("output/nankappa.png")
        #     sys.exit(0)

        if self.verbose:
            elapTime = time.time() - startTime
            print(("Time for core kappa was ", elapTime ," seconds."))

        if self.doCurl:
            OmAL = self.OmAL[XY]*fMask
            rawCurl = ifft(1.j*lx*kPy - 1.j*ly*kPx,axes=[-2,-1],normalize=True).real
            self.curl = -ifft(OmAL*fft(rawCurl,axes=[-2,-1]),axes=[-2,-1],normalize=True)
            return self.kappa, self.curl



        if returnFt:
            return self.kappa,kappaft
        else:
            return self.kappa





def Nlmv(Nleach,pols,centers,nlkk,bin_edges):
    # Nleach: dict of (ls,Nls) for each polComb
    # pols: list of polCombs to include
    # centers,nlkk: additonal Nl to add
    
    Nlmvinv = 0.
    for polComb in pols:
        ls,Nls = Nleach[polComb]
        nlfunc = interp1d(ls,Nls,bounds_error=False,fill_value=np.inf)
        Nleval = nlfunc(bin_edges)
        Nlmvinv += np.nan_to_num(1./Nleval)
        
    if nlkk is not None:
        nlfunc = interp1d(centers,nlkk,bounds_error=False,fill_value=np.inf)
        Nleval = nlfunc(bin_edges)
        Nlmvinv += np.nan_to_num(1./Nleval)
        
    return np.nan_to_num(1./Nlmvinv)


## HALOS

# g(x) = g(theta/thetaS) HuDeDeoVale 2007
gnfw = lambda x: np.piecewise(x, [x>1., x<1., x==1.], \
                            [lambda y: (1./(y*y - 1.)) * \
                             ( 1. - ( (2./np.sqrt(y*y - 1.)) * np.arctan(np.sqrt((y-1.)/(y+1.))) ) ), \
                             lambda y: (1./(y*y - 1.)) * \
                            ( 1. - ( (2./np.sqrt(-(y*y - 1.))) * np.arctanh(np.sqrt(-((y-1.)/(y+1.)))) ) ), \
                        lambda y: (1./3.)])

f_c = lambda c: np.log(1.+c) - (c/(1.+c))


def nfw_kappa(massOverh,modrmap_radians,cc,zL=0.7,concentration=3.2,overdensity=180.,critical=False,atClusterZ=False):
    sgn = 1. if massOverh>0. else -1.
    comS = cc.results.comoving_radial_distance(cc.cmbZ)*cc.h
    comL = cc.results.comoving_radial_distance(zL)*cc.h
    winAtLens = (comS-comL)/comS
    kappa,r500 = NFWkappa(cc,np.abs(massOverh),concentration,zL,modrmap_radians* 180.*60./np.pi,winAtLens,
                          overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)

    return sgn*kappa



def NFWkappa(cc,massOverh,concentration,zL,thetaArc,winAtLens,overdensity=500.,critical=True,atClusterZ=True):

    comL  = (cc.results.comoving_radial_distance(zL) )*cc.h

    

    c = concentration
    M = massOverh

    zdensity = 0.
    if atClusterZ: zdensity = zL

    if critical:
        r500 = cc.rdel_c(M,zdensity,overdensity).flatten()[0] # R500 in Mpc/h
    else:
        r500 = cc.rdel_m(M,zdensity,overdensity) # R500 in Mpc/h


    conv=np.pi/(180.*60.)
    theta = thetaArc*conv # theta in radians

    rS = r500/c

    thetaS = rS/ comL 


    const12 = 9.571e-20 # 2G/c^2 in Mpc / solar mass 
    fc = np.log(1.+c) - (c/(1.+c))    
    #const3 = comL * comLS * (1.+zL) / comS #  Mpc
    const3 = comL *  (1.+zL) *winAtLens #  Mpc
    const4 = M / (rS*rS) #solar mass / MPc^2
    const5 = 1./fc
    

    kappaU = gnfw(theta/thetaS)+theta*0. # added for compatibility with enmap

    consts = const12 * const3 * const4 * const5
    kappa = consts * kappaU

    if thetaArc.shape[0]%2==1 and thetaArc.shape[1]%2==1:
        Ny,Nx = thetaArc.shape
        cx = int(Nx/2.)
        cy = int(Ny/2.)
        kappa[cy,cx] = kappa[cy-1,cx]
        

    return kappa, r500



def NFWMatchedFilterSN(clusterCosmology,log10Moverh,c,z,ells,Nls,kellmax,overdensity=500.,critical=True,atClusterZ=True,arcStamp=100.,pxStamp=0.05,saveId=None,verbose=False,rayleighSigmaArcmin=None,returnKappa=False,winAtLens=None):
    if rayleighSigmaArcmin is not None: assert rayleighSigmaArcmin>=pxStamp
    M = 10.**log10Moverh

    
    shape,wcs = maps.rect_geometry(width_deg=arcStamp/60.,px_res_arcmin=pxStamp)
    kellmin = 2.*np.pi/arcStamp*np.pi/60./180.

    modLMap = enmap.modlmap(shape,wcs)
    xMap,yMap,modRMap,xx,yy  = maps.get_real_attributes(shape,wcs)
        
    cc = clusterCosmology

    cmb = False
    if winAtLens is None:
        cmb = True
        comS = cc.results.comoving_radial_distance(cc.cmbZ)*cc.h
        comL = cc.results.comoving_radial_distance(z)*cc.h
        winAtLens = (comS-comL)/comS

    kappaReal, r500 = NFWkappa(cc,M,c,z,modRMap*180.*60./np.pi,winAtLens,overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)
    
    dAz = cc.results.angular_diameter_distance(z) * cc.h
    # print ("daz " , dAz , " mpc")
    # print ("r500 " , r500 , " mpc")
    th500 = r500/dAz
    #fiveth500 = 10.*np.pi/180./60. #5.*th500
    fiveth500 = 5.*th500
    # print ("5theta500 " , fiveth500*180.*60./np.pi , " arcminutes")
    # print ("maximum theta " , modRMap.max()*180.*60./np.pi, " arcminutes")

    kInt = kappaReal.copy()
    kInt[modRMap>fiveth500] = 0.
    # print "mean kappa inside theta500 " , kInt[modRMap<fiveth500].mean()
    # print "area of th500 disc " , np.pi*fiveth500**2.*(180.*60./np.pi)**2.
    # print "estimated integral " , kInt[modRMap<fiveth500].mean()*np.pi*fiveth500**2.
    k500 = simps(simps(kInt, yy), xx)
    
    if verbose: print(("integral of kappa inside disc ",k500))
    kappaReal[modRMap>fiveth500] = 0. #### !!!!!!!!! Might not be necessary!
    # if cmb: print z,fiveth500*180.*60./np.pi
    Ukappa = kappaReal/k500


    
    # pl = Plotter()
    # pl.plot2d(Ukappa)
    # pl.done("output/kappa.png")

    ellmax = kellmax
    ellmin = kellmin

    
    
    Uft = fft(Ukappa,axes=[-2,-1])

    if rayleighSigmaArcmin is not None:
        Prayleigh = rayleigh(modRMap*180.*60./np.pi,rayleighSigmaArcmin)
        outDir = "/gpfs01/astro/www/msyriac/plots/"
        # io.quickPlot2d(Prayleigh,outDir+"rayleigh.png")
        rayK = fft(ifftshift(Prayleigh),axes=[-2,-1])
        rayK /= rayK[modLMap<1.e-3]
        Uft = Uft.copy()*rayK
    
    Upower = np.real(Uft*Uft.conjugate())

    

    # pl = Plotter()
    # pl.plot2d(fftshift(Upower))
    # pl.done("output/upower.png")


    
    Nls[Nls<0.]=0.
    s = splrep(ells,Nls,k=3)
    Nl2d = splev(modLMap,s) 
    
    Nl2d[modLMap<ellmin]=np.inf
    Nl2d[modLMap>ellmax] = np.inf

    Ny,Nx = shape
    pixScaleY,pixScaleX = enmap.pixshape(shape,wcs)
    area = Nx*Ny*pixScaleX*pixScaleY
    Upower = Upower *area / (Nx*Ny)**2
        
    filter = np.nan_to_num(Upower/Nl2d)
    #filter = np.nan_to_num(1./Nl2d)
    filter[modLMap>ellmax] = 0.
    filter[modLMap<ellmin] = 0.
    # pl = Plotter()
    # pl.plot2d(fftshift(filter))
    # pl.done("output/filter.png")
    # if (cmb): print Upower.sum()
    # if not(cmb) and z>2.5:
    #     bin_edges = np.arange(500,ellmax,100)
    #     binner = bin2D(modLMap, bin_edges)
    #     centers, nl2dells = binner.bin(Nl2d)
    #     centers, upowerells = binner.bin(np.nan_to_num(Upower))
    #     centers, filterells = binner.bin(filter)
    #     from orphics.tools.io import Plotter
    #     pl = Plotter(scaleY='log')
    #     pl.add(centers,upowerells,label="upower")
    #     pl.add(centers,nl2dells,label="noise")
    #     pl.add(centers,filterells,label="filter")
    #     pl.add(ells,Nls,ls="--")
    #     pl.legendOn(loc='upper right')
    #     #pl._ax.set_ylim(0,1e-8)
    #     pl.done("output/filterells.png")
    #     sys.exit()
    
    varinv = filter.sum()
    std = np.sqrt(1./varinv)
    sn = k500/std
    if verbose: print(sn)

    if saveId is not None:
        np.savetxt("data/"+saveId+"_m"+str(log10Moverh)+"_z"+str(z)+".txt",np.array([log10Moverh,z,1./sn]))

    if returnKappa:
        return sn,ifft(Uft,axes=[-2,-1],normalize=True).real*k500
    return sn, k500, std



    

def rayleigh(theta,sigma):
    sigmasq = sigma*sigma
    #return np.exp(-0.5*theta*theta/sigmasq)
    return theta/sigmasq*np.exp(-0.5*theta*theta/sigmasq)
        


def NFWkappa(cc,massOverh,concentration,zL,thetaArc,winAtLens,overdensity=500.,critical=True,atClusterZ=True):

    comL  = (cc.results.comoving_radial_distance(zL) )*cc.h

    

    c = concentration
    M = massOverh

    zdensity = 0.
    if atClusterZ: zdensity = zL

    if critical:
        r500 = cc.rdel_c(M,zdensity,overdensity).flatten()[0] # R500 in Mpc/h
    else:
        r500 = cc.rdel_m(M,zdensity,overdensity) # R500 in Mpc/h

    conv=np.pi/(180.*60.)
    theta = thetaArc*conv # theta in radians

    rS = r500/c

    thetaS = rS/ comL 


    const12 = 9.571e-20 # 2G/c^2 in Mpc / solar mass 
    fc = np.log(1.+c) - (c/(1.+c))    
    #const3 = comL * comLS * (1.+zL) / comS #  Mpc
    const3 = comL *  (1.+zL) *winAtLens #  Mpc
    const4 = M / (rS*rS) #solar mass / MPc^2
    const5 = 1./fc
    

    kappaU = gnfw(theta/thetaS)+theta*0. # added for compatibility with enmap

    consts = const12 * const3 * const4 * const5
    kappa = consts * kappaU


    return kappa, r500



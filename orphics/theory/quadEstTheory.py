import numpy as np
import orphics.analysis.flatMaps as fmaps 
import fftTools

from scipy.fftpack import fft2,ifft2,fftshift,ifftshift,fftfreq
#from numpy.fft import fft2,ifft2,fftshift,ifftshift,fftfreq

class QuadNorm(object):

    
    def __init__(self,templateMap,gradCut=None):
        '''

        templateFT is a template liteMap FFT object
    

    
        '''

        self.lxMap,self.lyMap,self.modLMap,self.thetaMap,self.lx,self.ly = fmaps.getFTAttributesFromLiteMap(templateMap)
        self.lxHatMap = np.nan_to_num(self.lxMap / self.modLMap)
        self.lyHatMap = np.nan_to_num(self.lyMap / self.modLMap)

        self.uClNow2d = {}
        self.uClFid2d = {}
        self.lClFid2d = {}
        self.noiseXX2d = {}
        self.noiseYY2d = {}
        self.fMaskXX = {}
        self.fMaskYY = {}

        self.lmax_T=9000.
        self.lmax_P=9000.
        self.defaultMaskT = fmaps.fourierMask(self.lx,self.ly,self.modLMap,lmin=2,lmax=self.lmax_T)
        self.defaultMaskP = fmaps.fourierMask(self.lx,self.ly,self.modLMap,lmin=2,lmax=self.lmax_P)
        self.bigell=9000.
        self.gradCut = self.bigell
        if gradCut is not None: self.gradCut = gradCut


        self.template = templateMap.copy()
        

    def addUnlensedFilter2DPower(self,XY,power2dData):
        '''
        XY = TT, TE, EE, EB or TB
        power2d is a flipper power2d object
        These Cls belong in the Wiener filters, and will not
        be perturbed if/when calculating derivatives.
        '''
        self.uClFid2d[XY] = power2dData.copy()
    def addUnlensedNorm2DPower(self,XY,power2dData):
        '''
        XY = TT, TE, EE, EB or TB
        power2d is a flipper power2d object
        These Cls belong in the CMB normalization, and will
        be perturbed if/when calculating derivatives.
        '''
        self.uClNow2d[XY] = power2dData.copy()
    def addLensedFilter2DPower(self,XY,power2dData):
        '''
        XY = TT, TE, EE, EB or TB
        power2d is a flipper power2d object
        These Cls belong in the Wiener filters, and will not
        be perturbed if/when calculating derivatives.
        '''
        self.lClFid2d[XY] = power2dData.copy()
    def addNoise2DPowerXX(self,XX,power2dData,fourierMask=None):
        '''
        Noise power for the X leg of the quadratic estimator
        XX = TT, EE, BB
        power2d is a flipper power2d object
        fourierMask is an int array that is 0 where noise is
        infinite and 1 where not. It should be in the same
        fftshift state as power2d.powerMap
        '''
        # check if fourier mask is int!
        self.noiseXX = power2dData.copy()
        if fourierMask is not None:
            self.noiseXX[fourierMask==0] = np.inf
            self.fMaskXX[XX] = fourierMask
        else:
            if XX=='TT':
                self.noiseXX[self.defaultMaskT==0] = np.inf
            else:
                self.noiseXX[self.defaultMaskP==0] = np.inf

    def addNoise2DPowerYY(self,YY,power2dData,fourierMask=None):
        '''
        Noise power for the Y leg of the quadratic estimator
        XX = TT, EE, BB
        power2d is a flipper power2d object
        fourierMask is an int array that is 0 where noise is
        infinite and 1 where not. It should be in the same
        fftshift state as power2d.powerMap
        '''
        # check if fourier mask is int!
        self.noiseYY = power2dData.copy()
        if fourierMask is not None:
            self.noiseYY[fourierMask==0] = np.inf
            self.fMaskYY[YY] = fourierMask
        else:
            if YY=='TT':
                self.noiseYY[self.defaultMaskT==0] = np.inf
            else:
                self.noiseYY[self.defaultMaskP==0] = np.inf
        
    def addClkk2DPower(self,power2dData):
        '''
        Fiducial Clkk power
        Used if delensing
        power2d is a flipper power2d object            
        '''
        self.clkk2d = power2dData.copy()


    def WXY(self,XY):
        X,Y = XY
        if Y=='B': Y='E'
        W = np.nan_to_num(self.uClFid2d[X+Y].copy()/(self.lClFid2d[X+X].copy()+self.noiseXX.copy()))*self.fMaskXX[X+X]
        W[self.modLMap>self.gradCut]=0.
        if X=='T':
            W[np.where(self.modLMap >= self.lmax_T)] = 0.
        else:
            W[np.where(self.modLMap >= self.lmax_P)] = 0.


        return W
        

    def WY(self,YY):
        W = np.nan_to_num(1./(self.lClFid2d[YY].copy()+self.noiseYY.copy()))*self.fMaskYY[YY]
        W[np.where(self.modLMap >= self.lmax_T)] = 0.
        if YY[0]=='T':
            W[np.where(self.modLMap >= self.lmax_T)] = 0.
        else:
            W[np.where(self.modLMap >= self.lmax_P)] = 0.
        return W

    def getCurlNlkk2d(self,XY,halo=False):
        pass
            
    def getNlkk2d(self,XY,halo=False):
        lx,ly = self.lxMap,self.lyMap
        lmap = self.modLMap

            
        h=0.

        allTerms = []
            
        if XY == 'TT':
            
            clunlenTTArrNow = self.uClNow2d['TT'].copy()
            clunlenTTArrNow[np.where(lmap >= self.lmax_T)] = 0.

                

            if halo:
            
                
                preG = self.WY('TT')
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTTArrNow*self.WXY('TT')
                    preFX = ell1*self.WXY('TT')
                    preGX = ell2*clunlenTTArrNow*self.WY('TT')
                    

                    calc = ell1*ell2*fft2(ifft2(preF)*ifft2(preG)+ifft2(preFX)*ifft2(preGX))
                    allTerms += [calc]
                    

            else:

                # IMPLEMENT GRADCUT IN NORM FOR LSS

                preG = self.WY('TT') #np.nan_to_num(1./cltotTTArrY)

                # from orphics.tools.output import Plotter
                # import sys
                # X = preG
                # pl = Plotter()
                # pl.plot2d(np.log10(fftshift(X)))
                # pl.done("debug.png")
                # sys.exit()
                
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTTArrNow*clunlenTTArr*np.nan_to_num(1./cltotTTArrX)/2.            
                    preFX = ell1*clunlenTTArrNow*np.nan_to_num(1./cltotTTArrX)
                    preGX = ell2*clunlenTTArr*np.nan_to_num(1./cltotTTArrY)


                    
                    calc = 2.*ell1*ell2*fft2(ifft2(preF)*ifft2(preG)+ifft2(preFX)*ifft2(preGX)/2.)
                    allTerms += [calc]
          

        elif XY == 'EE':

            #check noise array!!!

            clunlenEEArrNow = self.uClNow2d['EE'].copy()
            clunlenEEArr = self.uClFid2d['EE'].copy()
            
            cltotEEArr =self.lClFid2d['EE'].copy() + self.noiseArray[1]
            cltotEEArr[np.where(lmap >= self.lmax_P)] = np.inf

            


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
            
                
                clunlenEEArrNow[np.where(lmap >= self.gradCut)] = 0.
                
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                        preF = trigfact*ell1*ell2*clunlenEEArrNow*clunlenEEArr*np.nan_to_num(1./cltotEEArr)
                        preG = trigfact*np.nan_to_num(1./cltotEEArr)
                        allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
                        
                        preFX = trigfact*ell1*clunlenEEArrNow*np.nan_to_num(1./cltotEEArr)
                        preGX = trigfact*ell2*clunlenEEArr*np.nan_to_num(1./cltotEEArr)

                        allTerms += [ell1*ell2*fft2(ifft2(preFX)*ifft2(preGX))]

                
            else:


                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                        preF = trigfact*ell1*ell2*clunlenEEArrNow*clunlenEEArr*np.nan_to_num(1./cltotEEArr)/2.
                        preG = trigfact*np.nan_to_num(1.//cltotEEArr)
                        preFX = trigfact*ell1*clunlenEEArrNow*np.nan_to_num(1./cltotEEArr)
                        preGX = trigfact*ell2*clunlenEEArr*np.nan_to_num(1./cltotEEArr)

                        allTerms += [2.*ell1*ell2*fft2(ifft2(preF)*ifft2(preG)+ifft2(preFX)*ifft2(preGX)/2.)]


            


        elif XY == 'EB':

            ########
            ## EB ##
            ########
            #check noise array!!!

            clunlenEEArrNow = self.uClNow2d['EE'].copy()
            clunlenEEArr = self.uClFid2d['EE'].copy()
            cltotEEArr =self.lClFid2d['EE'].copy() + self.noiseArray[1]
            cltotEEArr[np.where(lmap >= self.lmax_P)] = np.inf

            clunlenBBArrNow = self.uClNow2d['BB'].copy()
            clunlenBBArr = self.uClFid2d['BB'].copy()
            cltotBBArr =self.lClFid2d['BB'].copy() + self.noiseArray[2]
            cltotBBArr[np.where(lmap >= self.lmax_P)] = np.inf


            if True:
                if halo: clunlenEEArrNow[np.where(lmap >= self.gradCut)] = 0.

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
            
                for ellsq in [lx*lx,ly*ly,np.sqrt(2.)*lx*ly]:
                    preF = ellsq*clunlenEEArrNow*clunlenEEArr*np.nan_to_num(1./cltotEEArr)
                    preG = np.nan_to_num(1./cltotBBArr)

                    for termF,termG in zip(termsF,termsG):
                        allTerms += [ellsq*fft2(ifft2(termF(preF,lxhat,lyhat))*ifft2(termG(preG,lxhat,lyhat)))]
                    

        elif XY=='ET':

            clunlenTEArrNow = self.uClNow2d['TE'].copy()
            clunlenTEArr = self.uClFid2d['TE'].copy()

            cltotTTArr =self.lClFid2d['TT'].copy() + self.noiseArray[0]
            cltotTTArr[np.where(lmap >= self.lmax_T)] = np.inf
            cltotEEArr =self.lClFid2d['EE'].copy() + self.noiseArray[1]
            cltotEEArr[np.where(lmap >= self.lmax_P)] = np.inf





            if halo:
                sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
                cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

                lx = self.lxMap
                ly = self.lyMap

                clunlenTEArrNow[np.where(lmap >= self.gradCut)] = 0.

                lxhat = self.lxHatMap
                lyhat = self.lyHatMap

                sinf = sin2phi(lxhat,lyhat)
                sinsqf = sinf**2.
                cosf = cos2phi(lxhat,lyhat)
                cossqf = cosf**2.

                clunlenTEArrNow[np.where(lmap >= self.gradCut)] = 0.

                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTEArrNow*clunlenTEArr*np.nan_to_num(1./cltotEEArr)
                    preG = np.nan_to_num(1./cltotTTArr)
                    allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
                    for trigfact in [cosf,sinf]:

                        preFX = trigfact*ell1*clunlenTEArrNow*np.nan_to_num(1./cltotEEArr)
                        preGX = trigfact*ell2*clunlenTEArr*np.nan_to_num(1./cltotTTArr)

                        allTerms += [ell1*ell2*fft2(ifft2(preFX)*ifft2(preGX))]


            else:



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
                
                
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTEArrNow*clunlenTEArr*np.nan_to_num(1./cltotEEArr)
                    preG = np.nan_to_num(1./cltotTTArr)
                    allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
                    for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                        preF = np.nan_to_num(1./cltotEEArr)
                        preG = trigfact*ell1*ell2*clunlenTEArrNow*clunlenTEArr*np.nan_to_num(1./cltotTTArr)
                        allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
                    for trigfact in [cosf,sinf]:
                        
                        preFX = trigfact*ell1*clunlenTEArrNow*np.nan_to_num(1./cltotEEArr)
                        preGX = trigfact*ell2*clunlenTEArr*np.nan_to_num(1./cltotTTArr)

                        allTerms += [2.*ell1*ell2*fft2(ifft2(preFX)*ifft2(preGX))]

                    

        elif XY=='TE':

            clunlenTEArrNow = self.uClNow2d['TE'].copy()
            clunlenTEArr = self.uClFid2d['TE'].copy()

            cltotTTArr =self.lClFid2d['TT'].copy() + self.noiseArray[0] #
            cltotTTArr[np.where(lmap >= self.lmax_T)] = np.inf
            cltotEEArr =self.lClFid2d['EE'].copy() + self.noiseArray[1] #
            cltotEEArr[np.where(lmap >= self.lmax_P)] = np.inf






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
                
                clunlenTEArrNow[np.where(lmap >= self.gradCut)] = 0.
                
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                        preF = trigfact*ell1*ell2*clunlenTEArrNow*clunlenTEArr*np.nan_to_num(1./cltotTTArr)
                        preG = trigfact*np.nan_to_num(1./cltotEEArr)
                        allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
                    for trigfact in [cosf,sinf]:
                        
                        preFX = trigfact*ell1*clunlenTEArrNow*np.nan_to_num(1./cltotTTArr)
                        preGX = trigfact*ell2*clunlenTEArr*np.nan_to_num(1./cltotEEArr)

                        allTerms += [ell1*ell2*fft2(ifft2(preFX)*ifft2(preGX))]

                
            else:



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
                
                
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                        preF = trigfact*ell1*ell2*clunlenTEArrNow*clunlenTEArr*np.nan_to_num(1./cltotTTArr)
                        preG = trigfact*np.nan_to_num(1./cltotEEArr)
                        allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
                    preF = np.nan_to_num(1./cltotTTArr)
                    preG = ell1*ell2*clunlenTEArrNow*clunlenTEArr*np.nan_to_num(1./cltotEEArr)
                    allTerms += [ell1*ell2*fft2(ifft2(preF)*ifft2(preG))]
                    for trigfact in [cosf,sinf]:
                        
                        preFX = trigfact*ell1*clunlenTEArrNow*np.nan_to_num(1./cltotTTArr)
                        preGX = trigfact*ell2*clunlenTEArr*np.nan_to_num(1./cltotEEArr)

                        allTerms += [2.*ell1*ell2*fft2(ifft2(preFX)*ifft2(preGX))]


                

        elif XY == 'TB':

            ########
            ## TB ##
            ########
            #check noise array!!!
        
            clunlenTEArrNow = self.uClNow2d['TE'].copy()
            clunlenTEArr = self.uClFid2d['TE'].copy()


            clunlenTTArrNow = self.uClNow2d['TT'].copy()
            clunlenTTArr = self.uClFid2d['TT'].copy()
            cltotTTArr =self.lClFid2d['TT'].copy() + self.noiseArray[0]
            cltotTTArr[np.where(lmap >= self.lmax_T)] = np.inf

            clunlenBBArrNow = self.uClNow2d['BB'].copy()
            clunlenBBArr = self.uClFid2d['BB'].copy()
            cltotBBArr =self.lClFid2d['BB'].copy() + self.noiseArray[2]
            cltotBBArr[np.where(lmap >= self.lmax_P)] = np.inf



            if halo: clunlenTEArrNow[np.where(lmap >= self.gradCut)] = 0.
            
                
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
            
            for ellsq in [lx*lx,ly*ly,np.sqrt(2.)*lx*ly]:
                preF = ellsq*clunlenTEArrNow*clunlenTEArr*np.nan_to_num(1./cltotTTArr)
                preG = np.nan_to_num(1./cltotBBArr)

                for termF,termG in zip(termsF,termsG):
                    allTerms += [ellsq*fft2(ifft2(termF(preF,lxhat,lyhat))*ifft2(termG(preG,lxhat,lyhat)))]
                    

            


        else:
            print "ERROR: Unrecognized polComb"
            sys.exit(1)    
        
                        
        ALinv = np.real(np.sum( allTerms, axis = 0))
        NL = np.nan_to_num(lmap**2. * (lmap + 1.)**2. / 4. / ALinv)
        NL[np.where(np.logical_or(lmap >= self.bigell, lmap == 0.))] = 0.

        retval = np.nan_to_num(NL.real * self.template.pixScaleX*self.template.pixScaleY  )

        self.Nlkk = retval

        
        return np.nan_to_num(retval * 2. / lmap/(lmap+1.))
        
        
                  

        
      



    def delensClBB(self,binfile,halo=False):
        lmap = self.lmap

        clPPArr = self.clPPArr
        cltotPPArr = clPPArr + self.NlPPnowArr
        cltotPPArr[np.isnan(cltotPPArr)] = np.inf
        
        clunlenEEArr = self.uClFid2d['EE'].copy()
        clunlentotEEArr =self.uClFid2d['EE'].copy() + self.noiseArray[1]
        clunlentotEEArr[np.where(lmap >= self.lmax_P)] = np.inf
        clunlenEEArr[np.where(lmap >= self.lmax_P)] = 0.
        clPPArr[np.where(lmap >= self.lmax_P)] = 0.
        cltotPPArr[np.where(lmap >= self.lmax_P)] = np.inf

        if halo: clunlenEEArr[np.where(lmap >= self.gradCut)] = 0.
                
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

                allTerms += [trigfactOut*(fft2(ifft2(preF1)*ifft2(preG1) - ifft2(preF2)*ifft2(preG2)))]


        
        ClBBres = np.real(np.sum( allTerms, axis = 0))

        
        ClBBres[np.where(np.logical_or(self.lmap >= self.bigell, lmap == 0.))] = 0.
        ClBBres *= self.ftMap.Nx * self.ftMap.Ny #* (2.*np.pi)**2.
        ft = self.ftMap
        ClBBres[lmap>self.lmax_P]=0.
        
        ftHolder = self.ftMap.copy()
        ftHolder.kMap = np.sqrt(ClBBres)/self.ftMap.pixScaleX/self.ftMap.pixScaleY
        bbNoise2D = fftTools.powerFromFFT(ftHolder, ftHolder)
        self.lClFid2d['BB'] = bbNoise2D.powerMap
        lLower,lUpper,lBin,NlBinBB,clBinSd,binWeight = aveBinInAnnuli(bbNoise2D,binfile = binfile)

        return lBin,NlBinBB


                
    

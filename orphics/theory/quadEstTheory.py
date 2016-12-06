import numpy as np
import orphics.analysis.flatMaps as fmaps 

class QuadNorm(object):

    
    def __init__(self,templateMap,gradCut=None):
        '''

        templateFT is a template liteMap FFT object
    

    
        '''
        
        self.lx,self.ly,self.modLMap,self.thetaMap,dlx,dly = fmaps.getFTAttributesFromLiteMap(templateMap)
        self.lxHat = np.nan_to_num(self.lx / self.modLMap)
        self.lyHat = np.nan_to_num(self.ly / self.modLMap)

        self.uClNow2d = {}
        self.uClFid2d = {}
        self.lClFid2d = {}
        self.noiseXX2d = {}
        self.noiseYY2d = {}

        self.lmax_T=9000.
        self.lmax_P=9000.
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
        if fourierMask is not None: self.noiseXX[fourierMask==0] = np.inf
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
        if fourierMask is not None: self.noiseYY[fourierMask==0] = np.inf
        
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
        return self.uClFid2d[X+Y]/(self.lClFid2d[X+X]+self.noiseXX)        

    def WY(self,YY):
        return 1./(self.lClFid2d[YY]+self.noiseYY)        

    def getCurlNlkk2d(self,XY,halo=False):
        pass
            
    def getNlkk2d(self,XY,halo=False):
        lx,ly = self.lx,self.ly
        ftMap = self.template
        lmap = self.modLMap



            
        h=0.

        allTerms = []
            
        if XY == 'TT':
            
            clunlenTTArrNow = self.uClNow2d['TT'].copy()
            clunlenTTArr = self.uClFid2d['TT'].copy()

            cltotTTArrX =self.lClFid2d['TT'].copy() + self.noiseXX
            cltotTTArrX[np.where(lmap >= self.lmax_T)] = np.inf

            cltotTTArrY =self.lClFid2d['TT'].copy() + self.noiseYY
            cltotTTArrY[np.where(lmap >= self.lmax_T)] = np.inf

                

            if halo:
            
                clunlenTTArrNow[np.where(lmap >= self.gradCut)] = 0.

                
                preG = 1./cltotTTArrY
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTTArrNow*clunlenTTArr/cltotTTArrX
                    preFX = ell1*clunlenTTArrNow/cltotTTArrX
                    preGX = ell2*clunlenTTArr/cltotTTArrY
                    

                    calc = ell1*ell2*np.fft.fft2(np.fft.ifft2(preF)*np.fft.ifft2(preG)+np.fft.ifft2(preFX)*np.fft.ifft2(preGX))
                    allTerms += [calc]
                    

            else:
                preG = 1./cltotTTArrY
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTTArrNow*clunlenTTArr/cltotTTArrX/2.            
                    preFX = ell1*clunlenTTArrNow/cltotTTArrX
                    preGX = ell2*clunlenTTArr/cltotTTArrY

                    allTerms += [2.*ell1*ell2*np.fft.fft2(np.fft.ifft2(preF)*np.fft.ifft2(preG)+np.fft.ifft2(preFX)*np.fft.ifft2(preGX)/2.)]
          


        elif XY == 'EE':

            #check noise array!!!

            clunlenEEArrNow = self.uClNow2d['EE'].copy()
            clunlenEEArr = self.uClFid2d['EE'].copy()
            cltotEEArr =self.lClFid2d['EE'].copy() + self.noiseArray[1]
            cltotEEArr[np.where(lmap >= self.lmax_P)] = np.inf

            


            sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
            cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

            lx = self.lx
            ly = self.ly


            lxhat = self.lxHat
            lyhat = self.lyHat

            sinf = sin2phi(lxhat,lyhat)
            sinsqf = sinf**2.
            cosf = cos2phi(lxhat,lyhat)
            cossqf = cosf**2.
                                
            if halo:
            
                
                clunlenEEArrNow[np.where(lmap >= self.gradCut)] = 0.
                
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                        preF = trigfact*ell1*ell2*clunlenEEArrNow*clunlenEEArr/cltotEEArr
                        preG = trigfact/cltotEEArr
                        allTerms += [ell1*ell2*np.fft.fft2(np.fft.ifft2(preF)*np.fft.ifft2(preG))]
                        
                        preFX = trigfact*ell1*clunlenEEArrNow/cltotEEArr
                        preGX = trigfact*ell2*clunlenEEArr/cltotEEArr

                        allTerms += [ell1*ell2*np.fft.fft2(np.fft.ifft2(preFX)*np.fft.ifft2(preGX))]

                
            else:


                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                        preF = trigfact*ell1*ell2*clunlenEEArrNow*clunlenEEArr/cltotEEArr/2.            
                        preG = trigfact/cltotEEArr
                        preFX = trigfact*ell1*clunlenEEArrNow/cltotEEArr
                        preGX = trigfact*ell2*clunlenEEArr/cltotEEArr

                        allTerms += [2.*ell1*ell2*np.fft.fft2(np.fft.ifft2(preF)*np.fft.ifft2(preG)+np.fft.ifft2(preFX)*np.fft.ifft2(preGX)/2.)]


            


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

                lx = self.lx
                ly = self.ly

                termsF = []
                termsF.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
                termsF.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
                termsF.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )

                termsG = []
                termsG.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
                termsG.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
                termsG.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )
            
                lxhat = self.lxHat
                lyhat = self.lyHat
            
                for ellsq in [lx*lx,ly*ly,np.sqrt(2.)*lx*ly]:
                    preF = ellsq*clunlenEEArrNow*clunlenEEArr/cltotEEArr
                    preG = 1./cltotBBArr

                    for termF,termG in zip(termsF,termsG):
                        allTerms += [ellsq*np.fft.fft2(np.fft.ifft2(termF(preF,lxhat,lyhat))*np.fft.ifft2(termG(preG,lxhat,lyhat)))]
                    

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

                lx = self.lx
                ly = self.ly

                clunlenTEArrNow[np.where(lmap >= self.gradCut)] = 0.

                lxhat = self.lxHat
                lyhat = self.lyHat

                sinf = sin2phi(lxhat,lyhat)
                sinsqf = sinf**2.
                cosf = cos2phi(lxhat,lyhat)
                cossqf = cosf**2.

                clunlenTEArrNow[np.where(lmap >= self.gradCut)] = 0.

                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTEArrNow*clunlenTEArr/cltotEEArr
                    preG = 1./cltotTTArr
                    allTerms += [ell1*ell2*np.fft.fft2(np.fft.ifft2(preF)*np.fft.ifft2(preG))]
                    for trigfact in [cosf,sinf]:

                        preFX = trigfact*ell1*clunlenTEArrNow/cltotEEArr
                        preGX = trigfact*ell2*clunlenTEArr/cltotTTArr

                        allTerms += [ell1*ell2*np.fft.fft2(np.fft.ifft2(preFX)*np.fft.ifft2(preGX))]


            else:



                sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
                cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

                lx = self.lx
                ly = self.ly

            
                lxhat = self.lxHat
                lyhat = self.lyHat

                sinf = sin2phi(lxhat,lyhat)
                sinsqf = sinf**2.
                cosf = cos2phi(lxhat,lyhat)
                cossqf = cosf**2.
                
                
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTEArrNow*clunlenTEArr/cltotEEArr
                    preG = 1./cltotTTArr
                    allTerms += [ell1*ell2*np.fft.fft2(np.fft.ifft2(preF)*np.fft.ifft2(preG))]
                    for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                        preF = 1./cltotEEArr
                        preG = trigfact*ell1*ell2*clunlenTEArrNow*clunlenTEArr/cltotTTArr
                        allTerms += [ell1*ell2*np.fft.fft2(np.fft.ifft2(preF)*np.fft.ifft2(preG))]
                    for trigfact in [cosf,sinf]:
                        
                        preFX = trigfact*ell1*clunlenTEArrNow/cltotEEArr
                        preGX = trigfact*ell2*clunlenTEArr/cltotTTArr

                        allTerms += [2.*ell1*ell2*np.fft.fft2(np.fft.ifft2(preFX)*np.fft.ifft2(preGX))]

                    

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

                lx = self.lx
                ly = self.ly

            
                lxhat = self.lxHat
                lyhat = self.lyHat

                sinf = sin2phi(lxhat,lyhat)
                sinsqf = sinf**2.
                cosf = cos2phi(lxhat,lyhat)
                cossqf = cosf**2.
                
                clunlenTEArrNow[np.where(lmap >= self.gradCut)] = 0.
                
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                        preF = trigfact*ell1*ell2*clunlenTEArrNow*clunlenTEArr/cltotTTArr
                        preG = trigfact/cltotEEArr
                        allTerms += [ell1*ell2*np.fft.fft2(np.fft.ifft2(preF)*np.fft.ifft2(preG))]
                    for trigfact in [cosf,sinf]:
                        
                        preFX = trigfact*ell1*clunlenTEArrNow/cltotTTArr
                        preGX = trigfact*ell2*clunlenTEArr/cltotEEArr

                        allTerms += [ell1*ell2*np.fft.fft2(np.fft.ifft2(preFX)*np.fft.ifft2(preGX))]

                
            else:



                sin2phi = lambda lxhat,lyhat: (2.*lxhat*lyhat)
                cos2phi = lambda lxhat,lyhat: (lyhat*lyhat-lxhat*lxhat)

                lx = self.lx
                ly = self.ly

            
                lxhat = self.lxHat
                lyhat = self.lyHat

                sinf = sin2phi(lxhat,lyhat)
                sinsqf = sinf**2.
                cosf = cos2phi(lxhat,lyhat)
                cossqf = cosf**2.
                
                
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    for trigfact in [cossqf,sinsqf,np.sqrt(2.)*sinf*cosf]:
                        preF = trigfact*ell1*ell2*clunlenTEArrNow*clunlenTEArr/cltotTTArr
                        preG = trigfact/cltotEEArr
                        allTerms += [ell1*ell2*np.fft.fft2(np.fft.ifft2(preF)*np.fft.ifft2(preG))]
                    preF = 1./cltotTTArr
                    preG = ell1*ell2*clunlenTEArrNow*clunlenTEArr/cltotEEArr
                    allTerms += [ell1*ell2*np.fft.fft2(np.fft.ifft2(preF)*np.fft.ifft2(preG))]
                    for trigfact in [cosf,sinf]:
                        
                        preFX = trigfact*ell1*clunlenTEArrNow/cltotTTArr
                        preGX = trigfact*ell2*clunlenTEArr/cltotEEArr

                        allTerms += [2.*ell1*ell2*np.fft.fft2(np.fft.ifft2(preFX)*np.fft.ifft2(preGX))]


                

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

            lx = self.lx
            ly = self.ly

            termsF = []
            termsF.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsF.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )

            termsG = []
            termsG.append( lambda pre,lxhat,lyhat: pre * cos2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * sin2phi(lxhat,lyhat)**2. )
            termsG.append( lambda pre,lxhat,lyhat: pre * (1.j*np.sqrt(2.)*sin2phi(lxhat,lyhat)*cos2phi(lxhat,lyhat)) )
            
            lxhat = self.lxHat
            lyhat = self.lyHat
            
            for ellsq in [lx*lx,ly*ly,np.sqrt(2.)*lx*ly]:
                preF = ellsq*clunlenTEArrNow*clunlenTEArr/cltotTTArr
                preG = 1./cltotBBArr

                for termF,termG in zip(termsF,termsG):
                    allTerms += [ellsq*np.fft.fft2(np.fft.ifft2(termF(preF,lxhat,lyhat))*np.fft.ifft2(termG(preG,lxhat,lyhat)))]
                    

            


        else:
            print "ERROR: Unrecognized polComb"
            sys.exit(1)    
        
                        
        ALinv = np.real(np.sum( allTerms, axis = 0)) 
        NL = lmap**2. * (lmap + 1.)**2 / 4. / ALinv
        NL[np.where(np.logical_or(lmap >= self.bigell, lmap == 0.))] = 0.
        NL *= ftMap.Nx * ftMap.Ny
        ftHolder = ftMap.copy()
        ftHolder.kMap = np.sqrt(NL)
        kappaNoise2D = fftTools.powerFromFFT(ftHolder, ftHolder)
        #self.NlPPnowArr = kappaNoise2D.powerMap[:] *4./lmap**2./(lmap+1.)**2.



        
        return kappaNoise2D.powerMap
        
        
                  

        
      



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

        lx = self.lx
        ly = self.ly

            
        lxhat = self.lxHat
        lyhat = self.lyHat

        sinf = sin2phi(lxhat,lyhat)
        sinsqf = sinf**2.
        cosf = cos2phi(lxhat,lyhat)
        cossqf = cosf**2.

        
        allTerms = []
        for ellsq in [lx*lx,ly*ly,np.sqrt(2.)*lx*ly]:
            for trigfactOut,trigfactIn in zip([sinsqf,cossqf,1.j*np.sqrt(2.)*sinf*cosf],[cossqf,sinsqf,1.j*np.sqrt(2.)*sinf*cosf]):
                preF1 = trigfactIn*ellsq*clunlenEEArr
                preG1 = ellsq*clPPArr

                preF2 = trigfactIn*ellsq*clunlenEEArr**2./clunlentotEEArr
                preG2 = ellsq*clPPArr**2./cltotPPArr

                allTerms += [trigfactOut*(np.fft.fft2(np.fft.ifft2(preF1)*np.fft.ifft2(preG1) - np.fft.ifft2(preF2)*np.fft.ifft2(preG2)))]


        
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


                
    

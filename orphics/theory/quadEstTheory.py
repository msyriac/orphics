

class QuadNorm(object):

    
    def __init__(self,templateMap,gradCut=10000.):
        '''

        templateFT is a template liteMap FFT object
    

    
        '''
        
        self.uClNow2d = {}
        self.uClFid2d = {}
        self.lClFid2d = {}
        

    def addUnlensedFilter2DPower(self,XY,power2d):
        '''
        XY = TT, TE, EE, EB or TB
        power2d is a flipper power2d object
        These Cls belong in the Wiener filters, and will not
        be perturbed if/when calculating derivatives.
        '''
    def addUnlensedNorm2DPower(self,XY,power2d):
        '''
        XY = TT, TE, EE, EB or TB
        power2d is a flipper power2d object
        These Cls belong in the CMB normalization, and will
        be perturbed if/when calculating derivatives.
        '''
    def addLensedFilter2DPower(self,XY,power2d):
        '''
        XY = TT, TE, EE, EB or TB
        power2d is a flipper power2d object
        These Cls belong in the Wiener filters, and will not
        be perturbed if/when calculating derivatives.
        '''
    def addNoise2DPowerXX(self,XX,power2d,fourierMask):
        '''
        Noise power for the X leg of the quadratic estimator
        XX = TT, EE, BB
        power2d is a flipper power2d object
        fourierMask is an int array that is 0 where noise is
        infinite and 1 where not. It should be in the same
        fftshift state as power2d.powerMap
        '''
        
    def addNoise2DPowerYY(self,YY,power2d,fourierMask):
        '''
        Noise power for the Y leg of the quadratic estimator
        XX = TT, EE, BB
        power2d is a flipper power2d object
        fourierMask is an int array that is 0 where noise is
        infinite and 1 where not. It should be in the same
        fftshift state as power2d.powerMap
        '''
        
    def addClkk2DPower(self,power2d):
        '''
        Fiducial Clkk power
        Used if delensing
        power2d is a flipper power2d object            
        '''

            
        #self._TCMB = 2.7255e6
    

        self.ftMap = fftTools.fftFromLiteMap(templateMap)
        self.lx, self.ly = np.meshgrid(self.ftMap.lx, self.ftMap.ly)
        self.lmap = self.ftMap.modLMap
        self.lxHat = np.nan_to_num(self.lx / self.lmap)
        self.lyHat = np.nan_to_num(self.ly / self.lmap)


        self.templatePower = fftTools.powerFromLiteMap(templateMap)
        self.templatePower.powerMap *= 0.

        self._Noisy = False
        self.lmax_T = lmax_T
        self.lmax_P = lmax_P
        self.bigell = 2.*max(ellsUNow[-1],ellsUFid[-1],ellsLFid[-1])
        self.gradCut = gradCut
        

        # Initialize a 2d power map for the normalization
        for key in self._c:
            

            if uClsNow[self._c[key]] == None: continue
            
            self.uClNow2d[key] = makeTemplate(ellsUNow,uClsNow[self._c[key]],self.templatePower)
            self.uClFid2d[key] = makeTemplate(ellsUFid,uClsFid[self._c[key]],self.templatePower)
            self.lClFid2d[key] = makeTemplate(ellsLFid,lClsFid[self._c[key]],self.templatePower)
            
        

            self.clPPArr = makeTemplate(elldd,clpp,self.templatePower)

            # pl = Plotter(scaleX='log',scaleY='log')
            # pl.add(elldd,clpp)
            # pl.done('output/pp1d.png')

                
    def addPickled1DNoise(self,noiseFileT,noiseFileE,ellCutTuple,fgPowerFile=None,noiseCuts = None):
        TCMB = self._TCMB

        self._Noisy = True

        
        ellT, f_ellT = np.loadtxt(noiseFileT,usecols=[0,1],unpack=True)        
        ellE, f_ellE = np.loadtxt(noiseFileE,usecols=[0,1],unpack=True)        


        lxcut,lycut,lmin,lmax = ellCutTuple

        if noiseCuts!=None:
            lminT,lmaxT,lminE,lmaxE = noiseCuts
        else:
            lminT = lmin
            lminE = lmin
            lmaxT = lmax
            lmaxE = lmax    

        
        p2dNoise = self.templatePower.copy()
        noiseForFilter1 = p2dNoise.powerMap.copy()   # template noise map
        noiseForFilter2 = p2dNoise.powerMap.copy()
        
        noiseForFilter1[:] = 1. / (TCMB )**2    # add instrument noise to noise power map
        noiseForFilter2[:] = 1. / (TCMB )**2    # add instrument noise to noise power map
        
        filt2dT =   makeTemplate(ellT,f_ellT,p2dNoise)  # make template noise map
        filt2dE =   makeTemplate(ellE,f_ellE,p2dNoise)  # make template noise map

        #self.nonwhite = (ellT,f_ellT*noiseForFilter1[0,0])
        filterMaskT = fourierMask(p2dNoise, lxcut = lxcut, lycut = lycut, lmin = lminT, lmax = lmaxT)
        filterMaskE = fourierMask(p2dNoise, lxcut = lxcut, lycut = lycut, lmin = lminE, lmax = lmaxE)
        filt2dT[filterMaskT == 0] =  1.e90
        filt2dE[filterMaskE == 0] =  1.e90

        # pl = Plotter(figsize=(20, 20),dpi=400)
        # pl.plot2d(fftshift(filterMask))
        # pl.done("fmask.png")
        # sys.exit()
    
        noiseForFilter1[:] = noiseForFilter1[:] * filt2dT[:]
        noiseForFilter2[:] = noiseForFilter2[:] * filt2dE[:]


        if fgPowerFile!=None:
            fgPower = numpy.loadtxt(fgPowerFile)
            (ls_fg, fgPower) = numpy.loadtxt(fgPowerFile).transpose()
            fgPower2d = makeTemplate(ls_fg, fgPower / TCMB**2, p2dNoise)
            noiseForFilter1 += fgPower2d
            noiseForFilter2 += fgPower2d
                                 

        self.noiseArray = [noiseForFilter1,noiseForFilter2.copy(),noiseForFilter2.copy()]



    def addACT2DNoise(self,beamFileTxt,noisePower2DPklTT,noisePower2DPklEE,noisePower2DPklBB,ellCutTuple):
        self._Noisy = True
        lxcut,lycut,lmin,lmax = ellCutTuple

        TCMB = self._TCMB
        ell, f_ell = np.transpose(np.loadtxt(beamFileTxt))[0:2,:]
        filt = 1./(np.array(f_ell)**2.)       # add beam to filter
        p2dNoise = self.templatePower.copy()
        filt2d =   makeTemplate(ell,filt,p2dNoise)  # make template noise map       
        filterMask = fourierMask(p2dNoise, lxcut = lxcut, lycut = lycut, lmin = lmin, lmax = lmax)
        filt2d[filterMask == 0] =  1.e90
     
        noise2dTT = fftshift(pickle.load(open(noisePower2DPklTT,'r'))) * (np.pi / (180. * 60))**2./(TCMB)**2.
        noise2dEE = fftshift(pickle.load(open(noisePower2DPklEE,'r')))* (np.pi / (180. * 60))**2./(TCMB)**2.
        noise2dBB = fftshift(pickle.load(open(noisePower2DPklBB,'r')))* (np.pi / (180. * 60))**2./(TCMB)**2.

    
        noiseForFilter1 = noise2dTT[:] * filt2d[:]
        noiseForFilter2 = noise2dEE[:] * filt2d[:]
        noiseForFilter3 = noise2dBB[:] * filt2d[:]

        self.noiseArray = [noiseForFilter1,noiseForFilter2,noiseForFilter3]

    def addWhiteNoiseBeamFile(self,noiseLevelT,noiseLevelP,beamFile,ellCutTuple):
        TCMB = self._TCMB

        self._Noisy = True

        


        ell, f_ell = np.transpose(np.loadtxt(beamFile))[0:2,:]
        lxcut,lycut,lmin,lmax = ellCutTuple

        filt = 1./(np.array(f_ell)**2.)       # add beam to filter


        p2dNoise = self.templatePower.copy()
        noiseForFilter1 = p2dNoise.powerMap.copy()   # template noise map                                                                                                                                                            
        noiseForFilter2 = p2dNoise.powerMap.copy()
        noiseForFilter1[:] = (np.pi / (180. * 60))**2 / (TCMB )**2 * noiseLevelT**2   # add instrument noise to noise power map                                                                                                       
        noiseForFilter2[:] = (np.pi / (180. * 60))**2 / (TCMB )**2 * noiseLevelP**2   # add instrument noise to noise power map

        self.NlTT = (ell,filt*(np.pi / (180. * 60))**2. / (TCMB )**2. * noiseLevelT**2.)
        self.NlEE = (ell,filt*(np.pi / (180. * 60))**2. / (TCMB )**2. * noiseLevelP**2.)
        
        filt2d =   makeTemplate(ell,filt,p2dNoise)  # make template noise map

        #self.white = (ell,filt*noiseForFilter1[0,0])
    
        filterMask = fourierMask(p2dNoise, lxcut = lxcut, lycut = lycut, lmin = lmin, lmax = lmax)
        filt2d[filterMask == 0] =  1.e90
    
        noiseForFilter1[:] = noiseForFilter1[:] * filt2d[:]
        noiseForFilter2[:] = noiseForFilter2[:] * filt2d[:]

        self.noiseArray = [noiseForFilter1,noiseForFilter2.copy(),noiseForFilter2.copy()]


    def addWhiteNoise(self,noiseLevelT,noiseLevelP,beamArcmin,ellCutTuple,noiseCuts = None):
        TCMB = self._TCMB

        self._Noisy = True

        


        Sigma = beamArcmin *np.pi/60./180./ np.sqrt(8.*np.log(2.))     # radian
        ell = np.arange(0.,25000.,1.)
        filt = np.exp(ell*ell*Sigma*Sigma)

        #ell, f_ell = np.transpose(np.loadtxt(beamFile))[0:2,:]
        #filt = 1./(np.array(f_ell)**2.)       # add beam to filter
        
        lxcut,lycut,lmin,lmax = ellCutTuple


        if noiseCuts!=None:
            lminT,lmaxT,lminE,lmaxE = noiseCuts
        else:
            lminT = lmin
            lminE = lmin
            lmaxT = lmax
            lmaxE = lmax    

        p2dNoise = self.templatePower.copy()
        noiseForFilter1 = p2dNoise.powerMap.copy()   # template noise map
        noiseForFilter2 = p2dNoise.powerMap.copy()

        
        
        noiseForFilter1[:] = (np.pi / (180. * 60))**2 / (TCMB )**2 * noiseLevelT**2   # add instrument noise to noise power map                                                                                                       
        noiseForFilter2[:] = (np.pi / (180. * 60))**2 / (TCMB )**2 * noiseLevelP**2   # add instrument noise to noise power map

        self.NlTT = (ell,filt*(np.pi / (180. * 60))**2 / (TCMB )**2 * noiseLevelT**2)
        self.NlEE = (ell,filt*(np.pi / (180. * 60))**2 / (TCMB )**2 * noiseLevelP**2)

        
        filt2dT =   makeTemplate(ell,filt,p2dNoise)#,debug=True)  # make template noise map
        filt2dE =   makeTemplate(ell,filt,p2dNoise)#,debug=True)  # make template noise map

        #self.white = (ell,filt*noiseForFilter1[0,0])
    
        filterMaskT = fourierMask(p2dNoise, lxcut = lxcut, lycut = lycut, lmin = lminT, lmax = lmaxT)
        filterMaskE = fourierMask(p2dNoise, lxcut = lxcut, lycut = lycut, lmin = lminE, lmax = lmaxE)
        filt2dT[filterMaskT == 0] =  np.inf #1.e90
        filt2dE[filterMaskE == 0] =  np.inf #1.e90

        
        noiseForFilter1[:] = noiseForFilter1[:] * filt2dT[:]
        noiseForFilter2[:] = noiseForFilter2[:] * filt2dE[:]

        self.noiseArray = [noiseForFilter1,noiseForFilter2.copy(),noiseForFilter2.copy()]


        
    
    def getNlkk(self,binEdges=None,shiftEllMin=None,shiftEllMax=None,shiftH=0.,shiftCl=0,inverted=True,halo=False):
        XY = self._XY
        c = self._c
        lx,ly = self.lx,self.ly
        ftMap = self.ftMap
        lmap = self.lmap



            
        B=0.
        a = time.time()
            
        h=0.

        allTerms = []
            
        if XY == 'TT':
            
            clunlenTTArrNow = self.uClNow2d['TT'].copy()
            clunlenTTArr = self.uClFid2d['TT'].copy()
            cltotTTArr =self.lClFid2d['TT'].copy() + self.noiseArray[0]
            cltotTTArr[np.where(lmap >= self.lmax_T)] = np.inf

            # For derivatives
            if shiftEllMin!=None and shiftEllMax!=None:
                #print "shifting"
                annulus = clunlenTTArrNow[np.logical_and(self.lmap>shiftEllMin,self.lmap<shiftEllMax)].copy()
                annmean = annulus.mean()
                clunlenTTArrNow[np.logical_and(self.lmap>shiftEllMin,self.lmap<shiftEllMax)] = annulus + 0.5*shiftH * annmean
                h = shiftH*annmean



                
            lx = self.lx
            ly = self.ly


            if halo:
            
                clunlenTTArrNow[np.where(lmap >= self.gradCut)] = 0.

                
                preG = 1./cltotTTArr
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTTArrNow*clunlenTTArr/cltotTTArr
                    preFX = ell1*clunlenTTArrNow/cltotTTArr
                    preGX = ell2*clunlenTTArr/cltotTTArr
                    

                    calc = ell1*ell2*np.fft.fft2(np.fft.ifft2(preF)*np.fft.ifft2(preG)+np.fft.ifft2(preFX)*np.fft.ifft2(preGX))
                    allTerms += [calc]
                    

            else:
                preG = 1./cltotTTArr
                rfact = 2.**0.25
                for ell1,ell2 in [(lx,lx),(ly,ly),(rfact*lx,rfact*ly)]:
                    preF = ell1*ell2*clunlenTTArrNow*clunlenTTArr/cltotTTArr/2.            
                    preFX = ell1*clunlenTTArrNow/cltotTTArr
                    preGX = ell2*clunlenTTArr/cltotTTArr

                    allTerms += [2.*ell1*ell2*np.fft.fft2(np.fft.ifft2(preF)*np.fft.ifft2(preG)+np.fft.ifft2(preFX)*np.fft.ifft2(preGX)/2.)]
          


        elif XY == 'EE':

            #check noise array!!!

            clunlenEEArrNow = self.uClNow2d['EE'].copy()
            clunlenEEArr = self.uClFid2d['EE'].copy()
            cltotEEArr =self.lClFid2d['EE'].copy() + self.noiseArray[1]
            cltotEEArr[np.where(lmap >= self.lmax_P)] = np.inf

            
            # For derivatives
            if shiftEllMin!=None and shiftEllMax!=None:
                #print "shifting"
                annulus = clunlenEEArrNow[np.logical_and(self.lmap>shiftEllMin,self.lmap<shiftEllMax)].copy()
                annmean = annulus.mean()
                clunlenEEArrNow[np.logical_and(self.lmap>shiftEllMin,self.lmap<shiftEllMax)] = annulus + 0.5*shiftH * annmean
                h = shiftH*annmean


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

            if shiftEllMin!=None and shiftEllMax!=None:
                #print "shifting"
                if shiftCl==0:
                    #print "THIS SHOULDNT BE PRINTING!"
                    annulus = clunlenEEArrNow[np.logical_and(self.lmap>shiftEllMin,self.lmap<shiftEllMax)].copy()
                    annmean = annulus.mean()
                    clunlenEEArrNow[np.logical_and(self.lmap>shiftEllMin,self.lmap<shiftEllMax)] = annulus + 0.5*shiftH * annmean
                else:
                    #print "ok"
                    annulus = clunlenBBArrNow[np.logical_and(self.lmap>shiftEllMin,self.lmap<shiftEllMax)].copy()
                    annmean = annulus.mean()
                    clunlenBBArrNow[np.logical_and(self.lmap>shiftEllMin,self.lmap<shiftEllMax)] = annulus + 0.5*shiftH * annmean    
                h = shiftH*annmean



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


            if shiftEllMin!=None and shiftEllMax!=None:
                annulus = clunlenTEArrNow[np.logical_and(self.lmap>shiftEllMin,self.lmap<shiftEllMax)].copy()
                annmean = annulus.mean()
                clunlenTEArrNow[np.logical_and(self.lmap>shiftEllMin,self.lmap<shiftEllMax)] = annulus + 0.5*shiftH * annmean
                h = shiftH*annmean




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


            if shiftEllMin!=None and shiftEllMax!=None:
                annulus = clunlenTEArrNow[np.logical_and(self.lmap>shiftEllMin,self.lmap<shiftEllMax)].copy()
                annmean = annulus.mean()
                clunlenTEArrNow[np.logical_and(self.lmap>shiftEllMin,self.lmap<shiftEllMax)] = annulus + 0.5*shiftH * annmean
                h = shiftH*annmean




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

            if shiftEllMin!=None and shiftEllMax!=None:
                annulus = clunlenTEArrNow[np.logical_and(self.lmap>shiftEllMin,self.lmap<shiftEllMax)].copy()
                annmean = annulus.mean()
                clunlenTEArrNow[np.logical_and(self.lmap>shiftEllMin,self.lmap<shiftEllMax)] = annulus + 0.5*shiftH * annmean
                h = shiftH*annmean


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
        ftHolder = self.ftMap.copy()
        ftHolder.kMap = np.sqrt(NL)
        kappaNoise2D = fftTools.powerFromFFT(ftHolder, ftHolder)
        self.NlPPnowArr = kappaNoise2D.powerMap[:] *4./lmap**2./(lmap+1.)**2.

        if inverted:
            kappaNoise2D.powerMap[:] = 1. / kappaNoise2D.powerMap[:]
      

        ## Bin the 2D p.s. in annuli to obtain 1D curve. 
        lLower,lUpper,lBin,NlBinKK,clBinSd,binWeight = aveBinInAnnuli(kappaNoise2D,binfile = binfile)
        Btime = time.time()
        B += time.time() - Btime
            
        if verbose: print 'Total time for N(L) estimation: %.1f seconds' % (time.time() - a)

        
        return lBin, NlBinKK, h
        
        
                  

        
      



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


                
    

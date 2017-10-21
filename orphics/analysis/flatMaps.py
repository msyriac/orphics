import numpy as np
import copy
from scipy.interpolate import splrep,splev
from scipy.fftpack import fftshift
from scipy.interpolate import RectBivariateSpline,interp2d,interp1d
from orphics.tools.stats import timeit
import orphics.tools.cmb as cmb
import flipper.fftTools as ft
try:
    from flipper.fft import fft,ifft
except:
    import logging
    logging.warning("You seem to be using a version of flipper that does not have the 'flipper.fft' module, a wrapper for fast multi-threaded pyfftw FFTs. Please use the version of flipper maintained by the ACT Collaboration at https://github.com/ACTCollaboration/flipper .")


try:
    from enlib import enmap, lensing
except:
    import logging
    logging.warning("Couldn't load enlib. Some functionality may be missing.")


    
    
def get_taper(shape,taper_percent = 12.0,pad_percent = 3.0,weight=None):
    Ny,Nx = shape[-2:]
    if weight is None: weight = np.ones(shape[-2:])
    taper = cosineWindow(Ny,Nx,lenApodY=int(taper_percent*min(Ny,Nx)/100.),lenApodX=int(taper_percent*min(Ny,Nx)/100.),padY=int(pad_percent*min(Ny,Nx)/100.),padX=int(pad_percent*min(Ny,Nx)/100.))*weight
    w2 = np.mean(taper**2.)
    return taper,w2

    
class QuickPower(object):
    def __init__(self,modlmap,bin_edges):
        import orphics.tools.stats as stats
        self.binner = stats.bin2D(modlmap,bin_edges)
    def calc(self,map1,map2=None):
        p2d = get_simple_power_enmap(map1,enmap2=map2)
        cents, p1d = self.binner.bin(p2d)
        return cents, p1d
    

def minimum_ell(shape,wcs):
    """
    Returns the lowest angular wavenumber of an ndmap
    rounded down to the nearest integer.
    """
    modlmap = enmap.modlmap(shape,wcs)
    min_ell = modlmap[modlmap>0].min()
    return int(min_ell)
    

class PatchArray(object):
    def __init__(self,shape,wcs,dimensionless=False,TCMB=2.7255e6,cc=None,theory=None,lmax=None,skip_real=False,orphics_is_dimensionless=True):
        self.shape = shape
        self.wcs = wcs
        if not(skip_real): self.modrmap = enmap.modrmap(shape,wcs)
        self.lxmap,self.lymap,self.modlmap,self.angmap,self.lx,self.ly = get_ft_attributes_enmap(shape,wcs)
        self.pix_ells = np.arange(0.,self.modlmap.max(),1.)
        self.posmap = enmap.posmap(self.shape,self.wcs)
        self.dimensionless = dimensionless
        self.TCMB = TCMB

        if (theory is not None) or (cc is not None):
            self.add_theory(cc,theory,lmax,orphics_is_dimensionless)

    def add_theory(self,cc=None,theory=None,lmax=None,orphics_is_dimensionless=True):
        if cc is not None:
            self.cc = cc
            self.theory = cc.theory
            self.lmax = cc.lmax
            # assert theory is None
            # assert lmax is None
            if theory is None: theory = self.theory
            if lmax is None: lmax = self.lmax
        else:
            assert theory is not None
            assert lmax is not None
            self.theory = theory
            self.lmax = lmax
            
        #psl = cmb.enmap_power_from_orphics_theory(theory,lmax,lensed=True,dimensionless=self.dimensionless,TCMB=self.TCMB,orphics_dimensionless=orphics_is_dimensionless)
        psu = cmb.enmap_power_from_orphics_theory(theory,lmax,lensed=False,dimensionless=self.dimensionless,TCMB=self.TCMB,orphics_dimensionless=orphics_is_dimensionless)
        self.fine_ells = np.arange(0,lmax,1)
        pclkk = theory.gCl("kk",self.fine_ells)
        self.clkk = pclkk.copy()
        pclkk = pclkk.reshape((1,1,pclkk.size))
        #self.pclkk.resize((1,self.pclkk.size))
        self.ugenerator = enmap.MapGen(self.shape,self.wcs,psu)
        self.kgenerator = enmap.MapGen(self.shape[-2:],self.wcs,pclkk)


    def update_kappa(self,kappa):
        # Converts kappa map to pixel displacements
        import alhazen.lensTools as lt
        fphi = lt.kappa_to_fphi(kappa,self.modlmap)
        grad_phi = enmap.gradf(enmap.ndmap(fphi,self.wcs))
        pos = self.posmap + grad_phi
	self._displace_pix = enmap.sky2pix(self.shape,self.wcs,pos, safe=False)

    def get_lensed(self, unlensed, order=3, mode="spline", border="cyclic"):
        return lensing.displace_map(unlensed, self._displace_pix, order=order, mode=mode, border=border)


    def get_unlensed_cmb(self,seed=None,scalar=False):
        return self.ugenerator.get_map(seed=seed,scalar=scalar)
        
    def get_grf_kappa(self,seed=None,skip_update=False):
        kappa = self.kgenerator.get_map(seed=seed,scalar=True)
        if not(skip_update): self.update_kappa(kappa)
        return kappa


    def _fill_beam(self,beam_func):
        self.lbeam = beam_func(self.modlmap)
        self.lbeam[self.modlmap<2] = 1.
        
    def add_gaussian_beam(self,fwhm):
        if fwhm<1.e-5:
            bfunc = lambda x: x*0.+1.
        else:
            bfunc = lambda x : cmb.gauss_beam(x,fwhm)
        self._fill_beam(bfunc)
        
    def add_1d_beam(self,ells,bls,fill_value="extrapolate"):
        bfunc = interp1d(ells,bls,fill_value=fill_value)
        self._fill_beam(bfunc)

    def add_2d_beam(self,beam_2d):
        self.lbeam = beam_2d

    def add_white_noise_with_atm(self,noise_uK_arcmin_T,noise_uK_arcmin_P=None,lknee_T=0.,alpha_T=0.,lknee_P=0.,
                        alpha_P=0.):

        map_dimensionless=self.dimensionless
        TCMB=self.TCMB

        
        self.nT = cmb.white_noise_with_atm_func(self.modlmap,noise_uK_arcmin_T,lknee_T,alpha_T,
                                      map_dimensionless,TCMB)
        
        if noise_uK_arcmin_P is None and np.isclose(lknee_T,lknee_P) and np.isclose(alpha_T,alpha_P):
            self.nP = 2.*self.nT.copy()
        else:
            if noise_uK_arcmin_P is None: noise_uK_arcmin_P = np.sqrt(2.)*noise_uK_arcmin_T
            self.nP = cmb.white_noise_with_atm_func(self.modlmap,noise_uK_arcmin_P,lknee_P,alpha_P,
                                      map_dimensionless,TCMB)

        TCMBt = TCMB if map_dimensionless else 1.
        ps_noise = np.zeros((3,3,self.pix_ells.size))
        ps_noise[0,0] = self.pix_ells*0.+(noise_uK_arcmin_T*np.pi/180./60./TCMBt)**2.
        ps_noise[1,1] = self.pix_ells*0.+(noise_uK_arcmin_P*np.pi/180./60./TCMBt)**2.
        ps_noise[2,2] = self.pix_ells*0.+(noise_uK_arcmin_P*np.pi/180./60./TCMBt)**2.
        noisecov = ps_noise
        self.is_2d_noise = False
        self.ngenerator = enmap.MapGen(self.shape,self.wcs,noisecov)

            
    def add_noise_2d(self,nT,nP=None):
        self.nT = nT
        if nP is None: nP = 2.*nT
        self.nP = nP

        ps_noise = np.zeros((3,3,self.modlmap.shape[0],self.modlmap.shape[1]))
        ps_noise[0,0] = nT
        ps_noise[1,1] = nP
        ps_noise[2,2] = nP
        noisecov = ps_noise
        self.is_2d_noise = True
        self.ngenerator = enmap.MapGen(self.shape,self.wcs,noisecov)


    def get_noise_sim(self,seed=None,scalar=False):
        return self.ngenerator.get_map(seed=seed,scalar=scalar)
    
    
def pixel_window_function(modLMap,thetaMap,pixScaleX,pixScaleY):
    from scipy.special import j0
    #return j0(modLMap*pixScaleX*np.cos(thetaMap)/2.)*j0(modLMap*pixScaleY*np.sin(thetaMap)/2.) # are cos and sin orders correct?
    return np.sinc(modLMap*pixScaleX*np.cos(thetaMap)/2./np.pi)*np.sinc(modLMap*pixScaleY*np.sin(thetaMap)/2./np.pi) # are cos and sin orders correct?


def pixwin(l, pixsize):
    # pixsize = 0.5 arcmin for ACT
    pixsize = pixsize  * np.pi / (60. * 180)
    return np.sinc(l * pixsize / (2. * np.pi))**2


# def power_from_map_attributes(kmap1,kmap2):
#     kmap1 = fft(map1,axes=[-2,-1])
#     area =Nx*Ny*pixScaleX*pixScaleY
#     return np.real(np.conjugate(kmap1)*kmap2)*area/(Nx*Ny*1.0)**2.

#@timeit
def get_simple_power(map1,mask1=1.,map2=None,mask2=None):
    '''Mask a map (pair) and calculate its power spectrum
    with only a norm (w2) correction.
    '''
    mask1 = np.asarray(mask1)
    if mask2 is None: mask2=mask1.copy()
    
   
    pass1 = map1.copy()
    pass1.data = pass1.data * mask1
    if map2 is not None:
        pass2 = map2.copy()
        pass2.data = pass2.data * mask2
        power = ft.powerFromLiteMap(pass1,pass2)
    else:        
        power = ft.powerFromLiteMap(pass1)
    w2 = np.mean(mask1*mask2)
    power.powerMap /= w2    
    return power.powerMap

#@timeit
def get_simple_power_enmap(enmap1,mask1=1.,enmap2=None,mask2=None):
    '''Mask a map (pair) and calculate its power spectrum
    with only a norm (w2) correction.
    '''
    t1 = simple_flipper_template_from_enmap(enmap1.shape,enmap1.wcs)
    t1.data = enmap1
    if enmap2 is not None:
        t2 = simple_flipper_template_from_enmap(enmap2.shape,enmap2.wcs)
        t2.data = enmap2
    else:
        t2 = None
        
    return get_simple_power(t1,mask1,t2,mask2)

    
#Take divergence using fourier space gradients
def takeDiv(vecStampX,vecStampY,lxMap,lyMap):

    fX = fft(vecStampX,axes=[-2,-1])
    fY = fft(vecStampY,axes=[-2,-1])

    return ifft((lxMap*fX+lyMap*fY)*1j,axes=[-2,-1],normalize=True).real

#Take divergence using fourier space gradients
def takeGrad(stamp,lyMap,lxMap):

    f = fft(stamp,axes=[-2,-1])

    return ifft(lyMap*f*1j,axes=[-2,-1],normalize=True).real,ifft(lxMap*f*1j,axes=[-2,-1],normalize=True).real


#@timeit
def interpolateGrid(inGrid,inY,inX,outY,outX,regular=True,kind="cubic",kx=3,ky=3,**kwargs):
    '''
    if inGrid is [j,i]
    Assumes inY is along j axis
    Assumes inX is along i axis
    Similarly for outY/X
    '''

    if regular:
        interp_spline = RectBivariateSpline(inY,inX,inGrid,kx=kx,ky=ky,**kwargs)
        outGrid = interp_spline(outY,outX)
    else:
        interp_spline = interp2d(inX,inY,inGrid,kind=kind,**kwargs)
        outGrid = interp_spline(outX,outY)
    

    return outGrid


class GRFGen(object):

    def __init__(self,templateLiteMap,ell=None,Cell=None,power2d=None,bufferFactor=1):

        bufferFactor = int(bufferFactor)
        self.b = float(bufferFactor)

        self.Ny,self.Nx = templateLiteMap.Ny,templateLiteMap.Nx
        self.bNy = templateLiteMap.Ny*bufferFactor
        self.bNx = templateLiteMap.Nx*bufferFactor

        ly = np.fft.fftfreq(self.bNy,d = templateLiteMap.pixScaleY)*(2*np.pi)
        lx = np.fft.fftfreq(self.bNx,d = templateLiteMap.pixScaleX)*(2*np.pi)
        self.modLMap = np.zeros([self.bNy,self.bNx])
        iy, ix = np.mgrid[0:self.bNy,0:self.bNx]
        self.modLMap[iy,ix] = np.sqrt(ly[iy]**2+lx[ix]**2)        

        if Cell is not None:
            assert ell is not None
            Cell[Cell<0.]=0.
            s = splrep(ell,Cell,k=3) # maps will be uK fluctuations about zero
            kk = splev(self.modLMap,s)
            kk[self.modLMap>ell.max()] = 0.
        elif power2d is not None:
            kk = power2d.copy()
            
        self.power = kk.copy()
        kk[self.modLMap<2.]=0.
        
        area = self.bNx*self.bNy*templateLiteMap.pixScaleX*templateLiteMap.pixScaleY
        p = kk /area * (self.bNx*self.bNy)**2
      
        self.sqp = np.sqrt(p)

    def getMap(self,stepFilterEll=None):
        """
        Modified from sudeepdas/flipper
        Generates a GRF from an input power spectrum specified as ell, Cell 
        BufferFactor =1 means the map will be periodic boundary function
        BufferFactor > 1 means the map will be genrated on  a patch bufferFactor times 
        larger in each dimension and then cut out so as to have non-periodic bcs.

        Fills the data field of the map with the GRF realization
        """


        realPart = self.sqp*np.random.randn(self.bNy,self.bNx)
        imgPart = self.sqp*np.random.randn(self.bNy,self.bNx)


        kMap = realPart+1.j*imgPart
        
        if stepFilterEll is not None:
            kMap[self.modLMap>stepFilterEll]=0.



        data = np.real(ifft(kMap,axes=[-2,-1],normalize=True)) 

        data = data[int((self.b-1)/2)*self.Ny:int((self.b+1)/2)*self.Ny,int((self.b-1)/2)*self.Nx:int((self.b+1)/2)*self.Nx]

        return data - data.mean()


def stepFunctionFilterLiteMap(map2d,modLMap,ellMax,ellMin=None):

    kmap = fft(map2d.copy(),axes=[-2,-1])
    kmap[modLMap>ellMax]=0.
    if ellMin is not None:
        kmap[modLMap<ellMin]=0.
        
    retMap = ifft(kmap,axes=[-2,-1],normalize=True).real

    return retMap


def FourierTQUtoFourierTEB(fT,fQ,fU,modLMap,angLMap):

    
    fE=fT.copy()
    fB=fT.copy()
    fE[:]=fQ[:]*np.cos(2.*angLMap)+fU*np.sin(2.*angLMap)
    fB[:]=-fQ[:]*np.sin(2.*angLMap)+fU*np.cos(2.*angLMap)
    
    return(fT, fE, fB)


def TQUtoFourierTEB(T_map,Q_map,U_map,modLMap,angLMap):

    fT=fft(T_map,axes=[-2,-1])    
    fQ=fft(Q_map,axes=[-2,-1])        
    fU=fft(U_map,axes=[-2,-1])
    
    fE=fT.copy()
    fB=fT.copy()
    fE[:]=fQ[:]*np.cos(2.*angLMap)+fU*np.sin(2.*angLMap)
    fB[:]=-fQ[:]*np.sin(2.*angLMap)+fU*np.cos(2.*angLMap)
    
    return(fT, fE, fB)


def getRealAttributes(templateLM):
    '''
    Given a liteMap, return a coord
    system centered on it and a map
    of distances from center in
    radians
    '''

        
    Nx = templateLM.Nx
    Ny = templateLM.Ny
    pixScaleX = templateLM.pixScaleX 
    pixScaleY = templateLM.pixScaleY
    return get_real_attributes(Ny,Nx,pixScaleY,pixScaleX)
    
def get_real_attributes(Ny,Nx,pixScaleY,pixScaleX):
    
    xx =  (np.arange(Nx)-Nx/2.+0.5)*pixScaleX
    yy =  (np.arange(Ny)-Ny/2.+0.5)*pixScaleY
    
    ix = np.mod(np.arange(Nx*Ny),Nx)
    iy = np.arange(Nx*Ny)/Nx
    
    modRMap = np.zeros([Ny,Nx])
    modRMap[iy,ix] = np.sqrt(xx[ix]**2 + yy[iy]**2)
    

    xMap, yMap = np.meshgrid(xx, yy)  # is this the right order?

    return xMap,yMap,modRMap,xx,yy


def getFTAttributesFromLiteMap(templateLM):
    '''
    Given a liteMap, return the fourier frequencies,
    magnitudes and phases.
    '''

    from scipy.fftpack import fftfreq
        
    Nx = templateLM.Nx
    Ny = templateLM.Ny
    pixScaleX = templateLM.pixScaleX 
    pixScaleY = templateLM.pixScaleY
    return get_ft_attributes(Ny,Nx,pixScaleY,pixScaleX)


def get_ft_attributes_enmap(shape,wcs):
    Ny, Nx = shape[-2:]
    pixScaleY, pixScaleX = enmap.pixshape(shape,wcs)
    return get_ft_attributes(Ny,Nx,pixScaleY,pixScaleX)


def get_ft_attributes(Ny,Nx,pixScaleY,pixScaleX):
    '''
    Given a liteMap, return the fourier frequencies,
    magnitudes and phases.
    '''

    from scipy.fftpack import fftfreq
        
    lx =  2*np.pi  * fftfreq( Nx, d = pixScaleX )
    ly =  2*np.pi  * fftfreq( Ny, d = pixScaleY )
    
    ix = np.mod(np.arange(Nx*Ny),Nx)
    iy = np.arange(Nx*Ny)/Nx
    
    modLMap = np.zeros([Ny,Nx])
    modLMap[iy,ix] = np.sqrt(lx[ix]**2 + ly[iy]**2)
    
    thetaMap = np.zeros([Ny,Nx])
    thetaMap[iy[:],ix[:]] = np.arctan2(ly[iy[:]],lx[ix[:]])
    #thetaMap *=180./np.pi


    lxMap, lyMap = np.meshgrid(lx, ly)  # is this the right order?

    return lxMap,lyMap,modLMap,thetaMap,lx,ly






def makeTemplate(l,Fl,modLMap,k=1,debug=False):
    """                                                                                                                                    
    Given 1d function Fl of l, creates the 2d version                                                                                   
    of Fl on 2d k-space defined by ftMap                                                                                                   
    """
    from scipy.interpolate import splrep, splev

    #Fl[Fl>1.e90] = 1.e90

    Ny,Nx = modLMap.shape
    tck = splrep(l,Fl,k=k)
    lmap = modLMap
    lmapunravel = lmap.ravel()
    template1d = splev(lmapunravel,tck)
    template = np.reshape(template1d,[Ny,Nx])

    if debug:
        from ..tools.output import Plotter
        from scipy.interpolate import interp1d
        _func = interp1d(l,Fl,kind=k,bounds_error=False,fill_value = 0)
        print np.sort(lmapunravel)[1]
        print lmapunravel.min(),template1d[lmapunravel==lmapunravel.min()]
        print modLMap.ravel().min(),_func(modLMap.ravel()==modLMap.ravel().min())
        pl = Plotter()
        pl.add(lmapunravel,template1d*(lmapunravel+1.)**2.,label="splev unravel",ls="-",marker="o")
        pl.add(modLMap.ravel(),_func(modLMap.ravel())*(modLMap.ravel()+1.)**2.,label="interp1d unravel",ls="none",marker="x")
        pl.add(l,_func(l)*(l+1)**2.,label="interp1d func")
        pl.add(l,Fl*(l+1)**2.,label="true func")
        pl.legendOn(loc='upper right', labsize=10)
        pl._ax.set_xlim(0.,800.)
        pl.done("fl.png")

        pl = Plotter(scaleX='log',scaleY='log')
        pl.add(lmapunravel,template1d)
        pl.done('debug.png')

        
        #template[np.where(lmap <= 100.)] = 0.
        #template[np.where(lmap >= 1000.)] = 0.
        
        
        pl = Plotter()
        pl.plot2d(np.log10((fftshift(template))))
        pl.done("temp.png")
        sys.exit()
    
    return template



def whiteNoise2D(noiseLevels,beamArcmin,modLMap,TCMB = 2.7255e6,lknees=None,alphas=None,beamFile=None, \
                 noiseFuncs=None):
    # Returns 2d map noise in units of uK**0.
    # Despite the name of the function, there are options to add
    # a simplistic atmosphere noise model

    # If no atmosphere is specified, set lknee to zero and alpha to 1
    if lknees is None:
        lknees = (np.array(noiseLevels)*0.).tolist()
    if alphas is None:
        alphas = (np.array(noiseLevels)*0.+1.).tolist()

    # we'll loop over it, so make it a list if nothing is specified
    if noiseFuncs is None: noiseFuncs = [None]*len(noiseLevels)

        
    # if one of the noise files is not specified, we will need a beam
    if None in noiseFuncs:
        
        if beamFile is not None:
            ell, f_ell = np.transpose(np.loadtxt(beamFile))[0:2,:]
            filt = 1./(np.array(f_ell)**2.)
            bfunc = interp1d(ell,f_ell,bounds_error=False,fill_value=np.inf)
            filt2d = bfunc(modLMap)
        else:
            Sigma = beamArcmin *np.pi/60./180./ np.sqrt(8.*np.log(2.))  # radians
            filt2d = np.exp(-(modLMap**2.)*Sigma*Sigma)


    retList = []

    for noiseLevel,lknee,alpha,noiseFunc in zip(noiseLevels,lknees,alphas,noiseFuncs):
        if noiseFunc is not None:
            retList.append(nfunc(modLMap))
        else:
        
            noiseForFilter = (np.pi / (180. * 60))**2.  * noiseLevel**2. / TCMB**2.  

            if lknee>0.:
                atmFactor = (lknee*np.nan_to_num(1./modLMap))**(-alpha)
            else:
                atmFactor = 0.
                
            with np.errstate(divide='ignore'):
                retList.append(noiseForFilter*(atmFactor+1.)*np.nan_to_num(1./filt2d.copy()))

    return retList


    


def fourierMask(lx,ly,modLMap, lxcut = None, lycut = None, lmin = None, lmax = None):
    output = np.zeros(modLMap.shape, dtype = int)
    output[:] = 1
    if lmin != None:
        wh = np.where(modLMap <= lmin)
        output[wh] = 0
    if lmax != None:
        wh = np.where(modLMap >= lmax)
        output[wh] = 0
    if lxcut != None:
        wh = np.where(np.abs(lx) < lxcut)
        output[:,wh] = 0
    if lycut != None:
        wh = np.where(np.abs(ly) < lycut)
        output[wh,:] = 0
    return output

def taper(lm,win):
    lmret = lm.copy()
    lmret.data[:,:] *= win[:,:]
    #w2 = np.sqrt(np.mean(win**2.))
    #lmret.data[:,:] /= w2    
    return lmret

def taperData(data2d,win):
    data2d[:,:] *= win[:,:]
    w2 = np.sqrt(np.mean(win**2.))
    lmret.data[:,:] /= w2    
    return data2d

#@timeit
def cosineWindow(Ny,Nx,lenApodY=30,lenApodX=30,padY=0,padX=0):
    win=np.ones((Ny,Nx))
    
    i = np.arange(Nx) 
    j = np.arange(Ny)
    ii,jj = np.meshgrid(i,j)

    # ii is array of x indices
    # jj is array of y indices
    # numpy indexes (j,i)

    # xdirection
    if lenApodX>0:
        r=ii.astype(float)-padX
        sel = np.where(ii<=(lenApodX+padX))
        win[sel] = 1./2*(1-np.cos(-np.pi*r[sel]/lenApodX))
        sel = np.where(ii>=((Nx-1)-lenApodX-padX))
        r=((Nx-1)-ii-padX).astype(float)
        win[sel] = 1./2*(1-np.cos(-np.pi*r[sel]/lenApodX))
    # ydirection
    if lenApodY>0:
        r=jj.astype(float)-padY
        sel = np.where(jj<=(lenApodY+padY))
        win[sel] *= 1./2*(1-np.cos(-np.pi*r[sel]/lenApodY))
        sel = np.where(jj>=((Ny-1)-lenApodY-padY))
        r=((Ny-1)-jj-padY).astype(float)
        win[sel] *= 1./2*(1-np.cos(-np.pi*r[sel]/lenApodY))

    win[0:padY,:]=0
    win[:,0:padX]=0
    win[Ny-padY:,:]=0
    win[:,Nx-padX:]=0
    return win

def initializeCosineWindow(templateLiteMap,lenApodY=30,lenApodX=None,pad=0):

    if lenApodX is None: lenApodY=lenApodY
    print "WARNING: This function is deprecated and will be removed. \
    Please replace with the much faster flatMaps.cosineWindow function."
	
    Nx=templateLiteMap.Nx
    Ny=templateLiteMap.Ny
    win=templateLiteMap.copy()
    win.data[:]=1

    winX=win.copy()
    winY=win.copy()

    for j in range(pad,Ny-pad):
        for i in range(pad,Nx-pad):
            if i<=(lenApodX+pad):
                r=float(i)-pad
                winX.data[j,i]=1./2*(1-np.cos(-np.pi*r/lenApodX))
            if i>=(Nx-1)-lenApodX-pad:
                r=float((Nx-1)-i-pad)
                winX.data[j,i]=1./2*(1-np.cos(-np.pi*r/lenApodX))

    for i in range(pad,Nx-pad):
        for j in range(pad,Ny-pad):
            if j<=(lenApodY+pad):
                r=float(j)-pad
                winY.data[j,i]=1./2*(1-np.cos(-np.pi*r/lenApodY))
            if j>=(Ny-1)-lenApodY-pad:
                r=float((Ny-1)-j-pad)
                winY.data[j,i]=1./2*(1-np.cos(-np.pi*r/lenApodY))

    win.data[:]*=winX.data[:,:]*winY.data[:,:]
    win.data[0:pad,:]=0
    win.data[:,0:pad]=0
    win.data[Nx-pad:Nx,:]=0
    win.data[:,Nx-pad:Nx]=0

    return(win.data)

def stack_on_map(lite_map,width_stamp_arcminute,pix_scale,ra_range,dec_range,ras=None,decs=None,n_random_points=None):
    from skimage.transform import resize
    import orphics.tools.stats as stats

    width_stamp_degrees = width_stamp_arcminute /60.
    Np = np.int(width_stamp_arcminute/pix_scale+0.5)
    pad = np.int(Np/2+0.5)
    print ("Expected width in pixels = ", Np)

    lmap = lite_map
    stack=0
    N=0

    if ras is not None:
        looprange = range(0,len(ras))
        assert n_random_points is None
        random = False
    else:
        assert n_random_points is not None
        assert len(ra_range)==2
        assert len(dec_range)==2
        looprange = range(0,n_random_points)
        random = True
        
    for i in looprange:
        if random:
                        ra = np.random.uniform(*ra_range)
                        dec =np.random.uniform(*dec_range)
        else:
                        ra=ras[i] #catalog[i][1]
                        dec=decs[i] #catalog[i][2]
        ix, iy = lmap.skyToPix(ra,dec)
        if ix>=pad and ix<lmap.Nx-pad and iy>=pad and iy<lmap.Ny-pad:
            print(i)
            #print(ra,dec)
            smap = lmap.selectSubMap(ra-width_stamp_degrees/2.,ra+width_stamp_degrees/2.,dec-width_stamp_degrees/2.,dec+width_stamp_degrees/2.)
            #print (smap.data.shape)
            #cutout = zoom(smap.data.copy(),zoom=(float(Np)/smap.data.shape[0],float(Np)/smap.data.shape[1]))
            cutout = resize(smap.data.copy(),output_shape=(Np,Np))
            #print (cutout.shape)
            stack = stack + cutout
            xMap,yMap,modRMap,xx,yy = getRealAttributes(smap)
            N=N+1.
        else:
            print ("skip")
    stack=stack/N
    #print(stack.shape())
    #print(smap.data.shape)
    print(stack)
    print(N)
    # io.quickPlot2d(stack,out_dir+"stack.png")

    dt = pix_scale
    arcmax = 20.
    thetaRange = np.arange(0.,arcmax,dt)
    breal = stats.bin2D(modRMap*180.*60./np.pi,thetaRange)
    cents,recons = breal.bin(stack)
    # pl = Plotter(labelX='Distance from Center (arcminutes)',labelY='Temperature Fluctuation ($\mu K$)', ftsize=10)
    # pl.add(cents,recons)
    # pl._ax.axhline(y=0.,ls="--",alpha=0.5)
    # pl.done(out_dir+"profiles.png")
    return stack, cents, recons

def initializeCosineWindowData(Ny,Nx,lenApod=30,pad=0):
    print "WARNING: This function is deprecated and will be removed. \
    Please replace with the much faster flatMaps.cosineWindow function."
	
    win=np.ones((Ny,Nx))

    winX=win.copy()
    winY=win.copy()

    
    for j in range(pad,Ny-pad):
            for i in range(pad,Nx-pad):
                    if i<=(lenApod+pad):
                            r=float(i)-pad
                            winX[j,i]=1./2*(1-np.cos(-np.pi*r/lenApod))
                    if i>=(Nx-1)-lenApod-pad:
                            r=float((Nx-1)-i-pad)
                            winX[j,i]=1./2*(1-np.cos(-np.pi*r/lenApod))

    for i in range(pad,Nx-pad):
            for j in range(pad,Ny-pad):
                    if j<=(lenApod+pad):
                            r=float(j)-pad
                            winY[j,i]=1./2*(1-np.cos(-np.pi*r/lenApod))
                    if j>=(Ny-1)-lenApod-pad:
                            r=float((Ny-1)-j-pad)
                            winY[j,i]=1./2*(1-np.cos(-np.pi*r/lenApod))

    win[:]*=winX[:,:]*winY[:,:]
    win[0:pad,:]=0
    win[:,0:pad]=0
    win[Nx-pad:Nx,:]=0
    win[:,Nx-pad:Nx]=0

    return win

def deconvolveBeam(data,modLMap,beamTemplate,lowPass=None,returnFTOnly = False):



    kMap = fft(data,axes=[-2,-1])

    kMap[:,:] = (kMap[:,:] / beamTemplate[:,:])
    if lowPass is not None: kMap[modLMap>lowPass] = 0.
    if returnFTOnly:
        return kMap
    else:
        return ifft(kMap,axes=[-2,-1],normalize=True).real



def convolveBeam(data,modLMap,beamTemplate):
    kMap = fft(data,axes=[-2,-1])
    kMap[:,:] = (kMap[:,:] * beamTemplate[:,:])
    return ifft(kMap,axes=[-2,-1],normalize=True).real

#@timeit
def smooth(data,modLMap,gauss_sigma_arcmin):
    kMap = fft(data,axes=[-2,-1])
    sigma = np.deg2rad(gauss_sigma_arcmin / 60.)
    beamTemplate = np.nan_to_num(1./np.exp((sigma**2.)*(modLMap**2.) / (2.)))
    kMap[:,:] = np.nan_to_num(kMap[:,:] * beamTemplate[:,:])
    return ifft(kMap,axes=[-2,-1],normalize=True).real


#@timeit
def filter_map(data2d,filter2d,modLMap,lowPass=None,highPass=None,keep_mean=True):
    kMap = fft(data2d,axes=[-2,-1])
    if keep_mean:
        mean_val = kMap[modLMap<1]

    kMap[:,:] = np.nan_to_num(kMap[:,:] * filter2d[:,:])
    if lowPass is not None: kMap[modLMap>lowPass] = 0.
    if highPass is not None: kMap[modLMap<highPass] = 0.

    if keep_mean: kMap[modLMap<1] = mean_val
    return ifft(kMap,axes=[-2,-1],normalize=True).real


def simple_flipper_template_from_enmap(shape,wcs):
    Ny,Nx = shape[-2:]
    pixScaleY, pixScaleX = enmap.pixshape(shape,wcs)
    return simple_flipper_template(Ny,Nx,pixScaleY,pixScaleX)


def simple_flipper_template(Ny,Nx,pixScaleY,pixScaleX):
    class template:
        def copy(self):
            return copy.deepcopy(self)
    t = template()
    t.Ny = Ny
    t.Nx = Nx
    t.pixScaleY = pixScaleY
    t.pixScaleX = pixScaleX
    return t
    

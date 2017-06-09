import numpy as np

from scipy.interpolate import splrep,splev
from scipy.fftpack import fftshift
from scipy.interpolate import RectBivariateSpline,interp2d,interp1d
from orphics.tools.stats import timeit
import flipper.fftTools as ft
try:
    from flipper.fft import fft,ifft
except:
    import logging
    logging.warning("You seem to be using a version of flipper that does not have the 'flipper.fft' module, a wrapper for fast multi-threaded pyfftw FFTs. Please use the version of flipper maintained by the ACT Collaboration at https://github.com/ACTCollaboration/flipper .")


@timeit
def get_simple_power(map1,mask1,map2=None,mask2=None):
    '''Mask a map (pair) and calculate its power spectrum
    with only a norm (w2) correction.
    '''
    if mask2 is None: mask2=np.asarray(mask1).copy()
   
    pass1 = map1.copy()
    pass1.data = pass1.data * mask1
    #from orphics.tools.io import quickPlot2d
    #quickPlot2d(pass1.data,"temp.png")
    #sys.exit()
    if map2 is not None:
        pass2 = map2.copy()
        pass2.data = pass2.data * mask2
        power = ft.powerFromLiteMap(pass1,pass2)
    else:        
        power = ft.powerFromLiteMap(pass1)
    w2 = np.mean(mask1*mask2)
    power.powerMap /= w2    
    return power.powerMap

    
#Take divergence using fourier space gradients
def takeDiv(vecStampX,vecStampY,lxMap,lyMap):

    fX = fft(vecStampX,axes=[-2,-1])
    fY = fft(vecStampY,axes=[-2,-1])

    return ifft((lxMap*fX+lyMap*fY)*1j,axes=[-2,-1],normalize=True).real

#Take divergence using fourier space gradients
def takeGrad(stamp,lyMap,lxMap):

    f = fft(stamp,axes=[-2,-1])

    return ifft(lyMap*f*1j,axes=[-2,-1],normalize=True).real,ifft(lxMap*f*1j,axes=[-2,-1],normalize=True).real


@timeit
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

@timeit
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

@timeit
def smooth(data,modLMap,gauss_sigma_arcmin):
    kMap = fft(data,axes=[-2,-1])
    sigma = np.deg2rad(gauss_sigma_arcmin / 60.)
    beamTemplate = np.nan_to_num(1./np.exp((sigma**2.)*(modLMap**2.) / (2.)))
    kMap[:,:] = np.nan_to_num(kMap[:,:] * beamTemplate[:,:])
    return ifft(kMap,axes=[-2,-1],normalize=True).real


@timeit
def filter_map(data2d,filter2d,modLMap,lowPass=None,highPass=None):
    kMap = fft(data2d,axes=[-2,-1])
    kMap[:,:] = np.nan_to_num(kMap[:,:] * filter2d[:,:])
    if lowPass is not None: kMap[modLMap>lowPass] = 0.
    if highPass is not None: kMap[modLMap<highPass] = 0.
    return ifft(kMap,axes=[-2,-1],normalize=True).real


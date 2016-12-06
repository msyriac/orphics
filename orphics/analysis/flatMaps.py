import numpy as np



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

    Fl[Fl>1.e90] = 1.e90

    Ny,Nx = modLMap.shape
    tck = splrep(l,Fl,k=k)
    lmap = modLMap
    lmapunravel = lmap.ravel()
    template1d = splev(lmapunravel,tck)
    template = np.reshape(template1d,[Ny,Nx])

    if debug:
        from ..tools.output import Plotter

        pl = Plotter()
        pl.add(l,Fl)
        pl.done("fl.png")

        pl = Plotter(scaleX='log',scaleY='log')
        pl.add(lmapunravel,template1d)
        pl.done('debug.png')

        
        template[np.where(lmap <= 100.)] = 0.
        template[np.where(lmap >= 1000.)] = 0.
        
        
        pl = Plotter()
        pl.plot2d(np.log10((fftshift(template))))
        pl.done("temp.png")
        sys.exit()
    
    return template



def whiteNoise2D(noiseLevels,beamArcmin,modLMap):
    # Returns 2d map noise in units of uK**2.

    Sigma = beamArcmin *np.pi/60./180./ np.sqrt(8.*np.log(2.))  # radians
    ell = np.arange(0.,modLMap.max(),1.)
    filt = np.exp(ell*ell*Sigma*Sigma)


    retList = []
    filt2d =   makeTemplate(ell,filt,modLMap)#,debug=True)  # make template noise map

    for noiseLevel in noiseLevels:
        noiseForFilter = (np.pi / (180. * 60))**2.  * noiseLevel**2.   # add instrument noise to noise power map
        retList.append(filt2d.copy()*noiseForFilter)

    return retList


    


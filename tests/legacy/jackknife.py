import matplotlib
matplotlib.use('Agg')
import sys
from astLib import astWCS, astCoords
import liteMap
from mmUtils import Plotter
import numpy as np
import fftTools as ft
from numpy.fft import fftshift
from LikeRealAL import aveBinInAnnuli as bin2d, makeBinfile
import liteMapPol as lpol
from hankelTransform import getStats

def is_square(apositiveint):
    if apositiveint==1: return True
    x = apositiveint // 2
    seen = set([x])
    while x * x != apositiveint:
        x = (x + (apositiveint // x)) // 2
        if x in seen: return False
        seen.add(x)
    return True

def getTaperedMap(lkk,clkk,templateMapLoc = "../DerivGen/data/order5_lensedCMB_T_beam_cutout_3.fits",bufferFactor=2,taperWidth = 120,jackknife=36):
    # jackknife = (number of jackknife regions)
    
    assert is_square(jackknife)
    

    templateMap = liteMap.liteMapFromFits(templateMapLoc)
    templateMap.data[:,:] = 0.
    templateMap.fillWithGaussianRandomField(lkk,clkk,bufferFactor = bufferFactor)
    retMap = templateMap.copy()

    xa,xb = (templateMap.x0,templateMap.x1)
    ya,yb = (templateMap.y0,templateMap.y1)
    x0 = min(xa,xb)
    x1 = max(xa,xb)
    y0 = min(ya,yb)
    y1 = max(ya,yb)

    xl = x1-x0
    yl = y1-y0

    Neach = int(np.sqrt(jackknife))
    xeach = xl/Neach
    yeach = yl/Neach

    bufferx = 0.001
    buffery = 0.001

    smaps = []
    stapers = []
    for i in range(Neach):
        
        tx0 = x0+i*xeach
        tx1 = x0+(i+1)*xeach

        if i==0: tx0 += bufferx
        if i==Neach-1: tx1 -= bufferx
            
        for j in range(Neach):
            ty0 = y0+j*yeach
            ty1 = y0+(j+1)*yeach

            if j==0: ty0 += buffery
            if j==Neach-1: ty1 -= buffery
            
            
            print((tx0,tx1,ty0,ty1))
            smap = templateMap.selectSubMap(tx0,tx1,ty0,ty1, safe = False)
            #print smap.info()
            
            subtaper = lpol.initializeCosineWindow(smap,int(taperWidth/Neach),0)
            smap.data[:] = smap.data[:]*subtaper.data[:]
            pl = Plotter()
            pl.plot2d(smap.data)
            pl.done("kappa"+str(i)+str(j)+".png")

            smaps.append(smap)
            stapers.append(subtaper)
            
    #sys.exit()



    taper = lpol.initializeCosineWindow(retMap,taperWidth,0)
    retMap.data[:] = retMap.data[:]*taper.data[:]


    pl = Plotter()
    pl.plot2d(templateMap.data)
    pl.done("kappa.png")

    return retMap,taper,smaps,stapers


def getBinnedPower(templateMap,binFile,taperMap):
    p2d = ft.powerFromLiteMap(templateMap,applySlepianTaper=False)


    # pl = Plotter()
    # pl.plot2d(np.log(fftshift(p2d.powerMap)))
    # pl.done("power.png")

    lower, upper, center, bin_means, bin_stds, bincount = bin2d(p2d, binfile = binFile)
    w2 = np.mean(taperMap.data**2)
    return lower, upper, center, bin_means/w2


tempBinfile = "tempbin.txt"
dummy = makeBinfile(tempBinfile,2.,4000.,100.,redundant=True)

clkkFile = "../actpLens/data/fidkk.dat"
clkk = np.loadtxt(clkkFile)
lkk = np.arange(2,len(clkk)+2)

N = 20

estcls = []
for i in range(N):

    kappaMap,taperMap = getTaperedMap(lkk,clkk)
    print((kappaMap.data.shape))
    print((kappaMap.info()))
    sys.exit()
    lower, upper, center, bin_means = getBinnedPower(kappaMap,tempBinfile,taperMap)
    estcls.append(bin_means)
    print(i)

clmeans, covMean, cov, errMean,err,corrcoef = getStats(estcls,N)

        


pl = Plotter()
pl.add(lkk,lkk*clkk)
#pl.add(center,center*bin_means,ls="none",marker="x",color='red',markersize=8,mew=3)
pl.addErr(center,center*clmeans,yerr=center*errMean,ls="none",marker="o",color='red',markersize=8,mew=3)
pl._ax.set_xlim(0.,3500.)
pl.done("clpower.png")

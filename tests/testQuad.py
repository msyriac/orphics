print "Starting imports..."
import matplotlib
matplotlib.use('Agg')
from orphics.analysis.quadraticEstimator import Estimator
import orphics.analysis.flatMaps as fmaps 
#from orphics.theory.gaussianCov import TheorySpectra
import numpy as np
from astLib import astWCS, astCoords
import liteMap as lm
from orphics.tools.output import Plotter
from orphics.tools.stats import binInAnnuli
import sys

# hu reproduce
# beamArcmin = 7.0
# noiseT = 27.0
# noiseP = 56.6
# cmbellmin = 50
# cmbellmax = 3000
# kellmin = 50
# kellmax = 2000

# beamArcmin = 1.4
# noiseT = 10.0
# noiseP = 14.4
# cmbellmin = 1000
# cmbellmax = 3000
# kellmin = 80
# kellmax = 2100

beamArcmin = 0.
noiseT = 0.
noiseP = 0.
cmbellmin = 100
kellmin = 100
cmbellmax = 6000
kellmax = 3000

#polCombList = ['TT']
#polCombList = ['TT','TE']
#polCombList = ['TT','EE','ET','EB','TB']
#colorList = ['red','blue','green','orange','purple']
polCombList = ['TT','EE','ET','TE','EB','TB']
colorList = ['red','blue','green','cyan','orange','purple']

TCMB = 2.7255e6

cambRoot = "/astro/u/msyriac/repos/cmb-lensing-projections/data/TheorySpectra/ell28k_highacc"
dataFile = "/astro/astronfs01/workarea/msyriac/act/FinalScinetPaper/preparedMap_T_6.fits"

print "Loading map..."
templateMap = lm.liteMapFromFits(dataFile)
pl = Plotter()
pl.plot2d(templateMap.data)
pl.done("map.png")

print "Interpolating Cls..."

from orphics.tools.cmb import loadTheorySpectraFromCAMB
theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,TCMB = TCMB,lpad=9000)

ellkk = np.arange(2,9000,1)
Clkk = theory.gCl("kk",ellkk)    



lxMap,lyMap,modLMap,thetaMap,lx,ly  = fmaps.getFTAttributesFromLiteMap(templateMap)

print "Making white noise..."
nT,nP = fmaps.whiteNoise2D([noiseT,noiseP],beamArcmin,modLMap,TCMB=TCMB)
fMask = fmaps.fourierMask(lx,ly,modLMap,lmin=cmbellmin,lmax=cmbellmax)
fMaskK = fmaps.fourierMask(lx,ly,modLMap,lmin=kellmin,lmax=kellmax)

qest = Estimator(templateMap,
                 theory,
                 theorySpectraForNorm=None,
                 noiseX2dTEB=[nT,nP,nP],
                 noiseY2dTEB=[nT,nP,nP],
                 fmaskX2dTEB=[fMask]*3,
                 fmaskY2dTEB=[fMask]*3,
                 fmaskKappa=fMaskK,
                 doCurl=False,
                 TOnly=len(polCombList)==1,
                 halo=True,
                 gradCut=10000,verbose=True)


modLMap = qest.N.modLMap
bin_edges = np.arange(2,kellmax,10)
pl = Plotter(scaleY='log',scaleX='log')
pl.add(ellkk,4.*Clkk/2./np.pi)

# CHECK THAT NORM MATCHES HU/OK
for polComb,col in zip(polCombList,colorList):
    data2d = qest.N.Nlkk[polComb]
    centers, Nlbinned = binInAnnuli(data2d, modLMap, bin_edges)
    print Nlbinned[50]

    try:
        huFile = '/astro/u/msyriac/repos/cmb-lensing-projections/data/NoiseCurvesKK/hu_'+polComb.lower()+'.csv'
        huell,hunl = np.loadtxt(huFile,unpack=True,delimiter=',')
    except:
        huFile = '/astro/u/msyriac/repos/cmb-lensing-projections/data/NoiseCurvesKK/hu_'+polComb[::-1].lower()+'.csv'
        huell,hunl = np.loadtxt(huFile,unpack=True,delimiter=',')


    pl.add(centers,4.*Nlbinned/2./np.pi,color=col)
    pl.add(huell,hunl,ls='--',color=col)


pl.done("tests/output/testbin.png")
#sys.exit()

# print "Reconstructing..."
# qest.updateTEB_X(templateMap.data.astype(float)/2.7255e6)
# qest.updateTEB_Y()

# kappa = qest.getKappa(polComb)

# pl = Plotter()
# pl.plot2d(kappa.real)
# pl.done("kappa.png")


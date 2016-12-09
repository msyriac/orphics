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
# cmbellmax = 3000
# kellmax = 2000

beamArcmin = 1.4
noiseT = 10.0
noiseP = 14.4
cmbellmin = 1000
cmbellmax = 3000
kellmin = 80
kellmax = 2100

# beamArcmin = 0.
# noiseT = 0.
# noiseP = 0.
# cmbellmax = 8000
# kellmax = 8000

TCMB = 2.7255e6

cambRoot = "/astro/u/msyriac/repos/cmb-lensing-projections/data/TheorySpectra/ell28k_highacc"
dataFile = "/astro/astronfs01/workarea/msyriac/act/FinalScinetPaper/preparedMap_T_6.fits"
huFile = '/astro/u/msyriac/repos/cmb-lensing-projections/data/NoiseCurvesKK/hu_tt.csv'

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
                 TOnly=True,
                 halo=False,
                 gradCut=None,verbose=True)



# CHECK THAT NORM MATCHES HU/OK
data2d = qest.AL['TT']
modLMap = qest.N.modLMap
bin_edges = np.arange(2,kellmax,10)
centers, Nlbinned = binInAnnuli(data2d, modLMap, bin_edges)

huell,hunl = np.loadtxt(huFile,unpack=True,delimiter=',')

pl = Plotter(scaleY='log',scaleX='log')
pl.add(ellkk,4.*Clkk/2./np.pi)
pl.add(centers,4.*Nlbinned/2./np.pi)#,ls="none",marker="o")
pl.add(huell,hunl,ls='--')#,ls="none",marker="o")
pl.done("testbin.png")
#sys.exit()

print "Reconstructing..."
qest.updateTEB_X(templateMap.data.astype(float)/2.7255e6)
qest.updateTEB_Y()

kappa = qest.getKappa('TT')

pl = Plotter()
pl.plot2d(kappa.real)
pl.done("kappa.png")


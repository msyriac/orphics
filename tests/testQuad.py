print "Starting imports..."
from orphics.analysis.quadraticEstimator import Estimator
import orphics.analysis.flatMaps as fmaps 
from orphics.theory.gaussianCov import TheorySpectra
import numpy as np
from astLib import astWCS, astCoords
import liteMap as lm

beamArcmin = 7.0
noiseT = 27.0
noiseP = 56.6

hereRoot = "/astro/u/msyriac/repos/orphics/tests/"

print "Loading map..."
templateMap = lm.liteMapFromFits("/astro/astronfs01/workarea/msyriac/act/FinalScinetPaper/preparedMap_T_6.fits")

theory = TheorySpectra()

print "Interpolating Cls..."
for cmb in ['tt','te','ee','bb']:
    Cl = np.loadtxt(hereRoot+"data/fid"+cmb+".dat")
    ell = np.arange(2,len(Cl)+2)
    theory.loadCls(ell,Cl,cmb.upper(),lensed=False,interporder="linear",lpad=9000)
    theory.loadCls(ell,Cl,cmb.upper(),lensed=True,interporder="linear",lpad=9000)


modLMap = fmaps.getFTAttributesFromLiteMap(templateMap)[2]

print "Making white noise..."
nT,nP = fmaps.whiteNoise2D([noiseT,noiseP],beamArcmin,modLMap)


qest = Estimator(templateMap,
                 theory,
                 theorySpectraForNorm=None,
                 noiseX2dTEB=[nT,nP,nP],
                 noiseY2dTEB=[nT,nP,nP],
                 fmaskX2dTEB=[None,None,None],
                 fmaskY2dTEB=[None,None,None],
                 doCurl=False,
                 TOnly=True,
                 halo=False,
                 gradCut=None,verbose=True)


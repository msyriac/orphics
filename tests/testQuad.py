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


print "Loading map..."
templateMap = lm.liteMapFromFits("/home/mathew/data/preparedMap_T_6.fits")

theory = TheorySpectra()

print "Interpolating Cls..."
ell = {}
Cl = {}
for cmb in ['TT','TE','EE','BB']:
    Cl[cmb] = np.loadtxt("/home/mathew/data/fid"+cmb.lower()+".dat")
    ell[cmb] = np.arange(2,len(Cl[cmb])+2)
    theory.loadCls(ell[cmb],Cl[cmb],cmb,lensed=False,interporder="linear",lpad=9000)
    theory.loadCls(ell[cmb],Cl[cmb],cmb,lensed=True,interporder="linear",lpad=9000)
Clkk = np.loadtxt("/home/mathew/data/fidkk.dat")
ellkk = np.arange(2,len(Clkk)+2)


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

from orphics.tools.output import Plotter
from orphics.tools.stats import binInAnnuli


# data2d = qest.N.lClFid2d['TT']
# modLMap = qest.N.modLMap
# bin_edges = np.arange(2,2000,20)
# centers, Clbinned = binInAnnuli(data2d, modLMap, bin_edges)

# pl = Plotter(scaleY='log')
# pl.add(ell['TT'],Cl['TT'])
# pl.add(centers,Clbinned,ls="none",marker="o")
# pl.done("testbin.png")

data2d = qest.AL['TT']
modLMap = qest.N.modLMap
bin_edges = np.arange(2,2000,20)
centers, Nlbinned = binInAnnuli(data2d, modLMap, bin_edges)

pl = Plotter(scaleY='log')
pl.add(ellkk,Clkk)
pl.add(centers,Nlbinned,ls="none",marker="o")
pl.done("testbin.png")

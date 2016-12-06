print "Starting imports..."
import matplotlib
matplotlib.use('Agg')
from orphics.analysis.quadraticEstimator import Estimator
import orphics.analysis.flatMaps as fmaps 
from orphics.theory.gaussianCov import TheorySpectra
import numpy as np
from astLib import astWCS, astCoords
import liteMap as lm
from orphics.tools.output import Plotter

beamArcmin = 7.0
noiseT = 27.0
noiseP = 56.6


print "Loading map..."
templateMap = lm.liteMapFromFits("/home/mathew/data/preparedMap_T_6.fits")

theory = TheorySpectra()

print "Interpolating Cls..."
ell = {}
Cl = {}
TCMB = 2.725e6
for cmb in ['TT','TE','EE','BB']:
    Cl[cmb] = np.loadtxt("/home/mathew/data/fid"+cmb.lower()+".dat")#*TCMB**2.
    ell[cmb] = np.arange(2,len(Cl[cmb])+2)
    theory.loadCls(ell[cmb],Cl[cmb],cmb,lensed=False,interporder="linear",lpad=9000)
    theory.loadCls(ell[cmb],Cl[cmb],cmb,lensed=True,interporder="linear",lpad=9000)
Clkk = np.loadtxt("/home/mathew/data/fidkk.dat")
ellkk = np.arange(2,len(Clkk)+2)



lxMap,lyMap,modLMap,thetaMap,lx,ly  = fmaps.getFTAttributesFromLiteMap(templateMap)

print "Making white noise..."
nT,nP = fmaps.whiteNoise2D([noiseT,noiseP],beamArcmin,modLMap)
fMask = fmaps.fourierMask(lx,ly,modLMap,lmin=2,lmax=3000)

qest = Estimator(templateMap,
                 theory,
                 theorySpectraForNorm=None,
                 noiseX2dTEB=[nT,nP,nP],
                 noiseY2dTEB=[nT,nP,nP],
                 fmaskX2dTEB=[fMask]*3,
                 fmaskY2dTEB=[fMask]*3,
                 doCurl=False,
                 TOnly=True,
                 halo=False,
                 gradCut=None,verbose=True)

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
bin_edges = np.arange(2,2000,10)
centers, Nlbinned = binInAnnuli(data2d, modLMap, bin_edges)


huell,hunl = np.loadtxt('/home/mathew/repos/cmb-lensing-projections/data/NoiseCurvesKK/hu_tt.csv',unpack=True,delimiter=',')

pl = Plotter(scaleY='log',scaleX='log')
pl.add(ellkk,4.*Clkk/2./np.pi)
pl.add(centers,4.*Nlbinned/2./np.pi)#,ls="none",marker="o")
pl.add(huell,hunl,ls='--')#,ls="none",marker="o")
pl.done("testbin.png")

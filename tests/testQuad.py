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


loadFile = None
saveFile = None

# hu reproduce
# beamArcmin = 7.0
# noiseT = 27.0
# noiseP = 56.6
# cmbellmin = 50
# cmbellmax = 3000
# kellmin = 50
# kellmax = 2000

beamArcmin = 1.4
noiseT = 16.0
noiseP = np.sqrt(2)*16.
cmbellmin = 1000
cmbellmax = 3000
kellmin = 80
kellmax = 2100

# beamArcmin = 0.
# noiseT = 0.
# noiseP = 0.
# cmbellmin = 100
# kellmin = 100
# cmbellmax = 6000
# kellmax = 3000

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


templateMap.info()
print templateMap.data.shape

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
                 gradCut=10000,
                 verbose=True,                 
                 loadPickledNormAndFilters=loadFile,
                 savePickledNormAndFilters=saveFile)


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








cambRoot = "data/ell28k_highacc"
gradCut = None
halo = True
beam = 7.0
noiseT = 27.0
noiseP = 56.6
tellmin = 2
tellmax = 3000
gradCut = 10000

pellmin = 2
pellmax = 3000

deg = 10.
px = 0.5
arc = deg*60.

bin_edges = np.arange(10,3000,10)

theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,lpad=9000)
lmap = lm.makeEmptyCEATemplate(raSizeDeg=arc/60., decSizeDeg=arc/60.,pixScaleXarcmin=px,pixScaleYarcmin=px)
print lmap.data.shape
myNls = NlGenerator(lmap,theory,bin_edges,gradCut=gradCut)

myNls.updateNoise(beam,noiseT,noiseP,tellmin,tellmax,pellmin,pellmax)

#polCombList = ['TT','EE','ET','TE','EB','TB']
#colorList = ['red','blue','green','cyan','orange','purple']
polCombList = ['TT','EE','ET','EB','TB']
colorList = ['red','blue','green','orange','purple']
ellkk = np.arange(2,9000,1)
Clkk = theory.gCl("kk",ellkk)    


pl = Plotter(scaleY='log',scaleX='log')
pl.add(ellkk,4.*Clkk/2./np.pi)

# CHECK THAT NORM MATCHES HU/OK
for polComb,col in zip(polCombList,colorList):
    ls,Nls = myNls.getNl(polComb=polComb,halo=halo)
    try:
        huFile = 'data/hu_'+polComb.lower()+'.csv'
        huell,hunl = np.loadtxt(huFile,unpack=True,delimiter=',')
    except:
        huFile = 'data/hu_'+polComb[::-1].lower()+'.csv'
        huell,hunl = np.loadtxt(huFile,unpack=True,delimiter=',')


    pl.add(ls,4.*Nls/2./np.pi,color=col)
    pl.add(huell,hunl,ls='--',color=col)


pl.done("output/hucomp.png")





sys.exit()



cambRoot = "data/ell28k_highacc"
gradCut = None
halo = True
beamY = 1.5
noiseT = 1.0
noiseP = 1.4
tellmin = 1000
tellmax = 3000
gradCut = 10000

pellmin = 1000
pellmax = 5000

deg = 10.
px = 0.5
arc = deg*60.

bin_edges = np.arange(10,3000,10)

theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,lpad=9000)
lmap = lm.makeEmptyCEATemplate(raSizeDeg=arc/60., decSizeDeg=arc/60.,pixScaleXarcmin=px,pixScaleYarcmin=px)
print lmap.data.shape
myNls = NlGenerator(lmap,theory,bin_edges,gradCut=gradCut)


#polCombList = ['TT','EE','ET','TE','EB','TB']
colorList = ['red','blue','green','cyan','orange','purple']
polCombList = ['EB']
#colorList = ['red']
ellkk = np.arange(2,9000,1)
Clkk = theory.gCl("kk",ellkk)    


pl = Plotter(scaleY='log',scaleX='log')
pl.add(ellkk,4.*Clkk/2./np.pi)

for beamX in np.arange(1.5,10.,1.0):
    myNls.updateNoise(beamX,noiseT,noiseP,tellmin,tellmax,pellmin,pellmax,beamY=beamY)
    for polComb,col in zip(polCombList,colorList):
        ls,Nls = myNls.getNl(polComb=polComb,halo=halo)

        pl.add(ls,4.*Nls/2./np.pi,label=str(beamX))#polComb)#,color=col

pl.legendOn(loc='lower left',labsize=10)
pl.done("output/hucomp.png")





sys.exit()

import numpy as np
from astLib import astWCS
from flipper import liteMap as lm
from orphics.tools.cmb import loadTheorySpectraFromCAMB
from orphics.theory.quadEstTheory import NlGenerator
from orphics.tools.output import Plotter

cambRoot = "data/ell28k_highacc"
halo = True
beamY = 1.5
noiseT = 6.0
noiseP = np.sqrt(2.)*noiseT
tellmin = 200
tellmax = 3000
gradCut = 10000

pellmin = 200
pellmax = 5000
polComb = 'EB'

deg = 10.
px = 0.5
arc = deg*60.

bin_edges = np.arange(100,4000,10)

theory = loadTheorySpectraFromCAMB(cambRoot,unlensedEqualsLensed=False,useTotal=False,lpad=9000)
lmap = lm.makeEmptyCEATemplate(raSizeDeg=arc/60., decSizeDeg=arc/60.,pixScaleXarcmin=px,pixScaleYarcmin=px)
print((lmap.data.shape))
myNls = NlGenerator(lmap,theory,bin_edges,gradCut=gradCut)


ellkk = np.arange(2,9000,1)
Clkk = theory.gCl("kk",ellkk)    


pl = Plotter(scaleY='log',scaleX='log')
pl.add(ellkk,4.*Clkk/2./np.pi)

for beamX in np.arange(1.5,10.,1.0):
    myNls.updateNoise(beamX,noiseT,noiseP,tellmin,tellmax,pellmin,pellmax,beamY=beamY)
    ls,Nls = myNls.getNl(polComb=polComb,halo=halo)

    
    pl.add(ls,4.*Nls/2./np.pi,label=str(beamX))
pl.legendOn(loc='lower left',labsize=10)
pl.done("output/nlSO6.png")

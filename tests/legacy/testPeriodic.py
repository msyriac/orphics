import flipper.liteMap as lm
from orphics.theory.cosmology import Cosmology
import numpy as np
import os, sys
import orphics.tools.io as io
import orphics.tools.stats as stats
import orphics.analysis.flatMaps as fmaps

out_dir = os.environ['WWW']
cc = Cosmology(pickling=True,clTTFixFile = "../szar/data/cltt_lensed_Feb18.txt")


lmap = lm.makeEmptyCEATemplate(raSizeDeg=20., decSizeDeg=20.)

ells = np.arange(2,6000,1)
Cell = cc.clttfunc(ells) #cc.theory.lCl('TT',ells)

lmap.fillWithGaussianRandomField(ells,Cell,bufferFactor = 1)

io.highResPlot2d(lmap.data,out_dir+"map.png")


p2d = fmaps.get_simple_power(lmap,lmap.data*0.+1.)
lxMap,lyMap,modLMap,thetaMap,lx,ly = fmaps.getFTAttributesFromLiteMap(lmap)

bin_edges = np.arange(20,4000,40)
b = stats.bin2D(modLMap,bin_edges)
cents, cdat = b.bin(p2d)

pl = io.Plotter(scaleX='log',scaleY='log')
pl.add(ells,Cell*ells**2.)
pl.add(cents,cdat*cents**2.)
# pl.done(out_dir+"cls.png")


lmap.fillWithGaussianRandomField(ells,Cell,bufferFactor = 3)



p2d = fmaps.get_simple_power(lmap,lmap.data*0.+1.)

cents, cdat = b.bin(p2d)

pl.add(cents,cdat*cents**2.,ls="--")
pl.done(out_dir+"cls.png")

import flipper.liteMap as lm
from orphics.tools.cosmology import Cosmology
import numpy as np
import os, sys
import orphics.tools.io as io

out_dir = "./" #os.environ['WWW']
cc = Cosmology(pickling=True,clTTFixFile = "data/cltt_lensed_Feb18.txt")


lmap = lm.makeEmptyCEATemplate(raSizeDeg=10., decSizeDeg=10.)

ells = np.arange(2,6000,1)
Cell = cc.theory.lCl('TT',ells)

lmap.fillWithGaussianRandomField(ells,Cell,bufferFactor = 1)

io.highResPlot2d(lmap.data,out_dir+"map.png")

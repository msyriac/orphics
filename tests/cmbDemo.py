from orphics.theory.cosmology import Cosmology
import numpy as np
import orphics.tools.io as io

cc = Cosmology(lmax=6000,pickling=True)


ells = np.arange(2,6000,1)
ucltt = cc.theory.uCl('TT',ells)
lcltt = cc.theory.lCl('TT',ells)

pl = io.Plotter(scaleY='log',scaleX='log')
pl.add(ells,ucltt*ells**2.,ls="--")
pl.add(ells,lcltt*ells**2.)
pl.done("cmbdemo.png")

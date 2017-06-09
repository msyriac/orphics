import numpy as np
import orphics.theory.gaussianCov as gcov
from orphics.theory.cosmology import Cosmology
import orphics.tools.io as io

cc = Cosmology(lmax=6000,pickling=True)
theory = cc.theory


planck_file = "input/planck_nlkk.dat"
lp,nlplanck = np.loadtxt(planck_file,usecols=[0,1],unpack=True)

LF = gcov.LensForecast(theory)

ells = np.arange(2,6000,1)
clkk = theory.gCl('kk',ells)
pl = io.Plotter(scaleY='log')
pl.add(ells,clkk)
pl.add(lp,nlplanck,ls="-.")
pl._ax.set_ylim(5e-10,1e-5)
pl.done("output/clsn.png")


LF.loadGenericCls("kk",ells,clkk,lp,nlplanck)

ellBinEdges = np.arange(8,400,20)
fsky = 0.65
specType = "kk"

sn,errs = LF.sn(ellBinEdges,fsky,specType)

print sn

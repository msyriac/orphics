import numpy as np
import orphics.theory.gaussianCov as gcov
from orphics.theory.cosmology import Cosmology
import orphics.tools.io as io

# Initialize a cosmology object. pickling saves and loads Cls daily.
cc = Cosmology(lmax=6000,pickling=True)
theory = cc.theory

# we need an Nl curve
planck_file = "input/planck_nlkk.dat"
lp,nlplanck = np.loadtxt(planck_file,usecols=[0,1],unpack=True)

# Initialize a lensing forecast object with the theory object
LF = gcov.LensForecast(theory)

# get Clkk interpolated on your choice of ells
ells = np.arange(2,6000,1)
clkk = theory.gCl('kk',ells)

# Plot it
pl = io.Plotter(scaleY='log')
pl.add(ells,clkk)
pl.add(lp,nlplanck,ls="-.")
pl._ax.set_ylim(5e-10,1e-5)
pl.done("output/clsn.png")

# Load Clkk and Nl curve into forecasting object
LF.loadGenericCls("kk",ells,clkk,lp,nlplanck)

# Define the edges of your measured Clkk bins
ellBinEdges = np.arange(8,400,20)

fsky = 0.65
specType = "kk"

# get S/N and error bar in each bin
sn,errs = LF.sn(ellBinEdges,fsky,specType)

print sn

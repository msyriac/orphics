from orphics.theory.cosmology import Cosmology
import orphics.tools.cmb as cmb
import orphics.tools.io as io
import numpy as np


cc = Cosmology(lmax=3000,pickling=True)
ells = np.arange(2,3000,1)
bb = cc.theory.lCl("BB",ells)

uk_arcmin = 1.25*np.sqrt(600./350.) #*np.sqrt(2.)

TCMB = 2.7255e6
wn = cmb.white_noise_with_atm_func(ells,uk_arcmin,lknee=0,alpha=0,dimensionless=True,TCMB=TCMB)

ellwhere = 1000.
print((wn[np.logical_and(ells>998,ells<1002)]*ellwhere*(ellwhere+1.)/2./np.pi*TCMB**2.))


pl = io.Plotter(scaleY='log',scaleX='log')
pl.add(ells,bb*ells**2.*TCMB**2./2./np.pi)
pl.add(ells,wn*ells**2.*TCMB**2./2./np.pi,ls="--")
pl.done("wnoise.png")

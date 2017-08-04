import orphics.tools.io as io
import orphics.tools.cmb as cmb
from orphics.theory.cosmology import Cosmology
import numpy as np

cc = Cosmology(lmax=3000,pickling=True)

TCMB = 2.7255e6
ells = np.arange(2,3000,1)
cltt = cc.theory.lCl("TT",ells)*ells*(ells+1.)/2./np.pi*TCMB**2.

fwhm = 7.
rms_noise = 40.


nls = cmb.noise_func(ells,fwhm,rms_noise)*ells*(ells+1.)/2./np.pi

pl = io.Plotter(scaleY='log')
pl.add(ells,cltt)
pl.add(ells,nls)
pl._ax.set_xlim(0,2500)
pl._ax.set_ylim(1.e-7,1e7)

pl.done("nls.png")

 # noise curve planck
 # fisher with planck noise 40 uk 7'
 



    

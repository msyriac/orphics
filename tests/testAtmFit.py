
import numpy as np
import orphics.tools.cmb as cmb
import orphics.tools.io as io


fwhm = 0. #1.4 # beam convolved noise
rms_noise = 35.0
lknee = 2000.
alpha = -4.7

ell_fit = 5000.
lknee_guess = 500.
alpha_guess=-1.0


ells = np.arange(100.,8000.,30)
nls = cmb.noise_func(ells,fwhm,rms_noise,lknee,alpha)#*cmb.gauss_beam(ells,fwhm)**2.
nls += nls*np.random.normal(0.,0.05,size=nls.size)

pl = io.Plotter(scaleY='log')
pl.add(ells,nls*ells**2.)

noise_guess,lknee_fit,alpha_fit = cmb.fit_noise_power(ells,nls,ell_fit,lknee_guess,alpha_guess)
fit_nls = cmb.noise_func(ells,fwhm,noise_guess,lknee_fit,alpha_fit)#*cmb.gauss_beam(ells,fwhm)**2.
    
pl.add(ells,fit_nls*ells**2.,ls="--")


pl.done("noisetest.png")

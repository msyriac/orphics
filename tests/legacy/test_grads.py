from __future__ import print_function
from orphics import maps,io,cosmology
from enlib import enmap
import numpy as np
import os,sys



cc = cosmology.Cosmology(lmax=6000,pickling=True)

ells = np.arange(2,6000,1)

ucltt = cc.theory.uCl('TT',ells)
lcltt = cc.theory.lCl('TT',ells)

lmaxes = np.arange(10,6000,1)

def integral(fcls,lmax):
    ls = ells.copy()[np.logical_and(ells>1,ells<lmax)]
    cls = fcls.copy()[np.logical_and(ells>1,ells<lmax)]
    integ = (ls**3)*cls/2/np.pi

    return np.sqrt(np.trapz(integ,ls))

lgs = []
ugs = []
for lmax in lmaxes:


    lensed = integral(lcltt,lmax)
    unlensed = integral(ucltt,lmax)
    lgs.append(lensed)
    ugs.append(unlensed)
    print(lmax)

pl = io.Plotter(xscale='log',yscale='log')
pl.add(lmaxes,lgs,ls="--")
pl.add(lmaxes,ugs)
pl.done()


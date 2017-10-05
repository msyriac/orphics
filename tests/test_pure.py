from enlib import enmap
import orphics.analysis.flatMaps as fmaps
import orphics.analysis.pure as pure
import orphics.tools.io as io
import orphics.tools.stats as stats
from orphics.theory.cosmology import Cosmology
import numpy as np
import os, sys

out_dir = os.environ['WWW']+"plots/pureTest_"
cc = Cosmology(lmax=6000,pickling=True,dimensionless=False)
theory = cc.theory
deg = 20.
px = 1.0
shape, wcs = enmap.rect_geometry(deg*60.,px,pol=True)
pa = fmaps.PatchArray(shape,wcs,cc=cc,orphics_is_dimensionless=False)
ulensed = pa.get_unlensed_cmb()
kappa = pa.get_grf_kappa()
cmb = pa.get_lensed(ulensed,order=5)

# io.highResPlot2d(cmb[0],out_dir+"t.png")
# io.highResPlot2d(cmb[1],out_dir+"q.png")
# io.highResPlot2d(cmb[2],out_dir+"u.png")

modlmap = enmap.modlmap(shape,wcs)
fc = enmap.FourierCalc(shape,wcs)
lbin_edges = np.arange(200,6000,40)
lbinner = stats.bin2D(modlmap,lbin_edges)

def plot_powers(cmb,suffix,power=None,w2=1.):

    if power is None:
        power,lteb1,lteb2 = fc.power2d(cmb,pixel_units=False,skip_cross=True)
    power /= w2
    cents,dtt = lbinner.bin(power[0,0])
    cents,dee = lbinner.bin(power[1,1])
    cents,dbb = lbinner.bin(power[2,2])


    pl = io.Plotter(scaleY='log')
    ellrange = np.arange(200,6000,1)
    cltt = theory.lCl('TT',ellrange)
    clee = theory.lCl('EE',ellrange)
    clbb = theory.lCl('BB',ellrange)
    pl.add(ellrange,cltt*ellrange**2.,color="k")
    pl.add(ellrange,clee*ellrange**2.,color="k")
    pl.add(ellrange,clbb*ellrange**2.,color="k")
    pl.add(cents,dtt*cents**2.)
    pl.add(cents,dee*cents**2.)
    pl.add(cents,dbb*cents**2.)
    pl.done(out_dir+"powers_"+suffix+".png")


plot_powers(cmb,suffix="periodic",w2=1.)    
taper,w2 = fmaps.get_taper(shape,taper_percent = 12.0,pad_percent = 3.0,weight=None)
plot_powers(cmb*taper,suffix="tapered",w2=w2)    

print "Pure..."
windict = pure.initializeDerivativesWindowfuntions(taper,px*np.pi/180./60.)
lxMap,lyMap,modLMap,angLMap,lx,ly = fmaps.get_ft_attributes_enmap(shape,wcs)
fT, fE, fB = pure.TQUtoPureTEB(cmb[0],cmb[1],cmb[2],modlmap,angLMap,windowDict=windict,method='pure')
power = np.zeros((3,3,shape[-2],shape[-1]))
power[0,0,:,:] = fc.f2power(fT,fT)
power[1,1,:,:] = fc.f2power(fE,fE)
power[2,2,:,:] = fc.f2power(fB,fB)
plot_powers(None,suffix="pure",power=power,w2=w2)    
                 

from enlib import enmap
#import orphics.m as fmaps
import orphics.maps as maps
import orphics.io as io
import orphics.stats as stats
from orphics import cosmology
import numpy as np
import os, sys

out_dir = os.environ['WWW']+"plots/pureTest_"
#cc = Cosmology(lmax=6000,pickling=True,dimensionless=False)
theory = cosmology.default_theory()
deg = 20.
px = 1.0
shape, wcs = maps.rect_geometry(width_deg=deg,px_res_arcmin=px,pol=True)
#pa = maps.PatchArray(shape,wcs,theory=theory,orphics_is_dimensionless=False,lmax=6000)
#ulensed = pa.get_unlensed_cmb()
#kappa = pa.get_grf_kappa()
#cmb = pa.get_lensed(ulensed,order=5)

ells = np.arange(0,6000,1)
ps = cosmology.enmap_power_from_orphics_theory(theory,lmax=6000,ells=ells,lensed=True,dimensionless=False,orphics_dimensionless=False)
mgen = maps.MapGen(shape,wcs,ps)
cmb = mgen.get_map()

# io.highResPlot2d(cmb[0],out_dir+"t.png")
# io.highResPlot2d(cmb[1],out_dir+"q.png")
# io.highResPlot2d(cmb[2],out_dir+"u.png")

modlmap = enmap.modlmap(shape,wcs)
fc = maps.FourierCalc(shape,wcs)
lbin_edges = np.arange(200,6000,40)
lbinner = stats.bin2D(modlmap,lbin_edges)

def plot_powers(cmb,suffix,power=None,w2=1.):

    if power is None:
        power,lteb1,lteb2 = fc.power2d(cmb,pixel_units=False,skip_cross=True)
    power /= w2
    cents,dtt = lbinner.bin(power[0,0])
    cents,dee = lbinner.bin(power[1,1])
    cents,dbb = lbinner.bin(power[2,2])


    pl = io.Plotter(xlabel='l',ylabel='D',yscale='log')
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
taper,w2 = maps.get_taper(shape,taper_percent = 12.0,pad_percent = 3.0,weight=None)
plot_powers(cmb*taper,suffix="tapered",w2=w2)    

print("Pure...")
#windict = pure.init_deriv_window(taper,px*np.pi/180./60.)
windict = maps.init_deriv_window(taper,px*np.pi/180./60.)
lxMap,lyMap,modLMap,angLMap,lx,ly = maps.get_ft_attributes(shape,wcs)
#cmb *= taper
#fT, fE, fB =  pure.iqu_to_pure_lteb(cmb[0],cmb[1],cmb[2],modlmap,angLMap,windowDict=windict,method='pure')
#fT, fE, fB =  maps.iqu_to_pure_lteb(cmb[0]*taper,cmb[1]*taper,cmb[2]*taper,modlmap,angLMap,windowDict=windict,method='pure')
fT, fE, fB =  maps.iqu_to_pure_lteb(cmb[0],cmb[1],cmb[2],modlmap,angLMap,windowDict=windict,method='pure')

church = maps.Purify(shape,wcs,taper)
fT,fE,fB = church.lteb_from_iqu(cmb,method='pure',flip_q=False,iau=False)

power = np.zeros((3,3,shape[-2],shape[-1]))
power[0,0,:,:] = fc.f2power(fT,fT)
power[1,1,:,:] = fc.f2power(fE,fE)
power[2,2,:,:] = fc.f2power(fB,fB)
plot_powers(None,suffix="pure",power=power,w2=w2)    
                 

